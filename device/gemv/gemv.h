// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode gemv: one ring placement of a phase, for 1 to R verify rows, as the persistent decode kernel's warps
// compose it. The producer warp streams the SM's units of the placement through the ring (whole 16 KiB tiles,
// evict-first); the eight consumer warps each own one m16 slab of every 128-row tile, decode their 16 bytes per lane
// straight into a-fragments and run `mma.sync` against the staged activation rows; every finished output tile goes
// to an epilogue.
//
// Orientation: the weights are the mma's A operand (m16, rows of the output) and the activation rows its B operand
// (n8, one column per verify row). A weight tile's 128 rows fill eight m16 fragments with nothing padded, the baked
// a-fragment order needs no `ldmatrix`, and rows are bought in n8 tiles, so 1 to 32 rows cost 1 to 4 mmas per k16
// step; the other way round would pad every row count to 16 and waste half the tensor core at 8 and 24 rows.
//
// Arithmetic: exactly the order of order.h (the model's `Sum::Weights`). Within a unit the warp's mma chains run
// over the unit's k16 steps in ascending k; the e4m3 block scale (bf16 widened exactly) folds each chain into the
// SM's fp32 partial with one fma; an output tile held by several SMs is finished by its last holder, which adds the
// earlier holders' partials in SM order, then its own.
#pragma once

#include "gemv/layout.h"
#include "gemv/order.h"
#include "model.h"
#include "prims/fragment.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/numeric.h"
#include "prims/order.h"
#include "prims/reduce.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "prims/tile.h"
#include "prims/transcendental.h"
#include "schedule/geometry.h"

namespace engine::device::gemv {

/** Consumer warps: one per m16 slab of a 128-row tile. */
inline constexpr int consumer_warps = static_cast<int>(tile_rows / schedule::mma_rows);
inline constexpr int consumer_threads = consumer_warps * 32;
/** Producer threads: one warp, each lane arriving on its own copies of a stage. */
inline constexpr int producer_threads = 32;

static_assert(consumer_warps == static_cast<int>(schedule::decode_roles.consumer_warpgroups * schedule::warpgroup_warps));
static_assert(schedule::decode_roles.producer_warps * 32 == producer_threads);

/** A ring stage: one weight tile, 16-byte chunks. */
inline constexpr int stage_chunks = static_cast<int>(schedule::tile_bytes / 16);

/** The ring the decode gemv streams through: each fill is the producer warp's 32 lanes, each drain one consumer warp. */
template<int stages>
using Ring = prims::StageRing<stages, static_cast<unsigned>(producer_threads), static_cast<unsigned>(consumer_warps)>;

using layout::ActivationTile;

/** The fp32 accumulators of one warp's m16 slab for every n8 tile of rows. */
template<int row_tiles>
struct Slab {
	prims::Accumulator tile[row_tiles];
};

/** A warp's slab of one output tile for every tensor the unit interleaves. */
template<int tensors, int row_tiles>
struct Totals {
	Slab<row_tiles> tensor[tensors];
};

/** Everything a gemv instance is: the placement, the rows it serves, its ring and the consumers' barrier. */
template<Matrix matrix_, int row_tiles_, int stages_, class Barrier_>
struct Shape {
	static constexpr Matrix matrix = matrix_;
	static constexpr int row_tiles = row_tiles_;
	static constexpr int rows = 8 * row_tiles;
	static constexpr int tensors = static_cast<int>(matrix.tensors);
	static constexpr int k_tiles = static_cast<int>(matrix.k_tiles);
	/** 16-byte chunks of one activation row: the row stride of the activation buffer. */
	static constexpr int row_chunks = static_cast<int>(k_extent(matrix) / 8);
	using Ring = gemv::Ring<stages_>;
	using Activations = ActivationTile<matrix.format, row_tiles>;
	/** A named barrier over the consumer warps; the composing kernel picks its id. */
	using Barrier = Barrier_;
	/** fp32 values one SM's fixup slot holds: its partial of one output tile for every tensor and row. */
	static constexpr int fixup_floats = tensors * rows * static_cast<int>(tile_rows);

	static_assert(row_tiles >= 1 && row_tiles <= 5, "one to five n8 tiles of rows: U5 sweeps through 40");
	static_assert(tensors == 1 || tensors == 2);
	static_assert(Barrier::threads == consumer_threads, "the activation barrier is the consumer warps'");
	static_assert(prims::fragment::conflict_free<Activations>(), "the activation tile serves ldmatrix without bank conflicts");
	static_assert(fixups_fit(matrix));
};

/** Where one gemv reads and writes, apart from the ring. */
struct Operands {
	/** bf16 activation rows, `Shape::rows` of `k_extent` each, as 16-byte chunks. */
	prims::Chunk const* activations;
	/** Per tensor: the bf16 block scales of the placement, one row of `k_tiles` per output tile (e4m3 only). */
	unsigned short const* scales[2];
	/** The two activation tiles the consumers stage into, alternating per unit. */
	prims::SharedAddress activation_tiles[2];
	/** Every SM's fixup slot, `Shape::fixup_floats` each. */
	float* fixups;
	/** Every SM's fixup flag: the epoch its slot's contribution was published at. */
	unsigned long long* fixup_flags;
	/** This use of the placement: flags only grow, so a slot written for an older use is never mistaken. */
	unsigned long long epoch;
	unsigned sm;
};

// ---------------------------------------------------------------------------------------------------------------------
// The producer.

/**
 * The producer warp streams the units of the placement SM `sm` holds, every tile of each, into the ring: a pointer
 * bump and 32 16-byte copies per lane per stage. `run` is the phase's first chunk in the image.
 */
template<class S>
__attribute__((device, always_inline)) inline void produce(typename S::Ring const& ring, typename S::Ring::Cursor& cursor,
                                                           prims::SharedAddress stages, prims::Chunk const* run, unsigned sm) {
	using namespace engine::device::prims;
	Range const mine = held(S::matrix, sm);
	unsigned const lane_index = lane();
	unsigned const tiles = static_cast<unsigned>(mine.length) * static_cast<unsigned>(S::tensors);
	WeightStream<producer_threads> stream = {
	    .next = run + mine.begin * static_cast<Count>(S::tensors) * static_cast<Count>(stage_chunks) + lane_index,
	    .policy = policy<Eviction::First>(),
	};
	for (unsigned tile = 0; tile < tiles; ++tile) {
		ring.await_free(cursor, block_backoff);
		SharedAddress const stage = stages.plus(cursor.stage * static_cast<unsigned>(schedule::tile_bytes));
#pragma unroll 8
		for (unsigned copy = 0; copy < static_cast<unsigned>(stage_chunks / producer_threads); ++copy) {
			stream.issue(stage.plus((copy * producer_threads + lane_index) * 16U));
		}
		ring.filled_on_copies(cursor);
		cursor = cursor.next();
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// One warp's tile.

/**
 * One warp's m16 slab of an e4m3 tile against every row: two k16 steps per 16-byte chunk, four chunks, the next
 * chunk's weights and activation fragments loaded while the current one decodes and multiplies. `slab` accumulates
 * the chain (from +0 for a fresh unit) over the tile's 8 k16 steps in ascending k.
 */
template<int row_tiles, class Activations>
__attribute__((device, always_inline)) inline void warp_tile_e4m3(prims::SharedAddress tile, prims::SharedAddress activations,
                                                                  unsigned slab_index, Slab<row_tiles>& slab) {
	using namespace engine::device::prims;

	struct Step {
		Chunk weights;
		Matrices4 rows[row_tiles];
	};

	unsigned const lane_index = lane();
	double_buffered<4>(
	    [&](int pair) {
		    Step loaded{};
		    loaded.weights = load_shared(tile.plus(layout::weight_offset(slab_index, static_cast<unsigned>(pair), lane_index)));
#pragma unroll
		    for (int n = 0; n < row_tiles; ++n) {
			    int const offset = layout::activation_offset<Activations>(8 * n, pair, lane_index);
			    loaded.rows[n] = ldmatrix_x4<fragment::Transpose::Plain>(activations.plus(static_cast<unsigned>(offset)));
		    }
		    return loaded;
	    },
	    [&](int, Step const& step) {
		    AFragments2 const a = decode_chunk(step.weights);
#pragma unroll
		    for (int k16 = 0; k16 < 2; ++k16) {
#pragma unroll
			    for (int n = 0; n < row_tiles; ++n) {
				    BFragment const b = {.reg = {step.rows[n].reg[2 * k16], step.rows[n].reg[2 * k16 + 1]}};
				    mma(slab.tile[n], a.step[k16], b);
			    }
		    }
	    });
}

/**
 * One warp's m16 slab of a bf16 tile (128 rows x 64 k): one k16 step per 16-byte chunk, four chunks, two at a time
 * against one `ldmatrix.x4` per n8 tile. `slab` continues the SM's chain over the output tile's k16 steps.
 */
template<int row_tiles, class Activations>
__attribute__((device, always_inline)) inline void warp_tile_bf16(prims::SharedAddress tile, prims::SharedAddress activations,
                                                                  unsigned slab_index, Slab<row_tiles>& slab) {
	using namespace engine::device::prims;

	struct Step {
		Chunk weights[2];
		Matrices4 rows[row_tiles];
	};

	unsigned const lane_index = lane();
	double_buffered<2>(
	    [&](int pair) {
		    Step loaded{};
		    loaded.weights[0] = load_shared(tile.plus(layout::weight_offset(slab_index, static_cast<unsigned>(2 * pair), lane_index)));
		    loaded.weights[1] = load_shared(tile.plus(layout::weight_offset(slab_index, static_cast<unsigned>(2 * pair + 1), lane_index)));
#pragma unroll
		    for (int n = 0; n < row_tiles; ++n) {
			    int const offset = layout::activation_offset<Activations>(8 * n, pair, lane_index);
			    loaded.rows[n] = ldmatrix_x4<fragment::Transpose::Plain>(activations.plus(static_cast<unsigned>(offset)));
		    }
		    return loaded;
	    },
	    [&](int, Step const& step) {
#pragma unroll
		    for (int k16 = 0; k16 < 2; ++k16) {
			    AFragment const a = {
			        .reg = {step.weights[k16].word[0], step.weights[k16].word[1], step.weights[k16].word[2], step.weights[k16].word[3]}};
#pragma unroll
			    for (int n = 0; n < row_tiles; ++n) {
				    BFragment const b = {.reg = {step.rows[n].reg[2 * k16], step.rows[n].reg[2 * k16 + 1]}};
				    mma(slab.tile[n], a, b);
			    }
		    }
	    });
}

template<int row_tiles>
__attribute__((device, always_inline)) inline void clear(Slab<row_tiles>& slab) {
#pragma unroll
	for (int n = 0; n < row_tiles; ++n) {
#pragma unroll
		for (int reg = 0; reg < 4; ++reg) {
			slab.tile[n].reg[reg] = 0.0F;
		}
	}
}

/** partial = fma(chain, scale, partial), element by element: the block scale applied in fp32, rounded once. */
template<int row_tiles>
__attribute__((device, always_inline)) inline void fold(Slab<row_tiles>& partial, Slab<row_tiles> const& chain, float scale) {
#pragma unroll
	for (int n = 0; n < row_tiles; ++n) {
#pragma unroll
		for (int reg = 0; reg < 4; ++reg) {
			partial.tile[n].reg[reg] = prims::math::fma(chain.tile[n].reg[reg], scale, partial.tile[n].reg[reg]);
		}
	}
}

using layout::Element;
using layout::element;

/** Where an element sits in a slot of [tensor][row][128 outputs] fp32 values. */
template<class S>
[[nodiscard]] constexpr auto slot_index(int tensor, Element at) -> int {
	return layout::slot_index(tensor, S::rows, at);
}

// ---------------------------------------------------------------------------------------------------------------------
// The consumers.

template<class S>
__attribute__((device, always_inline)) inline void stage_activations(Operands const& operands, prims::SharedAddress tile, unsigned k_tile,
                                                                     unsigned consumer) {
	using namespace engine::device::prims;
	constexpr unsigned row_chunks = static_cast<unsigned>(S::Activations::chunks_per_row);
	constexpr unsigned chunks = row_chunks * static_cast<unsigned>(S::rows);
	auto const reused = policy<Eviction::Last>();
	for (unsigned index = consumer; index < chunks; index += consumer_threads) {
		unsigned const row = index / row_chunks;
		unsigned const chunk = index % row_chunks;
		Chunk const* source = operands.activations + static_cast<Count>(row) * static_cast<Count>(S::row_chunks) +
		                      static_cast<Count>(k_tile) * row_chunks + chunk;
		copy_async<Prefetch::None>(tile.plus(static_cast<unsigned>(S::Activations::offset(static_cast<int>(row), static_cast<int>(chunk)))),
		                           source, reused);
	}
	commit_copies();
}

template<class S>
__attribute__((device, always_inline)) inline void write_slot(float* slot, Totals<S::tensors, S::row_tiles> const& totals,
                                                              unsigned slab_index, unsigned lane_index) {
	using namespace engine::device::prims;
	auto const reused = policy<Eviction::Last>();
#pragma unroll
	for (int t = 0; t < S::tensors; ++t) {
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
#pragma unroll
			for (int reg = 0; reg < 4; ++reg) {
				auto* word = reinterpret_cast<unsigned*>(slot + slot_index<S>(t, element(slab_index, lane_index, n, reg)));
				store_word(word, __builtin_bit_cast(unsigned, totals.tensor[t].tile[n].reg[reg]), reused);
			}
		}
	}
}

/** Reads a slot back; with `Fold::Add` adds it into `totals` (totals + slot), else replaces them. */
enum class Fold : unsigned char { Replace, Add };

template<class S, Fold how>
__attribute__((device, always_inline)) inline void read_slot(float const* slot, Totals<S::tensors, S::row_tiles>& totals,
                                                             unsigned slab_index, unsigned lane_index) {
	using namespace engine::device::prims;
	auto const reused = policy<Eviction::Last>();
#pragma unroll
	for (int t = 0; t < S::tensors; ++t) {
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
#pragma unroll
			for (int reg = 0; reg < 4; ++reg) {
				auto const* word = reinterpret_cast<unsigned const*>(slot + slot_index<S>(t, element(slab_index, lane_index, n, reg)));
				float const value = __builtin_bit_cast(float, load_word(word, reused));
				float& total = totals.tensor[t].tile[n].reg[reg];
				if constexpr (how == Fold::Add) {
					total = total + value;
				} else {
					total = value;
				}
			}
		}
	}
}

/**
 * The consumer warps' side of one placement on SM `operands.sm`: for each unit held, stage its activations (the next
 * unit's copy overlapping this one's math), take each of its tiles from the ring, chain and fold; hand every output
 * tile the SM finishes to `epilogue(tile, totals, slab)`, which every consumer warp calls at the same point.
 *
 * Cross-SM: a tile this SM only contributes to (its last) is written to its fixup slot and released at gpu scope with
 * the epoch; a tile it finishes after others (its first) is stashed in its own slot, read back at the end of the
 * share, then its contribution is written and released before the wait for the earlier holders, so no SM's release
 * waits on another's. The slot is reused by the next placement the SM contributes to: a composing kernel must not
 * start that write before every finisher of this one has read it (the phase dependency counters order it).
 *
 * The activations must be ready (the producing phase complete) before the call.
 */
template<class S, class Epilogue>
__attribute__((device, always_inline)) inline void consume(typename S::Ring const& ring, typename S::Ring::Cursor& cursor,
                                                           prims::SharedAddress stages, Operands const& operands, unsigned consumer_warp,
                                                           Epilogue& epilogue) {
	using namespace engine::device::prims;
	constexpr Matrix matrix = S::matrix;
	using Barrier = typename S::Barrier;
	Range const mine = held(matrix, operands.sm);
	if (mine.length == 0) {
		return;
	}
	unsigned const lane_index = lane();
	unsigned const consumer = consumer_warp * 32U + lane_index;
	float* const own_slot = operands.fixups + static_cast<Count>(operands.sm) * static_cast<Count>(S::fixup_floats);
	auto const once = policy<Eviction::First>();

	Totals<S::tensors, S::row_tiles> partial{};
	unsigned const first_relative = static_cast<unsigned>(mine.begin - matrix.first);
	unsigned const units = static_cast<unsigned>(mine.length);
	unsigned const first_tile = first_relative / static_cast<unsigned>(S::k_tiles);
	unsigned const last_tile = (first_relative + units - 1) / static_cast<unsigned>(S::k_tiles);
	bool const stash_first = holding(matrix, first_tile, operands.sm) == Holding::Finishes;
	bool const contribute_last = holding(matrix, last_tile, operands.sm) == Holding::Contributes;

	stage_activations<S>(operands, operands.activation_tiles[0], first_relative % static_cast<unsigned>(S::k_tiles), consumer);
	for (unsigned index = 0; index < units; ++index) {
		unsigned const relative = first_relative + index;
		unsigned const tile = relative / static_cast<unsigned>(S::k_tiles);
		unsigned const k_tile = relative % static_cast<unsigned>(S::k_tiles);
		float scales[S::tensors] = {};
		if constexpr (matrix.format == model::Format::E4m3) {
#pragma unroll
			for (int t = 0; t < S::tensors; ++t) {
				unsigned const at = tile * static_cast<unsigned>(S::k_tiles) + k_tile;
				auto const* words = reinterpret_cast<unsigned const*>(operands.scales[t]);
				unsigned const word = load_word(words + at / 2U, once);
				scales[t] = widen(Bf16{.bits = static_cast<unsigned short>((word >> (16U * (at % 2U))) & 0xffffU)});
			}
		}
		if (k_tile == 0 || index == 0) {
#pragma unroll
			for (int t = 0; t < S::tensors; ++t) {
				clear(partial.tensor[t]);
			}
		}
		// This unit's activations have landed for every consumer, and every consumer is done with the previous unit's
		// tile, which the next copy overwrites.
		wait_copies<0>();
		Barrier::sync();
		if (index + 1 < units) {
			stage_activations<S>(operands, (index % 2U == 0 ? operands.activation_tiles[1] : operands.activation_tiles[0]),
			                     (relative + 1) % static_cast<unsigned>(S::k_tiles), consumer);
		}
		SharedAddress const activations = index % 2U == 0 ? operands.activation_tiles[0] : operands.activation_tiles[1];
#pragma unroll
		for (int t = 0; t < S::tensors; ++t) {
			ring.await_filled(cursor, block_backoff);
			SharedAddress const stage = stages.plus(cursor.stage * static_cast<unsigned>(schedule::tile_bytes));
			if constexpr (matrix.format == model::Format::E4m3) {
				Slab<S::row_tiles> chain{};
				clear(chain);
				warp_tile_e4m3<S::row_tiles, typename S::Activations>(stage, activations, consumer_warp, chain);
				fold(partial.tensor[t], chain, scales[t]);
			} else {
				warp_tile_bf16<S::row_tiles, typename S::Activations>(stage, activations, consumer_warp, partial.tensor[t]);
			}
			sync_warp();
			if (lane_index == 0) {
				ring.drained(cursor);
			}
			cursor = cursor.next();
		}
		bool const tile_ends = k_tile + 1 == static_cast<unsigned>(S::k_tiles);
		if (tile_ends && holding(matrix, tile, operands.sm) == Holding::Whole) {
			epilogue(tile, partial, consumer_warp);
		} else if (tile_ends && tile == first_tile && stash_first) {
			write_slot<S>(own_slot, partial, consumer_warp, lane_index);
		}
	}

	// The end of the share: the stash comes back to registers before the slot takes this SM's contribution, which is
	// released before waiting on anyone, so the releases never chain from SM to SM.
	Totals<S::tensors, S::row_tiles> finished{};
	if (stash_first) {
		read_slot<S, Fold::Replace>(own_slot, finished, consumer_warp, lane_index);
	}
	if (contribute_last) {
		write_slot<S>(own_slot, partial, consumer_warp, lane_index);
		Barrier::sync();
		if (consumer == 0) {
			store_release<Scope::Gpu>(operands.fixup_flags + operands.sm, operands.epoch);
		}
	}
	if (stash_first) {
		Range const holders = tile_holders(matrix, first_tile);
		Totals<S::tensors, S::row_tiles> total{};
		for (unsigned holder_sm = static_cast<unsigned>(holders.begin); holder_sm < operands.sm; ++holder_sm) {
			EpochReader<Scope::Gpu>{.flag = operands.fixup_flags + holder_sm}.await(operands.epoch, remote_backoff);
			float const* slot = operands.fixups + static_cast<Count>(holder_sm) * static_cast<Count>(S::fixup_floats);
			if (holder_sm == holders.begin) {
				read_slot<S, Fold::Replace>(slot, total, consumer_warp, lane_index);
			} else {
				read_slot<S, Fold::Add>(slot, total, consumer_warp, lane_index);
			}
		}
#pragma unroll
		for (int t = 0; t < S::tensors; ++t) {
#pragma unroll
			for (int n = 0; n < S::row_tiles; ++n) {
#pragma unroll
				for (int reg = 0; reg < 4; ++reg) {
					total.tensor[t].tile[n].reg[reg] = total.tensor[t].tile[n].reg[reg] + finished.tensor[t].tile[n].reg[reg];
				}
			}
		}
		epilogue(first_tile, total, consumer_warp);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Epilogues.

/**
 * Stores each finished output tile's fp32 sums as [tile][tensor][row][128 outputs]: the layout one cross-card release
 * carries, and what the self-checks compare.
 */
template<class S>
struct StoreTiles {
	float* out;

	__attribute__((device, always_inline)) void operator()(unsigned tile, Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab_index) const {
		using namespace engine::device::prims;
		write_slot<S>(out + static_cast<Count>(tile) * static_cast<Count>(S::fixup_floats), totals, slab_index, lane());
	}
};

/**
 * silu(gate) * up for one lane's n8 accumulators of gate and up: both fp32, silu by the engine's fixed routine, the
 * product rounded once to bf16 (the model's `MlpProduct`). Lane 4g + t holds outputs g and g + 8 of rows 2t and
 * 2t + 1; it swaps with its partner across the quad (g ^ 1) so each lane emits two adjacent outputs of one row: the
 * even quad row 2t, the odd quad row 2t + 1. `place(reg)` names the element of accumulator register `reg`;
 * `emit(at, pair)` gets the pair's row and first output.
 */
template<class Place, class Emit>
__attribute__((device, always_inline)) inline void product_pairs(prims::Accumulator const& gate, prims::Accumulator const& up,
                                                                 Place const& place, Emit const& emit) {
	using namespace engine::device::prims;
	bool const even = ((lane() / 4U) % 2U) == 0;
	float product[4] = {};
#pragma unroll
	for (int reg = 0; reg < 4; ++reg) {
		product[reg] = model::mlp_product(gate.reg[reg], up.reg[reg]);
	}
	// Registers: 0 (g, 2t), 1 (g, 2t + 1), 2 (g + 8, 2t), 3 (g + 8, 2t + 1).
#pragma unroll
	for (int half = 0; half < 2; ++half) {
		float const give = product[static_cast<Count>(2 * half + (even ? 1 : 0))];
		float const keep = product[static_cast<Count>(2 * half + (even ? 0 : 1))];
		float const got = __builtin_bit_cast(float, exchange(__builtin_bit_cast(unsigned, give), 4U));
		Bf16x2 const pair = even ? pack_bf16x2(keep, got) : pack_bf16x2(got, keep);
		Element const at = place(2 * half + (even ? 0 : 1));
		emit(Element{.output = at.output - (even ? 0 : 1), .row = at.row}, pair);
	}
}

/** Interleaved gate_proj (tensor 0) and up_proj (tensor 1) into down_proj's bf16 activation rows [row][`width`]. */
template<class S, int width>
struct GateUpProduct {
	/** bf16 rows of `width`, as 32-bit words of two adjacent outputs. */
	unsigned* out;

	static_assert(S::tensors == 2, "the product needs gate and up interleaved");
	static_assert(width % 2 == 0);

	/** The word of the pair whose first element is `at`, in output tile `tile`. */
	[[nodiscard]] static constexpr auto word_index(unsigned tile, Element at) -> int {
		int const column = static_cast<int>(tile) * static_cast<int>(tile_rows) + at.output;
		return (at.row * width + column) / 2;
	}

	__attribute__((device, always_inline)) void operator()(unsigned tile, Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab_index) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const reused = policy<Eviction::Last>();
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
			product_pairs(
			    totals.tensor[0].tile[n], totals.tensor[1].tile[n],
			    [&](int reg) {
				    return element(slab_index, lane_index, n, reg);
			    },
			    [&](Element at, Bf16x2 pair) {
				    store_word(out + word_index(tile, at), pair.bits, reused);
			    });
		}
	}
};

/**
 * The residual's update for one output tile, run by one warp once both cards' sums of the tile are in: per row,
 * residual + (card 0 + card 1) (fp32 addition is commutative, so either card computes the same bits), and the tile's
 * sum of squares of the new residual in lanes of 128 (lane l folds outputs 4l .. 4l + 3 by fma from +0, then the
 * butterfly), the model's `HiddenSquares` per tile. `own` and `peer` are [row][128] fp32; `residual` is
 * [row][hidden]; `squares` gets one fp32 per row of the tile.
 */
template<int rows>
__attribute__((device, always_inline)) inline void add_to_residual(prims::Chunk const* own, prims::Chunk const* peer,
                                                                   prims::Chunk* residual, unsigned tile, float* squares) {
	using namespace engine::device::prims;
	constexpr unsigned tile_chunks = static_cast<unsigned>(tile_rows / 4);
	constexpr unsigned residual_chunks = static_cast<unsigned>(model::hidden / 4);
	auto const reused = policy<Eviction::Last>();
	auto const once = policy<Eviction::First>();
	unsigned const lane_index = lane();
	for (unsigned row = 0; row < static_cast<unsigned>(rows); ++row) {
		Chunk const mine = load(own + static_cast<Count>(row) * tile_chunks + lane_index, once);
		Chunk const theirs = load(peer + static_cast<Count>(row) * tile_chunks + lane_index, once);
		Chunk* const at = residual + static_cast<Count>(row) * residual_chunks + static_cast<Count>(tile) * tile_chunks + lane_index;
		Chunk const before = load(at, reused);
		Chunk after{};
		float sum = 0.0F;
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			float const card_sum = __builtin_bit_cast(float, mine.word[word]) + __builtin_bit_cast(float, theirs.word[word]);
			float const updated = __builtin_bit_cast(float, before.word[word]) + card_sum;
			after.word[word] = __builtin_bit_cast(unsigned, updated);
			sum = prims::math::fma(updated, updated, sum);
		}
		store(at, after, reused);
		float const total = warp_sum(sum);
		if (lane_index == 0) {
			store_word(reinterpret_cast<unsigned*>(squares + row), __builtin_bit_cast(unsigned, total), reused);
		}
	}
}

}

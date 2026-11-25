// SPDX-License-Identifier: AGPL-3.0-only
//
// The prefill gemm: bf16 activation rows x e4m3 (or bf16) weights -> fp32, for a chunk of prompt rows. A block takes
// work items of one 128-row output tile by one block of 32 rows; a producer warp copies each unit's weight tiles and
// its 32 x k-span activation tile (xor-swizzled) into a stage of the ring, and eight consumer warps each own 32
// outputs x 16 rows: two m16 slabs against two n8 tiles, the weights decoded in registers straight into a-fragments
// and the activations read by `ldmatrix`.
//
// Arithmetic: the decode's own order (order.h). The k tiles of an output tile are walked in ascending k and cut into
// the segments the decode's static split gives the SMs; within a segment each unit's chain is folded into the
// segment's partial exactly as the decode folds it, and at each segment's end the partial joins the total left to
// right. So for the same weights and activation row the prefill computes the decode's bits.
#pragma once

#include "gemv/gemv.h"
#include "gemv/order.h"
#include "model.h"
#include "prims/budget.h"
#include "prims/fragment.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "schedule/geometry.h"

namespace engine::device::gemv::prefill {

using layout::prefill::block_rows;
using layout::prefill::element;
using layout::prefill::output_groups;
using layout::prefill::row_groups;
using layout::prefill::warp_row_tiles;
using layout::prefill::warp_slabs;

static_assert(output_groups * row_groups == consumer_warps, "the eight consumer warps tile 128 outputs by 32 rows");

/** One stage: a unit's weight tiles, then its activation tile. */
template<Matrix matrix>
struct Stage {
	static constexpr int weight_bytes = static_cast<int>(matrix.tensors * schedule::tile_bytes);
	using Activations = layout::prefill::ActivationTile<matrix.format>;
	static constexpr int bytes = weight_bytes + Activations::bytes;
	static_assert(prims::fragment::conflict_free<Activations>());
	static_assert(weight_bytes % 128 == 0);
};

/** Stages that fit a block beside the ring's barriers: as many as fit, at most four. */
template<Matrix matrix>
inline constexpr int stages = [] {
	int const room = (prims::sm86.max_shared_per_block - 1024) / Stage<matrix>::bytes;
	return room < 4 ? room : 4;
}();

template<Matrix matrix>
using Ring = prims::StageRing<stages<matrix>, static_cast<unsigned>(producer_threads), static_cast<unsigned>(consumer_warps)>;

/** A consumer warp's accumulators: per tensor, two m16 slabs by two n8 tiles. */
template<int tensors>
struct Warp {
	Slab<warp_row_tiles> tensor[tensors][warp_slabs];
};

/** What a work item reads and writes. */
struct Operands {
	/** bf16 activation rows of the chunk, `k_extent` each, as 16-byte chunks. */
	prims::Chunk const* activations;
	/** The matrix's tiles in stream order: unit u at u * tensors tiles. */
	prims::Chunk const* tiles;
	/** Per tensor: the bf16 block scales, one row of `k_tiles` per output tile (e4m3 only). */
	unsigned short const* scales[2];
};

/** The work items: every output tile by every row block, tile-major so a tile's weights are reused while in l2. */
struct Item {
	unsigned tile;
	unsigned rows;
};

[[nodiscard]] constexpr auto item(unsigned index, unsigned row_blocks) -> Item {
	return {.tile = index / row_blocks, .rows = index % row_blocks};
}

/**
 * The producer: for each work item of this block (items block, block + blocks, ...), every unit of the output tile in
 * ascending k, its weight tiles evict-first, its activation tile evict-last (the row block is read once per output
 * tile).
 */
template<Matrix matrix>
__attribute__((device, always_inline)) inline void produce(Ring<matrix> const& ring, typename Ring<matrix>::Cursor& cursor,
                                                           prims::SharedAddress stages, Operands const& operands, unsigned row_blocks,
                                                           unsigned blocks) {
	using namespace engine::device::prims;
	using Activations = typename Stage<matrix>::Activations;
	constexpr unsigned tile_chunks = static_cast<unsigned>(stage_chunks);
	constexpr unsigned weight_chunks = tile_chunks * static_cast<unsigned>(matrix.tensors);
	constexpr unsigned row_chunks = static_cast<unsigned>(Activations::chunks_per_row);
	constexpr unsigned activation_row_chunks = static_cast<unsigned>(k_extent(matrix) / 8);
	unsigned const lane_index = lane();
	auto const once = policy<Eviction::First>();
	auto const reused = policy<Eviction::Last>();
	unsigned const items = static_cast<unsigned>(matrix.output_tiles) * row_blocks;
	for (unsigned index = block(); index < items; index += blocks) {
		Item const at = item(index, row_blocks);
		for (unsigned k_tile = 0; k_tile < static_cast<unsigned>(matrix.k_tiles); ++k_tile) {
			ring.await_free(cursor, block_backoff);
			SharedAddress const stage = stages.plus(cursor.stage * static_cast<unsigned>(Stage<matrix>::bytes));
			unsigned const unit = at.tile * static_cast<unsigned>(matrix.k_tiles) + k_tile;
			Chunk const* weights = operands.tiles + static_cast<Count>(unit) * weight_chunks;
			for (unsigned chunk = lane_index; chunk < weight_chunks; chunk += producer_threads) {
				copy_async<Prefetch::Chunk256>(stage.plus(chunk * 16U), weights + chunk, once);
			}
			SharedAddress const activations = stage.plus(static_cast<unsigned>(Stage<matrix>::weight_bytes));
			for (unsigned chunk = lane_index; chunk < row_chunks * block_rows; chunk += producer_threads) {
				unsigned const row = chunk / row_chunks;
				unsigned const column = chunk % row_chunks;
				Chunk const* source = operands.activations + static_cast<Count>(at.rows * block_rows + row) * activation_row_chunks +
				                      static_cast<Count>(k_tile) * row_chunks + column;
				copy_async<Prefetch::None>(
				    activations.plus(static_cast<unsigned>(Activations::offset(static_cast<int>(row), static_cast<int>(column)))), source,
				    reused);
			}
			ring.filled_on_copies(cursor);
			cursor = cursor.next();
		}
	}
}

/** One consumer warp's two slabs of one weight tile against its two n8 tiles of rows, chained in ascending k. */
template<Matrix matrix>
__attribute__((device, always_inline)) inline void warp_tile(prims::SharedAddress weights, prims::SharedAddress activations, unsigned warp,
                                                             Slab<warp_row_tiles> (&chain)[warp_slabs]) {
	using namespace engine::device::prims;
	using Activations = typename Stage<matrix>::Activations;
	unsigned const lane_index = lane();
	if constexpr (matrix.format == model::Format::E4m3) {
		struct Step {
			Chunk weights[warp_slabs];
			Matrices4 rows[warp_row_tiles];
		};

		double_buffered<4>(
		    [&](int pair) {
			    Step loaded{};
#pragma unroll
			    for (int slab = 0; slab < warp_slabs; ++slab) {
				    loaded.weights[slab] = load_shared(weights.plus(
				        layout::weight_offset(layout::prefill::tile_slab(warp, slab), static_cast<unsigned>(pair), lane_index)));
			    }
#pragma unroll
			    for (int n = 0; n < warp_row_tiles; ++n) {
				    int const offset = layout::activation_offset<Activations>(layout::prefill::first_row(warp, n), pair, lane_index);
				    loaded.rows[n] = ldmatrix_x4<fragment::Transpose::Plain>(activations.plus(static_cast<unsigned>(offset)));
			    }
			    return loaded;
		    },
		    [&](int, Step const& step) {
#pragma unroll
			    for (int slab = 0; slab < warp_slabs; ++slab) {
				    AFragments2 const a = decode_chunk(step.weights[slab]);
#pragma unroll
				    for (int k16 = 0; k16 < 2; ++k16) {
#pragma unroll
					    for (int n = 0; n < warp_row_tiles; ++n) {
						    BFragment const b = {.reg = {step.rows[n].reg[2 * k16], step.rows[n].reg[2 * k16 + 1]}};
						    mma(chain[slab].tile[n], a.step[k16], b);
					    }
				    }
			    }
		    });
	} else {
		struct Step {
			Chunk weights[warp_slabs][2];
			Matrices4 rows[warp_row_tiles];
		};

		double_buffered<2>(
		    [&](int pair) {
			    Step loaded{};
#pragma unroll
			    for (int slab = 0; slab < warp_slabs; ++slab) {
#pragma unroll
				    for (int k16 = 0; k16 < 2; ++k16) {
					    loaded.weights[slab][k16] = load_shared(weights.plus(layout::weight_offset(
					        layout::prefill::tile_slab(warp, slab), static_cast<unsigned>(2 * pair + k16), lane_index)));
				    }
			    }
#pragma unroll
			    for (int n = 0; n < warp_row_tiles; ++n) {
				    int const offset = layout::activation_offset<Activations>(layout::prefill::first_row(warp, n), pair, lane_index);
				    loaded.rows[n] = ldmatrix_x4<fragment::Transpose::Plain>(activations.plus(static_cast<unsigned>(offset)));
			    }
			    return loaded;
		    },
		    [&](int, Step const& step) {
#pragma unroll
			    for (int slab = 0; slab < warp_slabs; ++slab) {
#pragma unroll
				    for (int k16 = 0; k16 < 2; ++k16) {
					    Chunk const& w = step.weights[slab][k16];
					    AFragment const a = {.reg = {w.word[0], w.word[1], w.word[2], w.word[3]}};
#pragma unroll
					    for (int n = 0; n < warp_row_tiles; ++n) {
						    BFragment const b = {.reg = {step.rows[n].reg[2 * k16], step.rows[n].reg[2 * k16 + 1]}};
						    mma(chain[slab].tile[n], a, b);
					    }
				    }
			    }
		    });
	}
}

/**
 * The consumers: for each work item, the output tile's units in ascending k, each unit's chain folded into the current
 * segment's partial; at each segment's end (where the decode's split hands the tile to the next SM) the partial joins
 * the total. `epilogue(item, totals, warp)` takes each finished item; every consumer warp calls it at the same point.
 */
template<Matrix matrix, class Epilogue>
__attribute__((device, always_inline)) inline void consume(Ring<matrix> const& ring, typename Ring<matrix>::Cursor& cursor,
                                                           prims::SharedAddress stages, Operands const& operands, unsigned row_blocks,
                                                           unsigned blocks, unsigned warp, Epilogue& epilogue) {
	using namespace engine::device::prims;
	constexpr int tensors = static_cast<int>(matrix.tensors);
	unsigned const lane_index = lane();
	auto const once = policy<Eviction::First>();
	unsigned const items = static_cast<unsigned>(matrix.output_tiles) * row_blocks;
	for (unsigned index = block(); index < items; index += blocks) {
		Item const at = item(index, row_blocks);
		Range const holders = tile_holders(matrix, at.tile);
		Warp<tensors> total{};
		Warp<tensors> partial{};
		unsigned holder_sm = static_cast<unsigned>(holders.begin);
		Range part = segment(matrix, at.tile, holder_sm);
#pragma unroll
		for (int t = 0; t < tensors; ++t) {
#pragma unroll
			for (int slab = 0; slab < warp_slabs; ++slab) {
				clear(partial.tensor[t][slab]);
			}
		}
		for (unsigned k_tile = 0; k_tile < static_cast<unsigned>(matrix.k_tiles); ++k_tile) {
			float scales[tensors] = {};
			if constexpr (matrix.format == model::Format::E4m3) {
#pragma unroll
				for (int t = 0; t < tensors; ++t) {
					unsigned const scale_at = at.tile * static_cast<unsigned>(matrix.k_tiles) + k_tile;
					auto const* words = reinterpret_cast<unsigned const*>(operands.scales[t]);
					unsigned const word = load_word(words + scale_at / 2U, once);
					scales[t] = widen(Bf16{.bits = static_cast<unsigned short>((word >> (16U * (scale_at % 2U))) & 0xffffU)});
				}
			}
			ring.await_filled(cursor, block_backoff);
			SharedAddress const stage = stages.plus(cursor.stage * static_cast<unsigned>(Stage<matrix>::bytes));
			SharedAddress const activations = stage.plus(static_cast<unsigned>(Stage<matrix>::weight_bytes));
#pragma unroll
			for (int t = 0; t < tensors; ++t) {
				SharedAddress const weights = stage.plus(static_cast<unsigned>(t) * static_cast<unsigned>(schedule::tile_bytes));
				if constexpr (matrix.format == model::Format::E4m3) {
					Slab<warp_row_tiles> chain[warp_slabs] = {};
					warp_tile<matrix>(weights, activations, warp, chain);
#pragma unroll
					for (int slab = 0; slab < warp_slabs; ++slab) {
						fold(partial.tensor[t][slab], chain[slab], scales[t]);
					}
				} else {
					warp_tile<matrix>(weights, activations, warp, partial.tensor[t]);
				}
			}
			sync_warp();
			if (lane_index == 0) {
				ring.drained(cursor);
			}
			cursor = cursor.next();
			// The decode's SM boundary inside this tile: the segment's partial joins the total, left to right.
			if (k_tile + 1 == static_cast<unsigned>(part.begin + part.length)) {
				bool const first_segment = holder_sm == static_cast<unsigned>(holders.begin);
#pragma unroll
				for (int t = 0; t < tensors; ++t) {
#pragma unroll
					for (int slab = 0; slab < warp_slabs; ++slab) {
#pragma unroll
						for (int n = 0; n < warp_row_tiles; ++n) {
#pragma unroll
							for (int reg = 0; reg < 4; ++reg) {
								float& sum = total.tensor[t][slab].tile[n].reg[reg];
								float const added = partial.tensor[t][slab].tile[n].reg[reg];
								sum = first_segment ? added : sum + added;
							}
						}
						clear(partial.tensor[t][slab]);
					}
				}
				++holder_sm;
				if (holder_sm < static_cast<unsigned>(holders.begin + holders.length)) {
					part = segment(matrix, at.tile, holder_sm);
				}
			}
		}
		epilogue(at, total, warp);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Epilogues.

/** fp32 sums as rows of the whole output: [row][output_tiles * 128], tensor by tensor ([tensor][row][...]). */
template<Matrix matrix>
struct StoreRows {
	float* out;
	unsigned rows;

	/** Where element `e` of tensor `tensor` of item `at` lands. */
	[[nodiscard]] static constexpr auto index(Item at, Element e, int tensor, unsigned total_rows) -> unsigned {
		constexpr unsigned width = static_cast<unsigned>(matrix.output_tiles * tile_rows);
		unsigned const row = at.rows * block_rows + static_cast<unsigned>(e.row);
		unsigned const column = at.tile * static_cast<unsigned>(tile_rows) + static_cast<unsigned>(e.output);
		return (static_cast<unsigned>(tensor) * total_rows + row) * width + column;
	}

	__attribute__((device, always_inline)) void operator()(Item at, Warp<static_cast<int>(matrix.tensors)> const& totals,
	                                                       unsigned warp) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const reused = policy<Eviction::Last>();
#pragma unroll
		for (int t = 0; t < static_cast<int>(matrix.tensors); ++t) {
#pragma unroll
			for (int slab = 0; slab < warp_slabs; ++slab) {
#pragma unroll
				for (int n = 0; n < warp_row_tiles; ++n) {
#pragma unroll
					for (int reg = 0; reg < 4; ++reg) {
						auto* word = reinterpret_cast<unsigned*>(out + index(at, element(warp, lane_index, slab, n, reg), t, rows));
						store_word(word, __builtin_bit_cast(unsigned, totals.tensor[t][slab].tile[n].reg[reg]), reused);
					}
				}
			}
		}
	}
};

/** silu(gate) * up rounded once to bf16, as down_proj's activation rows [row][width] (the decode's `GateUpProduct`). */
template<Matrix matrix, int width>
struct GateUpRows {
	unsigned* out;

	/** The word of the pair whose first element is `e`, in item `at`. */
	[[nodiscard]] static constexpr auto word_index(Item at, Element e) -> unsigned {
		unsigned const row = at.rows * block_rows + static_cast<unsigned>(e.row);
		unsigned const column = at.tile * static_cast<unsigned>(tile_rows) + static_cast<unsigned>(e.output);
		return (row * static_cast<unsigned>(width) + column) / 2U;
	}

	static_assert(matrix.tensors == 2);
	static_assert(width % 2 == 0);

	__attribute__((device, always_inline)) void operator()(Item at, Warp<2> const& totals, unsigned warp) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const reused = policy<Eviction::Last>();
#pragma unroll
		for (int slab = 0; slab < warp_slabs; ++slab) {
#pragma unroll
			for (int n = 0; n < warp_row_tiles; ++n) {
				product_pairs(
				    totals.tensor[0][slab].tile[n], totals.tensor[1][slab].tile[n],
				    [&](int reg) {
					    return element(warp, lane_index, slab, n, reg);
				    },
				    [&](Element e, Bf16x2 pair) {
					    store_word(out + word_index(at, e), pair.bits, reused);
				    });
			}
		}
	}
};

}

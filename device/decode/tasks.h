// SPDX-License-Identifier: AGPL-3.0-only
//
// The tasks of the walk as every SM of one card runs them between two grid barriers: the embedding gather, the rms
// norms, the weight matmuls with their epilogues and the cross-card exchange, the deltanet and attention cores and
// attention's merge, the exact ranking of the lm_head's rows, the verdict and the drafts, and the mtp head's inputs.
// Each composes the building blocks (gemv, attn, deltanet, sampling) under the decode kernel's roles and its one shared
// layout; the arithmetic is theirs and the model description's. Device code only.
#pragma once

#include "attn/core.h"
#include "attn/merge.h"
#include "decode/machine.h"
#include "deltanet/core.h"
#include "gemv/gemv.h"
#include "gemv/order.h"
#include "model.h"
#include "prims/mma.h"
#include "prims/numeric.h"
#include "prims/reduce.h"
#include "sampling/draw.h"

namespace engine::device::decode {

using prims::Bf16;
using prims::Policy;
using schedule::Tensors;

inline constexpr Tensors const& tensor = schedule::tensors;
inline constexpr Count hidden_tiles = model::hidden / schedule::tile_rows;

/** The address of side read `index` of image phase `image_phase`. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto side(Machine const& machine, Count image_phase, Count index) -> Address {
	return machine.image(image_table.phase[image_phase].side[index]);
}

// ---------------------------------------------------------------------------------------------------------------------
// Rows of the hidden state, one warp per (row, 128-feature tile), four features a lane: the residual's and every
// norm's mapping, so a tile's squares are the model's `hidden_tile` sum.

/** One lane's four features of a bf16 row, widened. */
template<Eviction eviction>
__attribute__((device, always_inline)) inline void load_four(Bf16 const* row, Count first, Policy<eviction> policy, float (&values)[4]) {
	/* SAFETY: bf16 rows start 256-byte aligned and `first` is a multiple of four, so both words are whole. */
	auto const* words = reinterpret_cast<unsigned const*>(row + first);
	unsigned const low = prims::load_word(words, policy);
	unsigned const high = prims::load_word(words + 1, policy);
	values[0] = prims::widen(Bf16{.bits = static_cast<unsigned short>(low & 0xffffU)});
	values[1] = prims::widen(Bf16{.bits = static_cast<unsigned short>(low >> 16U)});
	values[2] = prims::widen(Bf16{.bits = static_cast<unsigned short>(high & 0xffffU)});
	values[3] = prims::widen(Bf16{.bits = static_cast<unsigned short>(high >> 16U)});
}

/** The warp's sum of its lanes' four squares: the tile's `model::hidden_tile` sum. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto tile_squares(float const (&values)[4]) -> float {
	float partial = 0.0F;
#pragma unroll
	for (int index = 0; index < 4; ++index) {
		partial = model::fixed::fma(values[index], values[index], partial);
	}
	return prims::warp_sum(partial);
}

/** A row's sum of squares from its 40 tile sums ([tile][max_rows]), folded left to right from tile 0. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto row_squares(float const* squares, unsigned row) -> float {
	auto const reused = prims::policy<Eviction::Last>();
	/* SAFETY: squares are fp32 words of the workspace. */
	auto const* words = reinterpret_cast<unsigned const*>(squares);
	float total = __builtin_bit_cast(float, prims::load_word(words + row, reused));
	for (Count tile = 1; tile < hidden_tiles; ++tile) {
		total = total + __builtin_bit_cast(float, prims::load_word(words + tile * max_rows + row, reused));
	}
	return total;
}

__attribute__((device, always_inline)) inline void store_floats(float* target, float const (&values)[4]) {
	Chunk const packed = {.word = {__builtin_bit_cast(unsigned, values[0]), __builtin_bit_cast(unsigned, values[1]),
	                               __builtin_bit_cast(unsigned, values[2]), __builtin_bit_cast(unsigned, values[3])}};
	/* SAFETY: fp32 rows start 256-byte aligned and every lane's four features start on a 16-byte boundary. */
	prims::store(reinterpret_cast<Chunk*>(target), packed, prims::policy<Eviction::Last>());
}

/** Four bf16 values rounded once and stored as two words. */
__attribute__((device, always_inline)) inline void store_four(Bf16* target, float const (&values)[4]) {
	auto const reused = prims::policy<Eviction::Last>();
	/* SAFETY: bf16 rows start 256-byte aligned and every lane's four features start on an 8-byte boundary. */
	auto* words = reinterpret_cast<unsigned*>(target);
	prims::store_word(words, prims::pack_bf16x2(values[0], values[1]).bits, reused);
	prims::store_word(words + 1, prims::pack_bf16x2(values[2], values[3]).bits, reused);
}

__attribute__((device, always_inline)) inline void store_square(float* squares, Count tile, unsigned row, float total) {
	if (prims::lane() == 0) {
		/* SAFETY: squares are fp32 words of the workspace. */
		prims::store_word(reinterpret_cast<unsigned*>(squares + tile * max_rows + row), __builtin_bit_cast(unsigned, total),
		                  prims::policy<Eviction::Last>());
	}
}

/** The embedding row of `token`: bf16 [hidden] in the host image, read across pcie once. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto embedding_row(Machine const& machine, unsigned token) -> Bf16 const* {
	return at<Bf16 const>(machine.params.embedding) + Count{token} * model::hidden;
}

/** The verify rows' embeddings widened into the residual, with the tiles' squares; rows past the cycle's are zero. */
__attribute__((device)) inline void embed(Machine const& machine) {
	PassRows const rows = pass_rows(machine, {.kind = PassKind::Verify, .draft = 0});
	auto const once = prims::policy<Eviction::First>();
	float* const residual = machine.local<float>(workspace.residual);
	float* const squares = machine.local<float>(workspace.squares);
	Count const lane_first = Count{prims::lane()} * 4;
	for (unsigned item = global_warp(); item < max_rows * hidden_tiles; item += grid_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		float values[4] = {};
		if (row < rows.rows) {
			load_four(embedding_row(machine, verify_token(machine, rows.cycle, row)), tile * schedule::tile_rows + lane_first, once, values);
		}
		store_floats(residual + Count{row} * model::hidden + tile * schedule::tile_rows + lane_first, values);
		store_square(squares, tile, row, tile_squares(values));
	}
}

/** The prompt's last row as the lm_head's row 0, the other rows zero: prefill already took it through the final norm. */
__attribute__((device)) inline void prompt_row(Machine const& machine) {
	auto const reused = prims::policy<Eviction::Last>();
	constexpr Count row_chunks = model::hidden * sizeof(Bf16) / 16;
	auto const* prompt = machine.local<Chunk const>(workspace.prompt);
	auto* rows = machine.local<Chunk>(workspace.final_normed);
	unsigned const thread = global_warp() * 32U + prims::lane();
	for (Count chunk = thread; chunk < max_rows * row_chunks; chunk += Count{grid_warps} * 32) {
		prims::store(rows + chunk, chunk < row_chunks ? prims::load(prompt + chunk, reused) : Chunk{}, reused);
	}
}

/** Where a norm reads and writes. */
struct NormRoute {
	float const* residual;
	float const* squares;
	Bf16 const* weight;
	Bf16* out;
};

/** x * factor * (1 + w), rounded once, for every row the phase computes; rows past the pass's are zero. */
__attribute__((device)) inline void normalize(NormRoute const& route, unsigned live_rows, unsigned computed_rows) {
	auto const reused = prims::policy<Eviction::Last>();
	Count const lane_first = Count{prims::lane()} * 4;
	for (unsigned item = global_warp(); item < computed_rows * hidden_tiles; item += grid_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		Count const first = tile * schedule::tile_rows + lane_first;
		float values[4] = {};
		if (row < live_rows) {
			float const factor = model::rms_factor(row_squares(route.squares, row), model::hidden);
			float weights[4] = {};
			load_four(route.weight, first, reused, weights);
			/* SAFETY: the residual's rows are fp32 chunks of the workspace. */
			Chunk const x = prims::load(reinterpret_cast<Chunk const*>(route.residual + Count{row} * model::hidden + first), reused);
#pragma unroll
			for (int index = 0; index < 4; ++index) {
				values[index] = model::one_plus_scaled(__builtin_bit_cast(float, x.word[index]), factor, weights[index]);
			}
		}
		store_four(route.out + Count{row} * model::hidden + first, values);
	}
}

/** The norm a phase begins with: which residual of the pass, which weight, and which rows it fills. */
__attribute__((device)) inline void norm_task(Machine const& machine, Point const& point) {
	bool const trunk = point.pass.kind == PassKind::Verify;
	PassRows const rows = pass_rows(machine, point.pass);
	Bf16* out = machine.local<Bf16>(workspace.normed);
	Count weight = 0;
	switch (point.step.op) {
	case Op::DeltanetIn:
		weight = side_index(Op::DeltanetIn, tensor.input_norm, Part::Whole);
		break;
	case Op::AttentionIn:
		weight = side_index(Op::AttentionIn, tensor.input_norm, Part::Whole);
		break;
	case Op::MlpIn:
		weight = side_index(Op::MlpIn, tensor.post_norm, Part::Whole);
		break;
	case Op::Head:
		weight = side_index(Op::Head, tensor.final_norm, Part::Whole);
		out = machine.local<Bf16>(workspace.final_normed);
		break;
	case Op::DraftHead:
		weight = side_index(Op::DraftHead, tensor.mtp_norm, Part::Whole);
		out = machine.local<Bf16>(workspace.mtp_normed);
		break;
	default:
		// Not reached: the walk gives a norm task only to the phases `normalizes` names.
		expect(false);
		break;
	}
	NormRoute const route = {
	    .residual = machine.local<float const>(trunk ? workspace.residual : workspace.mtp_residual),
	    .squares = machine.local<float const>(trunk ? workspace.squares : workspace.mtp_squares),
	    .weight = at<Bf16 const>(side(machine, point.image_phase, weight)),
	    .out = out,
	};
	normalize(route, rows.rows, static_cast<unsigned>(point.row_tiles * model::row_tile));
}

// ---------------------------------------------------------------------------------------------------------------------
// Epilogues of the weight matmuls.

using gemv::layout::Element;

/**
 * Adjacent outputs of one row from a lane's n8 accumulator, as product_pairs pairs them: lane 4g + t swaps with its
 * partner across the quad (g ^ 1), so each lane emits outputs (2h, 2h + 1) of one verify row: the even quad row 2t,
 * the odd one row 2t + 1. `emit(first, a, b)` gets the pair's first element and both values.
 */
template<class Place, class Emit>
__attribute__((device, always_inline)) inline void adjacent(prims::Accumulator const& values, Place const& place, Emit const& emit) {
	bool const even = ((prims::lane() / 4U) % 2U) == 0;
#pragma unroll
	for (int half = 0; half < 2; ++half) {
		float const give = values.reg[2 * half + (even ? 1 : 0)];
		float const keep = values.reg[2 * half + (even ? 0 : 1)];
		float const got = __builtin_bit_cast(float, prims::exchange(__builtin_bit_cast(unsigned, give), 4U));
		Element const at_keep = place(2 * half + (even ? 0 : 1));
		Element const first = {.output = at_keep.output - (even ? 0 : 1), .row = at_keep.row};
		if (even) {
			emit(first, keep, got);
		} else {
			emit(first, got, keep);
		}
	}
}

/** Each finished tile as fp32 rows of `width`, from column `column`: projections that stay fp32, and logits. */
template<class S>
struct RowsF32 {
	float* out;
	Count width;
	Count column;

	__attribute__((device, always_inline)) void operator()(unsigned tile, gemv::Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab) const {
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const lane = prims::lane();
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
#pragma unroll
			for (int reg = 0; reg < 4; ++reg) {
				Element const at_element = gemv::element(slab, lane, n, reg);
				Count const index = Count(at_element.row) * width + column + Count{tile} * schedule::tile_rows + Count(at_element.output);
				/* SAFETY: fp32 rows of the workspace. */
				prims::store_word(reinterpret_cast<unsigned*>(out + index), __builtin_bit_cast(unsigned, totals.tensor[0].tile[n].reg[reg]), reused);
			}
		}
	}
};

/** Each finished tile rounded once to bf16 rows of `width`: in_proj_qkv's projections. */
template<class S>
struct RowsBf16 {
	unsigned* words;
	Count width;

	__attribute__((device, always_inline)) void operator()(unsigned tile, gemv::Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab) const {
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const lane = prims::lane();
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
			adjacent(
			    totals.tensor[0].tile[n],
			    [&](int reg) {
				    return gemv::element(slab, lane, n, reg);
			    },
			    [&](Element first, float a, float b) {
				    Count const index = (Count(first.row) * width + Count{tile} * schedule::tile_rows + Count(first.output)) / 2;
				    prims::store_word(words + index, prims::pack_bf16x2(a, b).bits, reused);
			    });
		}
	}
};

/**
 * in_proj_z's tile gates the core's normed output it lines up with: normed * silu(z), rounded once, into out_proj's
 * input rows (`model::Value::DeltanetGated`). z itself is never stored; the core ran before this phase.
 */
template<class S>
struct GatedCore {
	float const* core;
	unsigned* gated;

	__attribute__((device, always_inline)) void operator()(unsigned tile, gemv::Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab) const {
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const lane = prims::lane();
		constexpr Count width = deltanet::value_features;
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
			adjacent(
			    totals.tensor[0].tile[n],
			    [&](int reg) {
				    return gemv::element(slab, lane, n, reg);
			    },
			    [&](Element first, float z_first, float z_second) {
				    Count const index = Count(first.row) * width + Count{tile} * schedule::tile_rows + Count(first.output);
				    /* SAFETY: the normed rows are fp32 words of the workspace. */
				    auto const* normed = reinterpret_cast<unsigned const*>(core + index);
				    float const a = model::deltanet::gated(__builtin_bit_cast(float, prims::load_word(normed, reused)), z_first);
				    float const b = model::deltanet::gated(__builtin_bit_cast(float, prims::load_word(normed + 1, reused)), z_second);
				    prims::store_word(gated + index / 2, prims::pack_bf16x2(a, b).bits, reused);
			    });
		}
	}
};

/** One exchange of partial sums in flight: the receive slot both cards use, and its epoch. */
struct ExchangeSlot {
	Count slot;
	unsigned long long epoch;
};

[[nodiscard]] __attribute__((device, always_inline)) inline auto partial_tile_offset(ExchangeSlot exchange, unsigned tile) -> Count {
	return workspace.exchange.partials + (exchange.slot * schedule::exchange_tiles + tile) * schedule::partial_tile_bytes(budget);
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto partial_flag_offset(ExchangeSlot exchange, unsigned tile) -> Count {
	return workspace.exchange.partial_flags + (exchange.slot * schedule::exchange_tiles + tile) * schedule::flag_bytes;
}

/**
 * A finished output tile's fp32 partial: kept as this card's own and pushed whole into the peer's receive slot, then
 * released with the exchange's epoch, once per tile.
 */
template<class S>
struct PushPartial {
	Machine const* machine;
	ExchangeSlot exchange;

	__attribute__((device, always_inline)) void operator()(unsigned tile, gemv::Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab) const {
		static_assert(S::tensors == 1);
		float* const own = machine->local<float>(workspace.own_partials) + Count{tile} * static_cast<Count>(S::fixup_floats);
		gemv::write_slot<S>(own, totals, slab, prims::lane());
		ConsumerBarrier::sync();
		prims::PeerOutbox const outbox = {
		    .slots = machine->peer<Chunk>(partial_tile_offset(exchange, tile)),
		    .flag = {.flag = machine->peer<unsigned long long>(partial_flag_offset(exchange, tile))},
		};
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const thread = slab * 32U + prims::lane();
		constexpr unsigned chunks = static_cast<unsigned>(S::fixup_floats / 4);
		/* SAFETY: the own partial is fp32 chunks of the workspace. */
		auto const* own_chunks = reinterpret_cast<Chunk const*>(own);
		for (unsigned chunk = thread; chunk < chunks; chunk += static_cast<unsigned>(gemv::consumer_threads)) {
			outbox.push(chunk, prims::load(own_chunks + chunk, reused));
		}
		ConsumerBarrier::sync();
		if (thread == 0) {
			outbox.publish(exchange.epoch);
		}
	}
};

/** How the cards' sum of an exchange meets its residual. */
enum class Residual : unsigned char {
	/** residual + (card 0 + card 1): every layer's two additions. */
	Add,
	/** card 0 + card 1 starts the residual: the mtp fc's output. */
	Start,
};

/** The mtp fc's sum starts the mtp residual: card 0 + card 1, and the tile's squares as `add_to_residual` takes them. */
template<int rows>
__attribute__((device, always_inline)) inline void start_residual(Chunk const* own, Chunk const* peer, Chunk* residual, unsigned tile, float* squares) {
	constexpr unsigned tile_chunks = static_cast<unsigned>(schedule::tile_rows / 4);
	constexpr unsigned residual_chunks = static_cast<unsigned>(model::hidden / 4);
	auto const reused = prims::policy<Eviction::Last>();
	auto const once = prims::policy<Eviction::First>();
	unsigned const lane = prims::lane();
	for (unsigned row = 0; row < static_cast<unsigned>(rows); ++row) {
		Chunk const mine = prims::load(own + Count{row} * tile_chunks + lane, once);
		Chunk const theirs = prims::load(peer + Count{row} * tile_chunks + lane, once);
		Chunk sum{};
		float values[4] = {};
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			values[word] = __builtin_bit_cast(float, mine.word[word]) + __builtin_bit_cast(float, theirs.word[word]);
			sum.word[word] = __builtin_bit_cast(unsigned, values[word]);
		}
		prims::store(residual + Count{row} * residual_chunks + Count{tile} * tile_chunks + lane, sum, reused);
		float const total = tile_squares(values);
		if (lane == 0) {
			/* SAFETY: squares are fp32 words of the workspace. */
			prims::store_word(reinterpret_cast<unsigned*>(squares + row), __builtin_bit_cast(unsigned, total), reused);
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The weight matmuls.

/** One SM's progress through a weight phase's placements, by one role. */
struct GemvRun {
	GemvRing ring;
	prims::SharedAddress stages;
	prims::SharedAddress activation_tiles[2];
	Chunk const* run;
	GemvRing::Cursor cursor;
	unsigned long long epoch;
	unsigned warp;
	unsigned sm;
};

/** One ring placement: the producer streams the SM's units of it, the consumers compute and finish its tiles. */
template<class S, class Epilogue>
__attribute__((device, always_inline)) inline void placement(Machine const& machine, GemvRun& g, Count activations, Address scales_first,
                                                             Address scales_second, Epilogue& epilogue) {
	++g.epoch;
	if (Producer::holds(g.warp)) {
		gemv::produce<S>(g.ring, g.cursor, g.stages, g.run, g.sm);
	} else if (Consumers::holds(g.warp)) {
		gemv::Operands const operands = {
		    .activations = machine.local<Chunk const>(activations),
		    .scales = {at<unsigned short const>(scales_first), at<unsigned short const>(scales_second)},
		    .activation_tiles = {g.activation_tiles[0], g.activation_tiles[1]},
		    .fixups = machine.local<float>(workspace.exchange.fixups),
		    .fixup_flags = machine.local<unsigned long long>(workspace.exchange.fixup_flags),
		    .epoch = g.epoch,
		    .sm = g.sm,
		};
		gemv::consume<S>(g.ring, g.cursor, g.stages, operands, Consumers::rank(g.warp), epilogue);
	}
}

/**
 * After an exchange phase's matmul: for each output tile this SM finished, the peer's partial is awaited and both
 * cards' sums meet the residual, one consumer warp per tile.
 */
template<class S, Residual how>
__attribute__((device, always_inline)) inline void meet(Machine const& machine, GemvRun const& g, ExchangeSlot exchange, Count residual,
                                                        Count squares) {
	if (!Consumers::holds(g.warp)) {
		return;
	}
	constexpr gemv::Matrix matrix = S::matrix;
	gemv::Range const mine = gemv::held(matrix, g.sm);
	if (mine.length == 0) {
		return;
	}
	unsigned const first_tile = static_cast<unsigned>((mine.begin - matrix.first) / matrix.k_tiles);
	unsigned const last_tile = static_cast<unsigned>((mine.begin + mine.length - 1 - matrix.first) / matrix.k_tiles);
	unsigned finished = 0;
	for (unsigned tile = first_tile; tile <= last_tile; ++tile) {
		if (gemv::holding(matrix, tile, g.sm) == gemv::Holding::Contributes) {
			continue;
		}
		if (finished % static_cast<unsigned>(Consumers::count) == Consumers::rank(g.warp)) {
			prims::EpochReader<prims::Scope::System>{.flag = machine.local<unsigned long long const>(partial_flag_offset(exchange, tile))}.await(
			    exchange.epoch, prims::remote_backoff);
			auto const* own = machine.local<Chunk const>(workspace.own_partials) + Count{tile} * static_cast<Count>(S::fixup_floats / 4);
			auto const* peer = machine.local<Chunk const>(partial_tile_offset(exchange, tile));
			float* const tile_squares_out = machine.local<float>(squares) + Count{tile} * max_rows;
			if constexpr (how == Residual::Add) {
				gemv::add_to_residual<S::rows>(own, peer, machine.local<Chunk>(residual), tile, tile_squares_out);
			} else {
				start_residual<S::rows>(own, peer, machine.local<Chunk>(residual), tile, tile_squares_out);
			}
		}
		++finished;
	}
}

template<Op op, model::Stream stream, Count ring, int rows>
using Shape = gemv::Shape<gemv::matrix(first_phase(op, stream), ring), rows, ring_stages, ConsumerBarrier>;

static_assert(sizeof(gemv::Ring<ring_stages>::Storage) == sizeof(GemvShared::barriers));

/**
 * in_proj_a and in_proj_b on the mixer input, read whole: one mma chain per (16 outputs, 8 rows) over the whole hidden
 * in ascending k, on side warp 0 of the first twelve SMs while the qkv tiles stream. The chain's m16 rows are eight
 * value heads' a rows then the same heads' b rows, so each lane holds a and b of one head for two rows, which the
 * deltanet core reads as fp32 a then b per head.
 */
__attribute__((device)) inline void decay_inputs(Machine const& machine, Point const& point, unsigned warp) {
	constexpr unsigned head_slabs = static_cast<unsigned>(deltanet::value_heads / 8);
	constexpr unsigned chains = head_slabs * static_cast<unsigned>(row_tiles);
	static_assert(deltanet::value_heads % 8 == 0 && model::hidden % 16 == 0);
	unsigned const chain = prims::block();
	if (warp != static_cast<unsigned>(Side::first) || chain >= chains) {
		return;
	}
	unsigned const slab = chain / static_cast<unsigned>(row_tiles);
	unsigned const n = chain % static_cast<unsigned>(row_tiles);
	unsigned const lane = prims::lane();
	unsigned const g = lane / 4U;
	unsigned const t = lane % 4U;
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	constexpr Count row_words = model::hidden / 2;
	auto const* a = at<unsigned const>(side(machine, point.image_phase, side_index(Op::DeltanetIn, tensor.a, Part::Whole))) +
	                Count{8U * slab + g} * row_words;
	auto const* b = at<unsigned const>(side(machine, point.image_phase, side_index(Op::DeltanetIn, tensor.b, Part::Whole))) +
	                Count{8U * slab + g} * row_words;
	auto const* x = machine.local<unsigned const>(workspace.normed) + Count{8U * n + g} * row_words;
	prims::Accumulator sum = {.reg = {0.0F, 0.0F, 0.0F, 0.0F}};
	for (Count step = 0; step < model::hidden / 16; ++step) {
		Count const low = step * 8 + t;
		Count const high = low + 4;
		prims::AFragment const weights = {.reg = {prims::load_word(a + low, once), prims::load_word(b + low, once), prims::load_word(a + high, once),
		                                          prims::load_word(b + high, once)}};
		prims::BFragment const rows = {.reg = {prims::load_word(x + low, reused), prims::load_word(x + high, reused)}};
		prims::mma(sum, weights, rows);
	}
	auto* gates = machine.local<unsigned>(workspace.gates);
	Count const head = 8U * slab + g;
	Count const row = 8U * n + 2U * t;
	// Registers: 0 (a, row), 1 (a, row + 1), 2 (b, row), 3 (b, row + 1).
	for (Count reg = 0; reg < 4; ++reg) {
		Count const word = ((row + reg % 2) * deltanet::value_heads + head) * 2 + reg / 2;
		prims::store_word(gates + word, __builtin_bit_cast(unsigned, sum.reg[reg]), reused);
	}
}

/** The weight phase of `point`: its placements in stream order, each with its epilogue, then the exchange's sums. */
template<int rows>
__attribute__((device)) inline void gemv_task(Machine const& machine, Shared& shared, Epochs& epochs, Point const& point) {
	auto const& entry = image_table.phase[point.image_phase];
	unsigned const warp = prims::warp_in_block();
	GemvRun g = {
	    .ring = GemvRing::at(shared.gemv.barriers),
	    .stages = prims::shared_address(shared.gemv.stages),
	    .activation_tiles = {prims::shared_address(shared.gemv.activations[0]), prims::shared_address(shared.gemv.activations[1])},
	    .run = at<Chunk const>(machine.image(entry.ring[0])),
	    .cursor = GemvRing::Cursor::start(),
	    .epoch = epochs.fixups,
	    .warp = warp,
	    .sm = prims::block(),
	};
	if (prims::thread_in_block() == 0) {
		g.ring.init();
	}
	prims::sync_block();
	auto const scales = [&](Count ring, Count which) {
		return machine.image(entry.scales[ring][which]);
	};
	bool const trunk = point.pass.kind == PassKind::Verify;
	Count const residual = trunk ? workspace.residual : workspace.mtp_residual;
	Count const squares = trunk ? workspace.squares : workspace.mtp_squares;
	auto const next_exchange = [&] {
		++epochs.partials;
		return ExchangeSlot{.slot = exchange_slot(epochs.partials), .epoch = epochs.partials};
	};
	switch (point.step.op) {
	case Op::DeltanetIn: {
		using S = Shape<Op::DeltanetIn, model::Stream::Body, 0, static_cast<int>(row_tiles)>;
		RowsBf16<S> out = {.words = machine.local<unsigned>(workspace.qkv), .width = static_cast<Count>(deltanet::channels)};
		placement<S>(machine, g, workspace.normed, scales(0, 0), scales(0, 0), out);
		decay_inputs(machine, point, warp);
		break;
	}
	case Op::DeltanetZ: {
		using S = Shape<Op::DeltanetZ, model::Stream::Body, 0, static_cast<int>(row_tiles)>;
		GatedCore<S> out = {.core = machine.local<float const>(workspace.core), .gated = machine.local<unsigned>(workspace.gated)};
		placement<S>(machine, g, workspace.normed, scales(0, 0), scales(0, 0), out);
		break;
	}
	case Op::DeltanetOut: {
		using S = Shape<Op::DeltanetOut, model::Stream::Body, 0, static_cast<int>(row_tiles)>;
		ExchangeSlot const exchange = next_exchange();
		PushPartial<S> out = {.machine = &machine, .exchange = exchange};
		placement<S>(machine, g, workspace.gated, scales(0, 0), scales(0, 0), out);
		meet<S, Residual::Add>(machine, g, exchange, residual, squares);
		break;
	}
	case Op::AttentionIn: {
		auto const project = [&]<model::Stream stream>() {
			using Q = Shape<Op::AttentionIn, stream, 0, rows>;
			using K = Shape<Op::AttentionIn, stream, 1, rows>;
			using V = Shape<Op::AttentionIn, stream, 2, rows>;
			RowsF32<Q> query = {.out = machine.local<float>(workspace.query), .width = attn::query_row_floats, .column = 0};
			RowsF32<K> key = {.out = machine.local<float>(workspace.key), .width = attn::kv_row_floats, .column = 0};
			RowsF32<V> value = {.out = machine.local<float>(workspace.value), .width = attn::kv_row_floats, .column = 0};
			placement<Q>(machine, g, workspace.normed, scales(0, 0), scales(0, 0), query);
			placement<K>(machine, g, workspace.normed, scales(1, 0), scales(1, 0), key);
			placement<V>(machine, g, workspace.normed, scales(2, 0), scales(2, 0), value);
		};
		if (trunk) {
			project.template operator()<model::Stream::Body>();
		} else {
			project.template operator()<model::Stream::Mtp>();
		}
		break;
	}
	case Op::AttentionGate: {
		auto const project = [&]<model::Stream stream>() {
			using S = Shape<Op::AttentionGate, stream, 0, rows>;
			RowsF32<S> out = {.out = machine.local<float>(workspace.attention_gate), .width = attn::gate_row_floats, .column = 0};
			placement<S>(machine, g, workspace.normed, scales(0, 0), scales(0, 0), out);
		};
		if (trunk) {
			project.template operator()<model::Stream::Body>();
		} else {
			project.template operator()<model::Stream::Mtp>();
		}
		break;
	}
	case Op::AttentionOut: {
		ExchangeSlot const exchange = next_exchange();
		auto const project = [&]<model::Stream stream>() {
			using S = Shape<Op::AttentionOut, stream, 0, rows>;
			PushPartial<S> out = {.machine = &machine, .exchange = exchange};
			placement<S>(machine, g, workspace.attention_output, scales(0, 0), scales(0, 0), out);
			meet<S, Residual::Add>(machine, g, exchange, residual, squares);
		};
		if (trunk) {
			project.template operator()<model::Stream::Body>();
		} else {
			project.template operator()<model::Stream::Mtp>();
		}
		break;
	}
	case Op::MlpIn: {
		auto const project = [&]<model::Stream stream>() {
			using S = Shape<Op::MlpIn, stream, 0, rows>;
			gemv::GateUpProduct<S, static_cast<int>(model::ffn_per_card)> out = {.out = machine.local<unsigned>(workspace.product)};
			placement<S>(machine, g, workspace.normed, scales(0, 0), scales(0, 1), out);
		};
		if (trunk) {
			project.template operator()<model::Stream::Body>();
		} else {
			project.template operator()<model::Stream::Mtp>();
		}
		break;
	}
	case Op::MlpOut: {
		ExchangeSlot const exchange = next_exchange();
		auto const project = [&]<model::Stream stream>() {
			using S = Shape<Op::MlpOut, stream, 0, rows>;
			PushPartial<S> out = {.machine = &machine, .exchange = exchange};
			placement<S>(machine, g, workspace.product, scales(0, 0), scales(0, 0), out);
			meet<S, Residual::Add>(machine, g, exchange, residual, squares);
		};
		if (trunk) {
			project.template operator()<model::Stream::Body>();
		} else {
			project.template operator()<model::Stream::Mtp>();
		}
		break;
	}
	case Op::Head: {
		using Draft = Shape<Op::Head, model::Stream::Head, 0, static_cast<int>(row_tiles)>;
		using Other = Shape<Op::Head, model::Stream::Head, 1, static_cast<int>(row_tiles)>;
		RowsF32<Draft> draft = {.out = machine.local<float>(workspace.logits), .width = model::vocab_per_card, .column = 0};
		RowsF32<Other> other = {.out = machine.local<float>(workspace.logits), .width = model::vocab_per_card, .column = model::draft_vocab_per_card};
		placement<Draft>(machine, g, workspace.final_normed, 0, 0, draft);
		placement<Other>(machine, g, workspace.final_normed, 0, 0, other);
		break;
	}
	case Op::MtpIn: {
		using S = Shape<Op::MtpIn, model::Stream::Mtp, 0, rows>;
		ExchangeSlot const exchange = next_exchange();
		PushPartial<S> out = {.machine = &machine, .exchange = exchange};
		placement<S>(machine, g, workspace.mtp_input, 0, 0, out);
		meet<S, Residual::Start>(machine, g, exchange, workspace.mtp_residual, workspace.mtp_squares);
		break;
	}
	case Op::DraftHead: {
		using S = Shape<Op::DraftHead, model::Stream::Mtp, 0, rows>;
		RowsF32<S> out = {.out = machine.local<float>(workspace.logits), .width = model::vocab_per_card, .column = 0};
		placement<S>(machine, g, workspace.mtp_normed, 0, 0, out);
		break;
	}
	default:
		// Not reached: the walk gives a weight task only to the phases with ring reads.
		expect(false);
		break;
	}
	if (Producer::holds(warp)) {
		prims::wait_all_copies();
	}
	epochs.fixups = g.epoch;
}

// ---------------------------------------------------------------------------------------------------------------------
// The cores.

/** The kv layer of a full-attention phase: the trunk's in layer order, then the mtp head's. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto kv_layer(Point const& point) -> unsigned {
	if (point.pass.kind != PassKind::Verify) {
		return static_cast<unsigned>(model::kv_layers - 1);
	}
	unsigned ordinal = 0;
	for (Count layer = 0; layer < point.step.layer; ++layer) {
		ordinal += model::trunk_mixers[layer] == model::Mixer::FullAttention ? 1U : 0U;
	}
	return ordinal;
}

/** The core's jobs on this block: its assignment of the cycle's groups, held in shared memory. */
template<class Tile>
struct AssignedJobs {
	attn::CoreShared<Tile> const* shared;

	[[nodiscard]] __attribute__((device)) auto count() const -> unsigned {
		return shared->assignment.count;
	}

	[[nodiscard]] __attribute__((device)) auto at(unsigned index) const -> attn::Job {
		attn::Item const item = shared->assignment.item[index];
		attn::Cycle const& cycle = shared->cycle;
		return {.work = cycle.work[item.work],
		        .head = item.head,
		        .first_group = item.first_group,
		        .groups = item.groups,
		        .record_base = static_cast<unsigned>(attn::record_base(cycle, item.work, item.head))};
	}
};

/** Split-k attention over each slot's kv, the pass's own rows appended first, every group's record left for the merge. */
template<class Tile>
__attribute__((device)) inline void attention_core(Machine const& machine, attn::CoreShared<Tile>& shared, Point const& point) {
	attn::KvRing const ring = attn::kv_ring(shared);
	if (prims::thread_in_block() == 0) {
		shared.cycle = pass_rows(machine, point.pass).cycle;
		expect(attn::valid(shared.cycle, max_rows));
		ring.init();
		attn::Assignment& mine = shared.assignment;
		mine.count = 0;
		attn::for_each_item(shared.cycle, prims::block(), sms, [&mine](attn::Item item) {
			mine.item[mine.count] = item;
			++mine.count;
		});
	}
	prims::sync_block();
	Count const projection = point.image_phase - 1;
	attn::KvPool const kv = {
	    .blocks = machine.params.pool, .tables = machine.address(workspace.tables), .layer = kv_layer(point), .unused = 0};
	attn::Projections const rows = {
	    .query = machine.address(workspace.query),
	    .key = machine.address(workspace.key),
	    .value = machine.address(workspace.value),
	    .gate = machine.address(workspace.attention_gate),
	};
	attn::Records const records = {.output = machine.address(record_outputs), .running = machine.address(record_running)};
	AssignedJobs<Tile> const jobs = {.shared = &shared};
	unsigned const warp = prims::warp_in_block();
	using Core = attn::Core<Tile>;
	if (Producer::holds(warp)) {
		[[maybe_unused]] attn::Cursor const streamed = Core::produce(shared, jobs, ring, attn::Cursor::start(), kv);
		prims::wait_all_copies();
	} else if (Consumers::holds(warp)) {
		[[maybe_unused]] attn::Cursor const consumed = Core::template consume<attn::Sink::Records, attn::Fresh::Appended, attn::PvParts::Three>(
		    shared, jobs, ring, attn::Cursor::start(), records, rows, 0);
	} else {
		attn::Preparation const preparation = {
		    .query = side(machine, projection, side_index(Op::AttentionIn, tensor.q_norm, Part::Whole)),
		    .key = side(machine, projection, side_index(Op::AttentionIn, tensor.k_norm, Part::Whole)),
		    .rotations = machine.address(workspace.exchange.rope_table),
		};
		Core::template prepare<attn::Fresh::Appended>(shared, jobs, kv, rows, preparation);
	}
}

/** Every (row, query head) of the pass folds its groups' records, gated by sigmoid(gate): one warp each. */
__attribute__((device)) inline void attention_merge(Machine const& machine, Point const& point) {
	attn::Cycle const cycle = pass_rows(machine, point.pass).cycle;
	attn::Records const records = {.output = machine.address(record_outputs), .running = machine.address(record_running)};
	Count const units = attn::merge_units(cycle);
	for (Count unit = global_warp(); unit < units; unit += grid_warps) {
		attn::merge(records, machine.address(workspace.attention_gate), machine.address(workspace.attention_output), cycle, attn::merge_unit(cycle, unit));
	}
}

/** One deltanet item on this SM: its state streamed, its events run, its committed state stored, its rows normalized. */
__attribute__((device)) inline void run_item(Machine const& machine, deltanet::DecodeShared& shared, deltanet::Item const& item, Count image_phase) {
	deltanet::Weights const weights = {
	    .conv = {.value = at<Bf16 const>(side(machine, image_phase, side_index(Op::DeltanetCore, tensor.conv, Part::Whole)))},
	    .a_log = {.value = at<Bf16 const>(side(machine, image_phase, side_index(Op::DeltanetCore, tensor.a_log, Part::Whole)))},
	    .dt_bias = {.value = at<Bf16 const>(side(machine, image_phase, side_index(Op::DeltanetCore, tensor.dt_bias, Part::Whole)))},
	    .norm = {.value = at<Bf16 const>(side(machine, image_phase, side_index(Op::DeltanetCore, tensor.deltanet_norm, Part::Whole)))},
	};
	deltanet::Rows const vectors = {
	    .qkv = {.value = machine.local<Bf16 const>(workspace.qkv)},
	    .gates = {.value = machine.local<float const>(workspace.gates)},
	    .outputs = {.value = machine.local<float>(workspace.core)},
	};
	deltanet::StateRing const ring = deltanet::state_ring(shared);
	prims::Mbarrier const prepared = {.address = prims::shared_address(&shared.prepared)};
	if (prims::thread_in_block() == 0) {
		ring.init();
		prepared.init(static_cast<unsigned>(Side::count));
	}
	prims::sync_block();
	unsigned const warp = prims::warp_in_block();
	if (Producer::holds(warp)) {
		deltanet::produce_state(item, ring, shared.stages);
	} else if (Consumers::holds(warp)) {
		unsigned const rank = Consumers::rank(warp);
		prepared.wait(0, prims::block_backoff);
		deltanet::consume_state(item, ring, shared.stages, shared.events, shared.outputs, static_cast<int>(rank));
		ConsumerBarrier::sync();
		deltanet::normalize_rows(item, weights, vectors, shared.outputs, rank * 32U + prims::lane());
	} else {
		using Preparers = prims::NamedBarrier<2, Side::count>;
		deltanet::prepare_events<Preparers>(item, weights, vectors, shared.events, prepared,
		                                    prims::thread_in_block() - static_cast<unsigned>(Side::first * 32));
	}
}

/** The deltanet item of slot `slot` (its state and ring) for one layer and key head group. */
[[nodiscard]] __attribute__((device)) inline auto slot_item(Machine const& machine, unsigned slot, int layer, int key_head, int replays, int rows,
                                                       int first_row) -> deltanet::Item {
	expect(replays >= 0 && replays <= deltanet::max_replays);
	return {
	    .image = at<unsigned char>(machine.params.states) + Count{slot} * deltanet::slot_image_bytes,
	    .ring = machine.local<unsigned char>(workspace.exchange.replay_rings) + Count{slot} * deltanet::ring_bytes_per_slot,
	    .layer = layer,
	    .key_head = key_head,
	    .replays = replays,
	    .rows = rows,
	    .first_row = first_row,
	};
}

/** The gated delta rule over each live slot's state: one item per (slot, key head group), each on its own SM. */
__attribute__((device)) inline void deltanet_core(Machine const& machine, deltanet::DecodeShared& shared, Point const& point) {
	Descriptor const& descriptor = machine.descriptor();
	unsigned const item_index = prims::block();
	unsigned const live = item_index / static_cast<unsigned>(deltanet::key_heads);
	if (live >= descriptor.live) {
		return;
	}
	attn::SlotWork const work = pass_rows(machine, point.pass).cycle.work[live];
	SlotRun const& run = descriptor.slots[live];
	deltanet::Item const item = slot_item(machine, run.slot, deltanet::deltanet_layer(point.step.layer),
	                                      static_cast<int>(item_index % static_cast<unsigned>(deltanet::key_heads)), static_cast<int>(run.replays),
	                                      static_cast<int>(work.rows), static_cast<int>(work.first_row));
	run_item(machine, shared, item, point.image_phase);
}

/**
 * Materializes a slot's pending replays: its stored deltanet state takes the `replays` accepted drafts waiting in its
 * ring, layer by layer, so a snapshot taken next holds the state after every token but the slot's last. The core
 * itself does it: the last pending entry is laid out as the one row of a cycle that replays the others first, and the
 * core commits after its first row; the row's output is not read.
 */
__attribute__((device)) inline void materialize(Machine const& machine, Shared& shared, Grid grid, unsigned slot, unsigned replays) {
	expect(replays >= 1 && replays <= static_cast<unsigned>(deltanet::max_replays));
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	auto* const ring = machine.local<unsigned char>(workspace.exchange.replay_rings) + Count{slot} * deltanet::ring_bytes_per_slot;
	auto* const qkv = machine.local<unsigned char>(workspace.qkv);
	auto* const gates = machine.local<unsigned>(workspace.gates);
	for (int layer = 0; layer < deltanet::layers; ++layer) {
		// The entry's raw qkv chunks go to their card channels in row 0; its gate words (a of each head, then b) to
		// row 0's a, b pairs.
		unsigned const key_head = prims::block();
		if (key_head < static_cast<unsigned>(deltanet::key_heads)) {
			Chunk const* const entry = deltanet::chunk_at(ring, deltanet::ring_offset(layer, static_cast<int>(key_head), static_cast<int>(replays) - 1));
			for (unsigned u = prims::thread_in_block(); u < static_cast<unsigned>(deltanet::group_chunks); u += static_cast<unsigned>(decode_budget.threads)) {
				Count const channel = static_cast<Count>(deltanet::card_channel(static_cast<int>(key_head), static_cast<int>(u) * deltanet::bf16_per_chunk));
				prims::store(deltanet::chunk_at(qkv, channel * sizeof(Bf16)), prims::load(entry + u, once), reused);
			}
			if (prims::thread_in_block() < static_cast<unsigned>(deltanet::entry_gate_floats)) {
				unsigned const index = prims::thread_in_block();
				unsigned const head = index % static_cast<unsigned>(deltanet::group);
				unsigned const which = index / static_cast<unsigned>(deltanet::group);
				/* SAFETY: an entry's gate chunks are fp32 words. */
				auto const* const words = reinterpret_cast<unsigned const*>(entry + deltanet::group_chunks);
				Count const value_head = Count{key_head} * deltanet::group + head;
				prims::store_word(gates + value_head * 2 + which, prims::load_word(words + index, once), reused);
			}
		}
		grid.sync(prims::remote_backoff);
		if (key_head < static_cast<unsigned>(deltanet::key_heads)) {
			deltanet::Item const item = slot_item(machine, slot, layer, static_cast<int>(key_head), static_cast<int>(replays) - 1, 1, 0);
			run_item(machine, shared.deltanet, item, deltanet_core_phase(layer));
		}
		grid.sync(prims::remote_backoff);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Ranking: exact top-64 per row by sorting keys, a warp per 512 keys at a time.

/** Sorts the warp's 512 keys in shared memory, largest first: a bitonic network, a fixed sequence of exchanges. */
__attribute__((device)) inline void sort_descending(Key* keys) {
	unsigned const lane = prims::lane();
	for (unsigned size = 2; size <= static_cast<unsigned>(segment_rows); size <<= 1U) {
		for (unsigned stride = size >> 1U; stride > 0; stride >>= 1U) {
			for (unsigned index = lane; index < static_cast<unsigned>(segment_rows); index += 32) {
				unsigned const partner = index ^ stride;
				if (partner > index) {
					Key const first = keys[index];
					Key const second = keys[partner];
					bool const descending = (index & size) == 0;
					if (descending ? first < second : first > second) {
						keys[index] = second;
						keys[partner] = first;
					}
				}
			}
			prims::sync_warp();
		}
	}
}

/** The rows a ranking phase ranks, and where each one's logits are. */
struct Ranked {
	unsigned rows;
	PassRows pass;
	Op op;
};

[[nodiscard]] __attribute__((device)) inline auto ranked(Machine const& machine, Point const& point) -> Ranked {
	PassRows const rows = pass_rows(machine, point.pass);
	return {.rows = point.step.op == Op::Head ? rows.rows : rows.cycle.live, .pass = rows, .op = point.step.op};
}

/** The logits row of ranked row `index`: every verify row, or each slot's last row of an mtp pass. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto logits_row(Ranked const& what, unsigned index) -> unsigned {
	if (what.op == Op::Head) {
		return index;
	}
	attn::SlotWork const& work = what.pass.cycle.work[index];
	return work.first_row + work.rows - 1;
}

/** Lists left after `level` folds and the buffer holding them. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto lists_buffer(Count level) -> Count {
	return workspace.lists[level % 2];
}

/**
 * A logits ticket's tail: every verify row's fp32 logits over the card's lm_head rows, copied as the lm_head left them
 * into the host's logits buffer for the ticket. The rows sit one after another in both, so the copy is one run of
 * chunks over the whole grid. The barrier after this task and the answer's system-scope release publish the copy.
 */
__attribute__((device)) inline void hand_logits(Machine const& machine, unsigned long long ticket) {
	PassRows const rows = pass_rows(machine, Pass{.kind = PassKind::Verify, .draft = 0});
	expect(rows.rows <= logits_rows);
	auto const once = prims::policy<Eviction::First>();
	auto const written = prims::policy<Eviction::Normal>();
	auto const* from = machine.local<Chunk const>(workspace.logits);
	/* SAFETY: a half's rows are whole chunks, contiguous, and the mapped buffer is chunk-aligned. */
	auto* to = reinterpret_cast<Chunk*>(&at<LogitsBuffer>(machine.params.logits)->half[logits_half(ticket)].rows[0][0]);
	Count const chunks = Count{rows.rows} * model::vocab_per_card * sizeof(float) / 16;
	Count const threads = static_cast<Count>(decode_budget.threads);
	for (Count chunk = Count{prims::block()} * threads + prims::thread_in_block(); chunk < chunks; chunk += Count{sms} * threads) {
		prims::store(to + chunk, prims::load(from + chunk, once), written);
	}
}

/** Waits until the host has released this cycle's masks; it has had the whole body to fill them. */
__attribute__((device)) inline void await_masks(Machine const& machine, unsigned long long ticket) {
	if (prims::thread_in_block() == 0) {
		auto* masks = at<Masks>(machine.params.masks);
		prims::EpochReader<prims::Scope::System>{.flag = &masks->flag[mask_half(ticket)].ticket}.await(ticket, prims::remote_backoff);
	}
	prims::sync_block();
}

/** Each row's best keys of each 512-row segment of the card's rows, masked rows excluded for the lm_head. */
__attribute__((device)) inline void rank_segments(Machine const& machine, RankShared& shared, Point const& point, unsigned long long ticket) {
	Ranked const what = ranked(machine, point);
	bool const masked = point.step.op == Op::Head;
	if (masked) {
		await_masks(machine, ticket);
	}
	unsigned const warp = prims::warp_in_block();
	unsigned const lane = prims::lane();
	Count const vocabulary = ranked_rows(point.step.op);
	Count const count = segments(point.step.op);
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	auto const* logits = machine.local<unsigned const>(workspace.logits);
	auto const* tokens = at<unsigned const>(machine.params.row_tokens);
	auto const* masks = at<Masks const>(machine.params.masks);
	Key* const keys = shared.keys[warp];
	Key* const lists = machine.local<Key>(lists_buffer(0));
	for (unsigned item = global_warp(); item < what.rows * count; item += grid_warps) {
		unsigned const row = item / static_cast<unsigned>(count);
		Count const segment = item % count;
		Count const logit_row = logits_row(what, row);
		for (unsigned index = lane; index < static_cast<unsigned>(segment_rows); index += 32) {
			Count const card_row = segment * segment_rows + index;
			Key key = no_candidate;
			if (card_row < vocabulary) {
				bool allowed = true;
				if (masked) {
					unsigned const word = prims::load_word(&masks->half[mask_half(ticket)].words[row][card_row / 32], once);
					allowed = ((word >> (card_row % 32)) & 1U) != 0;
				}
				unsigned const token = prims::load_word(tokens + card_row, reused);
				// A padding row scores no token the tokenizer has: it is never a candidate, masked or not.
				if (allowed && token < model::sampled_vocabulary) {
					float const logit = __builtin_bit_cast(float, prims::load_word(logits + logit_row * model::vocab_per_card + card_row, once));
					key = key_of(logit, token);
				}
			}
			keys[index] = key;
		}
		prims::sync_warp();
		sort_descending(keys);
		Key* const list = lists + (Count{row} * count + segment) * list_length;
		for (unsigned index = lane; index < static_cast<unsigned>(list_length); index += 32) {
			list[index] = keys[index];
		}
		prims::sync_warp();
	}
}

/** Loads up to eight lists of one row into the warp's keys (missing ones as no candidate) and sorts them. */
__attribute__((device)) inline void gather_lists(Key const* lists, Count first, Count present, Key* keys) {
	unsigned const lane = prims::lane();
	for (unsigned index = lane; index < static_cast<unsigned>(segment_rows); index += 32) {
		Count const list = index / list_length;
		keys[index] = list < present ? lists[(first + list) * list_length + index % list_length] : no_candidate;
	}
	prims::sync_warp();
	sort_descending(keys);
}

/** One fold: each row's lists, eight at a time, into one. */
__attribute__((device)) inline void rank_groups(Machine const& machine, RankShared& shared, Point const& point) {
	Ranked const what = ranked(machine, point);
	Count const before = lists_after(point.step.op, point.level - 1);
	Count const after = lists_after(point.step.op, point.level);
	Key const* const in = machine.local<Key const>(lists_buffer(point.level - 1));
	Key* const out = machine.local<Key>(lists_buffer(point.level));
	Key* const keys = shared.keys[prims::warp_in_block()];
	unsigned const lane = prims::lane();
	for (unsigned item = global_warp(); item < what.rows * after; item += grid_warps) {
		unsigned const row = item / static_cast<unsigned>(after);
		Count const group = item % after;
		Count const first = group * fold_lists;
		Count const present = before - first < fold_lists ? before - first : fold_lists;
		gather_lists(in + Count{row} * before * list_length, first, present, keys);
		Key* const list = out + (Count{row} * after + group) * list_length;
		for (unsigned index = lane; index < static_cast<unsigned>(list_length); index += 32) {
			list[index] = keys[index];
		}
		prims::sync_warp();
	}
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto candidate_slot(Count slot) -> Count {
	return workspace.exchange.candidates + slot * schedule::candidate_slot_bytes(budget);
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto candidate_flag(Count slot) -> Count {
	return workspace.exchange.candidate_flags + slot * schedule::flag_bytes;
}

/** Each row's best 64 of the card: kept as this card's candidates and pushed whole to the peer's receive slot. */
__attribute__((device)) inline void rank_final(Machine const& machine, RankShared& shared, Point const& point, Epochs& epochs) {
	++epochs.candidates;
	Count const slot = exchange_slot(epochs.candidates);
	Ranked const what = ranked(machine, point);
	Count const level = folds(point.step.op);
	Count const present = lists_after(point.step.op, level);
	Key const* const in = machine.local<Key const>(lists_buffer(level));
	Key* const keys = shared.keys[prims::warp_in_block()];
	auto* const own = machine.local<Chunk>(workspace.candidates);
	prims::PeerOutbox const outbox = {
	    .slots = machine.peer<Chunk>(candidate_slot(slot)),
	    .flag = {.flag = machine.peer<unsigned long long>(candidate_flag(slot))},
	};
	auto const reused = prims::policy<Eviction::Last>();
	unsigned const lane = prims::lane();
	constexpr unsigned row_chunks = static_cast<unsigned>(list_length / 2);
	static_assert(row_chunks == 32, "a row's candidates are one chunk a lane");
	for (unsigned row = global_warp(); row < what.rows; row += grid_warps) {
		gather_lists(in + Count{row} * present * list_length, 0, present, keys);
		sampling::Candidate const first = candidate_of(keys[Count{2} * lane]);
		sampling::Candidate const second = candidate_of(keys[(Count{2} * lane) + 1]);
		Chunk const pair = {.word = {__builtin_bit_cast(unsigned, first.logit), static_cast<unsigned>(first.id),
		                             __builtin_bit_cast(unsigned, second.logit), static_cast<unsigned>(second.id)}};
		prims::store(own + Count{row} * row_chunks + lane, pair, reused);
		outbox.push(row * row_chunks + lane, pair);
		prims::sync_warp();
	}
}

/**
 * The candidate exchange's release and arrival: the grid barrier before this task ordered every block's pushes, so one
 * release by block 0 publishes them all; then every block waits for the peer's.
 */
__attribute__((device)) inline void exchange_candidates(Machine const& machine, Epochs const& epochs) {
	Count const slot = exchange_slot(epochs.candidates);
	if (prims::thread_in_block() == 0) {
		if (prims::block() == 0) {
			prims::EpochWriter<prims::Scope::System>{.flag = machine.peer<unsigned long long>(candidate_flag(slot))}.publish(epochs.candidates);
		}
		prims::EpochReader<prims::Scope::System>{.flag = machine.local<unsigned long long const>(candidate_flag(slot))}.await(
		    epochs.candidates, prims::remote_backoff);
	}
	prims::sync_block();
}

/** Both cards' 64 candidates of one row merged into the row's 64, by key; returns how many are real. */
__attribute__((device)) inline auto merge_candidates(sampling::Candidate const* own, sampling::Candidate const* peer, sampling::Candidate* merged)
    -> int {
	auto const once = prims::policy<Eviction::First>();
	auto const read = [&](sampling::Candidate const* list, Count index) {
		/* SAFETY: candidate lists are pairs of words in the workspace. */
		auto const* words = reinterpret_cast<unsigned const*>(list + index);
		return sampling::Candidate{.logit = __builtin_bit_cast(float, prims::load_word(words, once)),
		                           .id = static_cast<int>(prims::load_word(words + 1, once))};
	};
	Count left = 0;
	Count right = 0;
	int available = 0;
	for (Count index = 0; index < list_length; ++index) {
		Key const mine = key_of(read(own, left));
		Key const theirs = key_of(read(peer, right));
		Key const taken = mine >= theirs ? mine : theirs;
		left += mine >= theirs ? 1 : 0;
		right += mine >= theirs ? 0 : 1;
		merged[index] = candidate_of(taken);
		available += taken != no_candidate ? 1 : 0;
	}
	return available;
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto own_candidates(Machine const& machine, unsigned row) -> sampling::Candidate const* {
	return machine.local<sampling::Candidate const>(workspace.candidates) + Count{row} * list_length;
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto peer_candidates(Machine const& machine, Epochs const& epochs, unsigned row)
    -> sampling::Candidate const* {
	return machine.local<sampling::Candidate const>(candidate_slot(exchange_slot(epochs.candidates))) + Count{row} * list_length;
}

// ---------------------------------------------------------------------------------------------------------------------
// The verdict and the drafts: one slot per block.

/**
 * Block `live` judges its slot: each verify row's merged candidates are its target distribution under the slot's rule,
 * its drafts' distributions come from its book, and `sampling::settle` decides the accepted drafts and the next token.
 */
__attribute__((device)) inline void verify(Machine const& machine, VerifyShared& shared, Epochs const& epochs) {
	exchange_candidates(machine, epochs);
	Descriptor const& descriptor = machine.descriptor();
	unsigned const live = prims::block();
	if (live >= descriptor.live) {
		return;
	}
	PassRows const rows = pass_rows(machine, {.kind = PassKind::Verify, .draft = 0});
	attn::SlotWork const work = rows.cycle.work[live];
	unsigned const warp = prims::warp_in_block();
	if (warp < work.rows && prims::lane() == 0) {
		unsigned const row = work.first_row + warp;
		shared.available[warp] = merge_candidates(own_candidates(machine, row), peer_candidates(machine, epochs, row), shared.candidates[warp]);
	}
	prims::sync_block();
	if (prims::thread_in_block() != 0) {
		return;
	}
	SlotRun const& run = descriptor.slots[live];
	Book const* book = machine.book(run.slot);
	for (unsigned row = 0; row < work.rows; ++row) {
		expect(shared.available[row] >= 1);
		shared.targets[row] = sampling::survivors(shared.candidates[row], shared.available[row], run.rule.filter);
	}
	sampling::Settled const settled =
	    sampling::settle(run.rule, run.position + 1U, shared.targets, book->proposal, book->chosen, static_cast<int>(run.drafts));
	SlotOutcome* outcome = machine.outcome(live);
	outcome->accepted = static_cast<unsigned>(settled.accepted);
	for (int index = 0; index < settled.accepted; ++index) {
		outcome->committed[index] = book->tokens[index];
	}
	outcome->committed[settled.accepted] = static_cast<unsigned>(settled.token);
}

/**
 * Block `live` proposes its slot's next draft from both cards' draft candidates, unmasked, under its rule: the
 * distribution and the choice go to its book, the token to its outcome.
 */
__attribute__((device)) inline void draft(Machine const& machine, VerifyShared& shared, Point const& point, Epochs const& epochs) {
	exchange_candidates(machine, epochs);
	Descriptor const& descriptor = machine.descriptor();
	unsigned const live = prims::block();
	if (live >= descriptor.live || prims::thread_in_block() != 0) {
		return;
	}
	shared.available[0] = merge_candidates(own_candidates(machine, live), peer_candidates(machine, epochs, live), shared.candidates[0]);
	SlotRun const& run = descriptor.slots[live];
	SlotOutcome* outcome = machine.outcome(live);
	Book* book = machine.book(run.slot);
	unsigned const step = static_cast<unsigned>(point.pass.draft) - 1U;
	expect(shared.available[0] >= 1);
	book->proposal[step] = sampling::survivors(shared.candidates[0], shared.available[0], run.rule.filter);
	// The next cycle verifies the token after its committed one first: that is position + accepted + 2.
	unsigned const first = run.position + outcome->accepted + 2U;
	int const chosen = sampling::propose(run.rule, first + step, step, book->proposal[step]);
	book->chosen[step] = chosen;
	book->tokens[step] = static_cast<unsigned>(book->proposal[step].id[chosen]);
	outcome->drafts[step] = book->tokens[step];
}

// ---------------------------------------------------------------------------------------------------------------------
// The mtp head's inputs.

/** The hidden row an mtp row reads: the trunk's final norm for the catch-up, the last mtp norm for a step. */
[[nodiscard]] __attribute__((device)) inline auto mtp_hidden_row(Machine const& machine, Pass pass, unsigned live, unsigned offset) -> Bf16 const* {
	if (pass.kind == PassKind::CatchUp) {
		attn::SlotWork const verify_work = pass_rows(machine, {.kind = PassKind::Verify, .draft = 0}).cycle.work[live];
		return machine.local<Bf16 const>(workspace.final_normed) + Count{verify_work.first_row + offset} * model::hidden;
	}
	unsigned row = live;
	if (pass.draft == 2) {
		attn::SlotWork const catch_up = pass_rows(machine, {.kind = PassKind::CatchUp, .draft = 1}).cycle.work[live];
		row = catch_up.first_row + catch_up.rows - 1;
	}
	return machine.local<Bf16 const>(workspace.mtp_normed) + Count{row} * model::hidden;
}

/** The token an mtp row embeds: the one after its position, the next committed token or the last draft. */
[[nodiscard]] __attribute__((device)) inline auto mtp_token(Machine const& machine, Pass pass, unsigned live, unsigned offset) -> unsigned {
	SlotOutcome const* outcome = machine.outcome(live);
	return pass.kind == PassKind::CatchUp ? outcome->committed[offset] : outcome->drafts[pass.draft - 2];
}

/** Each mtp row's embedding (fp32) and hidden input (bf16), each with its tiles' squares; rows past the pass's zero. */
__attribute__((device)) inline void mtp_gather(Machine const& machine, Point const& point) {
	PassRows const rows = pass_rows(machine, point.pass);
	unsigned const computed = static_cast<unsigned>(point.row_tiles * model::row_tile);
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	Count const lane_first = Count{prims::lane()} * 4;
	for (unsigned item = global_warp(); item < computed * hidden_tiles; item += grid_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		Count const first = tile * schedule::tile_rows + lane_first;
		float embedded[4] = {};
		float hidden[4] = {};
		if (row < rows.rows) {
			RowPlace const at_row = place(rows.cycle, row);
			load_four(embedding_row(machine, mtp_token(machine, point.pass, at_row.live, at_row.offset)), first, once, embedded);
			load_four(mtp_hidden_row(machine, point.pass, at_row.live, at_row.offset), first, reused, hidden);
		}
		store_floats(machine.local<float>(workspace.mtp_embedding) + Count{row} * model::hidden + first, embedded);
		store_four(machine.local<Bf16>(workspace.mtp_hidden) + Count{row} * model::hidden + first, hidden);
		store_square(machine.local<float>(workspace.embedding_squares), tile, row, tile_squares(embedded));
		store_square(machine.local<float>(workspace.hidden_squares), tile, row, tile_squares(hidden));
	}
}

/**
 * pre_fc_norm_embedding and pre_fc_norm_hidden over the whole hidden, each card keeping its half of both: the fc's
 * input row on card c is the embedding norm's features [c h/2, (c + 1) h/2) and then the hidden norm's.
 */
__attribute__((device)) inline void mtp_norm(Machine const& machine, Point const& point) {
	PassRows const rows = pass_rows(machine, point.pass);
	unsigned const computed = static_cast<unsigned>(point.row_tiles * model::row_tile);
	auto const reused = prims::policy<Eviction::Last>();
	Count const lane_first = Count{prims::lane()} * 4;
	constexpr Count half_tiles = hidden_tiles / model::cards;
	Count const own_first_tile = Count{machine.params.card} * half_tiles;
	Bf16 const* const weights[2] = {
	    at<Bf16 const>(side(machine, point.image_phase, side_index(Op::MtpIn, tensor.pre_fc_norm_embedding, Part::Whole))),
	    at<Bf16 const>(side(machine, point.image_phase, side_index(Op::MtpIn, tensor.pre_fc_norm_hidden, Part::Whole))),
	};
	for (unsigned item = global_warp(); item < computed * hidden_tiles; item += grid_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		Count const input = tile / half_tiles;
		Count const first = (own_first_tile + tile % half_tiles) * schedule::tile_rows + lane_first;
		float values[4] = {};
		if (row < rows.rows) {
			float const factor =
			    model::rms_factor(row_squares(machine.local<float const>(input == 0 ? workspace.embedding_squares : workspace.hidden_squares), row),
			                      model::hidden);
			float x[4] = {};
			if (input == 0) {
				/* SAFETY: the embeddings are fp32 chunks of the workspace. */
				Chunk const loaded =
				    prims::load(reinterpret_cast<Chunk const*>(machine.local<float const>(workspace.mtp_embedding) + Count{row} * model::hidden + first), reused);
#pragma unroll
				for (int index = 0; index < 4; ++index) {
					x[index] = __builtin_bit_cast(float, loaded.word[index]);
				}
			} else {
				load_four(machine.local<Bf16 const>(workspace.mtp_hidden) + Count{row} * model::hidden, first, reused, x);
			}
			float w[4] = {};
			load_four(weights[input], first, reused, w);
#pragma unroll
			for (int index = 0; index < 4; ++index) {
				values[index] = model::one_plus_scaled(x[index], factor, w[index]);
			}
		}
		store_four(machine.local<Bf16>(workspace.mtp_input) + Count{row} * model::hidden + tile * schedule::tile_rows + lane_first, values);
	}
}

}

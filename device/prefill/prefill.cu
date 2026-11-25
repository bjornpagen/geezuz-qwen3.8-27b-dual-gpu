// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill's kernels: the steps of a prompt chunk that attention's and the deltanet's images do not hold. A chunk runs
// them in a short fixed sequence per layer, each a launch over the chunk's rows on both cards at once: the embedding
// gather, the rms norms, the weight matmuls through the prefill gemm (each placement with its epilogue), in_proj_a and
// in_proj_b, the cross-card exchange of the row-parallel sums and their residual update, and the mtp head's inputs.
// Every value is computed exactly as the decode loop computes it for the same row (the model's rounding section, the
// gemm walking the decode's segments), so a prompt prefilled in chunks leaves the state its tokens decoded one per
// cycle would.
#include "gemv/gemm.h"
#include "gemv/kernels.h"
#include "gemv/order.h"
#include "model.h"
#include "prefill/image.h"
#include "prefill/layout.h"
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/numeric.h"
#include "prims/order.h"
#include "prims/reduce.h"
#include "prims/selfcheck.h"
#include "prims/sync.h"
#include "prims/thread.h"

namespace engine::device::prefill {

using prims::Bf16;
using prims::Chunk;
using prims::Eviction;
using prims::Policy;

/* SAFETY: every address a kernel reads or writes is a device allocation the host named in its parameters. */
template<class T>
[[nodiscard]] __attribute__((device, always_inline)) inline auto at(Address address) -> T* {
	return __builtin_bit_cast(T*, address);
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto global_warp() -> unsigned {
	return prims::block() * static_cast<unsigned>(rows_budget.threads / 32) + prims::warp_in_block();
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto global_thread() -> unsigned {
	return prims::block() * static_cast<unsigned>(rows_budget.threads) + prims::thread_in_block();
}

// ---------------------------------------------------------------------------------------------------------------------
// Rows of the hidden state: four features a lane, a warp per 128-feature tile, so a tile's squares are the model's
// `hidden_tile` sum.

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

/** The warp's sum of its lanes' four squares, folded by fma from +0 then the butterfly: one tile's `hidden_tile` sum. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto tile_squares(float const (&values)[4]) -> float {
	float partial = 0.0F;
#pragma unroll
	for (int index = 0; index < 4; ++index) {
		partial = model::fixed::fma(values[index], values[index], partial);
	}
	return prims::warp_sum(partial);
}

/** A row's sum of squares from its tile sums ([tile][max_rows]), folded left to right from tile 0. */
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

/** Four values rounded once to bf16 and stored as two words. */
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

[[nodiscard]] __attribute__((device, always_inline)) inline auto token_at(Address tokens, Count row) -> unsigned {
	return prims::load_word(at<unsigned const>(tokens) + row, prims::policy<Eviction::First>());
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto embedding_row(Address embedding, unsigned token) -> Bf16 const* {
	return at<Bf16 const>(embedding) + Count{token} * model::hidden;
}

// ---------------------------------------------------------------------------------------------------------------------
// Epilogues of the prefill gemm, beside gemm.h's `StoreRows` and `GateUpRows`.

using gemv::layout::Element;

/** The barrier of the gemm's eight consumer warps, apart from the ring's. */
using Consumers = gemv::kernels::Consumers;
using ItemBarrier = prims::NamedBarrier<2, gemv::consumer_warps>;

/**
 * Adjacent outputs of one row from a lane's n8 accumulator: lane 4g + t swaps with its partner across the quad
 * (g ^ 1), so each lane emits outputs (2h, 2h + 1) of one row. `emit(first, a, b)` gets the pair's first element.
 */
template<class Place, class Emit>
__attribute__((device, always_inline)) inline void adjacent(prims::Accumulator const& values, Place const& place, Emit const& emit) {
	bool const even = ((prims::lane() / 4U) % 2U) == 0;
#pragma unroll
	for (int half = 0; half < 2; ++half) {
		float const give = values.reg[2 * half + (even ? 1 : 0)];
		float const keep = values.reg[2 * half + (even ? 0 : 1)];
		float const got = __builtin_bit_cast(float, prims::exchange(__builtin_bit_cast(unsigned, give), 4U));
		Element const kept = place(2 * half + (even ? 0 : 1));
		Element const first = {.output = kept.output - (even ? 0 : 1), .row = kept.row};
		if (even) {
			emit(first, keep, got);
		} else {
			emit(first, got, keep);
		}
	}
}

/** The chunk row and output column of an element of item `at`. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto row_of(gemv::prefill::Item at, Element e) -> Count {
	return Count{at.rows} * block_rows + static_cast<Count>(e.row);
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto column_of(gemv::prefill::Item at, Element e) -> Count {
	return Count{at.tile} * schedule::tile_rows + static_cast<Count>(e.output);
}

/** Each finished item rounded once to bf16 rows of `width`: in_proj_qkv's projections (`DeltanetProjection`). */
template<gemv::Matrix matrix>
struct RowsBf16 {
	unsigned* words;

	__attribute__((device, always_inline)) void operator()(gemv::prefill::Item at, gemv::prefill::Warp<1> const& totals,
	                                                       unsigned warp) const {
		constexpr Count width = matrix.output_tiles * schedule::tile_rows;
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const lane = prims::lane();
#pragma unroll
		for (int slab = 0; slab < gemv::prefill::warp_slabs; ++slab) {
#pragma unroll
			for (int n = 0; n < gemv::prefill::warp_row_tiles; ++n) {
				adjacent(
				    totals.tensor[0][slab].tile[n],
				    [&](int reg) {
					    return gemv::prefill::element(warp, lane, slab, n, reg);
				    },
				    [&](Element first, float a, float b) {
					    prims::store_word(words + (row_of(at, first) * width + column_of(at, first)) / 2, prims::pack_bf16x2(a, b).bits,
						                  reused);
				    });
			}
		}
	}
};

/**
 * in_proj_z's item gates the deltanet core's normed output it lines up with: normed * silu(z), rounded once, into
 * out_proj's input rows (`DeltanetGated`). z itself is never stored; the core ran before this gemm.
 */
template<gemv::Matrix matrix>
struct GatedCore {
	float const* core;
	unsigned* gated;

	__attribute__((device, always_inline)) void operator()(gemv::prefill::Item at, gemv::prefill::Warp<1> const& totals,
	                                                       unsigned warp) const {
		constexpr Count width = deltanet::value_features;
		static_assert(matrix.output_tiles * schedule::tile_rows == width, "z lines up with the core's outputs");
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const lane = prims::lane();
#pragma unroll
		for (int slab = 0; slab < gemv::prefill::warp_slabs; ++slab) {
#pragma unroll
			for (int n = 0; n < gemv::prefill::warp_row_tiles; ++n) {
				adjacent(
				    totals.tensor[0][slab].tile[n],
				    [&](int reg) {
					    return gemv::prefill::element(warp, lane, slab, n, reg);
				    },
				    [&](Element first, float z_first, float z_second) {
					    Count const index = row_of(at, first) * width + column_of(at, first);
					    /* SAFETY: the normed rows are fp32 words of the workspace. */
					    auto const* normed = reinterpret_cast<unsigned const*>(core + index);
					    float const a = model::deltanet::gated(__builtin_bit_cast(float, prims::load_word(normed, reused)), z_first);
					    float const b = model::deltanet::gated(__builtin_bit_cast(float, prims::load_word(normed + 1, reused)), z_second);
					    prims::store_word(gated + index / 2, prims::pack_bf16x2(a, b).bits, reused);
				    });
			}
		}
	}
};

/**
 * A row-parallel matmul's finished item: kept whole as this card's partial and pushed whole into the peer's receive
 * slot, then released with the exchange's epoch, once per item. The pushes of one item overlap the gemm's later items.
 */
template<gemv::Matrix matrix>
struct PushRows {
	float* own;
	Chunk* peer;
	unsigned long long* peer_flags;
	unsigned long long epoch;

	static_assert(matrix.tensors == 1 && matrix.output_tiles == hidden_tiles, "a row-parallel matmul's outputs are the hidden state");

	__attribute__((device, always_inline)) void operator()(gemv::prefill::Item at, gemv::prefill::Warp<1> const& totals,
	                                                       unsigned warp) const {
		auto const reused = prims::policy<Eviction::Last>();
		unsigned const lane = prims::lane();
		Count const item = item_index(at.tile, at.rows);
		float* const mine = own + item * item_floats;
#pragma unroll
		for (int slab = 0; slab < gemv::prefill::warp_slabs; ++slab) {
#pragma unroll
			for (int n = 0; n < gemv::prefill::warp_row_tiles; ++n) {
#pragma unroll
				for (int reg = 0; reg < 4; ++reg) {
					Element const e = gemv::prefill::element(warp, lane, slab, n, reg);
					/* SAFETY: the own partials are fp32 words of the workspace. */
					prims::store_word(
					    reinterpret_cast<unsigned*>(mine + static_cast<Count>(e.row) * schedule::tile_rows + static_cast<Count>(e.output)),
					    __builtin_bit_cast(unsigned, totals.tensor[0][slab].tile[n].reg[reg]), reused);
				}
			}
		}
		ItemBarrier::sync();
		prims::PeerOutbox const outbox = {.slots = peer + item * (item_bytes / 16), .flag = {.flag = peer_flags + item}};
		unsigned const thread = warp * 32U + lane;
		/* SAFETY: the own partials are fp32 chunks of the workspace. */
		auto const* chunks = reinterpret_cast<Chunk const*>(mine);
		for (unsigned chunk = thread; chunk < static_cast<unsigned>(item_bytes / 16);
		     chunk += static_cast<unsigned>(gemv::consumer_threads)) {
			outbox.push(chunk, prims::load(chunks + chunk, reused));
		}
		ItemBarrier::sync();
		if (thread == 0) {
			outbox.publish(epoch);
		}
	}
};

[[nodiscard]] __attribute__((device, always_inline)) inline auto operands(Weights const& weights, Address activations)
    -> gemv::prefill::Operands {
	return {
	    .activations = at<Chunk const>(activations),
	    .tiles = at<Chunk const>(weights.tiles),
	    .scales = {at<unsigned short const>(weights.scales[0]), at<unsigned short const>(weights.scales[1])},
	};
}

template<gemv::Matrix matrix, class Epilogue>
__attribute__((device, always_inline)) inline void gemm(Weights const& weights, Address activations, unsigned row_blocks,
                                                        Epilogue& epilogue) {
	gemv::kernels::run_gemm<matrix>(operands(weights, activations), row_blocks, epilogue);
}

/** The geometry of ring placement `ring` of `op` in `stream`: every layer of an op in one stream shares it. */
template<Op op, model::Stream stream, Count ring>
inline constexpr gemv::Matrix matrix = gemv::matrix(first_phase(op, stream), ring);

template<Op op, model::Stream stream, Count ring>
inline constexpr prims::Budget gemm_budget = gemv::kernels::gemm_budget<matrix<op, stream, ring>>;

/** fp32 rows of `width` from the projection at `offset` of the workspace. */
template<Op op, model::Stream stream, Count ring>
__attribute__((device, always_inline)) inline void project_f32(GemmParams const& params, Count offset, Address activations) {
	constexpr gemv::Matrix m = matrix<op, stream, ring>;
	gemv::prefill::StoreRows<m> out = {.out = at<float>(params.workspace + offset), .rows = static_cast<unsigned>(max_rows)};
	gemm<m>(params.weights, activations, params.row_blocks, out);
}

template<Op op, model::Stream stream, Count ring>
__attribute__((device, always_inline)) inline void project_push(PushParams const& params, Address activations) {
	constexpr gemv::Matrix m = matrix<op, stream, ring>;
	Count const slot = exchange_slot(params.epoch);
	PushRows<m> out = {
	    .own = at<float>(params.workspace + workspace.own),
	    .peer = at<Chunk>(params.peer_workspace + workspace.receive + slot * exchange_floats * float_bytes),
	    .peer_flags = at<unsigned long long>(params.peer_workspace + workspace.flags + slot * exchange_items * schedule::flag_bytes),
	    .epoch = params.epoch,
	};
	gemm<m>(params.weights, activations, params.row_blocks, out);
}

template<model::Stream stream>
__attribute__((device, always_inline)) inline void mlp_in(GemmParams const& params) {
	constexpr gemv::Matrix m = matrix<Op::MlpIn, stream, 0>;
	gemv::prefill::GateUpRows<m, static_cast<int>(model::ffn_per_card)> out = {.out = at<unsigned>(params.workspace + workspace.product)};
	gemm<m>(params.weights, params.workspace + workspace.normed, params.row_blocks, out);
}

static_assert(gemm_blocks == gemv::selfcheck::blocks, "the gemm's work items are split over one block per SM");
static_assert(attn::query_row_floats == matrix<Op::AttentionIn, model::Stream::Body, 0>.output_tiles * schedule::tile_rows);
static_assert(attn::kv_row_floats == matrix<Op::AttentionIn, model::Stream::Body, 1>.output_tiles * schedule::tile_rows);
static_assert(attn::gate_row_floats == matrix<Op::AttentionGate, model::Stream::Body, 0>.output_tiles * schedule::tile_rows);
static_assert(deltanet::channels == static_cast<int>(matrix<Op::DeltanetIn, model::Stream::Body, 0>.output_tiles * schedule::tile_rows));

// ---------------------------------------------------------------------------------------------------------------------
// The residual's update once both cards' sums of an item are in.

/** How the cards' sum of an exchange meets the residual. */
enum class Residual : unsigned char {
	/** residual + (card 0 + card 1): every layer's two additions. */
	Add,
	/** card 0 + card 1 starts the residual: the mtp fc's output. */
	Start,
};

/**
 * One warp per (row, tile) of the chunk: waits for the peer's item, then adds (own + peer) (fp32 addition commutes,
 * so either card computes card 0 + card 1) to the residual, or starts it, and stores the tile's squares.
 */
template<Residual how>
__attribute__((device, always_inline)) inline void meet(ResidualParams const& params) {
	auto const reused = prims::policy<Eviction::Last>();
	auto const once = prims::policy<Eviction::First>();
	unsigned const lane = prims::lane();
	Count const slot = exchange_slot(params.epoch);
	auto const* flags = at<unsigned long long const>(params.workspace + workspace.flags + slot * exchange_items * schedule::flag_bytes);
	auto const* own = at<float const>(params.workspace + workspace.own);
	auto const* peer = at<float const>(params.workspace + workspace.receive + slot * exchange_floats * float_bytes);
	auto* const residual = at<float>(params.workspace + workspace.residual);
	auto* const squares = at<float>(params.workspace + workspace.squares);
	for (unsigned item = global_warp(); item < params.rows.count * static_cast<unsigned>(hidden_tiles); item += rows_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		prims::EpochReader<prims::Scope::System>{.flag = flags + item_index(tile, row / block_rows)}.await(params.epoch,
		                                                                                                   prims::remote_backoff);
		Count const from = exchange_offset(tile, row) + Count{lane} * 4;
		/* SAFETY: partials and the residual are fp32 chunks of the workspace. */
		Chunk const mine = prims::load(reinterpret_cast<Chunk const*>(own + from), once);
		Chunk const theirs = prims::load(reinterpret_cast<Chunk const*>(peer + from), once);
		auto* const target = reinterpret_cast<Chunk*>(residual + Count{row} * model::hidden + tile * schedule::tile_rows + Count{lane} * 4);
		Chunk before{};
		if constexpr (how == Residual::Add) {
			before = prims::load(target, reused);
		}
		float values[4] = {};
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			float const card_sum = __builtin_bit_cast(float, mine.word[word]) + __builtin_bit_cast(float, theirs.word[word]);
			if constexpr (how == Residual::Add) {
				values[word] = __builtin_bit_cast(float, before.word[word]) + card_sum;
			} else {
				values[word] = card_sum;
			}
		}
		store_floats(residual + Count{row} * model::hidden + tile * schedule::tile_rows + Count{lane} * 4, values);
		store_square(squares, tile, row, tile_squares(values));
	}
}

}

namespace prefill = engine::device::prefill;
namespace prims = engine::device::prims;
namespace gemv = engine::device::gemv;
namespace schedule = engine::device::schedule;
namespace deltanet = engine::device::deltanet;

using engine::model::Stream;
using prefill::Op;

// ---------------------------------------------------------------------------------------------------------------------
// Row-wise kernels.

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_zero = prefill::rows_budget;

/** Zeroes a run of whole chunks: a slot's deltanet state before its prompt's first chunk, the workspace once at start. */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_zero(prefill::ZeroParams params) {
	auto const written = prims::policy<prims::Eviction::Normal>();
	auto* const chunks = prefill::at<prims::Chunk>(params.target);
	constexpr unsigned long long threads = static_cast<unsigned long long>(prefill::rows_blocks) * prefill::rows_budget.threads;
	for (unsigned long long chunk = prefill::global_thread(); chunk < params.chunks; chunk += threads) {
		prims::store(chunks + chunk, prims::Chunk{}, written);
	}
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_embed = prefill::rows_budget;

/** The chunk's token embeddings widened into the fp32 residual, with each tile's squares; padding rows are zero. */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_embed(prefill::EmbedParams params) {
	using namespace engine::device::prefill;
	auto const once = prims::policy<Eviction::First>();
	float* const residual = at<float>(params.workspace + workspace.residual);
	float* const squares = at<float>(params.workspace + workspace.squares);
	Count const lane_first = Count{prims::lane()} * 4;
	for (unsigned item = global_warp(); item < params.rows.computed * static_cast<unsigned>(hidden_tiles); item += rows_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		float values[4] = {};
		if (row < params.rows.count) {
			load_four(embedding_row(params.embedding, token_at(params.tokens, row)), tile * schedule::tile_rows + lane_first, once, values);
		}
		store_floats(residual + Count{row} * engine::model::hidden + tile * schedule::tile_rows + lane_first, values);
		store_square(squares, tile, row, tile_squares(values));
	}
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_norm = prefill::rows_budget;

/** x * factor * (1 + w), rounded once, for every computed row; padding rows are zero. */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_norm(prefill::NormParams params) {
	using namespace engine::device::prefill;
	auto const reused = prims::policy<Eviction::Last>();
	auto const* const weight = at<Bf16 const>(params.weight);
	auto const* const residual = at<float const>(params.residual);
	auto const* const squares = at<float const>(params.squares);
	auto* const out = at<Bf16>(params.out);
	Count const lane_first = Count{prims::lane()} * 4;
	for (unsigned item = global_warp(); item < params.rows.computed * static_cast<unsigned>(hidden_tiles); item += rows_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		Count const first = tile * schedule::tile_rows + lane_first;
		float values[4] = {};
		if (row < params.rows.count) {
			float const factor = engine::model::rms_factor(row_squares(squares, row), engine::model::hidden);
			float weights[4] = {};
			load_four(weight, first, reused, weights);
			/* SAFETY: the residual's rows are fp32 chunks of the workspace. */
			Chunk const x = prims::load(reinterpret_cast<Chunk const*>(residual + Count{row} * engine::model::hidden + first), reused);
#pragma unroll
			for (int index = 0; index < 4; ++index) {
				values[index] = engine::model::one_plus_scaled(__builtin_bit_cast(float, x.word[index]), factor, weights[index]);
			}
		}
		store_four(out + Count{row} * engine::model::hidden + first, values);
	}
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_decay = prefill::rows_budget;

/**
 * in_proj_a and in_proj_b on the normed rows, read whole: one mma chain per (eight value heads, eight rows) over the
 * whole hidden in ascending k from +0, a warp each. The chain's m16 rows are eight heads' a rows then the same heads'
 * b rows, so a lane holds a and b of one head for two rows, stored as the deltanet reads them: fp32 a then b per head.
 */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_decay(prefill::DecayParams params) {
	using namespace engine::device::prefill;
	constexpr unsigned head_slabs = static_cast<unsigned>(deltanet::value_heads / 8);
	static_assert(deltanet::value_heads % 8 == 0 && engine::model::hidden % 16 == 0);
	unsigned const row_tiles = params.rows.computed / 8U;
	unsigned const lane = prims::lane();
	unsigned const g = lane / 4U;
	unsigned const t = lane % 4U;
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	constexpr Count row_words = engine::model::hidden / 2;
	auto* const gates = at<unsigned>(params.workspace + workspace.gates);
	for (unsigned chain = global_warp(); chain < head_slabs * row_tiles; chain += rows_warps) {
		unsigned const slab = chain / row_tiles;
		unsigned const n = chain % row_tiles;
		auto const* a = at<unsigned const>(params.a) + Count{8U * slab + g} * row_words;
		auto const* b = at<unsigned const>(params.b) + Count{8U * slab + g} * row_words;
		auto const* x = at<unsigned const>(params.workspace + workspace.normed) + Count{8U * n + g} * row_words;
		prims::Accumulator sum = {.reg = {0.0F, 0.0F, 0.0F, 0.0F}};
		for (Count step = 0; step < engine::model::hidden / 16; ++step) {
			Count const low = step * 8 + t;
			Count const high = low + 4;
			prims::AFragment const weights = {.reg = {prims::load_word(a + low, once), prims::load_word(b + low, once),
			                                          prims::load_word(a + high, once), prims::load_word(b + high, once)}};
			prims::BFragment const rows = {.reg = {prims::load_word(x + low, reused), prims::load_word(x + high, reused)}};
			prims::mma(sum, weights, rows);
		}
		Count const head = 8U * slab + g;
		Count const row = 8U * n + 2U * t;
		// Registers: 0 (a, row), 1 (a, row + 1), 2 (b, row), 3 (b, row + 1).
		for (Count reg = 0; reg < 4; ++reg) {
			Count const word = ((row + reg % 2) * deltanet::value_heads + head) * 2 + reg / 2;
			prims::store_word(gates + word, __builtin_bit_cast(unsigned, sum.reg[reg]), reused);
		}
	}
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_residual_add = prefill::rows_budget;

extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_residual_add(prefill::ResidualParams params) {
	prefill::meet<prefill::Residual::Add>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_residual_start = prefill::rows_budget;

extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_residual_start(prefill::ResidualParams params) {
	prefill::meet<prefill::Residual::Start>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_gather = prefill::rows_budget;

/**
 * The mtp head's entries: each one's embedding (fp32) and hidden input (bf16, the final norm at its position), each
 * with its tile squares; padding rows are zero.
 */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_mtp_gather(prefill::MtpParams params) {
	using namespace engine::device::prefill;
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	Count const lane_first = Count{prims::lane()} * 4;
	auto const* const normed = at<Bf16 const>(params.workspace + workspace.final_normed);
	auto const* const carried = at<Bf16 const>(params.workspace + workspace.carried) + Count{params.slot} * engine::model::hidden;
	for (unsigned item = global_warp(); item < params.rows.computed * static_cast<unsigned>(hidden_tiles); item += rows_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		Count const first = tile * schedule::tile_rows + lane_first;
		float embedded[4] = {};
		float hidden[4] = {};
		if (row < params.rows.count) {
			unsigned const entry = params.first + row;
			load_four(embedding_row(params.embedding, token_at(params.tokens, entry + 1U - params.from)), first, once, embedded);
			Bf16 const* source = entry < params.from ? carried : normed + Count{entry - params.from} * engine::model::hidden;
			load_four(source, first, reused, hidden);
		}
		store_floats(at<float>(params.workspace + workspace.mtp_embedding) + Count{row} * engine::model::hidden + first, embedded);
		store_four(at<Bf16>(params.workspace + workspace.mtp_hidden) + Count{row} * engine::model::hidden + first, hidden);
		store_square(at<float>(params.workspace + workspace.embedding_squares), tile, row, tile_squares(embedded));
		store_square(at<float>(params.workspace + workspace.hidden_squares), tile, row, tile_squares(hidden));
	}
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_norm = prefill::rows_budget;

/**
 * pre_fc_norm_embedding and pre_fc_norm_hidden over the whole hidden, each card keeping its half of both: the fc's
 * input row on card c is the embedding norm's features [c h/2, (c + 1) h/2), then the hidden norm's.
 */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_mtp_norm(prefill::MtpParams params) {
	using namespace engine::device::prefill;
	auto const reused = prims::policy<Eviction::Last>();
	Count const lane_first = Count{prims::lane()} * 4;
	constexpr Count half_tiles = hidden_tiles / engine::model::cards;
	Count const own_first_tile = Count{params.card} * half_tiles;
	Bf16 const* const weights[2] = {at<Bf16 const>(params.embedding_norm), at<Bf16 const>(params.hidden_norm)};
	float const* const squares[2] = {at<float const>(params.workspace + workspace.embedding_squares),
	                                 at<float const>(params.workspace + workspace.hidden_squares)};
	for (unsigned item = global_warp(); item < params.rows.computed * static_cast<unsigned>(hidden_tiles); item += rows_warps) {
		unsigned const row = item / static_cast<unsigned>(hidden_tiles);
		Count const tile = item % hidden_tiles;
		Count const input = tile / half_tiles;
		Count const first = (own_first_tile + tile % half_tiles) * schedule::tile_rows + lane_first;
		float values[4] = {};
		if (row < params.rows.count) {
			float const factor = engine::model::rms_factor(row_squares(squares[input], row), engine::model::hidden);
			float x[4] = {};
			if (input == 0) {
				/* SAFETY: the embeddings are fp32 chunks of the workspace. */
				Chunk const loaded =
				    prims::load(reinterpret_cast<Chunk const*>(at<float const>(params.workspace + workspace.mtp_embedding) +
					                                           Count{row} * engine::model::hidden + first),
					            reused);
#pragma unroll
				for (int index = 0; index < 4; ++index) {
					x[index] = __builtin_bit_cast(float, loaded.word[index]);
				}
			} else {
				load_four(at<Bf16 const>(params.workspace + workspace.mtp_hidden) + Count{row} * engine::model::hidden, first, reused, x);
			}
			float w[4] = {};
			load_four(weights[input], first, reused, w);
#pragma unroll
			for (int index = 0; index < 4; ++index) {
				values[index] = engine::model::one_plus_scaled(x[index], factor, w[index]);
			}
		}
		store_four(at<Bf16>(params.workspace + workspace.mtp_input) + Count{row} * engine::model::hidden + tile * schedule::tile_rows +
		               lane_first,
		           values);
	}
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_keep = prefill::rows_budget;

/** The chunk's last final-norm row, kept as the slot's carried row and as the prompt end's row. */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_keep(prefill::KeepParams params) {
	using namespace engine::device::prefill;
	auto const reused = prims::policy<Eviction::Last>();
	constexpr Count row_chunks = engine::model::hidden * bf16_bytes / 16;
	auto const* const from = at<Chunk const>(params.workspace + workspace.final_normed) + Count{params.row} * row_chunks;
	auto* const carried = at<Chunk>(params.workspace + workspace.carried) + Count{params.slot} * row_chunks;
	auto* const prompt = at<Chunk>(params.prompt);
	for (Count chunk = global_thread(); chunk < row_chunks; chunk += Count{rows_blocks} * rows_budget.threads) {
		Chunk const value = prims::load(from + chunk, reused);
		prims::store(carried + chunk, value, reused);
		prims::store(prompt + chunk, value, prims::policy<Eviction::Normal>());
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The weight matmuls: one launch per ring placement, 82 blocks of the decode's warp roles, each a producer warp
// streaming its work items' tiles and eight consumer warps computing them.

namespace engine::device::prefill {

/** The query, key, value and gate projections into their fp32 rows, as attention reads them. */
template<Op op, model::Stream stream, Count ring>
__attribute__((device, always_inline)) inline void attention_rows(GemmParams const& params, Count offset) {
	project_f32<op, stream, ring>(params, offset, params.workspace + workspace.normed);
}

}

// The trunk's matmuls.

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_deltanet_in =
    prefill::gemm_budget<Op::DeltanetIn, Stream::Body, 0>;

/** in_proj_qkv on the normed rows, rounded once to bf16: the convolution's inputs. */
extern "C"
    __attribute__((global, launch_bounds(engine_budget_prefill_deltanet_in.threads, engine_budget_prefill_deltanet_in.blocks_per_sm))) void
    prefill_deltanet_in(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	constexpr gemv::Matrix m = matrix<Op::DeltanetIn, Stream::Body, 0>;
	RowsBf16<m> out = {.words = at<unsigned>(params.workspace + workspace.qkv)};
	gemm<m>(params.weights, params.workspace + workspace.normed, params.row_blocks, out);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_deltanet_z =
    prefill::gemm_budget<Op::DeltanetZ, Stream::Body, 0>;

/** in_proj_z on the normed rows, gating the deltanet core's outputs into out_proj's input rows. */
extern "C"
    __attribute__((global, launch_bounds(engine_budget_prefill_deltanet_z.threads, engine_budget_prefill_deltanet_z.blocks_per_sm))) void
    prefill_deltanet_z(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	constexpr gemv::Matrix m = matrix<Op::DeltanetZ, Stream::Body, 0>;
	GatedCore<m> out = {.core = at<float const>(params.workspace + workspace.core),
	                    .gated = at<unsigned>(params.workspace + workspace.gated)};
	gemm<m>(params.weights, params.workspace + workspace.normed, params.row_blocks, out);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_deltanet_out =
    prefill::gemm_budget<Op::DeltanetOut, Stream::Body, 0>;

/** out_proj on the gated rows: this card's partial sums, pushed to the peer item by item. */
extern "C" __attribute__((global,
                          launch_bounds(engine_budget_prefill_deltanet_out.threads, engine_budget_prefill_deltanet_out.blocks_per_sm))) void
prefill_deltanet_out(prefill::PushParams params) {
	using namespace engine::device::prefill;
	project_push<Op::DeltanetOut, Stream::Body, 0>(params, params.workspace + workspace.gated);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_attention_query =
    prefill::gemm_budget<Op::AttentionIn, Stream::Body, 0>;

/** The query half of q_proj on the normed rows: fp32 query rows. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_attention_query.threads,
                                                engine_budget_prefill_attention_query.blocks_per_sm))) void
prefill_attention_query(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionIn, Stream::Body, 0>(params, workspace.query);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_attention_key =
    prefill::gemm_budget<Op::AttentionIn, Stream::Body, 1>;

/** k_proj on the normed rows: fp32 key rows. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_attention_key.threads,
                                                engine_budget_prefill_attention_key.blocks_per_sm))) void
prefill_attention_key(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionIn, Stream::Body, 1>(params, workspace.key);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_attention_value =
    prefill::gemm_budget<Op::AttentionIn, Stream::Body, 2>;

/** v_proj on the normed rows: fp32 value rows. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_attention_value.threads,
                                                engine_budget_prefill_attention_value.blocks_per_sm))) void
prefill_attention_value(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionIn, Stream::Body, 2>(params, workspace.value);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_attention_gate =
    prefill::gemm_budget<Op::AttentionGate, Stream::Body, 0>;

/** The gate half of q_proj on the normed rows: fp32 gate rows, which the prefill attention applies. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_attention_gate.threads,
                                                engine_budget_prefill_attention_gate.blocks_per_sm))) void
prefill_attention_gate(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionGate, Stream::Body, 0>(params, workspace.attention_gate);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_attention_out =
    prefill::gemm_budget<Op::AttentionOut, Stream::Body, 0>;

/** o_proj on the gated attention outputs: this card's partial sums, pushed to the peer item by item. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_attention_out.threads,
                                                engine_budget_prefill_attention_out.blocks_per_sm))) void
prefill_attention_out(prefill::PushParams params) {
	using namespace engine::device::prefill;
	project_push<Op::AttentionOut, Stream::Body, 0>(params, params.workspace + workspace.attention_output);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mlp_in =
    prefill::gemm_budget<Op::MlpIn, Stream::Body, 0>;

/** gate_proj and up_proj interleaved on the normed rows, into silu(gate) * up rounded once. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mlp_in.threads, engine_budget_prefill_mlp_in.blocks_per_sm))) void
prefill_mlp_in(prefill::GemmParams params) {
	prefill::mlp_in<Stream::Body>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mlp_out =
    prefill::gemm_budget<Op::MlpOut, Stream::Body, 0>;

/** down_proj on the product rows: this card's partial sums, pushed to the peer item by item. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mlp_out.threads, engine_budget_prefill_mlp_out.blocks_per_sm))) void
prefill_mlp_out(prefill::PushParams params) {
	using namespace engine::device::prefill;
	project_push<Op::MlpOut, Stream::Body, 0>(params, params.workspace + workspace.product);
}

// The mtp head's matmuls over its prompt entries.

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_fc =
    prefill::gemm_budget<Op::MtpIn, Stream::Mtp, 0>;

/** fc on the normed embedding and hidden halves: this card's partial sums of the mtp residual, pushed item by item. */
extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mtp_fc.threads, engine_budget_prefill_mtp_fc.blocks_per_sm))) void
prefill_mtp_fc(prefill::PushParams params) {
	using namespace engine::device::prefill;
	project_push<Op::MtpIn, Stream::Mtp, 0>(params, params.workspace + workspace.mtp_input);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_attention_query =
    prefill::gemm_budget<Op::AttentionIn, Stream::Mtp, 0>;

extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mtp_attention_query.threads,
                                                engine_budget_prefill_mtp_attention_query.blocks_per_sm))) void
prefill_mtp_attention_query(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionIn, Stream::Mtp, 0>(params, workspace.query);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_attention_key =
    prefill::gemm_budget<Op::AttentionIn, Stream::Mtp, 1>;

extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mtp_attention_key.threads,
                                                engine_budget_prefill_mtp_attention_key.blocks_per_sm))) void
prefill_mtp_attention_key(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionIn, Stream::Mtp, 1>(params, workspace.key);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_attention_value =
    prefill::gemm_budget<Op::AttentionIn, Stream::Mtp, 2>;

extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mtp_attention_value.threads,
                                                engine_budget_prefill_mtp_attention_value.blocks_per_sm))) void
prefill_mtp_attention_value(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionIn, Stream::Mtp, 2>(params, workspace.value);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_attention_gate =
    prefill::gemm_budget<Op::AttentionGate, Stream::Mtp, 0>;

extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mtp_attention_gate.threads,
                                                engine_budget_prefill_mtp_attention_gate.blocks_per_sm))) void
prefill_mtp_attention_gate(prefill::GemmParams params) {
	using namespace engine::device::prefill;
	attention_rows<Op::AttentionGate, Stream::Mtp, 0>(params, workspace.attention_gate);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_attention_out =
    prefill::gemm_budget<Op::AttentionOut, Stream::Mtp, 0>;

extern "C" __attribute__((global, launch_bounds(engine_budget_prefill_mtp_attention_out.threads,
                                                engine_budget_prefill_mtp_attention_out.blocks_per_sm))) void
prefill_mtp_attention_out(prefill::PushParams params) {
	using namespace engine::device::prefill;
	project_push<Op::AttentionOut, Stream::Mtp, 0>(params, params.workspace + workspace.attention_output);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_mlp_in =
    prefill::gemm_budget<Op::MlpIn, Stream::Mtp, 0>;

extern "C"
    __attribute__((global, launch_bounds(engine_budget_prefill_mtp_mlp_in.threads, engine_budget_prefill_mtp_mlp_in.blocks_per_sm))) void
    prefill_mtp_mlp_in(prefill::GemmParams params) {
	prefill::mlp_in<Stream::Mtp>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_mtp_mlp_out =
    prefill::gemm_budget<Op::MlpOut, Stream::Mtp, 0>;

extern "C"
    __attribute__((global, launch_bounds(engine_budget_prefill_mtp_mlp_out.threads, engine_budget_prefill_mtp_mlp_out.blocks_per_sm))) void
    prefill_mtp_mlp_out(prefill::PushParams params) {
	using namespace engine::device::prefill;
	project_push<Op::MlpOut, Stream::Mtp, 0>(params, params.workspace + workspace.product);
}

// ---------------------------------------------------------------------------------------------------------------------
// The self-check.

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prefill_check_words = prefill::rows_budget;

/**
 * Compares a region a chunk left on the card with the host emulation's bits for it, word by word: run after a chunk
 * at bring-up, it holds every launch of the sequence to the emulation the equality tests run. Each thread counts its
 * comparisons once and keeps its mismatches in the report's first slots.
 */
extern "C" __attribute__((global, launch_bounds(prefill::rows_budget.threads, prefill::rows_budget.blocks_per_sm))) void
prefill_check_words(prefill::CheckParams params) {
	using namespace engine::device::prefill;
	auto const once = prims::policy<Eviction::First>();
	auto const* const got = at<unsigned const>(params.got);
	auto const* const expected = at<unsigned const>(params.expected);
	auto* const report = at<prims::selfcheck::Report>(params.report);
	constexpr unsigned long long threads = static_cast<unsigned long long>(rows_blocks) * rows_budget.threads;
	unsigned checked = 0;
	for (unsigned long long word = global_thread(); word < params.words; word += threads) {
		unsigned const seen = prims::load_word(got + word, once);
		unsigned const wanted = prims::load_word(expected + word, once);
		++checked;
		if (seen != wanted) {
			unsigned const slot = prims::fetch_add(&report->mismatched, 1U);
			if (slot < prims::selfcheck::kept_mismatches) {
				report->first[slot] = {
				    .input = static_cast<unsigned>(word), .detail = static_cast<unsigned>(word >> 32U), .got = seen, .expected = wanted};
			}
		}
	}
	prims::count(&report->checked, checked);
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight matmuls' self-check kernels: each runs the decode gemv or the prefill gemm, exactly as the decode and
// prefill kernels compose them, over a real placement's geometry on host-written weights, activations and scales,
// and compares every output with what the host's emulation of every warp (`engine.gemv`, through the bit-exact mma
// model) says it must be. Built to ptx now; run at bring-up by the microbenchmark drivers, after U6 has confirmed the
// mma model. Every kernel runs on `selfcheck::blocks` blocks of `selfcheck::threads`; reports, flags and outputs start
// zeroed.
#include "gemv/kernels.h"

namespace kernels = engine::device::gemv::kernels;
namespace gemv = engine::device::gemv;
namespace prims = engine::device::prims;
using Report = prims::selfcheck::Report;

namespace engine::device::gemv::kernels {

/** Stores each finished tile like `StoreTiles` and compares each value's bits with the host's. */
template<class S>
struct CheckTiles {
	float* out;
	unsigned const* expected;
	Report* report;

	__attribute__((device, always_inline)) void operator()(unsigned tile, Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab_index) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const once = policy<Eviction::First>();
		auto const normal = policy<Eviction::Normal>();
#pragma unroll
		for (int t = 0; t < S::tensors; ++t) {
#pragma unroll
			for (int n = 0; n < S::row_tiles; ++n) {
#pragma unroll
				for (int reg = 0; reg < 4; ++reg) {
					unsigned const index = tile * static_cast<unsigned>(S::fixup_floats) +
					                       static_cast<unsigned>(slot_index<S>(t, element(slab_index, lane_index, n, reg)));
					unsigned const got = __builtin_bit_cast(unsigned, totals.tensor[t].tile[n].reg[reg]);
					unsigned const wanted = load_word(expected + index, once);
					store_word(reinterpret_cast<unsigned*>(out + index), got, normal);
					record(report, got == wanted, {.input = index, .detail = tile, .got = got, .expected = wanted});
				}
			}
		}
	}
};

/** Stores each finished tile's silu(gate) * up like `GateUpProduct` and compares each word with the host's. */
template<class S>
struct CheckProduct {
	unsigned* out;
	unsigned const* expected;
	Report* report;

	using Product = GateUpProduct<S, selfcheck::product_width>;

	__attribute__((device, always_inline)) void operator()(unsigned tile, Totals<S::tensors, S::row_tiles> const& totals,
	                                                       unsigned slab_index) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const once = policy<Eviction::First>();
		auto const normal = policy<Eviction::Normal>();
#pragma unroll
		for (int n = 0; n < S::row_tiles; ++n) {
			product_pairs(
			    totals.tensor[0].tile[n], totals.tensor[1].tile[n],
			    [&](int reg) {
				    return element(slab_index, lane_index, n, reg);
			    },
			    [&](Element at, Bf16x2 pair) {
				    auto const index = static_cast<unsigned>(Product::word_index(tile, at));
				    unsigned const wanted = load_word(expected + index, once);
				    store_word(out + index, pair.bits, normal);
				    record(report, pair.bits == wanted, {.input = index, .detail = tile, .got = pair.bits, .expected = wanted});
			    });
		}
	}
};

/** The prefill's `StoreRows`, comparing each value's bits with the host's. */
template<Matrix matrix>
struct CheckRows {
	float* out;
	unsigned const* expected;
	Report* report;
	unsigned rows;

	__attribute__((device, always_inline)) void operator()(prefill::Item at, prefill::Warp<static_cast<int>(matrix.tensors)> const& totals,
	                                                       unsigned warp) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const once = policy<Eviction::First>();
		auto const normal = policy<Eviction::Normal>();
#pragma unroll
		for (int t = 0; t < static_cast<int>(matrix.tensors); ++t) {
#pragma unroll
			for (int slab = 0; slab < prefill::warp_slabs; ++slab) {
#pragma unroll
				for (int n = 0; n < prefill::warp_row_tiles; ++n) {
#pragma unroll
					for (int reg = 0; reg < 4; ++reg) {
						unsigned const index =
						    prefill::StoreRows<matrix>::index(at, prefill::element(warp, lane_index, slab, n, reg), t, rows);
						unsigned const got = __builtin_bit_cast(unsigned, totals.tensor[t][slab].tile[n].reg[reg]);
						unsigned const wanted = load_word(expected + index, once);
						store_word(reinterpret_cast<unsigned*>(out + index), got, normal);
						record(report, got == wanted, {.input = index, .detail = at.tile, .got = got, .expected = wanted});
					}
				}
			}
		}
	}
};

/** The prefill's `GateUpRows`, comparing each word with the host's. */
template<Matrix matrix>
struct CheckProductRows {
	unsigned* out;
	unsigned const* expected;
	Report* report;

	using Product = prefill::GateUpRows<matrix, selfcheck::product_width>;

	__attribute__((device, always_inline)) void operator()(prefill::Item at, prefill::Warp<2> const& totals, unsigned warp) const {
		using namespace engine::device::prims;
		unsigned const lane_index = lane();
		auto const once = policy<Eviction::First>();
		auto const normal = policy<Eviction::Normal>();
#pragma unroll
		for (int slab = 0; slab < prefill::warp_slabs; ++slab) {
#pragma unroll
			for (int n = 0; n < prefill::warp_row_tiles; ++n) {
				product_pairs(
				    totals.tensor[0][slab].tile[n], totals.tensor[1][slab].tile[n],
				    [&](int reg) {
					    return prefill::element(warp, lane_index, slab, n, reg);
				    },
				    [&](Element e, Bf16x2 pair) {
					    unsigned const index = Product::word_index(at, e);
					    unsigned const wanted = load_word(expected + index, once);
					    store_word(out + index, pair.bits, normal);
					    record(report, pair.bits == wanted, {.input = index, .detail = at.tile, .got = pair.bits, .expected = wanted});
				    });
			}
		}
	}
};

using Down3 = Gemv<gemv::down, 3>;
using Down4 = Gemv<gemv::down, 4>;
using GateUp3 = Gemv<gemv::gate_up, 3>;
using GateUp4 = Gemv<gemv::gate_up, 4>;

static_assert(prims::check(gemv_budget<Down4>) == prims::Fit::Fits && prims::check(gemv_budget<GateUp4>) == prims::Fit::Fits);
static_assert(prims::check(gemm_budget<gemv::down>) == prims::Fit::Fits && prims::check(gemm_budget<gemv::gate_up>) == prims::Fit::Fits);

}

namespace engine::device::gemv::kernels {

template<class S>
__attribute__((device, always_inline)) inline void check_tiles(selfcheck::GemvCheckParams const& params) {
	CheckTiles<S> epilogue = {
	    .out = pointer<float>(params.out), .expected = pointer<unsigned const>(params.expected), .report = pointer<Report>(params.report)};
	run_gemv<S>(gemv_inputs(params, params.gate_scales, params.up_scales), epilogue);
}

template<class S>
__attribute__((device, always_inline)) inline void check_product(selfcheck::GemvCheckParams const& params) {
	CheckProduct<S> epilogue = {.out = pointer<unsigned>(params.out),
	                            .expected = pointer<unsigned const>(params.expected),
	                            .report = pointer<Report>(params.report)};
	run_gemv<S>(gemv_inputs(params, params.gate_scales, params.up_scales), epilogue);
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto gemm_operands(selfcheck::GemmCheckParams const& params)
    -> prefill::Operands {
	return {
	    .activations = pointer<prims::Chunk const>(params.activations),
	    .tiles = pointer<prims::Chunk const>(params.tiles),
	    .scales = {pointer<unsigned short const>(params.gate_scales), pointer<unsigned short const>(params.up_scales)},
	};
}

}

/**
 * The decode gemv over down_proj's geometry at 24 rows: the baked placement, activations [24][ffn_per_card] bf16 and
 * its block scales; `out` gets [tile][row][128] fp32 and every value is compared with `expected`.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_check_down_r3 = kernels::gemv_budget<kernels::Down3>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Down3>.threads, 1))) void
gemv_check_down_r3(gemv::selfcheck::GemvCheckParams params) {
	kernels::check_tiles<kernels::Down3>(params);
}

/** As `gemv_check_down_r3`, at 32 rows. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_check_down_r4 = kernels::gemv_budget<kernels::Down4>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Down4>.threads, 1))) void
gemv_check_down_r4(gemv::selfcheck::GemvCheckParams params) {
	kernels::check_tiles<kernels::Down4>(params);
}

/**
 * The decode gemv over gate_proj and up_proj interleaved at 24 rows, into silu(gate) * up: activations [24][hidden]
 * bf16; `out` gets the bf16 product rows [24][ffn_per_card] as words, each compared with `expected`.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_check_gate_up_r3 =
    kernels::gemv_budget<kernels::GateUp3>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::GateUp3>.threads, 1))) void
gemv_check_gate_up_r3(gemv::selfcheck::GemvCheckParams params) {
	kernels::check_product<kernels::GateUp3>(params);
}

/** As `gemv_check_gate_up_r3`, at 32 rows. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_check_gate_up_r4 =
    kernels::gemv_budget<kernels::GateUp4>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::GateUp4>.threads, 1))) void
gemv_check_gate_up_r4(gemv::selfcheck::GemvCheckParams params) {
	kernels::check_product<kernels::GateUp4>(params);
}

/**
 * The prefill gemm over down_proj's geometry for `selfcheck::prefill_check_rows` rows: `out` gets [row][hidden] fp32,
 * each value compared with `expected`; the decode's segment order makes it the decode's bits row for row.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemm_check_down = kernels::gemm_budget<gemv::down>;

extern "C" __attribute__((global, launch_bounds(kernels::gemm_budget<gemv::down>.threads, 1))) void
gemm_check_down(gemv::selfcheck::GemmCheckParams params) {
	constexpr unsigned rows = gemv::selfcheck::prefill_check_rows;
	kernels::CheckRows<gemv::down> epilogue = {.out = kernels::pointer<float>(params.out),
	                                           .expected = kernels::pointer<unsigned const>(params.expected),
	                                           .report = kernels::pointer<Report>(params.report),
	                                           .rows = rows};
	kernels::run_gemm<gemv::down>(kernels::gemm_operands(params), rows / gemv::prefill::block_rows, epilogue);
}

/** The prefill gemm over gate_proj and up_proj into silu(gate) * up rows, each word compared with `expected`. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemm_check_gate_up = kernels::gemm_budget<gemv::gate_up>;

extern "C" __attribute__((global, launch_bounds(kernels::gemm_budget<gemv::gate_up>.threads, 1))) void
gemm_check_gate_up(gemv::selfcheck::GemmCheckParams params) {
	constexpr unsigned rows = gemv::selfcheck::prefill_check_rows;
	kernels::CheckProductRows<gemv::gate_up> epilogue = {.out = kernels::pointer<unsigned>(params.out),
	                                                     .expected = kernels::pointer<unsigned const>(params.expected),
	                                                     .report = kernels::pointer<Report>(params.report)};
	kernels::run_gemm<gemv::gate_up>(kernels::gemm_operands(params), rows / gemv::prefill::block_rows, epilogue);
}

inline constexpr prims::Budget residual_budget = {.threads = 32, .registers_per_thread = 128, .shared_bytes = 0, .blocks_per_sm = 4};
static_assert(prims::check(residual_budget) == prims::Fit::Fits);

/**
 * The residual update and its tile squares: block t (of `selfcheck::residual_tiles`) adds both cards' sums of output
 * tile t into the residual's columns of the tile and writes the tile's squares per row; each new residual value and
 * each square is compared with the host's.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_check_residual = residual_budget;

extern "C" __attribute__((global, launch_bounds(residual_budget.threads, residual_budget.blocks_per_sm))) void
gemv_check_residual(gemv::selfcheck::ResidualCheckParams params) {
	using namespace engine::device::prims;
	constexpr int rows = gemv::selfcheck::residual_rows;
	constexpr unsigned tile_chunks = static_cast<unsigned>(gemv::tile_rows / 4);
	constexpr unsigned residual_chunks = static_cast<unsigned>(engine::model::hidden / 4);
	unsigned const tile = block();
	unsigned const lane_index = lane();
	auto* const residual = kernels::pointer<Chunk>(params.residual);
	auto* const squares = kernels::pointer<float>(params.squares) + static_cast<gemv::Count>(tile) * rows;
	gemv::Count const tile_offset = static_cast<gemv::Count>(tile) * rows * tile_chunks;
	gemv::add_to_residual<rows>(kernels::pointer<Chunk const>(params.own) + tile_offset,
	                            kernels::pointer<Chunk const>(params.peer) + tile_offset, residual, tile, squares);
	auto const once = policy<Eviction::First>();
	auto* const report = kernels::pointer<Report>(params.report);
	auto const* const expected_residual = kernels::pointer<Chunk const>(params.expected_residual);
	auto const* const expected_squares = kernels::pointer<unsigned const>(params.expected_squares);
	for (unsigned row = 0; row < static_cast<unsigned>(rows); ++row) {
		gemv::Count const at = static_cast<gemv::Count>(row) * residual_chunks + static_cast<gemv::Count>(tile) * tile_chunks + lane_index;
		Chunk const got = load(residual + at, once);
		Chunk const wanted = load_constant(expected_residual + at, once);
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			kernels::record(report, got.word[word] == wanted.word[word],
			                {.input = row, .detail = tile, .got = got.word[word], .expected = wanted.word[word]});
		}
		if (lane_index == 0) {
			unsigned const square = load_word(reinterpret_cast<unsigned const*>(squares + row), once);
			unsigned const wanted_square = load_word(expected_squares + static_cast<gemv::Count>(tile) * rows + row, once);
			kernels::record(report, square == wanted_square,
			                {.input = row, .detail = tile + 1000U, .got = square, .expected = wanted_square});
		}
	}
}

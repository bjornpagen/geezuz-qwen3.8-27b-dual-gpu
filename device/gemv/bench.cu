// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight matmuls' microbenchmark kernels, run by the ubench drivers on the cards:
// - U5: the real streaming gemv (ring, sequence-B decode, mma, scale fma, cross-SM finish) over down_proj's 44.6 MB at
//   8, 16, 24, 32 and 40 rows: `gemv_stream_r1` .. `gemv_stream_r5`. Flat GB/s through 32 rows and a drop at 40 is the
//   expectation that picks the row budget.
// - U22: the prefill gemm in situ, `selfcheck::prefill_rows` rows through down_proj (`gemm_prefill_down`) and through
//   gate_proj with up_proj into silu(gate) * up (`gemm_prefill_gate_up`).
// Each block's consumer warp 0 lane 0 writes its elapsed SM clocks (32 bits) to `elapsed[block]`. Grids are `selfcheck::blocks`
// blocks of `selfcheck::threads`; fixup flags start zeroed.
#include "gemv/kernels.h"

namespace kernels = engine::device::gemv::kernels;
namespace gemv = engine::device::gemv;
namespace prims = engine::device::prims;

namespace engine::device::gemv::kernels {

/** Records one block's elapsed clocks since `start`, from its first consumer's first lane. */
__attribute__((device, always_inline)) inline void record_elapsed(unsigned* elapsed, unsigned long long start) {
	using namespace engine::device::prims;
	if (Consumers::holds(warp_in_block()) && Consumers::rank(warp_in_block()) == 0 && lane() == 0) {
		store_word(elapsed + block(), static_cast<unsigned>(cycles() - start), policy<Eviction::Normal>());
	}
}

template<int row_tiles>
using Stream = Gemv<gemv::down, row_tiles>;

template<int row_tiles>
__attribute__((device, always_inline)) inline void stream(selfcheck::GemvStreamParams const& params) {
	StoreTiles<Stream<row_tiles>> epilogue = {.out = pointer<float>(params.out)};
	unsigned long long const start = prims::cycles();
	run_gemv<Stream<row_tiles>>(gemv_inputs(params, params.scales, params.scales), epilogue);
	record_elapsed(pointer<unsigned>(params.elapsed), start);
}

[[nodiscard]] __attribute__((device, always_inline)) inline auto gemm_operands(selfcheck::GemmBenchParams const& params)
    -> prefill::Operands {
	return {
	    .activations = pointer<prims::Chunk const>(params.activations),
	    .tiles = pointer<prims::Chunk const>(params.tiles),
	    .scales = {pointer<unsigned short const>(params.gate_scales), pointer<unsigned short const>(params.up_scales)},
	};
}

}

/**
 * U5 at 8 rows: down_proj's placement and activations [8][ffn_per_card] bf16 into fp32 tiles [tile][row][128]. The
 * r2 .. r5 kernels take 16 .. 40 rows.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_stream_r1 = kernels::gemv_budget<kernels::Stream<1>>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Stream<1>>.threads, 1))) void
gemv_stream_r1(gemv::selfcheck::GemvStreamParams params) {
	kernels::stream<1>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_stream_r2 = kernels::gemv_budget<kernels::Stream<2>>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Stream<2>>.threads, 1))) void
gemv_stream_r2(gemv::selfcheck::GemvStreamParams params) {
	kernels::stream<2>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_stream_r3 = kernels::gemv_budget<kernels::Stream<3>>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Stream<3>>.threads, 1))) void
gemv_stream_r3(gemv::selfcheck::GemvStreamParams params) {
	kernels::stream<3>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_stream_r4 = kernels::gemv_budget<kernels::Stream<4>>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Stream<4>>.threads, 1))) void
gemv_stream_r4(gemv::selfcheck::GemvStreamParams params) {
	kernels::stream<4>(params);
}

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemv_stream_r5 = kernels::gemv_budget<kernels::Stream<5>>;

extern "C" __attribute__((global, launch_bounds(kernels::gemv_budget<kernels::Stream<5>>.threads, 1))) void
gemv_stream_r5(gemv::selfcheck::GemvStreamParams params) {
	kernels::stream<5>(params);
}

/** U22: `selfcheck::prefill_rows` rows (bf16 [rows][ffn_per_card]) through down_proj into fp32 rows [rows][hidden]. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemm_prefill_down = kernels::gemm_budget<gemv::down>;

extern "C" __attribute__((global, launch_bounds(kernels::gemm_budget<gemv::down>.threads, 1))) void
gemm_prefill_down(gemv::selfcheck::GemmBenchParams params) {
	constexpr unsigned rows = gemv::selfcheck::prefill_rows;
	gemv::prefill::StoreRows<gemv::down> epilogue = {.out = kernels::pointer<float>(params.out), .rows = rows};
	unsigned long long const start = prims::cycles();
	kernels::run_gemm<gemv::down>(kernels::gemm_operands(params), rows / gemv::prefill::block_rows, epilogue);
	kernels::record_elapsed(kernels::pointer<unsigned>(params.elapsed), start);
}

/** U22 on gate_proj and up_proj: bf16 [rows][hidden] into the bf16 product rows [rows][ffn_per_card]. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_gemm_prefill_gate_up = kernels::gemm_budget<gemv::gate_up>;

extern "C" __attribute__((global, launch_bounds(kernels::gemm_budget<gemv::gate_up>.threads, 1))) void
gemm_prefill_gate_up(gemv::selfcheck::GemmBenchParams params) {
	constexpr unsigned rows = gemv::selfcheck::prefill_rows;
	gemv::prefill::GateUpRows<gemv::gate_up, gemv::selfcheck::product_width> epilogue = {.out = kernels::pointer<unsigned>(params.out)};
	unsigned long long const start = prims::cycles();
	kernels::run_gemm<gemv::gate_up>(kernels::gemm_operands(params), rows / gemv::prefill::block_rows, epilogue);
	kernels::record_elapsed(kernels::pointer<unsigned>(params.elapsed), start);
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight matmuls' standalone kernel images: the self-checks (`gemv_check_down_r3`, `_r4`, `gemv_check_gate_up_r3`,
// `_r4`, `gemm_check_down`, `gemm_check_gate_up`) and the microbenchmarks (`gemv_stream_r1` .. `_r5` for U5,
// `gemm_prefill_down` and `gemm_prefill_gate_up` for U22). Each kernel takes one parameter struct of `selfcheck` and
// carries its budget record in its image.
export module engine.gemv:images;

import std;
import engine.ptx;

namespace engine::gemv {

constexpr char check_text[] = {
#embed "gemv_check.ptx" suffix(, '\0')
};

constexpr char bench_text[] = {
#embed "gemv_bench.ptx" suffix(, '\0')
};

/** The self-check kernels. */
export inline constexpr ptx::Image check_image = ptx::image(check_text);
/** The U5 and U22 kernels. */
export inline constexpr ptx::Image bench_image = ptx::image(bench_text);

static_assert(check_image.header == ptx::abi.header && bench_image.header == ptx::abi.header,
              "every image shares the engine's isa and target");

}

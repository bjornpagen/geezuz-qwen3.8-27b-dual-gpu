// SPDX-License-Identifier: AGPL-3.0-only
//
// The build's proof that device code is compiled with contraction off: the ptx gate requires this kernel's body to
// round a * b + c as `mul.rn.f32` then `add.rn.f32`, with no fma, so a lost `-ffp-contract=off` fails the build.
#include "prims/budget.h"

inline constexpr engine::device::prims::Budget contraction_budget = {.threads = 32, .registers_per_thread = 32, .shared_bytes = 0, .blocks_per_sm = 1};
static_assert(engine::device::prims::fits(contraction_budget));

extern "C" __attribute__((device, used)) constexpr engine::device::prims::Budget engine_budget_prims_contraction_probe = contraction_budget;

extern "C" __attribute__((global, launch_bounds(contraction_budget.threads, contraction_budget.blocks_per_sm))) void prims_contraction_probe(float* values) {
	values[3] = values[0] * values[1] + values[2];
}

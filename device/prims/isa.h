// SPDX-License-Identifier: AGPL-3.0-only
//
// The level each primitive needs, stated beside it and checked against the build's target at compile time: a
// primitive whose ptx the target cannot express does not compile, rather than failing when the driver jits the image.
#pragma once

#include "target.h"

namespace engine::device::prims {

/** A ptx isa version (major * 10 + minor, as `ptx_isa`) and an sm level (major * 10 + minor, as `sm`). */
struct Isa {
	int ptx;
	int sm;
};

/** Whether the build's target can run what `need` names; every primitive `static_assert`s it. */
[[nodiscard]] consteval auto available(Isa need) -> bool {
	return ptx_isa >= need.ptx && sm >= need.sm;
}

// The levels the primitives cite, each the first ptx isa and sm that has every form the primitive spells.
namespace isa {
inline constexpr Isa special_registers = {.ptx = 20, .sm = 20};
inline constexpr Isa named_barrier = {.ptx = 60, .sm = 30};
inline constexpr Isa warp_sync = {.ptx = 60, .sm = 30};
inline constexpr Isa shuffle = {.ptx = 60, .sm = 30};
inline constexpr Isa vote = {.ptx = 60, .sm = 30};
inline constexpr Isa scoped_memory_order = {.ptx = 60, .sm = 70};
inline constexpr Isa scoped_fence = {.ptx = 60, .sm = 70};
inline constexpr Isa nanosleep = {.ptx = 63, .sm = 70};
inline constexpr Isa ldmatrix = {.ptx = 65, .sm = 75};
inline constexpr Isa integer_redux = {.ptx = 70, .sm = 80};
inline constexpr Isa cp_async = {.ptx = 70, .sm = 80};
inline constexpr Isa mbarrier = {.ptx = 70, .sm = 80};
inline constexpr Isa mbarrier_parity = {.ptx = 71, .sm = 80};
inline constexpr Isa mma_bf16 = {.ptx = 70, .sm = 80};
inline constexpr Isa fma_bf16x2 = {.ptx = 70, .sm = 80};
inline constexpr Isa cache_policy = {.ptx = 74, .sm = 80};
inline constexpr Isa l2_prefetch_size = {.ptx = 74, .sm = 80};
inline constexpr Isa l2_discard = {.ptx = 74, .sm = 80};
inline constexpr Isa integer_logic = {.ptx = 43, .sm = 50};
inline constexpr Isa byte_permute = {.ptx = 20, .sm = 20};
}

}

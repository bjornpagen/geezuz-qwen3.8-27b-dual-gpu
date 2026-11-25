// SPDX-License-Identifier: AGPL-3.0-only
//
// Where a thread is, from the special registers (ungated nvvm builtins, so no inline ptx), and the warp roles of a
// warp-specialized kernel as types: a role is a fixed range of warps chosen at compile time, and a kernel's roles
// must tile its block exactly.
#pragma once

#include "prims/budget.h"
#include "prims/isa.h"

namespace engine::device::prims {

static_assert(available(isa::special_registers));

[[nodiscard]] __attribute__((device)) inline auto lane() -> unsigned {
	return static_cast<unsigned>(__nvvm_read_ptx_sreg_laneid());
}

/** The thread's index in its block; blocks are one-dimensional. */
[[nodiscard]] __attribute__((device)) inline auto thread_in_block() -> unsigned {
	return static_cast<unsigned>(__nvvm_read_ptx_sreg_tid_x());
}

/** The warp's index in its block, from the thread index (`%warpid` moves under preemption and is not an index). */
[[nodiscard]] __attribute__((device)) inline auto warp_in_block() -> unsigned {
	return thread_in_block() / 32U;
}

/** The block's index in its grid; grids are one-dimensional. */
[[nodiscard]] __attribute__((device)) inline auto block() -> unsigned {
	return static_cast<unsigned>(__nvvm_read_ptx_sreg_ctaid_x());
}

[[nodiscard]] __attribute__((device)) inline auto lanes_below() -> unsigned {
	return static_cast<unsigned>(__nvvm_read_ptx_sreg_lanemask_lt());
}

/** The SM's cycle counter, for the self-check kernels' timings. */
[[nodiscard]] __attribute__((device)) inline auto cycles() -> unsigned long long {
	return static_cast<unsigned long long>(__nvvm_read_ptx_sreg_clock64());
}

/** A role: warps [first, first + count) of the block. */
template<int first_, int count_>
struct Warps {
	static constexpr int first = first_;
	static constexpr int count = count_;
	static constexpr int threads = count * 32;
	static_assert(first >= 0 && count >= 1, "a role is a nonempty range of warps");

	[[nodiscard]] __attribute__((device)) static auto holds(unsigned warp) -> bool {
		return warp >= static_cast<unsigned>(first) && warp < static_cast<unsigned>(first + count);
	}

	/** The warp's index within the role. */
	[[nodiscard]] __attribute__((device)) static auto rank(unsigned warp) -> unsigned {
		return warp - static_cast<unsigned>(first);
	}
};

/** Whether the roles, in order, tile warps [0, budget.threads / 32) without gap or overlap. */
template<class... Roles>
[[nodiscard]] consteval auto tiles(Budget const& budget) -> bool {
	int next = 0;
	bool contiguous = true;
	((contiguous = contiguous && Roles::first == next, next += Roles::count), ...);
	return contiguous && next == warps(budget);
}

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// A kernel's shared memory: one region sized by its budget and laid out by one struct the kernel defines, and the
// 32-bit shared-window addresses that `cp.async`, `ldmatrix` and the mbarriers take. A shared address is its own type,
// so a global pointer cannot reach an instruction that reads the shared window.
#pragma once

#include "prims/budget.h"
#include "prims/isa.h"

namespace engine::device::prims {

static_assert(available(isa::special_registers));

/** An address in the block's shared-memory window. */
struct SharedAddress {
	unsigned value;

	[[nodiscard]] constexpr auto plus(unsigned bytes) const -> SharedAddress {
		return {.value = value + bytes};
	}
};

/** The shared-window address of an object in shared memory, given its generic address. */
[[nodiscard]] __attribute__((device)) inline auto shared_address(void const* object) -> SharedAddress {
	unsigned window = 0;
	// Why ptx: the generic-to-shared conversion is `cvta`, which no ungated builtin spells.
	asm("{\n\t.reg .u64 saddr;\n\tcvta.to.shared.u64 saddr, %1;\n\tcvt.u32.u64 %0, saddr;\n\t}" : "=r"(window) : "l"(object));
	return {.value = window};
}

/**
 * The block's shared memory as the kernel's layout; the layout must fit the budget the kernel launches with. All of
 * it is the block's dynamic shared memory, sized by the budget's `shared_bytes`, which the typed launch passes:
 * dynamic because a static `__shared__` array is capped at 48 KiB.
 */
template<Budget budget, class Layout>
[[nodiscard]] __attribute__((device)) inline auto shared_layout() -> Layout& {
	// A shared-memory declaration has no initializer at all; the check mistakes the cuda attribute for static storage.
	// NOLINTNEXTLINE(bugprone-dynamic-static-initializers)
	extern __attribute__((shared, aligned(128))) unsigned char engine_dynamic_shared[];
	static_assert(sizeof(Layout) <= static_cast<decltype(sizeof(Layout))>(budget.shared_bytes), "the shared layout exceeds the budget");
	static_assert(alignof(Layout) <= 128, "shared memory is 128-byte aligned");
	/* SAFETY: the launch gives every block budget.shared_bytes of dynamic shared memory, which holds one Layout. */
	return *reinterpret_cast<Layout*>(engine_dynamic_shared);
}

}

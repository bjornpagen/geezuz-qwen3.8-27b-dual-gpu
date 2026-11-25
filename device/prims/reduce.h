// SPDX-License-Identifier: AGPL-3.0-only
//
// Warp reductions. Integers go through `redux.sync` (sm_86 has no float form). Floats go through a butterfly of
// `shfl.sync.bfly` in a fixed order: at distance 16, 8, 4, 2, 1 each lane adds its partner's value. fp32 addition is
// commutative, so partners compute identical sums and every lane ends with the same bits, which depend only on the
// 32 inputs and never on timing.
#pragma once

#include "prims/isa.h"

namespace engine::device::prims {

static_assert(available(isa::integer_redux) && available(isa::shuffle) && available(isa::vote));

inline constexpr unsigned full_warp = 0xffffffffU;

/** The value of lane `lane ^ distance`; every lane of the warp takes part. */
[[nodiscard]] __attribute__((device)) inline auto exchange(unsigned value, unsigned distance) -> unsigned {
	unsigned received = 0;
	// Why ptx: the shuffle builtins are gated at ptx 9.3.
	asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(received) : "r"(value), "r"(distance));
	return received;
}

/** Lane `source`'s value, on every lane. */
[[nodiscard]] __attribute__((device)) inline auto broadcast(unsigned value, unsigned source) -> unsigned {
	unsigned received = 0;
	asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(received) : "r"(value), "r"(source));
	return received;
}

/** The lanes whose `holds` is true, bit l for lane l. */
[[nodiscard]] __attribute__((device)) inline auto ballot(bool holds) -> unsigned {
	unsigned mask = 0;
	asm volatile("{\n\t.reg .pred pholds;\n\tsetp.ne.u32 pholds, %1, 0;\n\tvote.sync.ballot.b32 %0, pholds, 0xffffffff;\n\t}"
	             : "=r"(mask)
	             : "r"(holds ? 1U : 0U));
	return mask;
}

[[nodiscard]] __attribute__((device)) inline auto warp_sum(unsigned value) -> unsigned {
	unsigned total = 0;
	asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(total) : "r"(value));
	return total;
}

[[nodiscard]] __attribute__((device)) inline auto warp_min(unsigned value) -> unsigned {
	unsigned least = 0;
	asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(least) : "r"(value));
	return least;
}

[[nodiscard]] __attribute__((device)) inline auto warp_max(unsigned value) -> unsigned {
	unsigned most = 0;
	asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(most) : "r"(value));
	return most;
}

[[nodiscard]] __attribute__((device)) inline auto warp_min(int value) -> int {
	int least = 0;
	asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(least) : "r"(value));
	return least;
}

[[nodiscard]] __attribute__((device)) inline auto warp_max(int value) -> int {
	int most = 0;
	asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(most) : "r"(value));
	return most;
}

[[nodiscard]] __attribute__((device)) inline auto warp_or(unsigned value) -> unsigned {
	unsigned any = 0;
	asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(any) : "r"(value));
	return any;
}

[[nodiscard]] __attribute__((device)) inline auto warp_and(unsigned value) -> unsigned {
	unsigned all = 0;
	asm volatile("redux.sync.and.b32 %0, %1, 0xffffffff;" : "=r"(all) : "r"(value));
	return all;
}

/** The fp32 sum of the warp's 32 values, bit-identical on every lane and every run. */
[[nodiscard]] __attribute__((device)) inline auto warp_sum(float value) -> float {
	float total = value;
	for (unsigned distance = 16; distance != 0; distance /= 2) {
		float const partner = __builtin_bit_cast(float, exchange(__builtin_bit_cast(unsigned, total), distance));
		total = total + partner;
	}
	return total;
}

/** The warp's largest fp32 value (a nan only if every lane holds one), on every lane. */
[[nodiscard]] __attribute__((device)) inline auto warp_max(float value) -> float {
	float most = value;
	for (unsigned distance = 16; distance != 0; distance /= 2) {
		float const partner = __builtin_bit_cast(float, exchange(__builtin_bit_cast(unsigned, most), distance));
		most = __builtin_fmaxf(most, partner);
	}
	return most;
}

}

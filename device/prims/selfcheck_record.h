// SPDX-License-Identifier: AGPL-3.0-only
//
// The device half of the self-check reports: counting a comparison into a `Report`, and turning a parameter struct's
// 64-bit device addresses back into typed pointers. Device-only (it runs atomics), beside selfcheck.h, which the host
// shares.
#pragma once

#include "prims/order.h"
#include "prims/selfcheck.h"

namespace engine::device::prims::selfcheck {

/** The address a parameter struct carries, as the pointer the host filled it with. */
template<class T>
[[nodiscard]] __attribute__((device, always_inline)) inline auto pointer(Address address) -> T* {
	/* SAFETY: the host fills every parameter with the device address of a buffer of exactly this element type. */
	return __builtin_bit_cast(T*, address);
}

/** Counts one comparison into `report`, keeping the first `kept_mismatches` mismatches verbatim. */
__attribute__((device, always_inline)) inline void record(Report* report, bool agrees, Mismatch mismatch) {
	count(&report->checked, 1U);
	if (!agrees) {
		unsigned const slot = fetch_add(&report->mismatched, 1U);
		if (slot < kept_mismatches) {
			report->first[slot] = mismatch;
		}
	}
}

}

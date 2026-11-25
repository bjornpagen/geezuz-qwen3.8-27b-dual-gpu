// SPDX-License-Identifier: AGPL-3.0-only
//
// What every microbenchmark kernel shares on the device side: typed pointers from the parameter addresses, and the
// two clocks a measurement reads (the SM's cycle counter and the card's nanosecond timer), both special registers.
#pragma once

#include "prims/isa.h"
#include "prims/sync.h"
#include "ubench/shapes.h"

namespace engine::device::ubench {

static_assert(prims::available(prims::isa::special_registers));

/** The object a parameter's device address names. */
template<class T>
[[nodiscard]] __attribute__((device)) inline auto at(Address address) -> T* {
	/* SAFETY: the host passes the device address of an allocation laid out as T, by the typed launch that built it. */
	return __builtin_bit_cast(T*, address);
}

/** The card's global nanosecond timer: one time base for every SM, unlike the per-SM cycle counter. */
[[nodiscard]] __attribute__((device)) inline auto nanoseconds() -> unsigned long long {
	return static_cast<unsigned long long>(__nvvm_read_ptx_sreg_globaltimer());
}

/** The SM this thread runs on: what proves two blocks really sit on two SMs. */
[[nodiscard]] __attribute__((device)) inline auto sm_id() -> unsigned {
	return static_cast<unsigned>(__nvvm_read_ptx_sreg_smid());
}

/** The backoff a `Wait` names: the engine's own for its context, or a spin that sleeps not at all. */
template<Wait wait>
[[nodiscard]] constexpr auto backoff_for(prims::Backoff production) -> prims::Backoff {
	if constexpr (wait == Wait::Backoff) {
		return production;
	} else {
		return {.first_ns = 0, .ceiling_ns = 0};
	}
}

}

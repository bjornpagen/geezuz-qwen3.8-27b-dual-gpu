// SPDX-License-Identifier: AGPL-3.0-only
//
// Device addresses as attention's kernel parameters carry them: integers, because the host holds device memory by its
// address and fills the parameter structs it shares with the kernels. This is the one place they become pointers.
#pragma once

#include "attn/geometry.h"

namespace engine::device::attn {

template<class T>
[[nodiscard]] __attribute__((device, always_inline)) inline auto at(Address address) -> T* {
	// The address is the provenance: it comes from a device allocation the host made and passed in by value.
	/* SAFETY: the host fills every address from a device allocation holding values of T at that offset. */
	return reinterpret_cast<T*>(address); // NOLINT(performance-no-int-to-ptr)
}

}

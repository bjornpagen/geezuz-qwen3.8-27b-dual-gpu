// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's half of the resident ring. The slot's layout and protocol are the device header's, included inside the
// purview under one export so the host computes with the very definition the persistent kernel compiles; this unit
// adds only the host's release and acquire.
export module engine.driver:ring;

import std;

export extern "C++" {
#include "ring/ring.h"
}

namespace engine::driver {

/** The host's half of a `Run` publication: the request's fields, then its ticket, released. */
export template<class D, class R>
auto publish_run(device::ring::Slot<D, R>& slot, std::uint64_t ticket, D const& descriptor) -> void {
	slot.request.command = device::ring::Command::Run;
	slot.request.descriptor = descriptor;
	std::atomic_ref{slot.request.ticket}.store(ticket, std::memory_order_release);
}

/** The host's half of an `Exit` publication; the descriptor is not read. */
export template<class D, class R>
auto publish_exit(device::ring::Slot<D, R>& slot, std::uint64_t ticket) -> void {
	slot.request.command = device::ring::Command::Exit;
	std::atomic_ref{slot.request.ticket}.store(ticket, std::memory_order_release);
}

/** The ticket the card last answered in this slot, acquired: after it reads t, the slot's result is ticket t's. */
export template<class D, class R>
[[nodiscard]] auto answered(device::ring::Slot<D, R>& slot) -> std::uint64_t {
	return std::atomic_ref{slot.response.ticket}.load(std::memory_order_acquire);
}

}

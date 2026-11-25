// SPDX-License-Identifier: AGPL-3.0-only
module;

// offsetof is a macro: the slot layout checks need it.
#include <cstddef>

export module engine.driver:ring;

import std;
import :memory;

namespace engine::driver {

/** What a ring slot asks the persistent kernel to do. Zero is never written: fresh slots read as "nothing yet". */
export enum class Command : std::uint8_t {
	Run = 1,
	Exit = 2,
};

/** The granule host and card exchange through: a GPU L2 line, two host cache lines. */
export inline constexpr std::size_t ring_line = 128;

/**
 * One slot of the resident ring, in pinned host memory the card reads and writes over pcie. The host owns the request
 * half, the card the response half; each half starts on its own line, so neither side's writes ever share a line with
 * the other's.
 *
 * The protocol, per ticket t (1, 2, 3, ...; slot t mod capacity):
 * - the host writes `command` and `descriptor` with plain stores, then publishes `request = t` with a release store;
 * - the card polls `request` with `ld.acquire.sys` until it reads t, then reads `command` and `descriptor`;
 * - for `Run` the card writes `result` with plain stores; for `Run` and `Exit` alike it then publishes `response = t`
 *   with `st.release.sys`, and after `Exit` it leaves the loop;
 * - the host reads `response` with an acquire load; once it reads t, `result` is the card's answer to ticket t.
 *
 * The host submits ticket t + capacity only after it has collected ticket t, so neither side ever writes a half the
 * other is still reading. Tickets never repeat, so a slot's stale ticket is never mistaken for a new one.
 */
export template<Wire D, Wire R>
struct RingSlot {
	alignas(ring_line) std::uint64_t request;
	Command command;
	D descriptor;
	alignas(ring_line) std::uint64_t response;
	R result;
};

/** The byte offsets of a slot's fields: the layout the device side of the ring declares and checks against. */
export template<Wire D, Wire R>
struct RingLayout {
	using Slot = RingSlot<D, R>;

	static constexpr std::size_t request = offsetof(Slot, request);
	static constexpr std::size_t command = offsetof(Slot, command);
	static constexpr std::size_t descriptor = offsetof(Slot, descriptor);
	static constexpr std::size_t response = offsetof(Slot, response);
	static constexpr std::size_t result = offsetof(Slot, result);
	static constexpr std::size_t slot = sizeof(Slot);

	static_assert(request == 0 && response % ring_line == 0 && slot % ring_line == 0);
	static_assert(descriptor + sizeof(D) <= response, "the request half ends before the response half's line");
};

/** The slot ticket `ticket` lives in; both sides of the ring compute it this way. */
export [[nodiscard]] constexpr auto ring_index(std::uint64_t ticket, std::size_t capacity) -> std::size_t {
	return static_cast<std::size_t>(ticket % capacity);
}

/** The host's half of a `Run` publication: the request's fields, then its ticket, released. */
export template<Wire D, Wire R>
auto publish_run(RingSlot<D, R>& slot, std::uint64_t ticket, D const& descriptor) -> void {
	slot.command = Command::Run;
	slot.descriptor = descriptor;
	std::atomic_ref{slot.request}.store(ticket, std::memory_order_release);
}

/** The host's half of an `Exit` publication; the descriptor is not read. */
export template<Wire D, Wire R>
auto publish_exit(RingSlot<D, R>& slot, std::uint64_t ticket) -> void {
	slot.command = Command::Exit;
	std::atomic_ref{slot.request}.store(ticket, std::memory_order_release);
}

/** The ticket the card last answered in this slot, acquired: after it reads t, the slot's result is ticket t's. */
export template<Wire D, Wire R>
[[nodiscard]] auto answered(RingSlot<D, R>& slot) -> std::uint64_t {
	return std::atomic_ref{slot.response}.load(std::memory_order_acquire);
}

}

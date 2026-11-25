// SPDX-License-Identifier: AGPL-3.0-only
//
// The resident ring's slot, stated once for both of its ends: the persistent kernel includes it as device code, and the
// driver boundary re-exports it to the host. It uses no standard headers so that it compiles unchanged as device code
// and inside a module purview.
#pragma once

namespace engine::device::ring {

/** The granule host and card exchange through: a GPU L2 line, two host cache lines. */
inline constexpr unsigned long long line = 128;

/** What a slot asks the persistent kernel to do. Zero is never written: fresh, zeroed slots read as "nothing yet". */
enum class Command : unsigned char {
	Run = 1,
	Exit = 2,
};

/** A slot's request half: the host writes it, the card reads it. */
template<class D>
struct Request {
	static_assert(__is_trivially_copyable(D) && __is_standard_layout(D), "a descriptor crosses pcie as bytes");

	alignas(line) unsigned long long ticket;
	Command command;
	D descriptor;
};

/** A slot's response half: the card writes it, the host reads it. */
template<class R>
struct Response {
	static_assert(__is_trivially_copyable(R) && __is_standard_layout(R), "a result crosses pcie as bytes");

	alignas(line) unsigned long long ticket;
	R result;
};

/**
 * One slot of the resident ring, in pinned host memory the card reads and writes over pcie. The host owns the request
 * half, the card the response half; each half starts on its own line, so neither side's writes ever share a line with
 * the other's.
 *
 * The protocol, per ticket t (1, 2, 3, ...; slot `index(t, capacity)`):
 * - the host writes `request.command` and `request.descriptor` with plain stores, then publishes `request.ticket = t`
 *   with a release store;
 * - the card polls `request.ticket` with `ld.acquire.sys` until it reads t, then reads the command and descriptor;
 * - for `Run` the card writes `response.result` with plain stores; for `Run` and `Exit` alike it then publishes
 *   `response.ticket = t` with `st.release.sys`, and after `Exit` it leaves its loop;
 * - the host reads `response.ticket` with an acquire load; once it reads t, the result is the card's answer to t.
 *
 * The host submits ticket t + capacity only after it has collected ticket t, so neither side ever writes a half the
 * other is still reading. Tickets never repeat, so a slot's stale ticket is never mistaken for a new one.
 */
template<class D, class R>
struct Slot {
	Request<D> request;
	Response<R> response;
};

/** A slot's byte offsets: what both ends compute from this one definition, and what their checks compare. */
template<class D, class R>
struct Layout {
	static constexpr unsigned long long request = __builtin_offsetof(Slot<D, R>, request.ticket);
	static constexpr unsigned long long command = __builtin_offsetof(Slot<D, R>, request.command);
	static constexpr unsigned long long descriptor = __builtin_offsetof(Slot<D, R>, request.descriptor);
	static constexpr unsigned long long response = __builtin_offsetof(Slot<D, R>, response.ticket);
	static constexpr unsigned long long result = __builtin_offsetof(Slot<D, R>, response.result);
	static constexpr unsigned long long bytes = sizeof(Slot<D, R>);

	static_assert(request == 0 && response % line == 0 && bytes % line == 0, "each half starts on its own line");
	static_assert(descriptor + sizeof(D) <= response, "the request half ends before the response half's line");
};

/** The slot ticket `ticket` lives in. */
[[nodiscard]] constexpr auto index(unsigned long long ticket, unsigned long long capacity) -> unsigned long long {
	return ticket % capacity;
}

}

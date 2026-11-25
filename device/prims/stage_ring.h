// SPDX-License-Identifier: AGPL-3.0-only
//
// The phase arithmetic of a multi-stage ring guarded by two mbarriers per stage, "full" (the copy landed) and
// "empty" (every consumer is done). Pure constexpr, so the host proves the protocol on a model of the barrier and the
// device ring uses the very same cursor.
//
// An mbarrier's phase k completes when its expected arrivals are in; `test_wait.parity p` succeeds exactly when the
// phase with parity p is complete, i.e. while the barrier's current phase has the other parity. The n-th use of a
// stage (n counting from 0) waits for phase n of its full barrier, so consumers wait on parity n mod 2; the producer
// refilling it waits for phase n - 1 of its empty barrier, parity (n + 1) mod 2, which for n = 0 names the phase
// before the first and succeeds at once.
#pragma once

namespace engine::device::prims {

/** A position in a ring of `stages` stages: which stage, and the parity of its current use. */
template<int stages_>
struct StageCursor {
	static constexpr int stages = stages_;
	static_assert(stages >= 2, "a ring of one stage is a lock step, not a pipeline");

	unsigned stage;
	unsigned parity;

	[[nodiscard]] static constexpr auto start() -> StageCursor {
		return {.stage = 0, .parity = 0};
	}

	/** The next stage; the parity flips each time the ring wraps. */
	[[nodiscard]] constexpr auto next() const -> StageCursor {
		bool const wraps = stage + 1 == static_cast<unsigned>(stages);
		return {.stage = wraps ? 0U : stage + 1, .parity = wraps ? parity ^ 1U : parity};
	}

	/** The parity a consumer waits on in the stage's full barrier. */
	[[nodiscard]] constexpr auto full_parity() const -> unsigned {
		return parity;
	}

	/** The parity the producer waits on in the stage's empty barrier before refilling it. */
	[[nodiscard]] constexpr auto empty_parity() const -> unsigned {
		return parity ^ 1U;
	}

	[[nodiscard]] constexpr auto operator==(StageCursor const&) const -> bool = default;
};

}

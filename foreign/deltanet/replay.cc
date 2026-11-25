// SPDX-License-Identifier: AGPL-3.0-only
//
// A slot's lazy-replay ledger: what its stored deltanet state covers, and which accepted drafts wait in its replay ring.
// The stored state only ever holds tokens surely in the sequence. A cycle replays the drafts the last cycle accepted,
// then runs its rows; it stores the state after its first row (the token sampled last cycle, which is certain) and
// leaves its drafts in the ring until the next cycle knows how many were accepted. The ledger's states are three types,
// so a cycle cannot be planned from a slot mid-cycle, and a snapshot or a prompt chunk can only start from a slot
// whose state has nothing pending.
export module engine.deltanet:replay;

import std;
import engine.core;
import :arithmetic;

namespace engine::deltanet {

/** A slot whose stored state covers exactly its first `tokens` tokens: where snapshots are taken and prompts resume. */
export struct Committed {
	std::uint64_t tokens;

	[[nodiscard]] constexpr auto operator==(Committed const&) const -> bool = default;
};

/** A slot between cycles: its stored state covers `committed` tokens, and the `pending` tokens after them wait in its ring. */
export struct Settled {
	std::uint64_t committed;
	std::uint32_t pending;

	[[nodiscard]] constexpr auto operator==(Settled const&) const -> bool = default;
};

/** A slot's cycle in flight: it replays `replays` ring entries, then runs `rows` rows from position committed + replays. */
export struct Running {
	std::uint64_t committed;
	std::uint32_t replays;
	std::uint32_t rows;

	[[nodiscard]] constexpr auto operator==(Running const&) const -> bool = default;
};

/** A slot fresh from prefill or a snapshot load enters decode with nothing to replay. */
export [[nodiscard]] constexpr auto settled(Committed state) -> Settled {
	return {.committed = state.tokens, .pending = 0};
}

/** The cycle that replays the pending drafts and runs the committed token with `drafts` drafts. */
export [[nodiscard]] constexpr auto run(Settled slot, std::uint32_t drafts) -> Running {
	core::invariant(drafts <= static_cast<std::uint32_t>(max_drafts));
	core::invariant(slot.pending <= static_cast<std::uint32_t>(max_replays));
	return {.committed = slot.committed, .replays = slot.pending, .rows = 1 + drafts};
}

/** The position of a running cycle's first row: every row after it is a draft. */
export [[nodiscard]] constexpr auto first_position(Running cycle) -> std::uint64_t {
	return cycle.committed + cycle.replays;
}

/**
 * The slot after its cycle's verdict accepted `accepted` of its drafts: the replays and the first row are stored, the
 * accepted drafts pending.
 */
export [[nodiscard]] constexpr auto settle(Running cycle, std::uint32_t accepted) -> Settled {
	core::invariant(accepted < cycle.rows);
	return {.committed = cycle.committed + cycle.replays + 1, .pending = accepted};
}

/**
 * The slot's stored position, dropping what is pending: its pending tokens stay in its token sequence but are no longer
 * in the ring's care, so whatever continues the slot (a snapshot, a prompt chunk) runs them again as prompt tokens.
 */
export [[nodiscard]] constexpr auto stored(Settled slot) -> Committed {
	return {.tokens = slot.committed};
}

/** The device's share of a running cycle for slot `slot`, whose rows start at `first_row` of the cycle's rows. */
export [[nodiscard]] constexpr auto cycle_work(Running cycle, std::uint32_t slot, std::uint32_t first_row) -> SlotCycle {
	return {.slot = slot, .replays = cycle.replays, .rows = cycle.rows, .first_row = first_row};
}

/** A prompt chunk of `tokens` tokens continuing a committed slot, and the slot it leaves. */
export struct Chunk {
	SlotChunk work;
	Committed after;
};

export [[nodiscard]] constexpr auto prompt_chunk(Committed slot, std::uint32_t tokens, std::uint32_t slot_index, std::uint32_t first_row)
    -> Chunk {
	core::invariant(tokens >= 1);
	return {.work = {.slot = slot_index, .tokens = tokens, .first_row = first_row}, .after = {.tokens = slot.tokens + tokens}};
}

}

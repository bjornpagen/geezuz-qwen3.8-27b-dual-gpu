// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sched:status;

import std;
import engine.model;
import :request;

namespace engine::sched {

export struct Vacant {
	[[nodiscard]] constexpr auto operator==(Vacant const&) const -> bool = default;
};

/**
 * A slot reading its sequence: `prefilled` of `prompt_tokens` positions done, the first `cached_tokens` of the
 * request's first admission from the cache. A paused request resumes with what it generated as part of its prompt.
 */
export struct Prefilling {
	RequestHandle request;
	Count prompt_tokens;
	Count cached_tokens;
	Count prefilled_tokens;

	[[nodiscard]] constexpr auto operator==(Prefilling const&) const -> bool = default;
};

export struct Decoding {
	RequestHandle request;
	Count prompt_tokens;
	Count cached_tokens;
	Count generated_tokens;

	[[nodiscard]] constexpr auto operator==(Decoding const&) const -> bool = default;
};

/** A cancelled request whose slot is released at the next cycle boundary. */
export struct Finishing {
	RequestHandle request;

	[[nodiscard]] constexpr auto operator==(Finishing const&) const -> bool = default;
};

export using SlotFacts = std::variant<Vacant, Prefilling, Decoding, Finishing>;

/** The shared pool in blocks: `used` held by running requests, `cached` held only by the prefix cache. */
export struct PoolFacts {
	Count blocks;
	Count used_blocks;
	Count cached_blocks;

	[[nodiscard]] constexpr auto operator==(PoolFacts const&) const -> bool = default;
};

/** Counts since start. */
export struct Counters {
	std::uint64_t cycles;
	std::uint64_t drafted_tokens;
	std::uint64_t accepted_tokens;
	std::uint64_t prefill_chunks;
	std::uint64_t prefill_tokens;
	std::uint64_t preemptions;
	/** Snapshots the cache could not take for want of room: those states are recomputed when asked for again. */
	std::uint64_t snapshots_refused;

	[[nodiscard]] constexpr auto operator==(Counters const&) const -> bool = default;
};

/** What the scheduler is doing, for /status. */
export struct Facts {
	std::array<SlotFacts, model::slots> slots;
	/** Requests waiting for a slot, paused ones included. */
	Count queued;
	PoolFacts pool;
	Counters counters;

	[[nodiscard]] constexpr auto operator==(Facts const&) const -> bool = default;
};

}

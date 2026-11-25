// SPDX-License-Identifier: AGPL-3.0-only
export module engine.net:fact;

import std;

namespace engine::net {

/** Every wait is until an absolute point on the monotonic clock. */
export using Clock = std::chrono::steady_clock;
export using Deadline = Clock::time_point;

/** What the owner waits for on one socket; every watch is one-shot and is re-armed after its fact. */
export enum class Interest : std::uint8_t {
	Read,
	Write,
	ReadWrite,
};

export enum class Direction : std::uint8_t {
	Readable,
	Writable,
};

/**
 * A socket the owner watched under `token` may make progress in `direction`. A fact is a hint: the owner retries
 * its i/o and learns the truth from the call, so a stale or spurious fact costs one failed attempt and nothing else.
 */
export struct Ready {
	std::uint64_t token;
	Direction direction;

	[[nodiscard]] constexpr auto operator==(Ready const&) const -> bool = default;
};

/** SIGINT or SIGTERM arrived, once per delivery. */
export struct StopRequested {
	[[nodiscard]] constexpr auto operator==(StopRequested const&) const -> bool = default;
};

/** Another thread woke the owner: its mailbox holds messages. */
export struct Woken {
	[[nodiscard]] constexpr auto operator==(Woken const&) const -> bool = default;
};

/** Owner tokens stay below this bound; the backend's own descriptors may use the tokens above it. */
export inline constexpr std::uint64_t token_limit = std::uint64_t{1} << 63U;

export using Fact = std::variant<StopRequested, Woken, Ready>;

export inline constexpr std::size_t max_facts = 64;

/** One wait's facts, in the order the kernel reported them. */
export struct Facts {
	std::array<Fact, max_facts> items;
	std::size_t count;

	[[nodiscard]] auto view() const -> std::span<Fact const> {
		return std::span{items}.first(count);
	}
};

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The memory clock this box was accepted at. A memory overclock is safe only where a soak at temperature found no
// corruption, and a clock nobody soaked (an offset lost on a driver reload, a card that came up in another performance
// state) is a different machine from the one the soak vouched for. So the accepted clock is a fact of the box, fixed in
// the build at bring-up, and the engine refuses to start on a card whose clock under load is any other.
export module engine.platform:clock;

import std;
import engine.core;
import engine.device;
import engine.model;

namespace engine::platform {

/** A card's memory clock under load, as NVML reads it: megahertz, and the performance state holding it there. */
export struct AcceptedClock {
	std::uint32_t megahertz;
	device::PerformanceState state;

	[[nodiscard]] constexpr auto operator==(AcceptedClock const&) const -> bool = default;
};

/**
 * The memory clock the bring-up acceptance soak accepted on this box, for both cards: absent until a soak has accepted
 * one, and until then no card's clock is accepted. Set once, at bring-up, from the soak's report.
 */
export inline constexpr std::optional<AcceptedClock> accepted_memory_clock = std::nullopt;

/** No soak has accepted a memory clock on this box yet: nothing the cards run at is known to be safe. */
export struct NoneAccepted {
	[[nodiscard]] constexpr auto operator==(NoneAccepted const&) const -> bool = default;
};

/** A card's memory clock under load is not the accepted one. */
export struct ClockDiffers {
	model::Card card;
	AcceptedClock accepted;
	device::MemoryClock found;

	[[nodiscard]] constexpr auto operator==(ClockDiffers const&) const -> bool = default;
};

/**
 * Why the cards' memory clocks are refused. Both are the box's state, not a passing condition: retrying changes
 * nothing until bring-up applies the accepted clock (or a soak accepts the clock the cards run at).
 */
export struct [[nodiscard]] ClockRefusal {
	std::variant<NoneAccepted, ClockDiffers> reason;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(ClockRefusal const&) const -> bool = default;
};

/** Whether `found`, card `card`'s memory clock under load, is `accepted`; the refusal says why not. */
export [[nodiscard]] constexpr auto judge(std::optional<AcceptedClock> const& accepted, model::Card card, device::MemoryClock const& found)
    -> core::Result<void, ClockRefusal> {
	if (!accepted) {
		return std::unexpected(ClockRefusal{.reason = NoneAccepted{}});
	}
	if (found.megahertz != accepted->megahertz || found.state != std::optional{accepted->state}) {
		return std::unexpected(ClockRefusal{.reason = ClockDiffers{.card = card, .accepted = *accepted, .found = found}});
	}
	return {};
}

}

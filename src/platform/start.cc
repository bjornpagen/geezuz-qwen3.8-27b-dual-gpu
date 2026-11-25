// SPDX-License-Identifier: AGPL-3.0-only
//
// The memory-clock check the engine runs when it starts, before it loads a byte of its images. A card idle in its
// lowest performance state reads a memory clock far below its working one, so the cards are loaded first: both run
// the soak's pattern for a moment, and each card's clock is read while its last rounds are still running.
export module engine.platform:start;

import std;
import engine.core;
import engine.device;
import engine.model;
import engine.soak;
import :bench;
import :clock;
import :soak;

namespace engine::platform {

/** Pattern rounds queued at once while settling: about a tenth of a second of work per card. */
inline constexpr std::uint32_t queued_rounds = 8;

/** Why the engine may not start on these cards' memory clocks. */
export struct [[nodiscard]] ClockError {
	std::variant<device::DeviceError, Hung, ClockRefusal> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(ClockError const&) const -> bool = default;
};

template<class T>
[[nodiscard]] auto refuse_clock(T reason) -> std::unexpected<ClockError> {
	return std::unexpected(ClockError{.reason = std::move(reason)});
}

/** Queues `queued_rounds` pattern writes on every card, each card's completion recorded after its last. */
[[nodiscard]] auto queue_rounds(device::Devices const& devices, std::array<PatternCard, model::cards> const& cards, std::uint32_t first)
    -> core::Result<void, device::DeviceError> {
	auto queued = core::Result<void, device::DeviceError>{};
	for (auto const card : device::card_pair) {
		for (auto round = first; round < first + queued_rounds; ++round) {
			queued = queued.and_then([&] { return write_round(devices.context(card), cards.at(std::to_underlying(card)), round); });
		}
	}
	return queued;
}

/** Waits for every card's queued rounds, within the pattern lane's patience. */
template<Clock C>
[[nodiscard]] auto drain(C& clock, std::array<PatternCard, model::cards> const& cards) -> core::Result<void, ClockError> {
	auto const due = clock.now().steady + pattern_patience;
	for (auto const card : device::card_pair) {
		for (;;) {
			auto const state = cards.at(std::to_underlying(card)).done.query();
			if (!state) {
				return refuse_clock(state.error());
			}
			if (*state == device::EventState::Complete) {
				break;
			}
			auto const now = clock.now().steady;
			if (now > due) {
				return refuse_clock(Hung{.card = card, .lane = Lane::Pattern});
			}
			clock.sleep_until(now + poll_interval);
		}
	}
	return {};
}

/** Each card's memory clock read while both cards run the pattern, after `settle` of it. */
export template<Clock C>
[[nodiscard]] auto loaded_memory_clocks(device::Devices const& devices, C& clock) -> core::Result<std::array<device::MemoryClock, model::cards>, ClockError> {
	auto zero = pattern_card(devices.context(model::Card::Zero));
	auto one = zero.and_then([&](auto const&) { return pattern_card(devices.context(model::Card::One)); });
	if (!one) {
		return refuse_clock(one.error());
	}
	auto const cards = std::array{std::move(*zero), std::move(*one)};
	auto const until = clock.now().steady + settle;
	auto round = std::uint32_t{0};
	do {
		if (auto queued = queue_rounds(devices, cards, round); !queued) {
			return refuse_clock(queued.error());
		}
		round += queued_rounds;
		if (auto drained = drain(clock, cards); !drained) {
			return std::unexpected(drained.error());
		}
	} while (clock.now().steady < until);
	if (auto queued = queue_rounds(devices, cards, round); !queued) {
		return refuse_clock(queued.error());
	}
	auto const zero_clock = devices.memory_clock(model::Card::Zero);
	auto const one_clock = devices.memory_clock(model::Card::One);
	if (auto drained = drain(clock, cards); !drained) {
		return std::unexpected(drained.error());
	}
	if (!zero_clock || !one_clock) {
		return refuse_clock(!zero_clock ? zero_clock.error() : one_clock.error());
	}
	return std::array{*zero_clock, *one_clock};
}

/**
 * The engine's start check: both cards' memory clocks under load are the accepted one, or the refusal. Call it right
 * after the cards open and before anything is loaded onto them; it needs the cards to itself for `settle`.
 */
export template<Clock C>
[[nodiscard]] auto check_memory_clocks(device::Devices const& devices, C& clock) -> core::Result<std::array<device::MemoryClock, model::cards>, ClockError> {
	return loaded_memory_clocks(devices, clock).and_then([](std::array<device::MemoryClock, model::cards> const& found)
	                                                         -> core::Result<std::array<device::MemoryClock, model::cards>, ClockError> {
		for (auto const card : device::card_pair) {
			if (auto judged = judge(accepted_memory_clock, card, found.at(std::to_underlying(card))); !judged) {
				return refuse_clock(judged.error());
			}
		}
		return found;
	});
}

}

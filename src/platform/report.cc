// SPDX-License-Identifier: AGPL-3.0-only
//
// The operator's account of what the soak and the clock check found: every finding with its wall time (to set beside
// the junction temperature the gddr6 tool logs, which nothing here can read), the time into the soak, and both cards'
// clocks, power and slowdown counters right after it.
export module engine.platform:report;

import std;
import engine.core;
import engine.device;
import engine.file;
import engine.image;
import engine.model;
import engine.sha256;
import :bench;
import :clock;
import :soak;
import :start;
import :telemetry;

namespace engine::platform {

[[nodiscard]] auto card_name(model::Card card) -> std::string {
	return std::format("card {}", std::to_underlying(card));
}

[[nodiscard]] auto clock_name(device::MemoryClock const& clock) -> std::string {
	auto const state = clock.state.transform([](device::PerformanceState known) { return std::format("P{}", std::to_underlying(known)); });
	return std::format("{} MHz in {}", clock.megahertz, state.value_or("an unknown performance state"));
}

[[nodiscard]] auto counter_name(Counter counter) -> std::string_view {
	switch (counter) {
	case Counter::SwPowerCap:
		return "sw_power_cap";
	case Counter::SwThermalSlowdown:
		return "sw_thermal_slowdown";
	case Counter::HwThermalSlowdown:
		return "hw_thermal_slowdown";
	case Counter::HwPowerBrakeSlowdown:
		return "hw_power_brake_slowdown";
	}
	std::unreachable();
}

[[nodiscard]] auto lane_name(Lane lane) -> std::string_view {
	switch (lane) {
	case Lane::Image:
		return "image digest";
	case Lane::Pattern:
		return "pattern";
	}
	std::unreachable();
}

[[nodiscard]] auto milliseconds(std::optional<std::chrono::nanoseconds> time) -> std::string {
	return time.transform([](std::chrono::nanoseconds known) { return std::format("{:.3f} ms", std::chrono::duration<double, std::milli>{known}.count()); })
	    .value_or("untracked");
}

/** One card's readings on one line. */
export [[nodiscard]] auto render(model::Card card, Telemetry const& telemetry) -> std::string {
	auto const& counters = telemetry.counters;
	return std::format("{}: memory {}, SM {} MHz, {:.1f} W of a {:.1f} W limit, clock-event reasons {:#x}, slowdown counters sw_power_cap {}, "
	                   "sw_thermal {}, hw_thermal {}, hw_power_brake {}",
	                   card_name(card), clock_name(telemetry.memory), telemetry.sm.megahertz, telemetry.power.draw_milliwatts / 1000.0,
	                   telemetry.power.limit_milliwatts / 1000.0, telemetry.reasons.bits, milliseconds(counters.sw_power_cap),
	                   milliseconds(counters.sw_thermal_slowdown), milliseconds(counters.hw_thermal_slowdown),
	                   milliseconds(counters.hw_power_brake_slowdown));
}

struct Describe {
	[[nodiscard]] auto operator()(device::DeviceError const& failure) const -> std::string {
		return device::describe(failure);
	}

	[[nodiscard]] auto operator()(Hung const& hung) const -> std::string {
		return std::format("{}'s {} launch did not complete within its patience: the card is hung", card_name(hung.card), lane_name(hung.lane));
	}

	[[nodiscard]] auto operator()(Unwatchable const& missing) const -> std::string {
		return std::format("{}: NVML keeps no {} counter, so the soak cannot show the clocks were never held down", card_name(missing.card),
		                   counter_name(missing.counter));
	}

	[[nodiscard]] auto operator()(UploadStalled const& stalled) const -> std::string {
		return std::format("{}'s image did not finish copying to the card in time", card_name(stalled.card));
	}

	[[nodiscard]] auto operator()(image::ImageError const& error) const -> std::string {
		struct Cause {
			[[nodiscard]] auto operator()(file::FileError const& file) const -> std::string {
				return std::format("an image file cannot be mapped (failure {})", std::to_underlying(file.failure));
			}

			[[nodiscard]] auto operator()(image::Rejection const& rejection) const -> std::string {
				return std::format("an image is refused (failure {}): rebake it with this engine's geezuz-bake", std::to_underlying(rejection.failure));
			}
		};
		return std::visit(Cause{}, error.cause);
	}

	[[nodiscard]] auto operator()(ClockRefusal const& refusal) const -> std::string {
		struct Reason {
			[[nodiscard]] auto operator()(NoneAccepted /*none*/) const -> std::string {
				return "no memory clock is accepted on this box yet: soak the cards and set the accepted clock";
			}

			[[nodiscard]] auto operator()(ClockDiffers const& differs) const -> std::string {
				return std::format("{}'s memory runs at {} under load, not the accepted {}", card_name(differs.card), clock_name(differs.found),
				                   clock_name({.megahertz = differs.accepted.megahertz, .state = differs.accepted.state}));
			}
		};
		return std::visit(Reason{}, refusal.reason);
	}
};

export [[nodiscard]] auto describe(SoakError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

export [[nodiscard]] auto describe(SetupError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

export [[nodiscard]] auto describe(ClockError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

struct DescribeFinding {
	[[nodiscard]] auto operator()(ImageMismatch const& mismatch) const -> std::string {
		return std::format("{}'s weight image hashed on the card in pass {} to {}, not its baked digest {}", card_name(mismatch.card), mismatch.pass,
		                   mismatch.found, mismatch.expected);
	}

	[[nodiscard]] auto operator()(PatternMismatch const& mismatch) const -> std::string {
		return std::format("{}'s pattern round {} read back {} words wrong, the first in chunk {} ({} bytes into the region)", card_name(mismatch.card),
		                   mismatch.round, mismatch.words, mismatch.first_chunk, std::uint64_t{mismatch.first_chunk} * 16);
	}

	[[nodiscard]] auto operator()(ClockMoved const& moved) const -> std::string {
		return std::format("{}'s memory clock moved from {} to {}", card_name(moved.card), clock_name(moved.started), clock_name(moved.now));
	}

	[[nodiscard]] auto operator()(Throttled const& throttled) const -> std::string {
		return throttled.grew
		           .transform([&](std::chrono::nanoseconds grew) {
			           return std::format("{}'s {} counter moved by {}", card_name(throttled.card), counter_name(throttled.counter), milliseconds(grew));
		           })
		           .value_or(std::format("{}'s {} counter stopped being kept", card_name(throttled.card), counter_name(throttled.counter)));
	}
};

[[nodiscard]] auto when(Moment const& at, Moment const& started) -> std::string {
	auto const into = std::chrono::duration_cast<std::chrono::seconds>(at.steady - started.steady);
	return std::format("{:%F %T} UTC, {:%T} into the soak", std::chrono::floor<std::chrono::seconds>(at.wall), into);
}

[[nodiscard]] auto progress(std::array<Progress, model::cards> const& cards) -> std::string {
	auto text = std::string{};
	for (auto const card : device::card_pair) {
		auto const& done = cards.at(std::to_underlying(card));
		text += std::format("{}: {} whole image passes matched, {} pattern rounds read back clean\n", card_name(card), done.passes, done.rounds);
	}
	return text;
}

struct RenderVerdict {
	[[nodiscard]] auto operator()(Clean const& clean) const -> std::string {
		auto text = std::format("clean: nothing found from {} to {}\n", when(clean.started, clean.started), when(clean.ended, clean.started));
		for (auto const card : device::card_pair) {
			text += std::format("{}: memory held at {} throughout\n", card_name(card), clock_name(clean.clocks.at(std::to_underlying(card))));
		}
		return text + progress(clean.cards);
	}

	[[nodiscard]] auto operator()(Found const& found) const -> std::string {
		auto text = std::format("MISMATCH at {}: {}\n", when(found.sighting.at, found.started), std::visit(DescribeFinding{}, found.sighting.finding));
		for (auto const card : device::card_pair) {
			auto const& reading = found.sighting.cards.at(std::to_underlying(card));
			text += reading ? render(card, *reading) : std::format("{}: unreadable: {}", card_name(card), Describe{}(reading.error()));
			text += '\n';
		}
		return text + progress(found.cards);
	}

	[[nodiscard]] auto operator()(TooShort const& short_run) const -> std::string {
		return std::format("too short: a card finished no whole image pass between {} and {}, so the soak vouches for nothing\n{}",
		                   when(short_run.started, short_run.started), when(short_run.ended, short_run.started), progress(short_run.cards));
	}
};

/** The soak's verdict as the operator reads it. */
export [[nodiscard]] auto render(Verdict const& verdict) -> std::string {
	return std::visit(RenderVerdict{}, verdict);
}

/** Whether a verdict accepts the clock the cards ran at. */
export [[nodiscard]] auto accepts(Verdict const& verdict) -> bool {
	return std::holds_alternative<Clean>(verdict);
}

}

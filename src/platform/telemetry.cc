// SPDX-License-Identifier: AGPL-3.0-only
//
// What a card says about itself while the soak runs: its clocks, its power, and what has held its clocks down. A soak
// vouches only for the conditions it ran under, so it watches them stay put: the memory clock it started at, and no
// time added to the power-cap or thermal slowdown counters.
export module engine.platform:telemetry;

import std;
import engine.core;
import engine.device;
import engine.model;

namespace engine::platform {

/** One card's readings at one moment. */
export struct Telemetry {
	device::MemoryClock memory;
	device::SmClock sm;
	device::Power power;
	device::ClocksEventReasons reasons;
	device::ClocksEventCounters counters;

	[[nodiscard]] constexpr auto operator==(Telemetry const&) const -> bool = default;
};

/** Reads card `card`'s telemetry. */
export [[nodiscard]] auto read(device::Devices const& devices, model::Card card) -> core::Result<Telemetry, device::DeviceError> {
	auto telemetry = Telemetry{};
	auto const read_all = devices.memory_clock(card)
	                          .transform([&](device::MemoryClock const& memory) {
		                          telemetry.memory = memory;
	                          })
	                          .and_then([&] {
		                          return devices.sm_clock(card).transform([&](device::SmClock const& sm) {
			                          telemetry.sm = sm;
		                          });
	                          })
	                          .and_then([&] {
		                          return devices.power(card).transform([&](device::Power const& power) {
			                          telemetry.power = power;
		                          });
	                          })
	                          .and_then([&] {
		                          return devices.clocks_event_reasons(card).transform([&](device::ClocksEventReasons const& reasons) {
			                          telemetry.reasons = reasons;
		                          });
	                          })
	                          .and_then([&] {
		                          return devices.clocks_event_counters(card).transform([&](device::ClocksEventCounters const& counters) {
			                          telemetry.counters = counters;
		                          });
	                          });
	if (!read_all) {
		return std::unexpected(read_all.error());
	}
	return telemetry;
}

/** The slowdown counters a soak watches: time the power cap or heat took from the card's clocks. */
export enum class Counter : std::uint8_t {
	SwPowerCap,
	SwThermalSlowdown,
	HwThermalSlowdown,
	HwPowerBrakeSlowdown,
};

/**
 * The watched counters at the soak's start. The power cap and both thermal counters are required, because acceptance
 * means none of them moved and a counter NVML does not keep cannot show that; the power brake is watched where NVML
 * keeps it.
 */
export struct Watched {
	std::chrono::nanoseconds sw_power_cap;
	std::chrono::nanoseconds sw_thermal_slowdown;
	std::chrono::nanoseconds hw_thermal_slowdown;
	std::optional<std::chrono::nanoseconds> hw_power_brake_slowdown;

	[[nodiscard]] constexpr auto operator==(Watched const&) const -> bool = default;
};

/** NVML keeps no count for a counter acceptance needs. */
export struct Unwatchable {
	model::Card card;
	Counter counter;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Unwatchable const&) const -> bool = default;
};

/** What the soak holds a card to: the memory clock it started at, and its slowdown counters then. */
export struct Baseline {
	device::MemoryClock memory;
	Watched counters;

	[[nodiscard]] constexpr auto operator==(Baseline const&) const -> bool = default;
};

/** Card `card`'s baseline from its first readings, or the required counter NVML does not keep. */
export [[nodiscard]] constexpr auto baseline(model::Card card, Telemetry const& first) -> core::Result<Baseline, Unwatchable> {
	auto const& counters = first.counters;
	if (!counters.sw_power_cap) {
		return std::unexpected(Unwatchable{.card = card, .counter = Counter::SwPowerCap});
	}
	if (!counters.sw_thermal_slowdown) {
		return std::unexpected(Unwatchable{.card = card, .counter = Counter::SwThermalSlowdown});
	}
	if (!counters.hw_thermal_slowdown) {
		return std::unexpected(Unwatchable{.card = card, .counter = Counter::HwThermalSlowdown});
	}
	return Baseline{
	    .memory = first.memory,
	    .counters =
	        {
	            .sw_power_cap = *counters.sw_power_cap,
	            .sw_thermal_slowdown = *counters.sw_thermal_slowdown,
	            .hw_thermal_slowdown = *counters.hw_thermal_slowdown,
	            .hw_power_brake_slowdown = counters.hw_power_brake_slowdown,
	        },
	};
}

/** The memory clock is not the one the soak started at. */
export struct ClockMoved {
	model::Card card;
	device::MemoryClock started;
	device::MemoryClock now;

	[[nodiscard]] constexpr auto operator==(ClockMoved const&) const -> bool = default;
};

/**
 * A watched counter has grown since the soak started, by `grew`; absent when NVML stopped keeping the counter, which
 * hides a slowdown as surely as one happening.
 */
export struct Throttled {
	model::Card card;
	Counter counter;
	std::optional<std::chrono::nanoseconds> grew;

	[[nodiscard]] constexpr auto operator==(Throttled const&) const -> bool = default;
};

export using Drift = std::variant<ClockMoved, Throttled>;

/**
 * Whether a watched counter has moved from `start`: any change counts, growth or a reset (a driver reload restarts the
 * counts and drops every clock setting with them), and so does the counter vanishing.
 */
[[nodiscard]] constexpr auto throttled(model::Card card, Counter counter, std::chrono::nanoseconds start, std::optional<std::chrono::nanoseconds> now)
    -> std::optional<Throttled> {
	if (now && *now == start) {
		return std::nullopt;
	}
	return Throttled{.card = card, .counter = counter, .grew = now.transform([&](std::chrono::nanoseconds value) { return value - start; })};
}

/** The first way card `card`'s readings `now` have drifted from its baseline, if any. */
export [[nodiscard]] constexpr auto drift(model::Card card, Baseline const& start, Telemetry const& now) -> std::optional<Drift> {
	if (now.memory != start.memory) {
		return Drift{ClockMoved{.card = card, .started = start.memory, .now = now.memory}};
	}
	auto const& was = start.counters;
	auto const& is = now.counters;
	auto const found = throttled(card, Counter::SwPowerCap, was.sw_power_cap, is.sw_power_cap)
	                       .or_else([&] {
		                       return throttled(card, Counter::SwThermalSlowdown, was.sw_thermal_slowdown, is.sw_thermal_slowdown);
	                       })
	                       .or_else([&] {
		                       return throttled(card, Counter::HwThermalSlowdown, was.hw_thermal_slowdown, is.hw_thermal_slowdown);
	                       })
	                       .or_else([&] {
		                       // The power brake is watched only where NVML kept it at the start.
		                       return was.hw_power_brake_slowdown.and_then([&](std::chrono::nanoseconds brake) {
			                       return throttled(card, Counter::HwPowerBrakeSlowdown, brake, is.hw_power_brake_slowdown);
		                       });
	                       });
	return found.transform([](Throttled const& throttle) { return Drift{throttle}; });
}

}

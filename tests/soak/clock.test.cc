// SPDX-License-Identifier: AGPL-3.0-only
//
// The memory-clock refusal: the judgement against the accepted clock over every performance state, and the start-time
// check on the CUDA driver and NVML test double: it loads both cards with the pattern before it reads their clocks,
// reports what each card reads, refuses what the accepted clock does not allow, reports every failing vendor call, and
// leaves nothing allocated behind.
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.model;
import engine.platform;
import engine.soak;
import engine.testing;

namespace {

namespace platform = engine::platform;
namespace device = engine::device;
namespace fake = engine::driver_double;
namespace model = engine::model;

using model::Card;

/** A clock whose time moves on by a fixed tick each time it is read, and to the deadline when waited on. */
struct TickingClock {
	std::chrono::steady_clock::time_point steady = std::chrono::steady_clock::time_point{} + std::chrono::hours{1};
	std::chrono::milliseconds tick{50};

	[[nodiscard]] auto now() -> platform::Moment {
		auto const moment = platform::Moment{.wall = std::chrono::system_clock::time_point{}, .steady = steady};
		steady += tick;
		return moment;
	}

	auto sleep_until(std::chrono::steady_clock::time_point until) -> void {
		steady = std::max(steady, until);
	}
};

static_assert(platform::Clock<TickingClock>);

inline constexpr auto states = std::array{
    device::PerformanceState::P0,  device::PerformanceState::P1,  device::PerformanceState::P2,  device::PerformanceState::P3,
    device::PerformanceState::P4,  device::PerformanceState::P5,  device::PerformanceState::P6,  device::PerformanceState::P7,
    device::PerformanceState::P8,  device::PerformanceState::P9,  device::PerformanceState::P10, device::PerformanceState::P11,
    device::PerformanceState::P12, device::PerformanceState::P13, device::PerformanceState::P14, device::PerformanceState::P15,
};

/** Every reading: each state and none, at clocks around and at the ones the cards really use. */
[[nodiscard]] auto readings() -> std::vector<device::MemoryClock> {
	auto all = std::vector<device::MemoryClock>{};
	for (auto const megahertz : {0U, 405U, 9'500U, 9'501U, 9'502U, 9'751U, 10'001U, 10'501U, 0xffff'ffffU}) {
		all.push_back({.megahertz = megahertz, .state = std::nullopt});
		for (auto const state : states) {
			all.push_back({.megahertz = megahertz, .state = state});
		}
	}
	return all;
}

/** With an accepted clock, a reading passes exactly when it has the accepted megahertz in the accepted state. */
[[nodiscard]] auto judgement_is_equality() -> bool {
	auto const all = readings();
	auto holds = true;
	for (auto const& accepted : all) {
		if (!accepted.state) {
			continue;
		}
		auto const clock = platform::AcceptedClock{.megahertz = accepted.megahertz, .state = *accepted.state};
		for (auto const& found : all) {
			for (auto const card : device::card_pair) {
				auto const judged = platform::judge(clock, card, found);
				auto const same = found == accepted;
				holds = holds && judged.has_value() == same &&
				        (same || judged.error() == platform::ClockRefusal{.reason = platform::ClockDiffers{.card = card, .accepted = clock, .found = found}});
			}
		}
	}
	return holds;
}

/** With no clock accepted, every reading is refused as such. */
[[nodiscard]] auto nothing_accepted_refuses_everything() -> bool {
	return std::ranges::all_of(readings(), [](device::MemoryClock const& found) {
		auto const judged = platform::judge(std::nullopt, Card::One, found);
		return !judged && judged.error() == platform::ClockRefusal{.reason = platform::NoneAccepted{}};
	});
}

[[nodiscard]] auto machine_at(std::array<std::pair<std::uint32_t, int>, 2> const& clocks) -> fake::Machine {
	auto machine = fake::Machine{};
	for (auto index = std::size_t{0}; index < machine.cards.size(); ++index) {
		machine.cards.at(index).memory_clock = clocks.at(index).first;
		machine.cards.at(index).performance_state = clocks.at(index).second;
	}
	return machine;
}

/** Whether every pattern launch the double saw wrote the same region with consecutive rounds, on both cards, all SMs. */
[[nodiscard]] auto loads_both_cards(fake::Fake const& driver) -> bool {
	auto rounds = std::array<std::vector<std::uint32_t>, model::cards>{};
	auto regions = std::array<std::set<std::uint64_t>, model::cards>{};
	for (auto const& launch : driver.launches()) {
		if (launch.entry != "soak_write" || launch.parameters.size() != sizeof(engine::soak::WriteParams) || launch.grid.x != engine::soak::pattern_blocks ||
		    launch.block.x != static_cast<std::uint32_t>(engine::soak::pattern_budget.threads)) {
			return false;
		}
		auto bytes = std::array<std::byte, sizeof(engine::soak::WriteParams)>{};
		std::ranges::copy(launch.parameters, bytes.begin());
		auto const params = std::bit_cast<engine::soak::WriteParams>(bytes);
		rounds.at(static_cast<std::size_t>(launch.ordinal)).push_back(params.round);
		regions.at(static_cast<std::size_t>(launch.ordinal)).insert(params.region);
	}
	return std::ranges::all_of(std::views::iota(std::size_t{0}, model::cards), [&](std::size_t card) {
		auto const& seen = rounds.at(card);
		return seen.size() >= platform::settle / std::chrono::milliseconds{100} && regions.at(card).size() == 1 &&
		       std::ranges::equal(seen, std::views::iota(std::uint32_t{0}, static_cast<std::uint32_t>(seen.size())));
	});
}

/** On random clocks, the loaded readings are each card's own, after both cards ran the pattern, and nothing is left. */
[[nodiscard]] auto loaded_clocks_are_each_cards(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 6; ++trial) {
		auto const clocks = std::array{
		    std::pair{std::uniform_int_distribution<std::uint32_t>{405, 11'000}(random), std::uniform_int_distribution<int>{0, 15}(random)},
		    std::pair{std::uniform_int_distribution<std::uint32_t>{405, 11'000}(random), std::uniform_int_distribution<int>{0, 15}(random)},
		};
		auto const driver = fake::Fake{machine_at(clocks)};
		{
			auto const devices = device::Devices::open(driver.driver());
			if (!devices) {
				return false;
			}
			auto clock = TickingClock{};
			auto const found = platform::loaded_memory_clocks(*devices, clock);
			holds = holds && found.has_value() && loads_both_cards(driver) && driver.live(fake::Kind::DeviceMemory) == 0 &&
			        driver.live(fake::Kind::HostMemory) == 0 && driver.live(fake::Kind::Module) == 0 && driver.live(fake::Kind::Stream) == 0 &&
			        driver.live(fake::Kind::Event) == 0;
			for (auto const card : device::card_pair) {
				auto const& [megahertz, state] = clocks.at(std::to_underlying(card));
				holds = holds && found && found->at(std::to_underlying(card)) ==
				                              device::MemoryClock{.megahertz = megahertz, .state = static_cast<device::PerformanceState>(state)};
			}
		}
		holds = holds && driver.nothing_live() && driver.violations().empty();
	}
	return holds;
}

/** The engine's check refuses unless both cards run at the accepted clock under load. */
[[nodiscard]] auto start_check_holds_cards_to_the_accepted_clock() -> bool {
	auto const outcome = [](std::array<std::pair<std::uint32_t, int>, 2> const& clocks) {
		auto const driver = fake::Fake{machine_at(clocks)};
		auto const devices = device::Devices::open(driver.driver());
		engine::core::invariant(devices.has_value());
		auto clock = TickingClock{};
		return platform::check_memory_clocks(*devices, clock);
	};
	if constexpr (platform::accepted_memory_clock.has_value()) {
		auto const accepted = *platform::accepted_memory_clock;
		auto const at = std::pair{accepted.megahertz, static_cast<int>(std::to_underlying(accepted.state))};
		auto const off = std::pair{accepted.megahertz + 1, at.second};
		auto const refused = outcome({at, off});
		return outcome({at, at}).has_value() && !refused &&
		       refused.error() == platform::ClockError{.reason = platform::ClockRefusal{.reason = platform::ClockDiffers{
		                                                                       .card = Card::One,
		                                                                       .accepted = accepted,
		                                                                       .found = {.megahertz = off.first, .state = static_cast<device::PerformanceState>(off.second)},
		                                                                   }}};
	} else {
		auto const refused = outcome({std::pair{9'501U, 2}, std::pair{9'501U, 2}});
		return !refused && refused.error() == platform::ClockError{.reason = platform::ClockRefusal{.reason = platform::NoneAccepted{}}};
	}
}

/**
 * Every vendor call the loaded read makes, failing, fails the read with a device error and leaks nothing; except the
 * releases at its end, whose failure has no remedy and no caller to tell: those are the last calls, one per owner it
 * releases (each card's region, tallies, program, event and stream).
 */
[[nodiscard]] auto every_failing_call_is_reported() -> bool {
	constexpr auto releases = std::size_t{5} * model::cards;
	auto const counted = [] {
		auto const driver = fake::Fake{fake::Machine{}};
		auto const devices = device::Devices::open(driver.driver());
		engine::core::invariant(devices.has_value());
		auto const before = driver.calls();
		auto clock = TickingClock{};
		auto const found = platform::loaded_memory_clocks(*devices, clock);
		return std::pair{before, found ? driver.calls() : before};
	}();
	auto holds = counted.second > counted.first + releases;
	for (auto call = counted.first; call < counted.second && holds; ++call) {
		auto const driver = fake::Fake{fake::Machine{.failing_call = call}};
		auto reported = false;
		{
			auto const devices = device::Devices::open(driver.driver());
			auto clock = TickingClock{};
			auto const found = platform::loaded_memory_clocks(*devices, clock);
			reported = !found && std::holds_alternative<device::DeviceError>(found.error().reason);
		}
		holds = call + releases < counted.second ? reported && driver.nothing_live() : !reported;
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "an accepted clock passes exactly the reading with its megahertz and its state, on either card",
	        .holds = judgement_is_equality(),
	    },
	    engine::testing::Case{
	        .name = "with no clock accepted, every reading is refused as nothing accepted",
	        .holds = nothing_accepted_refuses_everything(),
	    },
	    engine::testing::Case{
	        .name = "random clocks: each card's reading is its own, taken after both cards ran the pattern, leaving nothing (seed 5)",
	        .holds = loaded_clocks_are_each_cards(5),
	    },
	    engine::testing::Case{
	        .name = "the engine's start check refuses cards not at the accepted clock, and with none accepted refuses all",
	        .holds = start_check_holds_cards_to_the_accepted_clock(),
	    },
	    engine::testing::Case{
	        .name = "every vendor call of the loaded read, failing, is reported and leaks nothing",
	        .holds = every_failing_call_is_reported(),
	    },
	};
	return engine::testing::report(cases);
}

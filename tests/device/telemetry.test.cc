// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.model;
import engine.testing;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;

using device::Devices;
using engine::model::Card;
using fake::CardModel;
using fake::Fake;
using fake::Machine;

[[nodiscard]] auto maybe_nanoseconds(std::mt19937_64& random) -> std::optional<std::chrono::nanoseconds> {
	if (std::uniform_int_distribution{0, 3}(random) == 0) {
		return std::nullopt;
	}
	return std::chrono::nanoseconds{std::uniform_int_distribution<std::int64_t>{0, std::int64_t{1125899906842624}}(random)};
}

[[nodiscard]] auto random_card(std::mt19937_64& random) -> CardModel {
	auto card = CardModel{};
	card.sm_clock = std::uniform_int_distribution<std::uint32_t>{210, 2100}(random);
	card.power_draw_milliwatts = std::uniform_int_distribution<std::uint32_t>{20'000, 450'000}(random);
	card.power_limit_milliwatts = std::uniform_int_distribution<std::uint32_t>{100'000, 400'000}(random);
	card.clocks_event_reasons = std::uniform_int_distribution<std::uint64_t>{0, 0x7ff}(random);
	card.clocks_event_counters = {
	    .sw_power_cap = maybe_nanoseconds(random),
	    .sync_boost = maybe_nanoseconds(random),
	    .sw_thermal_slowdown = maybe_nanoseconds(random),
	    .hw_thermal_slowdown = maybe_nanoseconds(random),
	    .hw_power_brake_slowdown = maybe_nanoseconds(random),
	};
	card.nvlink_version = std::uniform_int_distribution<unsigned int>{0, 12}(random);
	card.nvlink_megabytes_per_second = std::uniform_int_distribution<std::uint32_t>{1, 60'000}(random);
	card.peer_performance_rank = std::uniform_int_distribution{0, 3}(random);
	card.native_atomics = std::uniform_int_distribution{0, 1}(random) == 1;
	card.max_persisting_l2_bytes = std::uniform_int_distribution{0, 6 * 1024 * 1024}(random);
	card.max_access_policy_window_bytes = std::uniform_int_distribution{0, 1073741824}(random);
	return card;
}

/** Every read of a card returns exactly what the modelled card holds. */
[[nodiscard]] auto reads_return_what_the_card_holds(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto round = 0; round < 50; ++round) {
		auto machine = Machine{};
		for (auto& card : machine.cards) {
			card = random_card(random);
		}
		auto const cards = machine.cards;
		auto const driver = Fake{std::move(machine)};
		auto const opened = Devices::open(driver.driver());
		if (!opened) {
			return false;
		}
		for (auto const which : device::card_pair) {
			auto const& model = cards[std::to_underlying(which)];
			auto const expected_link = device::NvlinkLinkReading{
			    .version = static_cast<device::NvlinkVersion>(model.nvlink_version),
			    .megabytes_per_second = model.nvlink_megabytes_per_second,
			};
			auto const links = opened->nvlinks(which);
			holds = holds && opened->sm_clock(which) == device::SmClock{.megahertz = model.sm_clock} &&
			        opened->power(which) ==
			            device::Power{.draw_milliwatts = model.power_draw_milliwatts, .limit_milliwatts = model.power_limit_milliwatts} &&
			        opened->clocks_event_reasons(which) == device::ClocksEventReasons{.bits = model.clocks_event_reasons} &&
			        opened->clocks_event_counters(which) == model.clocks_event_counters && links &&
			        std::ranges::all_of(*links,
			                            [&](device::NvlinkLinkReading const& link) {
				                            return link == expected_link;
			                            }) &&
			        opened->peer_facts(which) ==
			            device::PeerFacts{.performance_rank = model.peer_performance_rank, .native_atomics = model.native_atomics} &&
			        opened->l2_persistence(which) ==
			            device::L2Persistence{
			                .max_persisting_bytes = static_cast<std::uint64_t>(model.max_persisting_l2_bytes),
			                .max_access_policy_window_bytes = static_cast<std::uint64_t>(model.max_access_policy_window_bytes),
			            };
		}
		holds = holds && driver.violations().empty();
	}
	return holds;
}

/** A reason set contains a reason exactly when its bit is set, for every set of the eleven reasons. */
[[nodiscard]] auto reasons_contain_exactly_their_bits() -> bool {
	constexpr auto reasons = std::array{
	    device::ClocksEventReason::GpuIdle,
	    device::ClocksEventReason::ApplicationsClocksSetting,
	    device::ClocksEventReason::SwPowerCap,
	    device::ClocksEventReason::HwSlowdown,
	    device::ClocksEventReason::SyncBoost,
	    device::ClocksEventReason::SwThermalSlowdown,
	    device::ClocksEventReason::HwThermalSlowdown,
	    device::ClocksEventReason::HwPowerBrakeSlowdown,
	    device::ClocksEventReason::DisplayClockSetting,
	    device::ClocksEventReason::BoardLimit,
	    device::ClocksEventReason::Reliability,
	};
	auto holds = true;
	for (auto bits = std::uint64_t{0}; bits < (std::uint64_t{1} << reasons.size()); ++bits) {
		auto const set = device::ClocksEventReasons{.bits = bits};
		for (auto index = std::size_t{0}; index < reasons.size(); ++index) {
			holds = holds && set.contains(reasons[index]) == (((bits >> index) & 1U) != 0);
		}
	}
	return holds;
}

/** Whether a read fails, as a reader the failure search can run. */
template<class Read>
[[nodiscard]] auto fails(Read read) -> std::pair<std::size_t, std::vector<bool>> {
	auto const run = [&](std::optional<std::size_t> failing) -> std::pair<std::size_t, std::size_t> {
		auto const driver = Fake{Machine{.failing_call = failing}};
		auto const opened = Devices::open(driver.driver());
		if (!opened) {
			return {0, 0};
		}
		auto const before = driver.calls();
		auto const failed = !read(*opened);
		return {before, failed ? 1 : 0};
	};
	auto const counted = [&] {
		auto const driver = Fake{Machine{}};
		auto const opened = Devices::open(driver.driver());
		auto const before = driver.calls();
		auto const ok = opened && read(*opened);
		return std::pair{before, ok ? driver.calls() - before : 0};
	}();
	auto outcomes = std::vector<bool>{};
	for (auto call = std::size_t{0}; call < counted.second; ++call) {
		outcomes.push_back(run(counted.first + call).second == 1);
	}
	return {counted.second, outcomes};
}

/** Every vendor call each read makes can fail, and whichever does, the read reports it. */
[[nodiscard]] auto every_failing_call_of_a_read_is_reported() -> bool {
	auto const results = std::array{
	    fails([](Devices const& devices) {
		    return devices.sm_clock(Card::One).has_value();
	    }),
	    fails([](Devices const& devices) {
		    return devices.power(Card::Zero).has_value();
	    }),
	    fails([](Devices const& devices) {
		    return devices.clocks_event_reasons(Card::One).has_value();
	    }),
	    fails([](Devices const& devices) {
		    return devices.clocks_event_counters(Card::Zero).has_value();
	    }),
	    fails([](Devices const& devices) {
		    return devices.nvlinks(Card::One).has_value();
	    }),
	    fails([](Devices const& devices) {
		    return devices.peer_facts(Card::Zero).has_value();
	    }),
	    fails([](Devices const& devices) {
		    return devices.l2_persistence(Card::One).has_value();
	    }),
	};
	return std::ranges::all_of(results, [](auto const& result) {
		return result.first > 0 && result.second.size() == result.first && std::ranges::all_of(result.second, std::identity{});
	});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "random cards: every clock, power, reason, counter, link, peer and l2 read is what the card holds (seed 51)",
	        .holds = reads_return_what_the_card_holds(51),
	    },
	    engine::testing::Case{
	        .name = "a clocks-event reason set contains a reason exactly when its bit is set (all 2048 sets)",
	        .holds = reasons_contain_exactly_their_bits(),
	    },
	    engine::testing::Case{
	        .name = "every vendor call each read makes, failing, makes that read fail",
	        .holds = every_failing_call_of_a_read_is_reported(),
	    },
	};
	return engine::testing::report(cases);
}

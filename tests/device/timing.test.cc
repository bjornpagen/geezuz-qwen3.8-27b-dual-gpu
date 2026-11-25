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
using fake::Fake;
using fake::Machine;

constexpr auto step = 0.25F;

/** The time between two timed markers is the work enqueued between them: k copies take k steps of the fake clock. */
[[nodiscard]] auto timed_markers_measure_the_work_between_them() -> bool {
	auto const driver = Fake{Machine{.milliseconds_per_operation = step}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Card::Zero);
	auto const stream = context.create_stream();
	auto staging = context.allocate_mapped<std::uint32_t, 256>();
	auto array = context.allocate<std::uint32_t, 256>();
	if (!stream || !staging || !array) {
		return false;
	}
	auto holds = true;
	for (auto copies = 0; copies <= 5; ++copies) {
		auto const start = context.create_timed_event();
		auto const end = context.create_timed_event();
		if (!start || !end || !start->record(*stream)) {
			return false;
		}
		for (auto copy = 0; copy < copies; ++copy) {
			auto transfer = context.upload(*stream, std::move(*array), std::move(*staging), 0);
			auto delivered = transfer.try_finish();
			if (!delivered || !delivered->copied) {
				return false;
			}
			array.emplace(std::move(delivered->destination));
			staging.emplace(std::move(delivered->source));
		}
		auto const elapsed = end->record(*stream).and_then([&] {
			return end->since(*start);
		});
		holds = holds && elapsed && *elapsed == std::chrono::duration<float, std::milli>{step * static_cast<float>(copies)};
	}
	return holds && driver.violations().empty();
}

/** While either marker's work still runs there is no time yet: absent, not a failure. */
[[nodiscard]] auto running_work_has_no_time_yet() -> bool {
	constexpr auto queries = std::size_t{2};
	auto const driver = Fake{Machine{.pending_queries = queries}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Card::One);
	auto const stream = context.create_stream();
	auto const start = context.create_timed_event();
	auto const end = context.create_timed_event();
	if (!stream || !start || !end || !start->record(*stream) || !end->record(*stream)) {
		return false;
	}
	auto holds = true;
	for (auto query = std::size_t{0}; query < queries; ++query) {
		auto const early = end->since(*start);
		holds = holds && early && !early->has_value();
		holds = holds && start->query() == device::EventState::Pending && end->query() == device::EventState::Pending;
	}
	auto const late = end->since(*start);
	return holds && late && late->has_value() && driver.violations().empty();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the time between two timed markers is exactly the work enqueued between them (0 to 5 copies)",
	        .holds = timed_markers_measure_the_work_between_them(),
	    },
	    engine::testing::Case{
	        .name = "while either marker's work runs, the elapsed time is absent rather than a failure",
	        .holds = running_work_has_no_time_yet(),
	    },
	};
	return engine::testing::report(cases);
}

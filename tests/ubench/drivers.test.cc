// SPDX-License-Identifier: AGPL-3.0-only
//
// Every driver against the driver double. The double runs no kernel: its timed markers advance a fake clock by a set
// time per launch, and device memory a kernel would write stays as the driver left it (zeroed). So an event-timed
// benchmark yields an outcome computed from the fake clock, whose gate the clock decides, and a benchmark that reads
// what its kernel wrote must refuse the untouched results as no reading. Either way the driver must launch exactly
// its declared kernels, misuse nothing, and leave nothing alive.
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.model;
import engine.testing;
import engine.ubench;
import engine.ubench_device;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;
namespace ubench = engine::ubench;
namespace ub = engine::device::ubench;

using engine::model::Card;
using ubench::Benchmark;
using ubench::Verdict;

inline constexpr auto calibration = ubench::Calibration{.megahertz = {1500.0, 1500.0}};

/**
 * Runs `check` over a machine started on the double, then requires that the engine misused nothing and released
 * everything it acquired.
 */
template<class Check>
[[nodiscard]] auto on_double(fake::Machine machine, Check check) -> bool {
	auto const driver = fake::Fake{std::move(machine)};
	auto held = false;
	{
		auto const opened = device::Devices::open(driver.driver());
		if (!opened) {
			return false;
		}
		auto zero = opened->context(Card::Zero).create_stream();
		auto one = opened->context(Card::One).create_stream();
		if (!zero || !one) {
			return false;
		}
		auto const streams = std::array{std::move(*zero), std::move(*one)};
		auto const bench = ubench::BenchView{.devices = *opened, .streams = streams, .calibration = calibration};
		held = check(bench, driver);
	}
	return held && driver.violations().empty() && driver.nothing_live();
}

/** What a driver must come to on the double. */
using Expected = std::variant<Verdict, ubench::NoReading, ubench::Hung>;

struct Comes {
	Expected expected;

	[[nodiscard]] auto operator()(Verdict verdict, ubench::Result<ubench::Outcome> const& result) const -> bool {
		return result && result->verdict == verdict;
	}

	[[nodiscard]] auto operator()(ubench::NoReading const& none, ubench::Result<ubench::Outcome> const& result) const -> bool {
		auto const* const got = result ? nullptr : std::get_if<ubench::NoReading>(&result.error().reason);
		return got != nullptr && got->kernel == none.kernel;
	}

	[[nodiscard]] auto operator()(ubench::Hung const& hung, ubench::Result<ubench::Outcome> const& result) const -> bool {
		return !result && result.error() == ubench::UbenchError{.reason = hung};
	}
};

[[nodiscard]] auto comes_to(Expected const& expected, ubench::Result<ubench::Outcome> const& result) -> bool {
	return std::visit(
	    [&](auto const& alternative) {
		    return Comes{.expected = expected}(alternative, result);
	    },
	    expected);
}

[[nodiscard]] constexpr auto none(std::string_view kernel) -> Expected {
	return ubench::NoReading{.kernel = kernel, .why = ""};
}

/** What every benchmark comes to on the double at one millisecond per launch: 2.06 GB in a millisecond passes the gate. */
[[nodiscard]] auto expected_on_double(Benchmark benchmark) -> Expected {
	switch (benchmark) {
	case Benchmark::ReadStream:
		return Verdict::Holds;
	case Benchmark::TileLayout:
	case Benchmark::MemoryClock:
	case Benchmark::RingStall:
	case Benchmark::OffsetStep:
		return Verdict::Measured;
	case Benchmark::LinkHealth:
		return Verdict::Holds;
	case Benchmark::TensorRate:
		return none("ubench_tensor_mma");
	case Benchmark::DecodeExact:
		return none("prims_check_decode");
	case Benchmark::BarrierCost:
		return none("ubench_mbarrier_backoff");
	case Benchmark::CrossSmHop:
		return none("ubench_hop_backoff");
	case Benchmark::PeerFlag:
		return none("ubench_flag_ping_release_backoff");
	case Benchmark::PeerPush:
		return none("ubench_receive_c4");
	case Benchmark::ReleaseCost:
		return none("ubench_release_idle");
	case Benchmark::HostFetch:
		return ubench::Hung{.kernel = "ubench_mailbox"};
	case Benchmark::L2Persistence:
		return none("ubench_l2_probe");
	case Benchmark::SustainedClock:
		return none("ubench_sustain");
	case Benchmark::TranscendentalRate:
		return none("ubench_exp");
	case Benchmark::StageHandoff:
		return none("ubench_stage_handoff");
	case Benchmark::L2PolicyReuse:
		return none("ubench_reuse_split");
	case Benchmark::LdmatrixSwizzle:
		return none("ubench_ldmatrix_swizzled");
	case Benchmark::MmaDoubleBuffer:
		return none("ubench_mma_double_buffered");
	case Benchmark::WarpReductions:
		return none("ubench_reduce_float");
	case Benchmark::CheckFragments:
		return none("prims_check_fragments");
	case Benchmark::CheckReduce:
		return none("prims_check_reduce");
	case Benchmark::CheckTranscendentals:
		return none("prims_digest_transcendentals");
	case Benchmark::CheckStageRing:
		return none("prims_check_stage_ring");
	case Benchmark::CheckOrder:
		return none("prims_check_order");
	case Benchmark::GemvRows:
		return none("gemv_stream_r1");
	case Benchmark::ProbeMma:
		return none("prims_probe_mma");
	case Benchmark::DeltanetChain:
		return none("deltanet_probe_chain_8");
	case Benchmark::SplitKAttention:
	case Benchmark::PrefillGemm:
		return Verdict::Measured;
	}
	std::unreachable();
}

/**
 * Every benchmark on the double: the event-timed ones yield their outcome from the fake clock, the ones that read
 * kernel results refuse the zeroed results naming the kernel, and the mailbox, which needs a running kernel to
 * answer, is hung.
 */
[[nodiscard]] auto every_driver_comes_to_its_reading_on_the_double() -> bool {
	auto holds = true;
	for (auto const& listed : ubench::catalogue) {
		holds = on_double(fake::Machine{}, [&](ubench::BenchView const& bench, fake::Fake const& driver) {
			auto const result = ubench::run(listed.benchmark, bench);
			auto const expected = expected_on_double(listed.benchmark);
			auto const declared = std::ranges::all_of(driver.launches(), [](fake::Launch const& launch) {
				return launch.entry.starts_with("ubench_") || launch.entry.starts_with("prims_") || launch.entry.starts_with("gemv_") ||
				       launch.entry.starts_with("gemm_") || launch.entry.starts_with("deltanet_") || launch.entry.starts_with("attn_");
			});
			if (!comes_to(expected, result) || !declared) {
				std::println("  {} did not come to its reading on the double{}", listed.name, result ? "" : ": " + ubench::describe(result.error()));
				return false;
			}
			return true;
		}) && holds;
	}
	return holds;
}

/** The read sweep launches every point on both cards, warm-up and timed launches each, on every SM. */
[[nodiscard]] auto the_read_sweep_launches_every_point() -> bool {
	return on_double(fake::Machine{}, [](ubench::BenchView const& bench, fake::Fake const& driver) {
		auto const result = ubench::read_stream(bench);
		auto const points = std::tuple_size_v<engine::ubench_device::ReadSweep>;
		auto const per_point = static_cast<std::size_t>(ubench::warmup_launches) + static_cast<std::size_t>(ubench::timed_launches);
		auto const launches = driver.launches();
		auto const on_card = [&](int ordinal) {
			return std::ranges::count_if(launches, [&](fake::Launch const& launch) {
				return launch.ordinal == ordinal && launch.entry.starts_with("ubench_read_p") && launch.grid.count() == ub::sms &&
				       launch.parameters.size() == sizeof(ub::stream::ReadParams);
			});
		};
		auto const distinct = std::set<std::string>(std::from_range, launches | std::views::transform(&fake::Launch::entry));
		return result && std::cmp_equal(on_card(0), points * per_point) && std::cmp_equal(on_card(1), points * per_point) && launches.size() == 2 * points * per_point &&
		       distinct.size() == points && result->readings.size() == 2 * points + 2;
	});
}

/**
 * The phase-b gate through the whole driver path: at t milliseconds per launch the stream moves 2.06 GB in t ms, and
 * the gate holds exactly when that is at least 93% of the 912 GB/s a 9501 MHz memory clock gives (t below 2.433).
 */
[[nodiscard]] auto the_read_gate_follows_the_measured_clock() -> bool {
	auto const verdict_at = [](float milliseconds, std::uint32_t megahertz) {
		auto card = fake::CardModel{};
		card.memory_clock = megahertz;
		auto verdict = std::optional<Verdict>{};
		auto const ran = on_double(fake::Machine{.cards = {card, card}, .milliseconds_per_operation = milliseconds},
		                           [&](ubench::BenchView const& bench, fake::Fake const&) {
			                           auto const result = ubench::read_stream(bench);
			                           verdict = result ? std::optional{result->verdict} : std::nullopt;
			                           return result.has_value();
		                           });
		return ran ? verdict : std::nullopt;
	};
	return verdict_at(2.375F, 9501) == Verdict::Holds && verdict_at(2.5F, 9501) == Verdict::Fails && verdict_at(1.0F, 9501) == Verdict::Holds &&
	       verdict_at(2.375F, 9751) == Verdict::Fails && verdict_at(2.25F, 9751) == Verdict::Holds;
}

/** The memory clock is read idle and under load on each card, as the management library reports it. */
[[nodiscard]] auto the_memory_clock_reads_each_card() -> bool {
	auto zero = fake::CardModel{};
	auto one = fake::CardModel{};
	one.memory_clock = 9751;
	one.performance_state = 0;
	return on_double(fake::Machine{.cards = {zero, one}}, [](ubench::BenchView const& bench, fake::Fake const&) {
		auto const result = ubench::memory_clock(bench);
		if (!result || result->readings.size() != 4) {
			return false;
		}
		auto const& readings = result->readings;
		return readings[0].value == 9501.0 && readings[1].value == 9501.0 && readings[2].value == 9751.0 && readings[3].value == 9751.0 &&
		       std::ranges::all_of(readings, [](ubench::Reading const& reading) {
			       return reading.unit == ubench::Unit::Megahertz;
		       });
	});
}

/** A run starts by calibrating the SM clocks, which reads what the clock kernel wrote: on the double, no reading. */
[[nodiscard]] auto a_run_stops_at_an_uncalibrated_clock() -> bool {
	return on_double(fake::Machine{}, [](ubench::BenchView const& bench, fake::Fake const& driver) {
		auto const selection = ubench::select({});
		auto const ran = selection ? ubench::run_all(bench.devices, *selection) : ubench::Result<bool>{false};
		auto const* const none = ran ? nullptr : std::get_if<ubench::NoReading>(&ran.error().reason);
		return none != nullptr && none->kernel == "ubench_clock" && !driver.launches().empty() &&
		       std::ranges::all_of(driver.launches(), [](fake::Launch const& launch) {
			       return launch.entry == "ubench_clock";
		       });
	});
}

/** A driver call that fails anywhere in a benchmark is that failure, reported as itself, not as a reading. */
[[nodiscard]] auto a_failed_driver_call_is_that_failure() -> bool {
	auto failed_launch = false;
	// Opening the machine takes the first few dozen calls; a failing call past them lands somewhere in the benchmark.
	for (auto call = std::size_t{40}; call < std::size_t{400} && !failed_launch; call += 7) {
		auto const driver = fake::Fake{fake::Machine{.failing_call = call}};
		auto const opened = device::Devices::open(driver.driver());
		if (!opened) {
			continue;
		}
		auto zero = opened->context(Card::Zero).create_stream();
		auto one = opened->context(Card::One).create_stream();
		if (!zero || !one) {
			continue;
		}
		auto const streams = std::array{std::move(*zero), std::move(*one)};
		auto const bench = ubench::BenchView{.devices = *opened, .streams = streams, .calibration = calibration};
		auto const result = ubench::tile_layout(bench);
		failed_launch = !result && std::holds_alternative<device::DeviceError>(result.error().reason) && driver.violations().empty();
	}
	return failed_launch;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every driver comes to its reading on the double, launching only declared kernels",
	                          .holds = every_driver_comes_to_its_reading_on_the_double()},
	    engine::testing::Case{.name = "the read sweep launches every point on both cards on every SM", .holds = the_read_sweep_launches_every_point()},
	    engine::testing::Case{.name = "the read gate holds at 93% of the measured memory clock's peak and fails below it",
	                          .holds = the_read_gate_follows_the_measured_clock()},
	    engine::testing::Case{.name = "the memory clock is read idle and under load on each card", .holds = the_memory_clock_reads_each_card()},
	    engine::testing::Case{.name = "a run stops at the calibration when the clock kernel wrote nothing", .holds = a_run_stops_at_an_uncalibrated_clock()},
	    engine::testing::Case{.name = "a driver call that fails inside a benchmark is that failure", .holds = a_failed_driver_call_is_that_failure()},
	};
	return engine::testing::report(cases);
}

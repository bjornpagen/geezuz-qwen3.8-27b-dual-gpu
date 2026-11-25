// SPDX-License-Identifier: AGPL-3.0-only
//
// The synchronisation benchmarks: the SM clock calibration every cycle count is converted with, the barrier costs
// inside a block, the cross-SM hop and the warp-specialised stage handoff.
export module engine.ubench:sync;

import std;
import engine.core;
import engine.device;
import engine.model;
import engine.ubench_device;
import :outcome;
import :protocol;

namespace engine::ubench {

namespace ub = engine::device::ubench;
namespace kernels = engine::ubench_device;
namespace shapes = engine::device::ubench::sync;

/** The median over blocks of each block's cycles per microsecond of the global timer. */
export [[nodiscard]] auto calibrated_megahertz(std::span<ub::Clocks const> blocks) -> Result<double> {
	auto rates = std::vector<double>{};
	for (auto const& clocks : blocks) {
		if (clocks.nanoseconds == 0 || clocks.cycles < shapes::calibration_cycles) {
			return no_reading(kernels::ClockKernel::entry, "a block did not spin its calibration cycles");
		}
		rates.push_back(static_cast<double>(clocks.cycles) * 1000.0 / static_cast<double>(clocks.nanoseconds));
	}
	return median(std::move(rates));
}

/** Each card's SM clock idle, measured on the card: the conversion from the kernels' cycle counts to time. */
export [[nodiscard]] auto calibrate(device::Devices const& devices, std::array<device::Stream, model::cards> const& streams) -> Result<Calibration> {
	auto calibration = Calibration{.megahertz = {}};
	auto const bench = BenchView{.devices = devices, .streams = streams, .calibration = calibration};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<kernels::ClockKernel>());
		auto clocks = Readback<ub::Clocks, ub::sms>::create(view.context);
		if (auto const failed = first_failure(program, clocks)) {
			return std::unexpected(*failed);
		}
		return run_once<kernels::ClockKernel>(view, *program, clocks->device())
		    .and_then([&] {
			    return clocks->fetch(view);
		    })
		    .and_then([](std::span<ub::Clocks const, ub::sms> blocks) {
			    return calibrated_megahertz(blocks);
		    })
		    .transform([&](double megahertz) {
			    calibration.megahertz.at(std::to_underlying(card)) = megahertz;
		    });
	});
	return ran.transform([&] {
		return calibration;
	});
}

/**
 * The least count a kernel writes to `cycles` over the protocol's launches. `between` resets what the kernel consumes
 * before each launch; a zero count is a kernel that did not run.
 */
template<class K, class... Ks, class Between, class... Arguments>
[[nodiscard]] auto least_cycles(CardView view, device::Program<Ks...> const& program, Readback<unsigned long long, 1>& cycles, Between between,
                                Arguments const&... arguments) -> Result<double> {
	auto least = std::numeric_limits<double>::infinity();
	for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
		auto const counted = between()
		                         .and_then([&] {
			                         return run_once<K>(view, program, arguments..., cycles.device());
		                         })
		                         .and_then([&] {
			                         return cycles.fetch(view);
		                         });
		if (!counted) {
			return std::unexpected(counted.error());
		}
		if (launch >= warmup_launches) {
			least = std::min(least, static_cast<double>(counted->front()));
		}
	}
	if (!(least > 0.0)) {
		return no_reading(K::entry, "the kernel counted no cycles");
	}
	return least;
}

/** One card's barrier readings: each kernel's least total cycles, and the sleep grain's spread per argument. */
export struct BarrierReading {
	model::Card card;
	double mbarrier_backoff;
	double mbarrier_spin;
	double named_barrier;
	double copy_completion;
	std::array<shapes::SleepReading, shapes::sleep_kinds> sleeps;
};

export [[nodiscard]] auto evaluate_barrier_cost(std::span<BarrierReading const> readings, Calibration calibration) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	auto const per = [&](model::Card card, double cycles, int count) {
		return calibration.nanoseconds(card, cycles / static_cast<double>(count));
	};
	for (auto const& reading : readings) {
		auto const card = reading.card;
		outcome.readings.push_back({.label = std::format("{}, mbarrier arrive + test_wait round trip, production backoff", on(card)),
		                            .value = per(card, reading.mbarrier_backoff, shapes::pingpong_iterations),
		                            .unit = Unit::Nanoseconds});
		outcome.readings.push_back({.label = std::format("{}, mbarrier arrive + test_wait round trip, spinning", on(card)),
		                            .value = per(card, reading.mbarrier_spin, shapes::pingpong_iterations),
		                            .unit = Unit::Nanoseconds});
		outcome.readings.push_back({.label = std::format("{}, bar.arrive + bar.sync round trip", on(card)),
		                            .value = per(card, reading.named_barrier, shapes::pingpong_iterations),
		                            .unit = Unit::Nanoseconds});
		outcome.readings.push_back({.label = std::format("{}, cp.async 16 B + mbarrier completion, l2-resident", on(card)),
		                            .value = per(card, reading.copy_completion, shapes::copy_iterations),
		                            .unit = Unit::Nanoseconds});
		for (auto kind = std::size_t{0}; kind < reading.sleeps.size(); ++kind) {
			auto const& sleep = reading.sleeps.at(kind);
			if (sleep.total == 0) {
				return no_reading(kernels::SleepGrainKernel::entry, "a sleep took no cycles");
			}
			auto const asked = std::span{shapes::sleeps}[kind];
			outcome.readings.push_back({.label = std::format("{}, nanosleep({}) mean", on(card), asked),
			                            .value = per(card, static_cast<double>(sleep.total), shapes::sleep_samples),
			                            .unit = Unit::Nanoseconds});
			outcome.readings.push_back({.label = std::format("{}, nanosleep({}) shortest", on(card), asked),
			                            .value = calibration.nanoseconds(card, static_cast<double>(sleep.least)),
			                            .unit = Unit::Nanoseconds});
			outcome.readings.push_back({.label = std::format("{}, nanosleep({}) longest", on(card), asked),
			                            .value = calibration.nanoseconds(card, static_cast<double>(sleep.most)),
			                            .unit = Unit::Nanoseconds});
		}
	}
	outcome.notes.emplace_back("the ring's slot-free signal is the cheaper of the mbarrier and the named-barrier round trips");
	return outcome;
}

using MbarrierBackoff = kernels::MbarrierKernel<ub::Wait::Backoff>;
using MbarrierSpin = kernels::MbarrierKernel<ub::Wait::Spin>;
using NamedBarrier = kernels::NamedBarrierKernel;
using CopyCompletion = kernels::CopyCompletionKernel;
using SleepGrain = kernels::SleepGrainKernel;

export [[nodiscard]] auto barrier_cost(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<BarrierReading>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<MbarrierBackoff, MbarrierSpin, NamedBarrier, CopyCompletion, SleepGrain>());
		auto cycles = Readback<unsigned long long, 1>::create(view.context);
		auto copied = Readback<ub::Word4, 32>::create(view.context);
		auto sleeps = Readback<shapes::SleepReading, shapes::sleep_kinds>::create(view.context);
		if (auto const failed = first_failure(program, cycles, copied, sleeps)) {
			return std::unexpected(*failed);
		}
		auto reading = BarrierReading{.card = card, .mbarrier_backoff = 0, .mbarrier_spin = 0, .named_barrier = 0, .copy_completion = 0, .sleeps = {}};
		auto const measured = least_cycles<MbarrierBackoff>(view, *program, *cycles, nothing)
		                          .and_then([&](double counted) {
			                          reading.mbarrier_backoff = counted;
			                          return least_cycles<MbarrierSpin>(view, *program, *cycles, nothing);
		                          })
		                          .and_then([&](double counted) {
			                          reading.mbarrier_spin = counted;
			                          return least_cycles<NamedBarrier>(view, *program, *cycles, nothing);
		                          })
		                          .and_then([&](double counted) {
			                          reading.named_barrier = counted;
			                          return copied->clear(view);
		                          })
		                          .and_then([&] {
			                          return least_cycles<CopyCompletion>(view, *program, *cycles, nothing, copied->device());
		                          })
		                          .and_then([&](double counted) {
			                          reading.copy_completion = counted;
			                          return run_once<SleepGrain>(view, *program, sleeps->device());
		                          })
		                          .and_then([&] {
			                          return sleeps->fetch(view);
		                          });
		return measured.transform([&](std::span<shapes::SleepReading const, shapes::sleep_kinds> spread) {
			std::ranges::copy(spread, reading.sleeps.begin());
			readings.push_back(reading);
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_barrier_cost(readings, bench.calibration);
}

/** One card's hop: the reading of the best launch under each wait. */
export struct HopReading {
	model::Card card;
	ub::Wait wait;
	shapes::HopReading reading;
};

/**
 * The one-way hop, half a round trip. Two blocks on one SM measure nothing across SMs; an echoed value other than the
 * one released is the release/acquire protocol failing, which fails the benchmark.
 */
export [[nodiscard]] auto evaluate_cross_sm_hop(std::span<HopReading const> hops, Calibration calibration) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Holds, .notes = {}};
	for (auto const& hop : hops) {
		auto const kernel = hop.wait == ub::Wait::Backoff ? kernels::HopKernel<ub::Wait::Backoff>::entry : kernels::HopKernel<ub::Wait::Spin>::entry;
		if (hop.reading.cycles == 0) {
			return no_reading(kernel, "the first block counted no cycles");
		}
		if (hop.reading.first_sm == hop.reading.second_sm) {
			return no_reading(kernel, "both blocks ran on one SM");
		}
		auto const round_trip = static_cast<double>(hop.reading.cycles) / static_cast<double>(shapes::hop_iterations);
		outcome.readings.push_back({.label = std::format("{}, one-way hop (store, release add, acquire poll, load), {}", on(hop.card),
		                                                 hop.wait == ub::Wait::Backoff ? "production backoff" : "spinning"),
		                            .value = calibration.nanoseconds(hop.card, round_trip / 2.0),
		                            .unit = Unit::Nanoseconds});
		auto const mismatches = hop.reading.first_mismatches + hop.reading.second_mismatches;
		if (mismatches != 0) {
			outcome.verdict = Verdict::Fails;
			outcome.notes.push_back(std::format("{}: {} values read after an acquire were not the ones released", on(hop.card), mismatches));
		}
	}
	return outcome;
}

using HopBackoff = kernels::HopKernel<ub::Wait::Backoff>;
using HopSpin = kernels::HopKernel<ub::Wait::Spin>;

export [[nodiscard]] auto cross_sm_hop(BenchView const& bench) -> Result<Outcome> {
	auto hops = std::vector<HopReading>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<HopBackoff, HopSpin>());
		auto scratch = Readback<shapes::HopScratch, 1>::create(view.context);
		auto reading = Readback<shapes::HopReading, 1>::create(view.context);
		if (auto const failed = first_failure(program, scratch, reading)) {
			return std::unexpected(*failed);
		}
		auto const measure = [&]<class K>(ub::Wait wait) -> Result<void> {
			auto best = std::optional<shapes::HopReading>{};
			for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
				auto const read = scratch->clear(view)
				                      .and_then([&] {
					                      return run_once<K>(view, *program, scratch->device(), reading->device());
				                      })
				                      .and_then([&] {
					                      return reading->fetch(view);
				                      });
				if (!read) {
					return std::unexpected(read.error());
				}
				if (launch >= warmup_launches && (!best || read->front().cycles < best->cycles)) {
					best = read->front();
				}
			}
			hops.push_back({.card = card, .wait = wait, .reading = *best});
			return {};
		};
		return measure.template operator()<HopBackoff>(ub::Wait::Backoff).and_then([&] {
			return measure.template operator()<HopSpin>(ub::Wait::Spin);
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_cross_sm_hop(hops, bench.calibration);
}

/** One card's least total cycles for the handoff's stages. */
export struct HandoffReading {
	model::Card card;
	double cycles;
};

export [[nodiscard]] auto evaluate_stage_handoff(std::span<HandoffReading const> readings, Calibration calibration) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		auto const per_stage = reading.cycles / static_cast<double>(shapes::handoff_iterations);
		outcome.readings.push_back({.label = std::format("{}, one stage through a 4-stage ring, one producer and two consumer warpgroups", on(reading.card)),
		                            .value = calibration.nanoseconds(reading.card, per_stage),
		                            .unit = Unit::Nanoseconds});
	}
	return outcome;
}

export [[nodiscard]] auto stage_handoff(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<HandoffReading>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<kernels::StageHandoffKernel>());
		auto cycles = Readback<unsigned long long, 1>::create(view.context);
		if (auto const failed = first_failure(program, cycles)) {
			return std::unexpected(*failed);
		}
		return least_cycles<kernels::StageHandoffKernel>(view, *program, *cycles, nothing).transform([&](double counted) {
			readings.push_back({.card = card, .cycles = counted});
		});
	});
	return ran.transform([&] {
		return evaluate_stage_handoff(readings, bench.calibration);
	});
}

}

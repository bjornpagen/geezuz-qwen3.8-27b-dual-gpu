// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight-stream benchmarks: the read stream and its phase-b gate, the tile layout, the memory clock under load and
// at an offset step, what l2 carries past the stream, the ring under stalls, the clock under a decode-like load and
// activation reuse under the stream's l2 priorities. Each driver runs its kernels and hands the raw results to an
// evaluation that is a plain function of them.
export module engine.ubench:stream;

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
namespace shapes = engine::device::ubench::stream;

/** The phase-b gate: the best read stream reaches this share of the measured memory clock's peak. */
export inline constexpr double read_gate = 0.93;

/** One timed sweep point of the read stream on one card. */
export struct ReadPoint {
	model::Card card;
	int producer_warps;
	int depth_kib;
	double milliseconds;
};

/**
 * Every point's bandwidth, each card's best, and the gate: every card's best at `read_gate` of the peak its own memory
 * clock gives.
 */
export [[nodiscard]] auto evaluate_read_stream(std::span<ReadPoint const> points, std::span<MemoryState const, model::cards> memory) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Holds, .notes = {}};
	for (auto const card : device::card_pair) {
		auto mine = std::vector<ReadPoint>{};
		std::ranges::copy_if(points, std::back_inserter(mine), [&](ReadPoint const& point) {
			return point.card == card;
		});
		core::invariant(!mine.empty());
		for (auto const& point : mine) {
			outcome.readings.push_back({
			    .label = std::format("{}, {} producer warp(s), {} KiB in flight", on(card), point.producer_warps, point.depth_kib),
			    .value = gigabytes_per_second(static_cast<double>(shapes::bytes), point.milliseconds),
			    .unit = Unit::GigabytesPerSecond,
			});
		}
		auto const best = std::ranges::min_element(mine, {}, &ReadPoint::milliseconds);
		auto const& state = memory[std::to_underlying(card)];
		auto const rate = gigabytes_per_second(static_cast<double>(shapes::bytes), best->milliseconds);
		auto const share = rate / state.peak_gigabytes_per_second();
		outcome.readings.push_back({.label = std::format("{}, best share of {} MHz's peak", on(card), state.clock.megahertz), .value = share, .unit = Unit::Ratio});
		outcome.notes.push_back(std::format("{}: best {:.1f} GB/s at {} producer warp(s) and {} KiB in flight, {:.1f}% of {:.0f} GB/s", on(card), rate,
		                                    best->producer_warps, best->depth_kib, share * 100.0, state.peak_gigabytes_per_second()));
		if (share < read_gate) {
			outcome.verdict = Verdict::Fails;
		}
	}
	return outcome;
}

/** Each card's memory clock now. */
[[nodiscard]] auto memory_states(BenchView const& bench) -> Result<std::array<MemoryState, model::cards>> {
	auto states = std::array<MemoryState, model::cards>{};
	auto const read = for_each_card([&](model::Card card) {
		return memory_state(bench, card).transform([&](MemoryState state) {
			states.at(std::to_underlying(card)) = state;
		});
	});
	return read.transform([&] {
		return states;
	});
}

template<template<class> class Kernel, class Sweep>
[[nodiscard]] auto load_sweep(device::Context const& context) {
	return std::apply(
	    [&](auto... shapes) {
		    return context.load<Kernel<decltype(shapes)>...>();
	    },
	    Sweep{});
}

export [[nodiscard]] auto read_stream(BenchView const& bench) -> Result<Outcome> {
	auto points = std::vector<ReadPoint>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(load_sweep<kernels::ReadKernel, kernels::ReadSweep>(view.context));
		auto const source = scratch<ub::Word4, shapes::chunks>(view.context);
		if (auto const failed = first_failure(program, source)) {
			return std::unexpected(*failed);
		}
		return for_each_shape<kernels::ReadSweep>([&](auto shape) -> Result<void> {
			using Shape = decltype(shape);
			return best_time<kernels::ReadKernel<Shape>>(view, *program, nothing, *source).transform([&](double milliseconds) {
				points.push_back({.card = card, .producer_warps = Shape::producer_warps, .depth_kib = Shape::depth_kib, .milliseconds = milliseconds});
			});
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return memory_states(bench).transform([&](std::array<MemoryState, model::cards> const& memory) {
		return evaluate_read_stream(points, memory);
	});
}

/** The canonical stream and its row-major twin on one card, in milliseconds. */
export struct LayoutTiming {
	model::Card card;
	double contiguous;
	double row_major;
};

export [[nodiscard]] auto evaluate_tile_layout(std::span<LayoutTiming const> timings) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& timing : timings) {
		auto const contiguous = gigabytes_per_second(static_cast<double>(shapes::bytes), timing.contiguous);
		auto const row_major = gigabytes_per_second(static_cast<double>(shapes::bytes), timing.row_major);
		outcome.readings.push_back({.label = std::format("{}, contiguous 16 KiB tiles", on(timing.card)), .value = contiguous, .unit = Unit::GigabytesPerSecond});
		outcome.readings.push_back({.label = std::format("{}, row-major tiles", on(timing.card)), .value = row_major, .unit = Unit::GigabytesPerSecond});
		outcome.readings.push_back({.label = std::format("{}, contiguous over row-major", on(timing.card)), .value = contiguous / row_major, .unit = Unit::Ratio});
	}
	return outcome;
}

using CanonicalKernel = kernels::ReadKernel<shapes::Canonical>;
using RowMajorKernel = kernels::ReadKernel<kernels::RowMajorRead>;

export [[nodiscard]] auto tile_layout(BenchView const& bench) -> Result<Outcome> {
	auto timings = std::vector<LayoutTiming>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<CanonicalKernel, RowMajorKernel>());
		auto const source = scratch<ub::Word4, shapes::chunks>(view.context);
		if (auto const failed = first_failure(program, source)) {
			return std::unexpected(*failed);
		}
		auto const contiguous = best_time<CanonicalKernel>(view, *program, nothing, *source);
		auto const row_major = contiguous.and_then([&](double) {
			return best_time<RowMajorKernel>(view, *program, nothing, *source);
		});
		return row_major.transform([&](double strided) {
			timings.push_back({.card = card, .contiguous = *contiguous, .row_major = strided});
		});
	});
	return ran.transform([&] {
		return evaluate_tile_layout(timings);
	});
}

/** How many canonical streams keep a card busy while its clock is read: a few hundred milliseconds. */
inline constexpr int loaded_launches = 64;

/** A card's memory clock idle and under the stream. */
export struct ClockUnderLoad {
	model::Card card;
	device::MemoryClock idle;
	device::MemoryClock loaded;
};

[[nodiscard]] auto state_name(std::optional<device::PerformanceState> state) -> std::string {
	return state ? std::format("P{}", std::to_underlying(*state)) : std::string{"an unknown p-state"};
}

/** The p2 memory clock CUDA work runs at on GeForce Ampere, and the p0 maximum. */
inline constexpr std::uint32_t p2_megahertz = 9501;
inline constexpr std::uint32_t p0_megahertz = 9751;

export [[nodiscard]] auto evaluate_memory_clock(std::span<ClockUnderLoad const> readings) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		outcome.readings.push_back({.label = std::format("{}, idle", on(reading.card)), .value = static_cast<double>(reading.idle.megahertz), .unit = Unit::Megahertz});
		outcome.readings.push_back({.label = std::format("{}, under the stream", on(reading.card)), .value = static_cast<double>(reading.loaded.megahertz), .unit = Unit::Megahertz});
		auto const base = reading.loaded.megahertz >= p0_megahertz ? "p0's 936 GB/s: the budget may count from 9751 MHz"
		                  : reading.loaded.megahertz == p2_megahertz ? "p2's 912 GB/s: the budget counts from 9501 MHz"
		                                                             : "neither 9501 nor 9751 MHz: an offset or a lock is in force";
		outcome.notes.push_back(std::format("{}: {} MHz in {} under CUDA work; base {}", on(reading.card), reading.loaded.megahertz, state_name(reading.loaded.state), base));
	}
	return outcome;
}

export [[nodiscard]] auto memory_clock(BenchView const& bench) -> Result<Outcome> {
	auto idle = memory_states(bench);
	if (!idle) {
		return std::unexpected(idle.error());
	}
	auto programs = std::vector<device::Program<CanonicalKernel>>{};
	auto sources = std::vector<kernels::StreamSource>{};
	auto const loaded = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto program = lift(view.context.load<CanonicalKernel>());
		auto source = scratch<ub::Word4, shapes::chunks>(view.context);
		if (auto const failed = first_failure(program, source)) {
			return std::unexpected(*failed);
		}
		for (auto launch = 0; launch < loaded_launches; ++launch) {
			if (auto queued = lift(kernels::launch<CanonicalKernel>(view.context, *program, view.stream, *source)); !queued) {
				return queued;
			}
		}
		programs.push_back(std::move(*program));
		sources.push_back(std::move(*source));
		return {};
	});
	if (!loaded) {
		return std::unexpected(loaded.error());
	}
	auto const busy = memory_states(bench);
	auto const drained = for_each_card([&](model::Card card) {
		return finish(bench.card(card), CanonicalKernel::entry);
	});
	if (!busy || !drained) {
		return std::unexpected(!busy ? busy.error() : drained.error());
	}
	auto readings = std::vector<ClockUnderLoad>{};
	for (auto const card : device::card_pair) {
		auto const index = std::to_underlying(card);
		readings.push_back({.card = card, .idle = idle->at(index).clock, .loaded = busy->at(index).clock});
	}
	return evaluate_memory_clock(readings);
}

/** One offset step's reading on one card: the canonical stream's time and the memory clock it ran at. */
export struct StepReading {
	model::Card card;
	double milliseconds;
	MemoryState memory;
};

export [[nodiscard]] auto evaluate_offset_step(std::span<StepReading const> readings) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		auto const rate = gigabytes_per_second(static_cast<double>(shapes::bytes), reading.milliseconds);
		outcome.readings.push_back({.label = std::format("{}, memory clock", on(reading.card)), .value = static_cast<double>(reading.memory.clock.megahertz), .unit = Unit::Megahertz});
		outcome.readings.push_back({.label = std::format("{}, pure read", on(reading.card)), .value = rate, .unit = Unit::GigabytesPerSecond});
		outcome.readings.push_back({.label = std::format("{}, share of the clock's peak", on(reading.card)), .value = rate / reading.memory.peak_gigabytes_per_second(), .unit = Unit::Ratio});
	}
	outcome.notes.emplace_back("one offset step: compare the share across steps; the knee is where it stops holding");
	return outcome;
}

export [[nodiscard]] auto offset_step(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<StepReading>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<CanonicalKernel>());
		auto const source = scratch<ub::Word4, shapes::chunks>(view.context);
		if (auto const failed = first_failure(program, source)) {
			return std::unexpected(*failed);
		}
		auto const took = best_time<CanonicalKernel>(view, *program, nothing, *source);
		auto const memory = took.and_then([&](double) {
			return memory_state(bench, card);
		});
		return memory.transform([&](MemoryState state) {
			readings.push_back({.card = card, .milliseconds = *took, .memory = state});
		});
	});
	return ran.transform([&] {
		return evaluate_offset_step(readings);
	});
}

/** Per 128 KiB segment, the best read-back cycles hot (just fetched), cold (never fetched) and after the stream. */
export struct SegmentTimes {
	std::array<double, ub::l2::segments> hot;
	std::array<double, ub::l2::segments> cold;
	std::array<double, ub::l2::segments> after;
};

/**
 * How much of the evict-last region survived the stream: each segment's share of the way from cold back to hot, in
 * MiB. A segment whose cold read is not slower than its hot read gives no contrast to measure against.
 */
export [[nodiscard]] auto evaluate_l2_persistence(std::span<SegmentTimes const> cards) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const card : device::card_pair) {
		auto const& times = cards[std::to_underlying(card)];
		auto surviving = 0.0;
		for (auto segment = std::size_t{0}; segment < times.hot.size(); ++segment) {
			auto const contrast = times.cold.at(segment) - times.hot.at(segment);
			if (!(contrast > 0.0)) {
				return no_reading(kernels::L2ProbeKernel::entry, "a cold segment read no slower than a hot one");
			}
			auto const kept = std::clamp((times.cold.at(segment) - times.after.at(segment)) / contrast, 0.0, 1.0);
			surviving += kept * static_cast<double>(ub::l2::segment_bytes) / (1024.0 * 1024.0);
		}
		outcome.readings.push_back({.label = std::format("{}, evict-last MiB surviving a 13.9 GB evict-first stream", on(card)), .value = surviving, .unit = Unit::Mebibytes});
	}
	return outcome;
}

using PrefetchKernel = kernels::L2PrefetchKernel;
using ProbeKernel = kernels::L2ProbeKernel;

/** Every segment's read-back cycles from one probe. */
[[nodiscard]] auto probe(CardView view, device::Program<PrefetchKernel, ProbeKernel> const& program, kernels::L2Region const& region,
                         Readback<ub::Rate, ub::l2::segments>& cycles) -> Result<std::array<double, ub::l2::segments>> {
	return run_once<ProbeKernel>(view, program, region, cycles.device())
	    .and_then([&] {
		    return cycles.fetch(view);
	    })
	    .transform([](std::span<ub::Rate const, ub::l2::segments> rates) {
		    auto times = std::array<double, ub::l2::segments>{};
		    std::ranges::transform(rates, times.begin(), [](ub::Rate const& rate) {
			    return static_cast<double>(rate.cycles);
		    });
		    return times;
	    });
}

auto keep_least(std::array<double, ub::l2::segments>& best, std::array<double, ub::l2::segments> const& seen) -> void {
	std::ranges::transform(best, seen, best.begin(), [](double kept, double now) {
		return std::min(kept, now);
	});
}

export [[nodiscard]] auto l2_persistence(BenchView const& bench) -> Result<Outcome> {
	auto cards = std::vector<SegmentTimes>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<PrefetchKernel, ProbeKernel>());
		auto const stream = lift(view.context.load<CanonicalKernel>());
		auto const fetched = scratch<ub::Word4, ub::l2::region_chunks>(view.context);
		auto const untouched = scratch<ub::Word4, ub::l2::region_chunks>(view.context);
		auto const source = scratch<ub::Word4, shapes::chunks>(view.context);
		auto cycles = Readback<ub::Rate, ub::l2::segments>::create(view.context);
		if (auto const failed = first_failure(program, stream, fetched, untouched, source, cycles)) {
			return std::unexpected(*failed);
		}
		constexpr auto never = std::numeric_limits<double>::infinity();
		auto times = SegmentTimes{};
		times.hot.fill(never);
		times.cold.fill(never);
		times.after.fill(never);
		for (auto repeat = 0; repeat < timed_launches; ++repeat) {
			auto const hot = run_once<PrefetchKernel>(view, *program, *fetched).and_then([&] {
				return probe(view, *program, *fetched, *cycles);
			});
			auto const cold = hot.and_then([&](auto const&) {
				return probe(view, *program, *untouched, *cycles);
			});
			auto streamed = cold.and_then([&](auto const&) {
				return run_once<PrefetchKernel>(view, *program, *fetched);
			});
			for (auto pass = 0; pass < ub::l2::stream_passes && streamed; ++pass) {
				streamed = run_once<CanonicalKernel>(view, *stream, *source);
			}
			auto const after = streamed.and_then([&] {
				return probe(view, *program, *fetched, *cycles);
			});
			if (!after) {
				return std::unexpected(after.error());
			}
			keep_least(times.hot, *hot);
			keep_least(times.cold, *cold);
			keep_least(times.after, *after);
		}
		cards.push_back(times);
		return {};
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_l2_persistence(cards);
}

/** One stall sweep point on one card, in milliseconds. */
export struct StallPoint {
	model::Card card;
	int ring_kib;
	int stall_us;
	double milliseconds;
};

/**
 * Per ring and stall: the time each stall exposed (the run's excess over the same ring unstalled, per stall) and the
 * share of the stall the ring hid.
 */
export [[nodiscard]] auto evaluate_ring_stall(std::span<StallPoint const> points) -> Result<Outcome> {
	constexpr auto stalls = static_cast<double>(shapes::tiles_per_block) / 16.0;
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& point : points) {
		if (point.stall_us == 0) {
			continue;
		}
		auto const baseline = std::ranges::find_if(points, [&](StallPoint const& candidate) {
			return candidate.card == point.card && candidate.ring_kib == point.ring_kib && candidate.stall_us == 0;
		});
		core::invariant(baseline != points.end());
		auto const exposed = std::max(0.0, (point.milliseconds - baseline->milliseconds) * 1000.0 / stalls);
		auto const label = std::format("{}, {} KiB ring, {} us stalls", on(point.card), point.ring_kib, point.stall_us);
		outcome.readings.push_back({.label = label + ", exposed per stall", .value = exposed, .unit = Unit::Microseconds});
		outcome.readings.push_back({.label = label + ", share hidden", .value = 1.0 - std::min(1.0, exposed / point.stall_us), .unit = Unit::Ratio});
	}
	return outcome;
}

static_assert(shapes::Stall<32, 4>::stall_period == 16, "the evaluation counts one stall per sixteen tiles");

export [[nodiscard]] auto ring_stall(BenchView const& bench) -> Result<Outcome> {
	auto points = std::vector<StallPoint>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const source = scratch<ub::Word4, shapes::chunks>(view.context);
		if (!source) {
			return std::unexpected(source.error());
		}
		return for_each_shape<kernels::StallSweep>([&](auto row) -> Result<void> {
			using Row = decltype(row);
			auto const program = lift(load_sweep<kernels::StallKernel, Row>(view.context));
			if (!program) {
				return std::unexpected(program.error());
			}
			return for_each_shape<Row>([&](auto shape) -> Result<void> {
				using Shape = decltype(shape);
				return best_time<kernels::StallKernel<Shape>>(view, *program, nothing, *source).transform([&](double milliseconds) {
					points.push_back({.card = card, .ring_kib = Shape::ring_kib, .stall_us = Shape::stall_us, .milliseconds = milliseconds});
				});
			});
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_ring_stall(points);
}

/** The SM clock a launch's blocks saw: the median over blocks of cycles over nanoseconds, in MHz. */
export [[nodiscard]] auto launch_megahertz(std::span<ub::Clocks const> blocks) -> Result<double> {
	auto rates = std::vector<double>{};
	for (auto const& clocks : blocks) {
		if (clocks.nanoseconds == 0 || clocks.cycles == 0) {
			return no_reading(kernels::SustainKernel::entry, "a block reported no elapsed time");
		}
		rates.push_back(static_cast<double>(clocks.cycles) * 1000.0 / static_cast<double>(clocks.nanoseconds));
	}
	return median(std::move(rates));
}

/** Each card's SM clock launch by launch through the soak. */
export [[nodiscard]] auto evaluate_sustained_clock(std::span<std::vector<double> const, model::cards> megahertz) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const card : device::card_pair) {
		auto const& series = megahertz[std::to_underlying(card)];
		core::invariant(!series.empty());
		auto const settled = std::vector<double>(series.begin() + static_cast<std::ptrdiff_t>(series.size() / 2), series.end());
		outcome.readings.push_back({.label = std::format("{}, first launch", on(card)), .value = series.front(), .unit = Unit::Megahertz});
		outcome.readings.push_back({.label = std::format("{}, lowest launch", on(card)), .value = std::ranges::min(series), .unit = Unit::Megahertz});
		outcome.readings.push_back({.label = std::format("{}, median of the second half", on(card)), .value = median(settled), .unit = Unit::Megahertz});
	}
	outcome.notes.emplace_back("the lock is one step below the second half's median, if the sw_power_cap counter stayed flat");
	return outcome;
}

using kernels::SustainKernel;

export [[nodiscard]] auto sustained_clock(BenchView const& bench) -> Result<Outcome> {
	struct Card {
		device::Program<SustainKernel> program;
		kernels::StreamSource source;
		Readback<ub::Clocks, ub::sms> clocks;
	};
	auto cards = std::vector<Card>{};
	auto const set_up = for_each_card([&](model::Card card) -> Result<void> {
		auto const& context = bench.card(card).context;
		auto program = lift(context.load<SustainKernel>());
		auto source = scratch<ub::Word4, shapes::chunks>(context);
		auto clocks = Readback<ub::Clocks, ub::sms>::create(context);
		if (auto const failed = first_failure(program, source, clocks)) {
			return std::unexpected(*failed);
		}
		cards.push_back({.program = std::move(*program), .source = std::move(*source), .clocks = std::move(*clocks)});
		return {};
	});
	if (!set_up) {
		return std::unexpected(set_up.error());
	}
	auto megahertz = std::array<std::vector<double>, model::cards>{};
	for (auto launch = 0; launch < shapes::Sustain::launches; ++launch) {
		// Both cards run at once, as they do in the engine: each card's power and clock are its own.
		auto const queued = for_each_card([&](model::Card card) {
			auto const& mine = cards.at(std::to_underlying(card));
			auto const view = bench.card(card);
			return lift(kernels::launch<SustainKernel>(view.context, mine.program, view.stream, mine.source, mine.clocks.device()));
		});
		auto const read = queued.and_then([&] {
			return for_each_card([&](model::Card card) {
				auto& mine = cards.at(std::to_underlying(card));
				auto const view = bench.card(card);
				return finish(view, SustainKernel::entry)
				    .and_then([&] {
					    return mine.clocks.fetch(view);
				    })
				    .and_then([](std::span<ub::Clocks const, ub::sms> blocks) {
					    return launch_megahertz(blocks);
				    })
				    .transform([&](double rate) {
					    megahertz.at(std::to_underlying(card)).push_back(rate);
				    });
			});
		});
		if (!read) {
			return std::unexpected(read.error());
		}
	}
	return evaluate_sustained_clock(megahertz);
}

/** One card's activation-load cycles per block under each priority assignment. */
export struct ReuseReading {
	model::Card card;
	double split_cycles;
	double uniform_cycles;
};

/** The mean cycles of one activation load across blocks, from each block's total over its tiles. */
export [[nodiscard]] auto load_cycles(std::span<ub::Rate const> blocks) -> Result<double> {
	auto total = 0.0;
	for (auto const& block : blocks) {
		if (block.cycles == 0) {
			return no_reading(kernels::ReuseKernel<shapes::Priorities::Split>::entry, "a block reported no load time");
		}
		total += static_cast<double>(block.cycles);
	}
	return total / static_cast<double>(blocks.size()) / static_cast<double>(shapes::tiles_per_block);
}

export [[nodiscard]] auto evaluate_l2_policy_reuse(std::span<ReuseReading const> readings, Calibration calibration) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		outcome.readings.push_back({.label = std::format("{}, activation load, stream evict-first and activations evict-last", on(reading.card)),
		                            .value = calibration.nanoseconds(reading.card, reading.split_cycles),
		                            .unit = Unit::Nanoseconds});
		outcome.readings.push_back({.label = std::format("{}, activation load, everything evict-normal", on(reading.card)),
		                            .value = calibration.nanoseconds(reading.card, reading.uniform_cycles),
		                            .unit = Unit::Nanoseconds});
		outcome.readings.push_back({.label = std::format("{}, uniform over split", on(reading.card)), .value = reading.uniform_cycles / reading.split_cycles, .unit = Unit::Ratio});
	}
	return outcome;
}

using SplitKernel = kernels::ReuseKernel<shapes::Priorities::Split>;
using UniformKernel = kernels::ReuseKernel<shapes::Priorities::Uniform>;

export [[nodiscard]] auto l2_policy_reuse(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<ReuseReading>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<SplitKernel, UniformKernel>());
		auto const source = scratch<ub::Word4, shapes::chunks>(view.context);
		auto const activations = scratch<ub::Word4, shapes::Reuse<shapes::Priorities::Split>::activation_chunks>(view.context);
		auto rates = Readback<ub::Rate, ub::sms>::create(view.context);
		if (auto const failed = first_failure(program, source, activations, rates)) {
			return std::unexpected(*failed);
		}
		auto const measure = [&]<class K> -> Result<double> {
			return best_time<K>(view, *program, nothing, *source, *activations, rates->device())
			    .and_then([&](double) {
				    return rates->fetch(view);
			    })
			    .and_then([](std::span<ub::Rate const, ub::sms> blocks) {
				    return load_cycles(blocks);
			    });
		};
		auto const split = measure.template operator()<SplitKernel>();
		auto const uniform = split.and_then([&](double) {
			return measure.template operator()<UniformKernel>();
		});
		return uniform.transform([&](double cycles) {
			readings.push_back({.card = card, .split_cycles = *split, .uniform_cycles = cycles});
		});
	});
	return ran.transform([&] {
		return evaluate_l2_policy_reuse(readings, bench.calibration);
	});
}

}

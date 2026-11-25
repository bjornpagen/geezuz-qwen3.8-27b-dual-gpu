// SPDX-License-Identifier: AGPL-3.0-only
//
// The runner: which benchmarks a command line names, running them in catalogue order over the started machine, and
// printing each outcome beside the review's expectation, the evidence behind it, the decision it settles and what the
// operator must have set. The pending list closes every report, so a measurement that could not run is never mistaken
// for one that passed.
export module engine.ubench:runner;

import std;
import engine.core;
import engine.device;
import engine.model;
import :outcome;
import :catalogue;
import :protocol;
import :stream;
import :sync;
import :compute;
import :link;
import :checks;

namespace engine::ubench {

/** A command-line name that names no benchmark. */
export struct UnknownBenchmark {
	std::string name;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto operator==(UnknownBenchmark const&) const -> bool = default;
};

/**
 * The benchmarks the arguments name, in catalogue order and each once; every benchmark when there are no arguments.
 * Names are the catalogue's, nothing else.
 */
export [[nodiscard]] auto select(std::span<std::string_view const> arguments) -> core::Result<std::vector<Benchmark>, UnknownBenchmark> {
	auto chosen = std::array<bool, catalogue.size()>{};
	for (auto const argument : arguments) {
		auto const benchmark = parse(argument);
		if (!benchmark) {
			return std::unexpected(UnknownBenchmark{.name = std::string{argument}});
		}
		chosen.at(std::to_underlying(*benchmark)) = true;
	}
	auto const everything = arguments.empty();
	auto selection = std::vector<Benchmark>{};
	for (auto const& candidate : catalogue) {
		if (everything || chosen.at(std::to_underlying(candidate.benchmark))) {
			selection.push_back(candidate.benchmark);
		}
	}
	return selection;
}

/** Runs one benchmark on the machine. */
export [[nodiscard]] auto run(Benchmark benchmark, BenchView const& bench) -> Result<Outcome> {
	switch (benchmark) {
	case Benchmark::ReadStream:
		return read_stream(bench);
	case Benchmark::TileLayout:
		return tile_layout(bench);
	case Benchmark::MemoryClock:
		return memory_clock(bench);
	case Benchmark::TensorRate:
		return tensor_rate(bench);
	case Benchmark::DecodeExact:
		return decode_exact(bench);
	case Benchmark::BarrierCost:
		return barrier_cost(bench);
	case Benchmark::CrossSmHop:
		return cross_sm_hop(bench);
	case Benchmark::PeerFlag:
		return peer_flag(bench);
	case Benchmark::PeerPush:
		return peer_push(bench);
	case Benchmark::ReleaseCost:
		return release_cost(bench);
	case Benchmark::LinkHealth:
		return link_health(bench);
	case Benchmark::HostFetch:
		return host_fetch(bench);
	case Benchmark::L2Persistence:
		return l2_persistence(bench);
	case Benchmark::RingStall:
		return ring_stall(bench);
	case Benchmark::SustainedClock:
		return sustained_clock(bench);
	case Benchmark::OffsetStep:
		return offset_step(bench);
	case Benchmark::TranscendentalRate:
		return transcendental_rate(bench);
	case Benchmark::StageHandoff:
		return stage_handoff(bench);
	case Benchmark::L2PolicyReuse:
		return l2_policy_reuse(bench);
	case Benchmark::LdmatrixSwizzle:
		return ldmatrix_swizzle(bench);
	case Benchmark::MmaDoubleBuffer:
		return mma_double_buffer(bench);
	case Benchmark::WarpReductions:
		return warp_reductions(bench);
	case Benchmark::CheckFragments:
		return check_fragments(bench);
	case Benchmark::CheckReduce:
		return check_reduce(bench);
	case Benchmark::CheckTranscendentals:
		return check_transcendentals(bench);
	}
	std::unreachable();
}

/** What a benchmark expects and settles: the review's row for a question, the catalogue's own words otherwise. */
struct Expectation {
	std::string heading;
	std::string_view expected;
	std::string_view settles;
};

[[nodiscard]] auto expectation(Entry const& entry, Table const& review) -> Expectation {
	struct Read {
		Entry const& entry;
		Table const& review;

		[[nodiscard]] auto operator()(Question question) const -> Expectation {
			auto const found = row(review, question.number);
			return {.heading = std::format("{} (U{}: {})", entry.name, question.number, found.measures), .expected = found.expected, .settles = found.settles};
		}

		[[nodiscard]] auto operator()(Primitive const& primitive) const -> Expectation {
			return {.heading = std::format("{} (a required primitive)", entry.name), .expected = primitive.expected, .settles = primitive.settles};
		}
	};
	return std::visit(Read{.entry = entry, .review = review}, entry.answers);
}

[[nodiscard]] constexpr auto verdict_name(Verdict verdict) -> std::string_view {
	switch (verdict) {
	case Verdict::Measured:
		return "measured (the decision is the operator's, against the expectation)";
	case Verdict::Holds:
		return "holds";
	case Verdict::Fails:
		return "FAILS";
	}
	std::unreachable();
}

/** Prints one benchmark's report. */
export void print(Benchmark benchmark, Result<Outcome> const& result, Table const& review) {
	auto const& listed = entry(benchmark);
	auto const said = expectation(listed, review);
	std::println("== {}", said.heading);
	std::println("expected: {}", said.expected);
	std::println("basis:    {}", listed.basis);
	std::println("settles:  {}", said.settles);
	if (listed.operator_sets) {
		std::println("operator: {}", *listed.operator_sets);
	}
	if (!result) {
		std::println("error:    {}", describe(result.error()));
		return;
	}
	for (auto const& reading : result->readings) {
		std::println("  {}", render(reading));
	}
	for (auto const& note : result->notes) {
		std::println("  note: {}", note);
	}
	std::println("verdict:  {}", verdict_name(result->verdict));
}

/** Prints the measurements that cannot run yet and what each waits for. */
export void print_pending() {
	std::println("== pending: measurements that did not run");
	for (auto const& waiting : pending) {
		if (waiting.question) {
			std::println("  {} (U{}): waits for {}", waiting.name, *waiting.question, waiting.waits_for);
		} else {
			std::println("  {}: waits for {}", waiting.name, waiting.waits_for);
		}
	}
}

/** Prints the clocks the run starts at: each card's memory clock and p-state, and its calibrated SM clock. */
export void print_clocks(BenchView const& bench) {
	for (auto const card : device::card_pair) {
		auto const memory = memory_state(bench, card);
		auto const sm = bench.calibration.megahertz.at(std::to_underlying(card));
		if (!memory) {
			std::println("{}: SM {:.0f} MHz idle; memory clock unreadable: {}", on(card), sm, describe(memory.error()));
			continue;
		}
		auto const state = memory->clock.state ? std::format("P{}", std::to_underlying(*memory->clock.state)) : std::string{"an unknown p-state"};
		std::println("{}: SM {:.0f} MHz idle (calibrated on the card); memory {} MHz in {}, a {:.0f} GB/s peak", on(card), sm, memory->clock.megahertz,
		             state, memory->peak_gigabytes_per_second());
	}
	std::println("clock locks, power limits and offsets in force are the operator's to note: the runner reads clocks, it sets none");
}

/** Whether a report passed: it ran, and its verdict does not fail. */
export [[nodiscard]] auto passed(Result<Outcome> const& result) -> bool {
	return result.has_value() && result->verdict != Verdict::Fails;
}

/**
 * A whole run: starts streams on both cards, calibrates their SM clocks, runs every selected benchmark and prints
 * every report, the pending list last. Whether every selected benchmark ran and none failed.
 */
export [[nodiscard]] auto run_all(device::Devices const& devices, std::span<Benchmark const> selection) -> Result<bool> {
	auto zero = lift(devices.context(model::Card::Zero).create_stream());
	auto one = lift(devices.context(model::Card::One).create_stream());
	if (auto const failed = first_failure(zero, one)) {
		return std::unexpected(*failed);
	}
	auto const streams = std::array{std::move(*zero), std::move(*one)};
	auto const calibration = calibrate(devices, streams);
	if (!calibration) {
		return std::unexpected(calibration.error());
	}
	auto const bench = BenchView{.devices = devices, .streams = streams, .calibration = *calibration};
	auto const table = review();
	print_clocks(bench);
	auto all_passed = true;
	for (auto const benchmark : selection) {
		auto const result = run(benchmark, bench);
		print(benchmark, result, table);
		all_passed = all_passed && passed(result);
	}
	print_pending();
	return all_passed;
}

}

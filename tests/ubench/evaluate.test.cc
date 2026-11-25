// SPDX-License-Identifier: AGPL-3.0-only
//
// The evaluations against synthetic results: what each benchmark makes of what its kernels report, the gates and the
// refusals included, each checked against the arithmetic it states.
import std;
import engine.device;
import engine.mma_model;
import engine.model;
import engine.prims;
import engine.testing;
import engine.ubench;
import engine.ubench_device;

namespace {

namespace ubench = engine::ubench;
namespace ub = engine::device::ubench;
namespace selfcheck = engine::device::prims::selfcheck;

using engine::model::Card;
using ubench::Verdict;

inline constexpr auto stream_bytes = static_cast<double>(ub::stream::bytes);

[[nodiscard]] auto near(double got, double expected) -> bool {
	return std::abs(got - expected) <= 1e-9 * std::max(1.0, std::abs(expected));
}

[[nodiscard]] auto memory_at(std::uint32_t megahertz) -> ubench::MemoryState {
	return {.clock = {.megahertz = megahertz, .state = engine::device::PerformanceState::P2}};
}

/**
 * The read gate holds exactly when each card's fastest point reaches 93% of the peak its own memory clock gives,
 * over random sweeps and clocks.
 */
[[nodiscard]] auto the_read_gate_is_each_cards_best_against_its_peak(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 500; ++trial) {
		auto points = std::vector<ubench::ReadPoint>{};
		auto best = std::array<double, 2>{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
		for (auto const card : engine::device::card_pair) {
			auto const count = std::uniform_int_distribution<int>{1, 18}(random);
			for (auto point = 0; point < count; ++point) {
				auto const milliseconds = std::uniform_real_distribution<double>{2.0, 4.0}(random);
				points.push_back({.card = card, .producer_warps = 1, .depth_kib = 4, .milliseconds = milliseconds});
				best.at(std::to_underlying(card)) = std::min(best.at(std::to_underlying(card)), milliseconds);
			}
		}
		std::ranges::shuffle(points, random);
		auto const memory = std::array{memory_at(std::uniform_int_distribution<std::uint32_t>{9000, 10500}(random)),
		                               memory_at(std::uniform_int_distribution<std::uint32_t>{9000, 10500}(random))};
		auto const outcome = ubench::evaluate_read_stream(points, memory);
		auto const reaches = [&](std::size_t card) {
			return stream_bytes / (best.at(card) * 1e6) >= ubench::read_gate * memory.at(card).peak_gigabytes_per_second();
		};
		auto const expected = reaches(0) && reaches(1) ? Verdict::Holds : Verdict::Fails;
		holds = holds && outcome.verdict == expected && outcome.readings.size() == points.size() + 2;
	}
	return holds;
}

/** The peak is 96 bytes per memory-clock cycle: 9501 MHz gives the 912 GB/s the budget counts from, 9751 gives 936. */
[[nodiscard]] auto the_peak_is_the_bus_at_the_clock() -> bool {
	return near(memory_at(9501).peak_gigabytes_per_second(), 912.096) && near(memory_at(9751).peak_gigabytes_per_second(), 936.096);
}

[[nodiscard]] auto report_with(unsigned checked, unsigned mismatched) -> selfcheck::Report {
	auto report = selfcheck::Report{.checked = checked, .mismatched = mismatched, .first = {}};
	for (auto index = 0U; index < std::min(mismatched, selfcheck::kept_mismatches); ++index) {
		std::span{report.first}[index] = {.input = index, .detail = 1, .got = 2, .expected = 3};
	}
	return report;
}

/**
 * A self-check report holds exactly when it compared something and nothing differed; a report that compared nothing
 * is no reading; a failing one shows its first mismatches, at most eight.
 */
[[nodiscard]] auto a_report_holds_only_without_mismatches(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = !ubench::evaluate_report(Card::Zero, report_with(0, 0), "k").has_value();
	for (auto trial = 0; trial < 1000; ++trial) {
		auto const checked = std::uniform_int_distribution<unsigned>{1, 1U << 20U}(random);
		auto const mismatched = trial % 2 == 0 ? 0U : std::uniform_int_distribution<unsigned>{1, checked}(random);
		auto const outcome = ubench::evaluate_report(Card::One, report_with(checked, mismatched), "k");
		holds = holds && outcome && outcome->verdict == (mismatched == 0 ? Verdict::Holds : Verdict::Fails) &&
		        outcome->notes.size() == std::min<std::size_t>(mismatched, 8) && outcome->readings.at(1).value == static_cast<double>(mismatched);
	}
	return holds;
}

/** Joining keeps every reading and note and the worst verdict. */
[[nodiscard]] auto joining_keeps_the_worst_verdict() -> bool {
	auto const part = [](Verdict verdict) {
		return ubench::Outcome{.readings = {{.label = "r", .value = 1.0, .unit = ubench::Unit::Count}}, .verdict = verdict, .notes = {"n"}};
	};
	auto const verdicts = std::array{Verdict::Measured, Verdict::Holds, Verdict::Fails};
	auto holds = true;
	for (auto const first : verdicts) {
		for (auto const second : verdicts) {
			auto const whole = ubench::joined({part(first), part(second)});
			holds = holds && whole.verdict == std::max(first, second) && whole.readings.size() == 2 && whole.notes.size() == 2;
		}
	}
	return holds;
}

inline constexpr auto digest_count = static_cast<std::size_t>(selfcheck::functions) * selfcheck::digest_blocks;

[[nodiscard]] constexpr auto oracle(selfcheck::Function function, std::uint32_t block) -> unsigned long long {
	return selfcheck::digest_term(std::to_underlying(function) * 4096U + block, block ^ 0x5a5aU);
}

/** Digests that differ from the host's fail their function, block by block; all-zero digests are no reading. */
[[nodiscard]] auto digests_count_every_differing_block(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto device = std::vector<unsigned long long>(digest_count);
	for (auto at = std::size_t{0}; at < digest_count; ++at) {
		device.at(at) = oracle(static_cast<selfcheck::Function>(at / selfcheck::digest_blocks), static_cast<std::uint32_t>(at % selfcheck::digest_blocks));
	}
	auto const exact = ubench::evaluate_digests(Card::Zero, std::span<unsigned long long const, digest_count>{device}, oracle);
	auto holds = exact && exact->verdict == Verdict::Holds;
	auto flipped = std::array<std::size_t, selfcheck::functions>{};
	auto chosen = std::set<std::size_t>{};
	for (auto flip = 0; flip < 40; ++flip) {
		auto const at = std::uniform_int_distribution<std::size_t>{0, digest_count - 1}(random);
		if (chosen.insert(at).second) {
			device.at(at) ^= 1ULL << (at % 64);
			++flipped.at(at / selfcheck::digest_blocks);
		}
	}
	auto const differing = ubench::evaluate_digests(Card::One, std::span<unsigned long long const, digest_count>{device}, oracle);
	holds = holds && differing && differing->verdict == Verdict::Fails;
	for (auto function = std::size_t{0}; function < flipped.size() && differing; ++function) {
		holds = holds && differing->readings.at(function).value == static_cast<double>(flipped.at(function));
	}
	auto const zeros = std::vector<unsigned long long>(digest_count);
	return holds && !ubench::evaluate_digests(Card::Zero, std::span<unsigned long long const, digest_count>{zeros}, oracle).has_value();
}

/** The hop is half a round trip at the calibrated clock; two blocks on one SM are no reading; a wrong echo fails. */
[[nodiscard]] auto the_hop_is_half_a_round_trip() -> bool {
	auto const calibration = ubench::Calibration{.megahertz = {1000.0, 2000.0}};
	auto const reading = [](unsigned long long cycles, unsigned second_sm, unsigned mismatches) {
		return ub::sync::HopReading{.cycles = cycles, .first_sm = 3, .second_sm = second_sm, .first_mismatches = mismatches, .second_mismatches = 0};
	};
	constexpr auto round_trips = static_cast<unsigned long long>(ub::sync::hop_iterations);
	auto const hops = std::array{
	    ubench::HopReading{.card = Card::Zero, .wait = ub::Wait::Spin, .reading = reading(round_trips * 1000, 7, 0)},
	    ubench::HopReading{.card = Card::One, .wait = ub::Wait::Backoff, .reading = reading(round_trips * 4000, 9, 0)},
	};
	auto const outcome = ubench::evaluate_cross_sm_hop(hops, calibration);
	auto const same_sm = std::array{ubench::HopReading{.card = Card::Zero, .wait = ub::Wait::Spin, .reading = reading(round_trips, 3, 0)}};
	auto const wrong = std::array{ubench::HopReading{.card = Card::Zero, .wait = ub::Wait::Spin, .reading = reading(round_trips, 4, 2)}};
	auto const failed = ubench::evaluate_cross_sm_hop(wrong, calibration);
	return outcome && outcome->verdict == Verdict::Holds && near(outcome->readings.at(0).value, 500.0) && near(outcome->readings.at(1).value, 1000.0) &&
	       !ubench::evaluate_cross_sm_hop(same_sm, calibration).has_value() && failed && failed->verdict == Verdict::Fails;
}

/**
 * Survival is each segment's way from cold back to hot: k segments read back as hot and the rest as cold leave k
 * segments' worth; a segment whose cold read is no slower than its hot one is no reading.
 */
[[nodiscard]] auto survival_counts_hot_segments(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 100; ++trial) {
		auto cards = std::array<ubench::SegmentTimes, 2>{};
		auto expected = std::array<double, 2>{};
		for (auto card = std::size_t{0}; card < 2; ++card) {
			for (auto segment = std::size_t{0}; segment < ub::l2::segments; ++segment) {
				auto const hot = std::uniform_real_distribution<double>{1000.0, 2000.0}(random);
				auto const cold = hot + std::uniform_real_distribution<double>{500.0, 3000.0}(random);
				auto const share = std::uniform_real_distribution<double>{0.0, 1.0}(random);
				cards.at(card).hot.at(segment) = hot;
				cards.at(card).cold.at(segment) = cold;
				cards.at(card).after.at(segment) = cold - share * (cold - hot);
				expected.at(card) += share * 0.125;
			}
		}
		auto const limits = std::array{engine::device::L2Persistence{.max_persisting_bytes = std::uint64_t{4608} * 1024, .max_access_policy_window_bytes = std::uint64_t{128} << 20U},
		                               engine::device::L2Persistence{.max_persisting_bytes = 0, .max_access_policy_window_bytes = 0}};
		auto const outcome = ubench::evaluate_l2_persistence(cards, limits);
		// Per card: the two limits, then the surviving MiB.
		holds = holds && outcome && near(outcome->readings.at(0).value, 4.5) && near(outcome->readings.at(1).value, 128.0) &&
		        std::abs(outcome->readings.at(2).value - expected[0]) < 1e-9 && std::abs(outcome->readings.at(5).value - expected[1]) < 1e-9;
		cards.at(1).cold.at(5) = cards.at(1).hot.at(5);
		holds = holds && !ubench::evaluate_l2_persistence(cards, limits).has_value();
	}
	return holds;
}

/** A stall's exposed time is the run's excess over the unstalled ring, per stall; what it did not expose was hidden. */
[[nodiscard]] auto stalls_expose_their_excess() -> bool {
	constexpr auto stalls = static_cast<double>(ub::stream::tiles_per_block) / 16.0;
	auto const points = std::array{
	    ubench::StallPoint{.card = Card::Zero, .ring_kib = 64, .stall_us = 0, .milliseconds = 3.0},
	    ubench::StallPoint{.card = Card::Zero, .ring_kib = 64, .stall_us = 8, .milliseconds = 3.0 + stalls * 2.0 / 1000.0},
	    ubench::StallPoint{.card = Card::Zero, .ring_kib = 64, .stall_us = 4, .milliseconds = 2.9},
	};
	auto const outcome = ubench::evaluate_ring_stall(points);
	return outcome && outcome->readings.size() == 4 && near(outcome->readings[0].value, 2.0) && near(outcome->readings[1].value, 0.75) &&
	       near(outcome->readings[2].value, 0.0) && near(outcome->readings[3].value, 1.0);
}

/**
 * The tensor rates: an mma kernel at 32 clocks per mma per scheduler reads 512 flop/clk/SM, and a mixed kernel that
 * takes 8 more clocks per fragment than its 16 mmas reads 8 extra clocks.
 */
[[nodiscard]] auto tensor_rates_follow_the_cycle_counts() -> bool {
	constexpr auto iterations = static_cast<double>(ub::compute::iterations);
	auto const mma = iterations * 8.0 * 32.0;
	auto const mixed = iterations * (16.0 * 32.0 + 4.0 * 8.0);
	auto const readings = std::array{ubench::TensorCycles{.card = Card::Zero, .mma = mma, .mma_decode = mixed, .decode = iterations * 256.0 * 8.0 / 12.8,
	                                                     .widen = iterations * 256.0 * 8.0 / 64.0}};
	auto const outcome = ubench::evaluate_tensor_rate(readings);
	return near(outcome.readings.at(0).value, 512.0) && near(outcome.readings.at(2).value, 8.0) && near(outcome.readings.at(3).value, 12.8) &&
	       near(outcome.readings.at(4).value, 64.0);
}

/** The calibrated clock is the median block's cycles per microsecond; a block short of its calibration spin is no reading. */
[[nodiscard]] auto calibration_is_the_median_block() -> bool {
	auto blocks = std::vector<ub::Clocks>(ub::sms);
	for (auto index = std::size_t{0}; index < blocks.size(); ++index) {
		auto const nanoseconds = 100'000'000ULL + index;
		blocks.at(index) = {.cycles = ub::sync::calibration_cycles + index * 1000, .nanoseconds = nanoseconds};
	}
	auto rates = std::vector<double>{};
	for (auto const& block : blocks) {
		rates.push_back(static_cast<double>(block.cycles) * 1000.0 / static_cast<double>(block.nanoseconds));
	}
	std::ranges::sort(rates);
	auto const calibrated = ubench::calibrated_megahertz(blocks);
	blocks.at(17).cycles = ub::sync::calibration_cycles - 1;
	return calibrated && near(*calibrated, rates.at(rates.size() / 2)) && !ubench::calibrated_megahertz(blocks).has_value();
}

/** The folded decode reports its losses per exponent; a warp that compared nothing is no reading. */
[[nodiscard]] auto folded_losses_are_per_exponent() -> bool {
	auto reading = ub::compute::FoldedReading{};
	for (auto index = std::size_t{0}; index < ub::compute::exponent_count; ++index) {
		std::span{reading.checked}[index] = 512;
		std::span{reading.mismatched}[index] = static_cast<unsigned>(index * 3);
	}
	auto const outcome = ubench::evaluate_folded(Card::Zero, reading);
	auto holds = outcome && outcome->readings.size() == ub::compute::exponent_count && outcome->verdict == Verdict::Measured;
	for (auto index = std::size_t{0}; holds && index < ub::compute::exponent_count; ++index) {
		holds = outcome->readings.at(index).value == static_cast<double>(index * 3);
	}
	std::span{reading.checked}[4] = 0;
	return holds && !ubench::evaluate_folded(Card::Zero, reading).has_value();
}

/** A pair benchmark reads both rates and the engine form's speed-up over the comparison. */
[[nodiscard]] auto pairs_compare_the_engine_form() -> bool {
	auto const readings = std::array{ubench::PairCycles{.card = Card::One, .engine = 1000.0, .comparison = 4000.0}};
	auto const outcome = ubench::evaluate_pair(readings, ubench::ldmatrix_measure);
	return outcome.readings.size() == 3 && near(outcome.readings[0].value, ubench::ldmatrix_measure.operations / 1000.0) &&
	       near(outcome.readings[1].value, ubench::ldmatrix_measure.operations / 4000.0) && near(outcome.readings[2].value, 4.0);
}

/**
 * U6 against the model: results equal to the model's prediction hold, every flipped bit is counted as one differing
 * result, and all-zero results are no reading.
 */
[[nodiscard]] auto mma_probes_are_judged_bit_for_bit(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const all = ubench::mma_probes(seed);
	auto const probes = std::vector<selfcheck::MmaProbe>(all.begin(), all.begin() + 8);
	auto results = std::vector<engine::device::prims::Accumulator>(probes.size() * 32);
	for (auto probe = std::size_t{0}; probe < probes.size(); ++probe) {
		auto operands = engine::mma_model::WarpOperands{};
		for (auto lane = std::size_t{0}; lane < 32; ++lane) {
			std::ranges::copy(std::span{std::span{probes[probe].a}[lane].reg}, operands.a.at(lane).begin());
			std::ranges::copy(std::span{std::span{probes[probe].b}[lane].reg}, operands.b.at(lane).begin());
			std::ranges::copy(std::span{std::span{probes[probe].c}[lane].reg}, operands.c.at(lane).begin());
		}
		auto const model = engine::mma_model::warp_mma(operands);
		for (auto lane = std::size_t{0}; lane < 32; ++lane) {
			std::ranges::copy(model.at(lane), std::span{results.at(probe * 32 + lane).reg}.begin());
		}
	}
	auto const agreeing = ubench::evaluate_probe_mma(Card::Zero, probes, results);
	auto holds = agreeing && agreeing->verdict == Verdict::Holds && agreeing->readings.at(1).value == 0.0;
	auto flipped = std::set<std::size_t>{};
	for (auto flip = 0; flip < 20; ++flip) {
		auto const at = std::uniform_int_distribution<std::size_t>{0, results.size() * 4 - 1}(random);
		if (flipped.insert(at).second) {
			auto& value = std::span{results.at(at / 4).reg}[at % 4];
			value = std::bit_cast<float>(std::bit_cast<std::uint32_t>(value) ^ 1U);
		}
	}
	auto const differing = ubench::evaluate_probe_mma(Card::One, probes, results);
	holds = holds && differing && differing->verdict == Verdict::Fails && differing->readings.at(1).value == static_cast<double>(flipped.size());
	auto const zeros = std::vector<engine::device::prims::Accumulator>(probes.size() * 32);
	return holds && !ubench::evaluate_probe_mma(Card::Zero, probes, zeros).has_value();
}

/** The probe set is the same for a seed, and its operands are finite: no infinity or nan reaches the model. */
[[nodiscard]] auto mma_probes_are_fixed_and_finite() -> bool {
	auto const first = ubench::mma_probes(7);
	auto const second = ubench::mma_probes(7);
	auto const finite_bf16 = [](std::uint32_t half) {
		return (half & 0x7f80U) != 0x7f80U;
	};
	auto finite = true;
	for (auto const& probe : first) {
		for (auto const& fragment : std::span{probe.a}) {
			for (auto const reg : std::span{fragment.reg}) {
				finite = finite && finite_bf16(reg & 0xffffU) && finite_bf16(reg >> 16U);
			}
		}
		for (auto const& accumulator : std::span{probe.c}) {
			for (auto const value : std::span{accumulator.reg}) {
				finite = finite && std::isfinite(value);
			}
		}
	}
	return finite && first.size() == selfcheck::mma_probes &&
	       std::ranges::equal(std::as_bytes(std::span{first}), std::as_bytes(std::span{second}));
}

/** A chain probe's latency is the median block's cycles over its token loop per token; an empty loop is no reading. */
[[nodiscard]] auto chain_latency_is_the_median_block() -> bool {
	auto counts = std::vector<unsigned long long>{};
	for (auto block = 0ULL; block < 9; ++block) {
		counts.push_back(1000 * block);
		counts.push_back(1000 * block + 1024 * (300 + block));
	}
	auto const latency = ubench::cycles_per_token(counts, 9, "probe");
	counts.at(9) = counts.at(8);
	return latency && near(*latency, 304.0) && !ubench::cycles_per_token(counts, 9, "probe").has_value();
}

/** Link versions are shown raw and only noted, never asserted; the start's proof is the verdict. */
[[nodiscard]] auto link_versions_are_noted_not_asserted() -> bool {
	using engine::device::NvlinkVersion;
	auto const link = [](NvlinkVersion version) {
		return engine::device::NvlinkLinkReading{.version = version, .megabytes_per_second = 14062};
	};
	auto const readings = std::array{
	    ubench::LinkReading{.card = Card::Zero, .links = {link(NvlinkVersion::V3_0), link(NvlinkVersion::V3_0), link(NvlinkVersion::V3_0), link(NvlinkVersion::V3_0)},
	                        .peer = {.performance_rank = 0, .native_atomics = false}},
	    ubench::LinkReading{.card = Card::One, .links = {link(NvlinkVersion::V2_2), link(NvlinkVersion::V3_0), link(NvlinkVersion::V3_0), link(NvlinkVersion::V3_0)},
	                        .peer = {.performance_rank = 0, .native_atomics = true}},
	};
	auto const outcome = ubench::evaluate_link_health(readings);
	auto const flagged = std::ranges::count_if(outcome.notes, [](std::string const& note) {
		return note.starts_with("card 1: link 0");
	});
	return outcome.verdict == Verdict::Holds && flagged == 1 && near(outcome.readings.at(1).value, 14.062) && outcome.readings.at(0).value == 4.0;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the read gate is each card's best point against its own clock's peak (seed 11)",
	                          .holds = the_read_gate_is_each_cards_best_against_its_peak(11)},
	    engine::testing::Case{.name = "the peak is 96 bytes per memory-clock cycle", .holds = the_peak_is_the_bus_at_the_clock()},
	    engine::testing::Case{.name = "a self-check report holds only without mismatches (seed 12)", .holds = a_report_holds_only_without_mismatches(12)},
	    engine::testing::Case{.name = "joining outcomes keeps the worst verdict", .holds = joining_keeps_the_worst_verdict()},
	    engine::testing::Case{.name = "digests count every differing block per function (seed 13)", .holds = digests_count_every_differing_block(13)},
	    engine::testing::Case{.name = "the cross-SM hop is half a round trip; one SM or a wrong echo refuses", .holds = the_hop_is_half_a_round_trip()},
	    engine::testing::Case{.name = "l2 survival is each segment's way from cold back to hot (seed 14)", .holds = survival_counts_hot_segments(14)},
	    engine::testing::Case{.name = "a stall exposes the run's excess over the unstalled ring", .holds = stalls_expose_their_excess()},
	    engine::testing::Case{.name = "tensor rates and the decode's extra clocks follow the cycle counts", .holds = tensor_rates_follow_the_cycle_counts()},
	    engine::testing::Case{.name = "the calibrated clock is the median block", .holds = calibration_is_the_median_block()},
	    engine::testing::Case{.name = "the folded decode's losses are read per exponent", .holds = folded_losses_are_per_exponent()},
	    engine::testing::Case{.name = "a pair benchmark compares the engine's form with the alternative", .holds = pairs_compare_the_engine_form()},
	    engine::testing::Case{.name = "mma probes are judged against the model bit for bit (seed 15)", .holds = mma_probes_are_judged_bit_for_bit(15)},
	    engine::testing::Case{.name = "the mma probe set is fixed per seed and finite", .holds = mma_probes_are_fixed_and_finite()},
	    engine::testing::Case{.name = "a chain's latency is the median block's cycles per token", .holds = chain_latency_is_the_median_block()},
	    engine::testing::Case{.name = "nvlink versions are noted for confirmation, never asserted", .holds = link_versions_are_noted_not_asserted()},
	};
	return engine::testing::report(cases);
}

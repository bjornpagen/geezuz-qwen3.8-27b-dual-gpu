// SPDX-License-Identifier: AGPL-3.0-only
//
// The primitives' self-checks on the hardware, and the decode question beside them: each check kernel compares a
// primitive against its oracle on the device and reports what disagreed, or (the transcendentals) folds what it
// computed into digests the host recomputes. Any mismatch fails; a report that checked nothing is no reading.
export module engine.ubench:checks;

import std;
import engine.core;
import engine.device;
import engine.mma_model;
import engine.model;
import engine.prims;
import engine.ubench_device;
import :outcome;
import :protocol;

namespace engine::ubench {

namespace ub = engine::device::ubench;
namespace kernels = engine::ubench_device;
namespace selfcheck = engine::device::prims::selfcheck;

using model::Card;

/** How many kept mismatches a failed report shows the operator. */
inline constexpr std::size_t shown_mismatches = 8;

/** A self-check report's verdict on one card: its counts, and its first mismatches when it found any. */
export [[nodiscard]] auto evaluate_report(model::Card card, selfcheck::Report const& report, std::string_view kernel) -> Result<Outcome> {
	if (report.checked == 0) {
		return no_reading(kernel, "the self-check compared nothing");
	}
	auto outcome = Outcome{
	    .readings =
	        {
	            {.label = std::format("{}, {} comparisons", on(card), kernel), .value = static_cast<double>(report.checked), .unit = Unit::Count},
	            {.label = std::format("{}, {} mismatches", on(card), kernel), .value = static_cast<double>(report.mismatched), .unit = Unit::Count},
	        },
	    .verdict = report.mismatched == 0 ? Verdict::Holds : Verdict::Fails,
	    .notes = {},
	};
	auto const kept = std::span{report.first}.first(std::min<std::size_t>({report.mismatched, selfcheck::kept_mismatches, shown_mismatches}));
	for (auto const& mismatch : kept) {
		outcome.notes.push_back(std::format("{}: input {:#010x} (detail {}) gave {:#010x}, expected {:#010x}", on(card), mismatch.input, mismatch.detail,
		                                    mismatch.got, mismatch.expected));
	}
	return outcome;
}

/** Joins per-card outcomes: every reading and note, and the worst verdict. */
export [[nodiscard]] auto joined(std::vector<Outcome> parts) -> Outcome {
	auto whole = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto& part : parts) {
		std::ranges::move(part.readings, std::back_inserter(whole.readings));
		std::ranges::move(part.notes, std::back_inserter(whole.notes));
		whole.verdict = std::max(whole.verdict, part.verdict);
	}
	return whole;
}

static_assert(Verdict::Measured < Verdict::Holds && Verdict::Holds < Verdict::Fails, "joining keeps the worst verdict");

/** Runs one reporting self-check on one card and evaluates its report. */
template<class K>
[[nodiscard]] auto reporting_check(CardView view) -> Result<Outcome> {
	auto const program = lift(view.context.load<K>());
	auto report = Readback<selfcheck::Report, 1>::create(view.context);
	if (auto const failed = first_failure(program, report)) {
		return std::unexpected(*failed);
	}
	return report->clear(view)
	    .and_then([&] {
		    return run_once<K>(view, *program, report->device());
	    })
	    .and_then([&] {
		    return report->fetch(view);
	    })
	    .and_then([&](std::span<selfcheck::Report const, 1> read) {
		    return evaluate_report(view.card, read.front(), K::entry);
	    });
}

template<class K>
[[nodiscard]] auto reporting_check(BenchView const& bench) -> Result<Outcome> {
	auto parts = std::vector<Outcome>{};
	auto const ran = for_each_card([&](model::Card card) {
		return reporting_check<K>(bench.card(card)).transform([&](Outcome part) {
			parts.push_back(std::move(part));
		});
	});
	return ran.transform([&] {
		return joined(std::move(parts));
	});
}

export [[nodiscard]] auto check_fragments(BenchView const& bench) -> Result<Outcome> {
	return reporting_check<kernels::CheckFragmentsKernel>(bench);
}

export [[nodiscard]] auto check_reduce(BenchView const& bench) -> Result<Outcome> {
	return reporting_check<kernels::CheckReduceKernel>(bench);
}

/**
 * The folded decode on one card: per activation exponent, how many of the 256 codes' results lost bits. The fold is
 * what the engine refuses; this shows where it would have lost them.
 */
export [[nodiscard]] auto evaluate_folded(model::Card card, ub::compute::FoldedReading const& reading) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	auto const exponents = std::span{ub::compute::activation_exponents};
	auto const checked = std::span{reading.checked};
	auto const mismatched = std::span{reading.mismatched};
	for (auto index = std::size_t{0}; index < exponents.size(); ++index) {
		if (checked[index] == 0) {
			return no_reading(kernels::DecodeFoldedKernel::entry, "an exponent's warp compared nothing");
		}
		outcome.readings.push_back({.label = std::format("{}, folded scale, activations 2^{}: results that lost bits", on(card), exponents[index]),
		                            .value = static_cast<double>(mismatched[index]),
		                            .unit = Unit::Count});
	}
	return outcome;
}

/** The exact decode's self-check (the gate) and the folded variant's losses beside it, on every card. */
export [[nodiscard]] auto decode_exact(BenchView const& bench) -> Result<Outcome> {
	auto parts = std::vector<Outcome>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const exact = reporting_check<kernels::CheckDecodeKernel>(view);
		auto const program = lift(view.context.load<kernels::DecodeFoldedKernel>());
		auto reading = Readback<ub::compute::FoldedReading, 1>::create(view.context);
		if (auto const failed = first_failure(exact, program, reading)) {
			return std::unexpected(*failed);
		}
		auto const folded = run_once<kernels::DecodeFoldedKernel>(view, *program, reading->device())
		                        .and_then([&] {
			                        return reading->fetch(view);
		                        })
		                        .and_then([&](std::span<ub::compute::FoldedReading const, 1> read) {
			                        return evaluate_folded(card, read.front());
		                        });
		return folded.transform([&](Outcome measured) {
			parts.push_back(*exact);
			parts.push_back(std::move(measured));
		});
	});
	return ran.transform([&] {
		auto whole = joined(std::move(parts));
		whole.notes.emplace_back("the gate is the exact path; folded losses at small exponents confirm the no-fold rule");
		return whole;
	});
}

inline constexpr auto digest_functions = std::to_array<selfcheck::Function>({selfcheck::Function::Exp, selfcheck::Function::Log, selfcheck::Function::Rsqrt,
                                                                             selfcheck::Function::Sigmoid, selfcheck::Function::Silu,
                                                                             selfcheck::Function::Softplus});
static_assert(digest_functions.size() == selfcheck::functions);

inline constexpr auto digests = static_cast<std::size_t>(selfcheck::functions) * selfcheck::digest_blocks;

/**
 * The device's digests against `expected(function, block)`: a device whose digests are all zero did not run (a sum of
 * four million mixed terms is zero by chance with probability 2^-64); any differing block fails its function.
 */
export template<class Expected>
[[nodiscard]] auto evaluate_digests(model::Card card, std::span<unsigned long long const, digests> device, Expected expected) -> Result<Outcome> {
	if (std::ranges::all_of(device, [](unsigned long long digest) {
		    return digest == 0;
	    })) {
		return no_reading(kernels::DigestKernel::entry, "every digest is zero");
	}
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Holds, .notes = {}};
	for (auto const function : digest_functions) {
		auto differing = std::size_t{0};
		for (auto block = std::uint32_t{0}; block < selfcheck::digest_blocks; ++block) {
			auto const at = static_cast<std::size_t>(std::to_underlying(function)) * selfcheck::digest_blocks + block;
			differing += device[at] == expected(function, block) ? 0U : 1U;
		}
		outcome.readings.push_back({.label = std::format("{}, function {}: blocks whose digest differs from the host's", on(card), std::to_underlying(function)),
		                            .value = static_cast<double>(differing),
		                            .unit = Unit::Count});
		if (differing != 0) {
			outcome.verdict = Verdict::Fails;
		}
	}
	return outcome;
}

/** The host's digest of every (function, block), in the device's order: minutes of work, done once per run. */
[[nodiscard]] auto host_digests() -> std::vector<unsigned long long> {
	auto expected = std::vector<unsigned long long>(digests);
	for (auto const function : digest_functions) {
		for (auto block = std::uint32_t{0}; block < selfcheck::digest_blocks; ++block) {
			expected.at(static_cast<std::size_t>(std::to_underlying(function)) * selfcheck::digest_blocks + block) = engine::prims::digest(function, block);
		}
	}
	return expected;
}

export [[nodiscard]] auto check_transcendentals(BenchView const& bench) -> Result<Outcome> {
	auto device_digests = std::array<std::vector<unsigned long long>, model::cards>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<kernels::DigestKernel>());
		auto readback = Readback<unsigned long long, digests>::create(view.context);
		if (auto const failed = first_failure(program, readback)) {
			return std::unexpected(*failed);
		}
		return readback->clear(view)
		    .and_then([&] {
			    return run_once<kernels::DigestKernel>(view, *program, readback->device());
		    })
		    .and_then([&] {
			    return readback->fetch(view);
		    })
		    .and_then([&](std::span<unsigned long long const, digests> read) -> Result<void> {
			    // A card that did not run is known before the host spends minutes on its side.
			    if (std::ranges::all_of(read, [](unsigned long long digest) {
				        return digest == 0;
			        })) {
				    return no_reading(kernels::DigestKernel::entry, "every digest is zero");
			    }
			    device_digests.at(std::to_underlying(card)).assign(read.begin(), read.end());
			    return {};
		    });
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	auto const expected = host_digests();
	auto parts = std::vector<Outcome>{};
	for (auto const card : device::card_pair) {
		auto const evaluated = evaluate_digests(card, std::span<unsigned long long const, digests>{device_digests.at(std::to_underlying(card))},
		                                        [&](selfcheck::Function function, std::uint32_t block) {
			                                        return expected.at(static_cast<std::size_t>(std::to_underlying(function)) * selfcheck::digest_blocks + block);
		                                        });
		if (!evaluated) {
			return std::unexpected(evaluated.error());
		}
		parts.push_back(*evaluated);
	}
	return joined(std::move(parts));
}

/** The probe set's seed: the same probes every run, so two runs compare. */
inline constexpr std::uint64_t probe_seed = 0x6d6d615f7536ULL;

/**
 * A finite bf16 with a random sign and significand whose exponent is drawn from [-30, 30] around 2^0, a subnormal one
 * time in sixteen and a zero one time in thirty-two: the inputs that exercise the alignment, the truncation and the
 * carries.
 */
[[nodiscard]] auto probe_bf16(std::mt19937_64& random) -> std::uint32_t {
	auto const bits = static_cast<std::uint32_t>(random());
	auto const sign = bits & 0x8000U;
	auto const mantissa = (bits >> 16U) & 0x7fU;
	auto const kind = (bits >> 24U) & 31U;
	if (kind == 0) {
		return sign;
	}
	if (kind < 3) {
		return sign | (mantissa == 0 ? 1U : mantissa);
	}
	auto const exponent = 127U - 30U + (bits >> 8U) % 61U;
	return sign | (exponent << 7U) | mantissa;
}

/** A finite fp32 addend of either sign within 2^±40, or zero one time in eight. */
[[nodiscard]] auto probe_float(std::mt19937_64& random) -> float {
	auto const bits = static_cast<std::uint32_t>(random());
	if ((bits & 7U) == 0) {
		return 0.0F;
	}
	auto const exponent = 127U - 40U + (bits >> 3U) % 81U;
	return std::bit_cast<float>((bits & 0x807fffffU) | (exponent << 23U));
}

/** The probes U6 runs: one per warp, every operand register random as `probe_bf16` and `probe_float` draw them. */
export [[nodiscard]] auto mma_probes(std::uint64_t seed) -> std::vector<selfcheck::MmaProbe> {
	auto random = std::mt19937_64{seed};
	auto probes = std::vector<selfcheck::MmaProbe>(selfcheck::mma_probes);
	for (auto& probe : probes) {
		for (auto lane = std::size_t{0}; lane < 32; ++lane) {
			for (auto& reg : std::span{std::span{probe.a}[lane].reg}) {
				reg = probe_bf16(random) | (probe_bf16(random) << 16U);
			}
			for (auto& reg : std::span{std::span{probe.b}[lane].reg}) {
				reg = probe_bf16(random) | (probe_bf16(random) << 16U);
			}
			for (auto& reg : std::span{std::span{probe.c}[lane].reg}) {
				reg = probe_float(random);
			}
		}
	}
	return probes;
}

/** What the bit-exact model says one probe's warp returns. */
[[nodiscard]] auto predicted(selfcheck::MmaProbe const& probe) -> engine::mma_model::WarpResult {
	auto operands = engine::mma_model::WarpOperands{};
	for (auto lane = std::size_t{0}; lane < 32; ++lane) {
		std::ranges::copy(std::span{std::span{probe.a}[lane].reg}, operands.a.at(lane).begin());
		std::ranges::copy(std::span{std::span{probe.b}[lane].reg}, operands.b.at(lane).begin());
		std::ranges::copy(std::span{std::span{probe.c}[lane].reg}, operands.c.at(lane).begin());
	}
	return engine::mma_model::warp_mma(operands);
}

/**
 * U6: every lane's D of every probe against the model, bit for bit. Any difference fails the model (and so the
 * reference's noise-floor sizing); the first few are shown as (probe, lane, register).
 */
export [[nodiscard]] auto evaluate_probe_mma(Card card, std::span<selfcheck::MmaProbe const> probes, std::span<engine::device::prims::Accumulator const> results)
    -> Result<Outcome> {
	core::invariant(results.size() == probes.size() * 32);
	if (std::ranges::all_of(results, [](engine::device::prims::Accumulator const& d) {
		    return std::ranges::all_of(std::span{d.reg}, [](float value) {
			    return std::bit_cast<std::uint32_t>(value) == 0;
		    });
	    })) {
		return no_reading(kernels::ProbeMmaKernel::entry, "every result is +0");
	}
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Holds, .notes = {}};
	auto differing = std::size_t{0};
	for (auto probe = std::size_t{0}; probe < probes.size(); ++probe) {
		auto const model = predicted(probes[probe]);
		for (auto lane = std::size_t{0}; lane < 32; ++lane) {
			auto const& got = results[probe * 32 + lane];
			for (auto reg = std::size_t{0}; reg < 4; ++reg) {
				auto const hardware = std::bit_cast<std::uint32_t>(std::span{got.reg}[reg]);
				auto const expected = std::bit_cast<std::uint32_t>(model.at(lane).at(reg));
				if (hardware == expected) {
					continue;
				}
				if (differing < shown_mismatches) {
					outcome.notes.push_back(std::format("{}: probe {} lane {} register {} gave {:#010x}, the model {:#010x}", on(card), probe, lane, reg, hardware, expected));
				}
				++differing;
			}
		}
	}
	outcome.readings.push_back({.label = std::format("{}, mma results compared with the model", on(card)), .value = static_cast<double>(results.size() * 4), .unit = Unit::Count});
	outcome.readings.push_back({.label = std::format("{}, results differing from the model", on(card)), .value = static_cast<double>(differing), .unit = Unit::Count});
	if (differing != 0) {
		outcome.verdict = Verdict::Fails;
	}
	return outcome;
}

export [[nodiscard]] auto probe_mma(BenchView const& bench) -> Result<Outcome> {
	auto const probes = mma_probes(probe_seed);
	auto parts = std::vector<Outcome>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<kernels::ProbeMmaKernel>());
		auto staged = Readback<selfcheck::MmaProbe, selfcheck::mma_probes>::create(view.context);
		auto results = Readback<engine::device::prims::Accumulator, static_cast<std::size_t>(selfcheck::mma_probes) * 32>::create(view.context);
		if (auto const failed = first_failure(program, staged, results)) {
			return std::unexpected(*failed);
		}
		return staged->put(view, probes)
		    .and_then([&] {
			    return run_once<kernels::ProbeMmaKernel>(view, *program, staged->device(), results->device());
		    })
		    .and_then([&] {
			    return results->fetch(view);
		    })
		    .and_then([&](auto const& read) {
			    return evaluate_probe_mma(card, probes, read);
		    })
		    .transform([&](Outcome part) {
			    parts.push_back(std::move(part));
		    });
	});
	return ran.transform([&] {
		return joined(std::move(parts));
	});
}

/** The ring check's source: word i of the stream is `ring_word(i)`. */
[[nodiscard]] auto ring_source() -> std::vector<engine::device::prims::Chunk> {
	auto source = std::vector<engine::device::prims::Chunk>(selfcheck::stage_ring_source_chunks);
	for (auto chunk = std::size_t{0}; chunk < source.size(); ++chunk) {
		for (auto word = std::size_t{0}; word < 4; ++word) {
			std::span{source[chunk].word}[word] = selfcheck::ring_word(static_cast<unsigned>(chunk * 4 + word));
		}
	}
	return source;
}

/** The ring check's report, and the median block's consumer time per 4 KiB tile beside it. */
export [[nodiscard]] auto evaluate_stage_ring(Card card, selfcheck::Report const& report, std::span<unsigned long long const> elapsed, Calibration calibration)
    -> Result<Outcome> {
	return evaluate_report(card, report, kernels::StageRingCheckKernel::entry).and_then([&](Outcome outcome) -> Result<Outcome> {
		auto cycles = std::vector<double>(elapsed.begin(), elapsed.end());
		if (std::ranges::any_of(cycles, [](double counted) {
			    return !(counted > 0.0);
		    })) {
			return no_reading(kernels::StageRingCheckKernel::entry, "a block's consumers timed nothing");
		}
		auto const per_tile = median(std::move(cycles)) / static_cast<double>(selfcheck::stage_ring_tiles_per_block);
		outcome.readings.push_back({.label = std::format("{}, one checked 4 KiB tile through the ring, median block", on(card)),
		                            .value = calibration.nanoseconds(card, per_tile),
		                            .unit = Unit::Nanoseconds});
		return outcome;
	});
}

export [[nodiscard]] auto check_stage_ring(BenchView const& bench) -> Result<Outcome> {
	auto const source = ring_source();
	auto parts = std::vector<Outcome>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<kernels::StageRingCheckKernel>());
		auto words = Readback<engine::device::prims::Chunk, selfcheck::stage_ring_source_chunks>::create(view.context);
		auto report = Readback<selfcheck::Report, 1>::create(view.context);
		auto elapsed = Readback<unsigned long long, selfcheck::stage_ring_blocks>::create(view.context);
		if (auto const failed = first_failure(program, words, report, elapsed)) {
			return std::unexpected(*failed);
		}
		auto const checked = words->put(view, source)
		                         .and_then([&] {
			                         return report->clear(view);
		                         })
		                         .and_then([&] {
			                         return run_once<kernels::StageRingCheckKernel>(view, *program, words->device(), report->device(), elapsed->device());
		                         })
		                         .and_then([&] {
			                         return report->fetch(view);
		                         });
		auto const timed = checked.and_then([&](auto const&) {
			return elapsed->fetch(view);
		});
		if (!timed) {
			return std::unexpected(timed.error());
		}
		return evaluate_stage_ring(card, checked->front(), *timed, bench.calibration).transform([&](Outcome part) {
			parts.push_back(std::move(part));
		});
	});
	return ran.transform([&] {
		return joined(std::move(parts));
	});
}

export [[nodiscard]] auto check_order(BenchView const& bench) -> Result<Outcome> {
	auto parts = std::vector<Outcome>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<kernels::OrderCheckKernel>());
		auto scratch_memory = Readback<selfcheck::OrderScratch, 1>::create(view.context);
		auto report = Readback<selfcheck::Report, 1>::create(view.context);
		if (auto const failed = first_failure(program, scratch_memory, report)) {
			return std::unexpected(*failed);
		}
		return scratch_memory->clear(view)
		    .and_then([&] {
			    return report->clear(view);
		    })
		    .and_then([&] {
			    return run_once<kernels::OrderCheckKernel>(view, *program, scratch_memory->device(), report->device());
		    })
		    .and_then([&] {
			    return report->fetch(view);
		    })
		    .and_then([&](std::span<selfcheck::Report const, 1> read) {
			    return evaluate_report(card, read.front(), kernels::OrderCheckKernel::entry);
		    })
		    .transform([&](Outcome part) {
			    parts.push_back(std::move(part));
		    });
	});
	return ran.transform([&] {
		return joined(std::move(parts));
	});
}

}

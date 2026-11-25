// SPDX-License-Identifier: AGPL-3.0-only
//
// The primitives' self-checks on the hardware, and the decode question beside them: each check kernel compares a
// primitive against its oracle on the device and reports what disagreed, or (the transcendentals) folds what it
// computed into digests the host recomputes. Any mismatch fails; a report that checked nothing is no reading.
export module engine.ubench:checks;

import std;
import engine.core;
import engine.device;
import engine.model;
import engine.prims;
import engine.ubench_device;
import :outcome;
import :protocol;

namespace engine::ubench {

namespace ub = engine::device::ubench;
namespace kernels = engine::ubench_device;
namespace selfcheck = engine::device::prims::selfcheck;

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

}

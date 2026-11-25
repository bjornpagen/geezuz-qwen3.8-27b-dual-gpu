// SPDX-License-Identifier: AGPL-3.0-only
//
// The arithmetic benchmarks: the tensor core beside the weight decode, the shared transcendentals, ldmatrix from
// swizzled and plain tiles, register double buffering and the warp reductions. Every rate kernel runs a fixed amount
// of work per block, one block per SM, so a rate per SM is that work over the block's cycles.
export module engine.ubench:compute;

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
namespace shapes = engine::device::ubench::compute;
using engine::device::prims::selfcheck::Function;

/** The flop of one `mma.sync.m16n8k16`. */
inline constexpr double mma_flop = 16.0 * 8.0 * 16.0 * 2.0;
inline constexpr double block_threads = shapes::rate_budget.threads;
inline constexpr double block_warps = shapes::warps;
inline constexpr double iterations = shapes::iterations;

/** The mean over blocks of one launch's cycles; a block that counted none did not run. */
export [[nodiscard]] auto block_cycles(std::span<ub::Rate const> blocks, std::string_view kernel) -> Result<double> {
	auto total = 0.0;
	for (auto const& block : blocks) {
		if (block.cycles == 0) {
			return no_reading(kernel, "a block counted no cycles");
		}
		total += static_cast<double>(block.cycles);
	}
	return total / static_cast<double>(blocks.size());
}

/** The least per-launch mean block cycles of rate kernel `K` over the protocol's timed launches. */
template<class K, class... Ks>
[[nodiscard]] auto rate_cycles(CardView view, device::Program<Ks...> const& program, Readback<ub::Rate, ub::sms>& rates) -> Result<double> {
	auto least = std::numeric_limits<double>::infinity();
	for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
		auto const cycles = run_once<K>(view, program, rates.device())
		                        .and_then([&] {
			                        return rates.fetch(view);
		                        })
		                        .and_then([](std::span<ub::Rate const, ub::sms> blocks) {
			                        return block_cycles(blocks, K::entry);
		                        });
		if (!cycles) {
			return std::unexpected(cycles.error());
		}
		if (launch >= warmup_launches) {
			least = std::min(least, *cycles);
		}
	}
	return least;
}

/** Runs every kernel of `Ks` on every card and collects its least mean block cycles, in order. */
template<class... Ks>
[[nodiscard]] auto rate_sweep(BenchView const& bench) -> Result<std::array<std::array<double, sizeof...(Ks)>, model::cards>> {
	auto cycles = std::array<std::array<double, sizeof...(Ks)>, model::cards>{};
	auto const ran = for_each_card([&](model::Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = lift(view.context.load<Ks...>());
		auto rates = Readback<ub::Rate, ub::sms>::create(view.context);
		if (auto const failed = first_failure(program, rates)) {
			return std::unexpected(*failed);
		}
		auto result = Result<void>{};
		auto index = std::size_t{0};
		auto const one = [&]<class K> {
			result = rate_cycles<K>(view, *program, *rates).transform([&](double counted) {
				cycles.at(std::to_underlying(card)).at(index++) = counted;
			});
			return result.has_value();
		};
		std::ignore = (one.template operator()<Ks>() && ...);
		return result;
	});
	return ran.transform([&] {
		return cycles;
	});
}

/** One card's least mean block cycles for the four tensor mixes, in the enum's order. */
export struct TensorCycles {
	model::Card card;
	double mma;
	double mma_decode;
	double decode;
	double widen;
};

/**
 * The rates per SM, and the decode's cost in tensor clocks: per scheduler and iteration the mixed kernel runs 16 mmas
 * and decodes 4 fragments of 256 bytes (2 warps, 2 k16 steps each); whatever it takes beyond 16 mmas at the pure
 * kernel's rate is the decode's, per fragment.
 */
export [[nodiscard]] auto evaluate_tensor_rate(std::span<TensorCycles const> readings) -> Outcome {
	constexpr auto mmas_per_scheduler = 8.0;
	constexpr auto mixed_mmas_per_scheduler = 16.0;
	constexpr auto fragments_per_scheduler = 4.0;
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		auto const card = reading.card;
		outcome.readings.push_back({.label = std::format("{}, mma.sync m16n8k16 bf16 -> fp32 alone", on(card)),
		                            .value = block_warps * iterations * 4.0 * mma_flop / reading.mma,
		                            .unit = Unit::FlopPerClockPerSm});
		outcome.readings.push_back({.label = std::format("{}, mma beside the decode at the 32-row ratio", on(card)),
		                            .value = block_warps * iterations * 8.0 * mma_flop / reading.mma_decode,
		                            .unit = Unit::FlopPerClockPerSm});
		auto const clocks_per_mma = reading.mma / (iterations * mmas_per_scheduler);
		auto const extra = (reading.mma_decode / iterations - mixed_mmas_per_scheduler * clocks_per_mma) / fragments_per_scheduler;
		outcome.readings.push_back({.label = std::format("{}, tensor clocks the decode adds per 256-byte fragment", on(card)), .value = extra, .unit = Unit::Cycles});
		outcome.readings.push_back({.label = std::format("{}, decode alone, 4-code words", on(card)),
		                            .value = block_threads * iterations * 8.0 / reading.decode,
		                            .unit = Unit::PerClockPerSm});
		outcome.readings.push_back({.label = std::format("{}, prmt widening alone", on(card)),
		                            .value = block_threads * iterations * 8.0 / reading.widen,
		                            .unit = Unit::PerClockPerSm});
	}
	outcome.notes.emplace_back("0 extra clocks: the decode's fma.bf16x2 issue off the tensor pipe; about 8: they share it, and R = 32's margin shrinks");
	return outcome;
}

export [[nodiscard]] auto tensor_rate(BenchView const& bench) -> Result<Outcome> {
	using shapes::Mix;
	return rate_sweep<kernels::TensorKernel<Mix::Mma>, kernels::TensorKernel<Mix::MmaDecode>, kernels::TensorKernel<Mix::Decode>,
	                  kernels::TensorKernel<Mix::Widen>>(bench)
	    .transform([](auto const& cycles) {
		    auto readings = std::vector<TensorCycles>{};
		    for (auto const card : device::card_pair) {
			    auto const& mine = cycles.at(std::to_underlying(card));
			    readings.push_back({.card = card, .mma = mine[0], .mma_decode = mine[1], .decode = mine[2], .widen = mine[3]});
		    }
		    return evaluate_tensor_rate(readings);
	    });
}

inline constexpr auto function_names = std::to_array<std::string_view>({"exp", "log", "rsqrt", "sigmoid", "silu", "softplus"});
static_assert(function_names.size() == device::prims::selfcheck::functions);

/** One card's least mean block cycles per transcendental, in the digest's function order. */
export struct TranscendentalCycles {
	model::Card card;
	std::array<double, device::prims::selfcheck::functions> cycles;
};

export [[nodiscard]] auto evaluate_transcendental_rate(std::span<TranscendentalCycles const> readings) -> Outcome {
	constexpr auto evaluations = block_threads * iterations * 4.0;
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		for (auto function = std::size_t{0}; function < function_names.size(); ++function) {
			outcome.readings.push_back({.label = std::format("{}, {} evaluations", on(reading.card), function_names.at(function)),
			                            .value = evaluations / reading.cycles.at(function),
			                            .unit = Unit::PerClockPerSm});
		}
	}
	outcome.notes.emplace_back("mufu's documented 16/clk/SM is the comparison; the gate keeps .approx out of every image");
	return outcome;
}

export [[nodiscard]] auto transcendental_rate(BenchView const& bench) -> Result<Outcome> {
	return rate_sweep<kernels::TranscendentalKernel<Function::Exp>, kernels::TranscendentalKernel<Function::Log>,
	                  kernels::TranscendentalKernel<Function::Rsqrt>, kernels::TranscendentalKernel<Function::Sigmoid>,
	                  kernels::TranscendentalKernel<Function::Silu>, kernels::TranscendentalKernel<Function::Softplus>>(bench)
	    .transform([](auto const& cycles) {
		    auto readings = std::vector<TranscendentalCycles>{};
		    for (auto const card : device::card_pair) {
			    readings.push_back({.card = card, .cycles = cycles.at(std::to_underlying(card))});
		    }
		    return evaluate_transcendental_rate(readings);
	    });
}

/** One card's least mean block cycles for a pair of variants: the engine's form first, the comparison second. */
export struct PairCycles {
	model::Card card;
	double engine;
	double comparison;
};

/** What a pair benchmark counts per block, and how it names its two variants. */
export struct PairMeasure {
	double operations;
	Unit unit;
	std::string_view engine;
	std::string_view comparison;
};

export [[nodiscard]] auto evaluate_pair(std::span<PairCycles const> readings, PairMeasure measure) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		outcome.readings.push_back({.label = std::format("{}, {}", on(reading.card), measure.engine), .value = measure.operations / reading.engine, .unit = measure.unit});
		outcome.readings.push_back({.label = std::format("{}, {}", on(reading.card), measure.comparison), .value = measure.operations / reading.comparison, .unit = measure.unit});
		outcome.readings.push_back({.label = std::format("{}, {} over {}", on(reading.card), measure.engine, measure.comparison), .value = reading.comparison / reading.engine, .unit = Unit::Ratio});
	}
	return outcome;
}

template<class Engine, class Comparison>
[[nodiscard]] auto pair(BenchView const& bench, PairMeasure measure) -> Result<Outcome> {
	return rate_sweep<Engine, Comparison>(bench).transform([&](auto const& cycles) {
		auto readings = std::vector<PairCycles>{};
		for (auto const card : device::card_pair) {
			auto const& mine = cycles.at(std::to_underlying(card));
			readings.push_back({.card = card, .engine = mine[0], .comparison = mine[1]});
		}
		return evaluate_pair(readings, measure);
	});
}

export inline constexpr PairMeasure ldmatrix_measure = {
    .operations = block_warps * iterations, .unit = Unit::PerClockPerSm, .engine = "ldmatrix.x4, xor-swizzled tile", .comparison = "ldmatrix.x4, row-major tile"};

export inline constexpr PairMeasure double_buffer_measure = {.operations = block_warps * iterations * 2.0 * mma_flop,
                                                              .unit = Unit::FlopPerClockPerSm,
                                                              .engine = "ldmatrix.trans + mma, double-buffered",
                                                              .comparison = "ldmatrix.trans + mma, sequential"};

export inline constexpr PairMeasure reduction_measure = {.operations = block_warps * iterations * 4.0,
                                                          .unit = Unit::PerClockPerSm,
                                                          .engine = "fixed-order float butterfly warp sums",
                                                          .comparison = "integer redux.sync warp sums"};

static_assert(shapes::iterations % shapes::pipeline_steps == 0, "the double-buffer kernels run whole rounds of steps");

export [[nodiscard]] auto ldmatrix_swizzle(BenchView const& bench) -> Result<Outcome> {
	return pair<kernels::LdmatrixKernel<shapes::Swizzle::Xor>, kernels::LdmatrixKernel<shapes::Swizzle::None>>(bench, ldmatrix_measure);
}

export [[nodiscard]] auto mma_double_buffer(BenchView const& bench) -> Result<Outcome> {
	return pair<kernels::PipelineKernel<shapes::Pipelining::Double>, kernels::PipelineKernel<shapes::Pipelining::Sequential>>(bench, double_buffer_measure);
}

export [[nodiscard]] auto warp_reductions(BenchView const& bench) -> Result<Outcome> {
	return pair<kernels::ReductionKernel<shapes::Reduction::FloatButterfly>, kernels::ReductionKernel<shapes::Reduction::IntegerRedux>>(bench,
	                                                                                                                              reduction_measure);
}

}

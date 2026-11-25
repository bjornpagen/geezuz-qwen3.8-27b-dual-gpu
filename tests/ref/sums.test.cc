// SPDX-License-Identifier: AGPL-3.0-only
//
// The fixed summation orders and the norms on them: the butterfly against an independent halving fold, every sum
// against float64, and sums of exact terms exactly.
import std;
import engine.model;
import engine.ref;
import engine.testing;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;

using model::Count;

/** The butterfly's lane 0, written as the halving fold it is: pairs `distance` apart, the widest first. */
[[nodiscard]] auto halving(std::vector<float> values) -> float {
	while (values.size() > 1) {
		auto const half = values.size() / 2;
		for (auto index = Count{0}; index < half; ++index) {
			values[index] = values[index] + values[index + half];
		}
		values.resize(half);
	}
	return values.front();
}

[[nodiscard]] auto draws(std::mt19937_64& generator, Count count, float scale) -> std::vector<float> {
	auto value = std::uniform_real_distribution<float>{-scale, scale};
	auto out = std::vector<float>(count);
	for (auto& each : out) {
		each = value(generator);
	}
	return out;
}

/** Terms spanning many binades, so that a different order gives different bits. */
[[nodiscard]] auto wild(std::mt19937_64& generator, Count count) -> std::vector<float> {
	auto mantissa = std::uniform_real_distribution<float>{-1.0F, 1.0F};
	auto exponent = std::uniform_int_distribution<int>{-30, 30};
	auto out = std::vector<float>(count);
	for (auto& each : out) {
		each = std::ldexp(mantissa(generator), exponent(generator));
	}
	return out;
}

[[nodiscard]] auto butterfly_is_the_halving_fold(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 2000; ++trial) {
		auto const values = wild(generator, ref::lanes);
		auto lanes = std::array<float, ref::lanes>{};
		std::ranges::copy(values, lanes.begin());
		if (std::bit_cast<std::uint32_t>(ref::butterfly(lanes)) != std::bit_cast<std::uint32_t>(halving(values))) {
			return false;
		}
	}
	return true;
}

/** Lanes of 128: each lane's four products by fma from zero, then the fold. */
[[nodiscard]] auto lane_dot_is_its_order(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 500; ++trial) {
		auto const x = wild(generator, 128);
		auto const y = wild(generator, 128);
		auto partials = std::vector<float>(ref::lanes);
		for (auto lane = Count{0}; lane < ref::lanes; ++lane) {
			auto partial = 0.0F;
			for (auto term = 4 * lane; term < (4 * lane) + 4; ++term) {
				partial = std::fma(x[term], y[term], partial);
			}
			partials[lane] = partial;
		}
		auto const got = ref::lane_dot<128>(std::span<float const, 128>{x}, std::span<float const, 128>{y});
		if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(halving(partials))) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto close(double got, double want, double relative) -> bool {
	return std::abs(got - want) <= relative * std::max(std::abs(want), 1e-30);
}

[[nodiscard]] auto tiled_squares_match_float64(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 50; ++trial) {
		auto const x = draws(generator, model::hidden, 4.0F);
		auto want = 0.0;
		for (auto const each : x) {
			want += static_cast<double>(each) * static_cast<double>(each);
		}
		if (!close(static_cast<double>(ref::tiled_squares<model::hidden>(std::span<float const, model::hidden>{x})), want, 1e-6)) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto exact_terms_sum_exactly(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto small = std::uniform_int_distribution<int>{-64, 64};
	for (auto trial = 0; trial < 200; ++trial) {
		auto x = std::vector<float>(256);
		auto want = 0;
		for (auto& each : x) {
			auto const value = small(generator);
			each = static_cast<float>(value);
			want += value * value;
		}
		if (ref::lane_squares<256>(std::span<float const, 256>{x}) != static_cast<float>(want)) {
			return false;
		}
	}
	return true;
}

/** The hidden norm against float64, within the one bf16 rounding of its output. */
[[nodiscard]] auto hidden_norm_matches_float64(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 20; ++trial) {
		auto const x = draws(generator, model::hidden, 30.0F);
		auto const raw = draws(generator, model::hidden, 0.5F);
		auto weight = std::vector<ref::Bf16>{};
		for (auto const each : raw) {
			weight.push_back(ref::round(each));
		}
		auto squares = 0.0;
		for (auto const each : x) {
			squares += static_cast<double>(each) * static_cast<double>(each);
		}
		auto const factor = 1.0 / std::sqrt((squares / static_cast<double>(model::hidden)) + static_cast<double>(model::rms_norm_eps));
		auto const got = ref::hidden_norm<model::hidden>(std::span<float const, model::hidden>{x}, std::span<ref::Bf16 const, model::hidden>{weight});
		for (auto index = Count{0}; index < model::hidden; ++index) {
			auto const want = static_cast<double>(x[index]) * factor * (1.0 + static_cast<double>(ref::widen(weight[index])));
			if (!close(static_cast<double>(ref::widen(got[index])), want, 0x1p-8)) {
				return false;
			}
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the butterfly is the halving fold, bit for bit (seed 1)", .holds = butterfly_is_the_halving_fold(1)},
	    engine::testing::Case{.name = "lanes of 128 are per-lane fma chains then the fold, bit for bit (seed 2)", .holds = lane_dot_is_its_order(2)},
	    engine::testing::Case{.name = "the tiled hidden sum of squares is float64's within 1e-6 (seed 3)", .holds = tiled_squares_match_float64(3)},
	    engine::testing::Case{.name = "sums of exactly representable terms are exact (seed 4)", .holds = exact_terms_sum_exactly(4)},
	    engine::testing::Case{.name = "the hidden rms norm is float64's within its bf16 rounding (seed 5)", .holds = hidden_norm_matches_float64(5)},
	};
	return engine::testing::report(cases);
}

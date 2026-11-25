// SPDX-License-Identifier: AGPL-3.0-only
//
// The rounding points' constants and fixed algorithms, each against an independent oracle: double-precision libm for
// the rope table and rotations, exact reconstruction for the attention weight split.
import std;
import engine.model;
import engine.testing;

namespace {

namespace model = engine::model;
namespace rope = engine::model::rope;

using model::Count;

[[nodiscard]] auto next_up(float value) -> float {
	return std::nextafter(value, std::numeric_limits<float>::infinity());
}

[[nodiscard]] auto next_down(float value) -> float {
	return std::nextafter(value, -std::numeric_limits<float>::infinity());
}

/** Whether `candidate` is an fp32 nearest to `exact`: no neighbour is strictly closer. */
[[nodiscard]] auto nearest(float candidate, double exact) -> bool {
	auto const distance = [exact](float value) {
		return std::abs(static_cast<double>(value) - exact);
	};
	return distance(candidate) <= distance(next_up(candidate)) && distance(candidate) <= distance(next_down(candidate));
}

[[nodiscard]] auto inverse_frequencies_are_nearest() -> bool {
	for (auto pair = Count{0}; pair < model::rotary_dim / 2; ++pair) {
		auto const exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(model::rotary_dim);
		if (!nearest(rope::inverse_frequency[pair], std::pow(static_cast<double>(model::rope_theta), exponent))) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto ulps_apart(float left, float right) -> std::int64_t {
	auto const ordered = [](float value) {
		auto const bits = static_cast<std::int64_t>(std::bit_cast<std::uint32_t>(value));
		return bits >= 0x80000000LL ? 0x80000000LL - bits : bits;
	};
	return std::abs(ordered(left) - ordered(right));
}

struct RotationCheck {
	Count checked;
	Count within_one_ulp;
	Count equal_to_libm_rounded;
};

/** Every pair at the given positions against fp32(cos) and fp32(sin) of the same fp32 angle, computed by libm. */
[[nodiscard]] auto check_rotations(std::span<Count const> positions) -> RotationCheck {
	auto result = RotationCheck{.checked = 0, .within_one_ulp = 0, .equal_to_libm_rounded = 0};
	for (auto const position : positions) {
		for (auto pair = Count{0}; pair < model::rotary_dim / 2; ++pair) {
			auto const turn = rope::rotation(position, pair);
			auto const x = static_cast<double>(rope::angle(position, pair));
			auto const cosine = static_cast<float>(std::cos(x));
			auto const sine = static_cast<float>(std::sin(x));
			result.checked += 2;
			result.within_one_ulp += (ulps_apart(turn.cosine, cosine) <= 1 ? 1 : 0) + (ulps_apart(turn.sine, sine) <= 1 ? 1 : 0);
			result.equal_to_libm_rounded += (turn.cosine == cosine ? 1 : 0) + (turn.sine == sine ? 1 : 0);
		}
	}
	return result;
}

[[nodiscard]] auto sampled_positions(std::uint64_t seed) -> std::vector<Count> {
	auto generator = engine::testing::generator(seed);
	auto draw = std::uniform_int_distribution<Count>{0, model::max_positions - 1};
	auto positions = std::vector<Count>{0, 1, 2, 3, model::max_positions - 1, model::max_positions - 2};
	for (auto index = 0; index < 20'000; ++index) {
		positions.push_back(draw(generator));
	}
	return positions;
}

/** Whether the three parts are bf16 values summing exactly to `scaled`, largest first. */
[[nodiscard]] auto split_is_exact(float scaled) -> bool {
	auto const parts = model::split_weight(scaled);
	auto const widen = [](unsigned short bits) {
		return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
	};
	// Exact in double: three fp32 values of at most 24 bits within 48 binades of each other.
	auto const sum = static_cast<double>(widen(parts.high)) + static_cast<double>(widen(parts.middle)) +
	                 static_cast<double>(widen(parts.low));
	auto const low = parts.low;
	auto const rest = scaled - widen(parts.high) - widen(parts.middle);
	return sum == static_cast<double>(scaled) && std::bit_cast<std::uint32_t>(rest) == static_cast<std::uint32_t>(low) << 16U;
}

/** Weights in [0, 1]: every binade's edges and a random fill, scaled or not. */
[[nodiscard]] auto weights(std::uint64_t seed) -> std::vector<float> {
	auto generator = engine::testing::generator(seed);
	auto values = std::vector<float>{0.0F, 1.0F, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min()};
	auto mantissa = std::uniform_int_distribution<std::uint32_t>{0, 0x7fffffU};
	for (auto field = std::uint32_t{0}; field < 127; ++field) {
		values.push_back(std::bit_cast<float>(field << 23U));
		values.push_back(std::bit_cast<float>((field << 23U) | 0x7fffffU));
		for (auto draw = 0; draw < 4096; ++draw) {
			values.push_back(std::bit_cast<float>((field << 23U) | mantissa(generator)));
		}
	}
	return values;
}

}

auto main() -> int {
	auto const positions = sampled_positions(0x5eed'0001);
	auto const rotations = check_rotations(positions);
	auto const probes = weights(0x5eed'0002);
	auto const scaled_exact = std::ranges::all_of(probes, [](float p) {
		return split_is_exact(p * model::attention_weight_scale);
	});
	auto const unscaled_inexact_below_the_bound = std::ranges::any_of(probes, [](float p) {
		return p < 0x1p-110F && !split_is_exact(p);
	});
	auto const unscaled_exact_above_the_bound = std::ranges::all_of(probes, [](float p) {
		return p < 0x1p-110F || split_is_exact(p);
	});
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every rope inverse frequency is the fp32 nearest rope_theta^(-2j / rotary_dim)",
	        .holds = inverse_frequencies_are_nearest(),
	    },
	    engine::testing::Case{
	        .name = "rope cos and sin are within one ulp of libm's at every pair of 20,006 positions (seed 0x5eed0001)",
	        .holds = rotations.within_one_ulp == rotations.checked,
	    },
	    engine::testing::Case{
	        .name = "rope cos and sin equal libm's double rounded to fp32 but for near-ties (at most 1 in 10^5)",
	        .holds = (rotations.checked - rotations.equal_to_libm_rounded) * 100'000 <= rotations.checked,
	    },
	    engine::testing::Case{
	        .name = "the deltanet query scale is the fp32 nearest 128^(-1/2)",
	        .holds = nearest(model::deltanet_query_scale, 1.0 / std::sqrt(static_cast<double>(model::key_dim))),
	    },
	    engine::testing::Case{
	        .name = "every scaled weight in [0, 1] splits into three bf16 that sum to it exactly (seed 0x5eed0002)",
	        .holds = scaled_exact,
	    },
	    engine::testing::Case{
	        .name = "unscaled, the split is exact from 2^-110 up and loses bits below it: the scale is needed",
	        .holds = unscaled_exact_above_the_bound && unscaled_inexact_below_the_bound,
	    },
	};
	return engine::testing::report(cases);
}

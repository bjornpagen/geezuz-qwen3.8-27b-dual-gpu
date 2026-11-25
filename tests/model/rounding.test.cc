// SPDX-License-Identifier: AGPL-3.0-only
//
// The rounding points' constants and fixed algorithms, each against an independent oracle: double-precision libm for
// the rope table and rotations, exact reconstruction for the attention weight split, each lane tree's order written out
// independently, double precision for the merge of attention's groups.
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
	auto const parts = model::attention::split_weight(scaled);
	auto const widen = [](unsigned short bits) {
		return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
	};
	// Exact in double: three fp32 values of at most 24 bits within 48 binades of each other.
	auto const sum =
	    static_cast<double>(widen(parts.high)) + static_cast<double>(widen(parts.middle)) + static_cast<double>(widen(parts.low));
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

/** A finite float of either sign whose exponent spans 2^±30, so sums both cancel and round. */
[[nodiscard]] auto wild(std::mt19937_64& generator) -> float {
	auto const value = std::normal_distribution<float>{0.0F, 1.0F}(generator);
	return std::ldexp(value, std::uniform_int_distribution<int>{-30, 30}(generator));
}

/**
 * Each tree against its order written out independently: every lane's fma chain from +0 over its consecutive terms,
 * then the halving fold at the widest distance first (lane 0's butterfly), bit for bit.
 */
template<model::Lanes tree>
[[nodiscard]] auto tree_is_its_order(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 2000; ++trial) {
		auto x = std::array<float, tree.width>{};
		auto y = std::array<float, tree.width>{};
		std::ranges::generate(x, [&] {
			return wild(generator);
		});
		std::ranges::generate(y, [&] {
			return wild(generator);
		});
		auto level = std::vector<float>(tree.lanes);
		auto const per = tree.width / tree.lanes;
		for (auto lane = Count{0}; lane < tree.lanes; ++lane) {
			auto partial = 0.0F;
			for (auto term = lane * per; term < (lane + 1) * per; ++term) {
				partial = std::fma(x.at(term), y.at(term), partial);
			}
			level.at(lane) = partial;
		}
		for (auto width = tree.lanes / 2; width != 0; width /= 2) {
			for (auto lane = Count{0}; lane < width; ++lane) {
				level.at(lane) = level.at(lane) + level.at(lane + width);
			}
		}
		if (std::bit_cast<std::uint32_t>(model::lane_sum<tree>(x, y, 0)) != std::bit_cast<std::uint32_t>(level.front())) {
			return false;
		}
	}
	return true;
}

/**
 * The merge of two groups rounds once: the group with the larger maximum is the addend, unscaled, because exp(0) is
 * exactly 1; the result equals left + right scaled to the larger maximum, computed exactly in double and rounded once.
 */
[[nodiscard]] auto merge_rounds_once(std::uint64_t seed) -> bool {
	namespace attention = model::attention;
	auto generator = engine::testing::generator(seed);
	auto holds = engine::device::prims::math::exp(0.0F) == 1.0F;
	for (auto trial = 0; trial < 100'000; ++trial) {
		auto const left_maximum = std::uniform_real_distribution<float>{-20.0F, 20.0F}(generator);
		auto const right_maximum = trial % 7 == 0 ? left_maximum : std::uniform_real_distribution<float>{-20.0F, 20.0F}(generator);
		auto const left = std::uniform_real_distribution<float>{0.0F, 1e12F}(generator);
		auto const right = std::uniform_real_distribution<float>{0.0F, 1e12F}(generator);
		auto const how = attention::merge(left_maximum, right_maximum);
		auto const got = attention::merged(left, right, how);
		auto const larger = std::max(left_maximum, right_maximum);
		auto const scaled_left = how.scaled == attention::Side::Left;
		// The double product of two floats is exact; its sum with a float rounds at most once more, far below a float ulp.
		auto const exact = scaled_left ? static_cast<double>(left) * static_cast<double>(how.factor) + static_cast<double>(right)
		                               : static_cast<double>(right) * static_cast<double>(how.factor) + static_cast<double>(left);
		auto const ulp = static_cast<double>(std::nextafter(got, std::numeric_limits<float>::infinity()) - got);
		holds = holds && how.maximum == larger && std::abs(static_cast<double>(got) - exact) <= 0.5 * ulp * (1.0 + 0x1p-20);
		holds =
		    holds && (left_maximum == right_maximum ? !scaled_left && how.factor == 1.0F : scaled_left == (right_maximum > left_maximum));
	}
	return holds;
}

/** The groups tile the chunks: 8 chunks each below position 32768, 64 from there, and `group_of` inverts `group_begin`. */
[[nodiscard]] auto groups_tile_the_chunks() -> bool {
	namespace attention = model::attention;
	auto holds = attention::group_begin(0) == 0;
	for (auto chunk = Count{0}; chunk < attention::max_chunks; ++chunk) {
		auto const group = attention::group_of(chunk);
		holds = holds && attention::group_begin(group) <= chunk && chunk < attention::group_begin(group + 1);
		auto const length = attention::group_begin(group + 1) - attention::group_begin(group);
		holds = holds && length == (chunk * attention::chunk_positions < 32768 ? Count{8} : Count{64});
	}
	return holds && attention::groups_seen(model::max_positions - 1) == attention::group_of(attention::max_chunks - 1) + 1;
}

}

auto main() -> int {
	auto const positions = sampled_positions(0x5eed'0001);
	auto const rotations = check_rotations(positions);
	auto const probes = weights(0x5eed'0002);
	auto const scaled_exact = std::ranges::all_of(probes, [](float p) {
		return split_is_exact(p * model::attention::weight_scale);
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
	        .name = "the deltanet query divisor is the fp32 nearest sqrt(128)",
	        .holds = nearest(model::deltanet::query_divisor, std::sqrt(static_cast<double>(model::key_dim))),
	    },
	    engine::testing::Case{
	        .name = "every scaled weight in [0, 1] splits into three bf16 that sum to it exactly (seed 0x5eed0002)",
	        .holds = scaled_exact,
	    },
	    engine::testing::Case{
	        .name = "unscaled, the split is exact from 2^-110 up and loses bits below it: the scale is needed",
	        .holds = unscaled_exact_above_the_bound && unscaled_inexact_below_the_bound,
	    },
	    engine::testing::Case{.name = "a hidden tile's tree is its fma chains and halving fold (seed 0x5eed0003)",
		                      .holds = tree_is_its_order<model::hidden_tile>(0x5eed'0003)},
	    engine::testing::Case{.name = "an attention head's tree is its fma chains and halving fold (seed 0x5eed0004)",
		                      .holds = tree_is_its_order<model::attention_head>(0x5eed'0004)},
	    engine::testing::Case{.name = "a deltanet vector's tree is its fma chains and halving fold (seed 0x5eed0005)",
		                      .holds = tree_is_its_order<model::deltanet_vector>(0x5eed'0005)},
	    engine::testing::Case{.name = "merging two groups rounds once, the larger maximum's group unscaled (seed 0x5eed0006)",
		                      .holds = merge_rounds_once(0x5eed'0006)},
	    engine::testing::Case{.name = "attention's groups tile the chunks, 8 below position 32768 and 64 from there",
		                      .holds = groups_tile_the_chunks()},
	};
	return engine::testing::report(cases);
}

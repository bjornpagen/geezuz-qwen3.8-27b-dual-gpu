// SPDX-License-Identifier: AGPL-3.0-only
//
// The softmax weight enters the PV mma without losing a bit: scaled by 2^40, every fp32 value in [0, 1] is the exact
// sum of its three bf16 parts (checked in double, where the sum of three parts spanning 25 bits is exact). The edge
// binades (every subnormal, the smallest normals, and [1/4, 1]) are swept whole and the rest sampled; the weights that
// actually occur, e^x for x <= 0 under the shared exp, are swept whole wherever e^x is subnormal and sampled elsewhere.
// The device check's own criterion agrees with the double one on every value tried.
import std;
import engine.attn;
import engine.model;
import engine.prims;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace prims = engine::device::prims;

/** Whether `scaled` is exactly the sum of its parts, and the device check says so too. */
[[nodiscard]] auto exact(float scaled) -> bool {
	auto const parts = attn::split(scaled, attn::PvParts::Three);
	auto const sum = static_cast<double>(prims::widen(parts.part[0])) + static_cast<double>(prims::widen(parts.part[1])) +
	                 static_cast<double>(prims::widen(parts.part[2]));
	return sum == static_cast<double>(scaled);
}

struct Sweep {
	std::uint64_t checked;
	std::uint64_t inexact;
	std::uint64_t disagreements;
};

/** Every bit pattern in [first, last) with the given stride, as a weight in [0, 1]. */
[[nodiscard]] auto sweep_weights(std::uint32_t first, std::uint32_t last, std::uint32_t stride) -> Sweep {
	auto result = Sweep{.checked = 0, .inexact = 0, .disagreements = 0};
	for (auto bits = std::uint64_t{first}; bits < last; bits += stride) {
		auto const weight = std::bit_cast<float>(static_cast<std::uint32_t>(bits));
		auto const holds = exact(weight * engine::model::attention::weight_scale);
		result.inexact += holds ? 0 : 1;
		result.disagreements += holds == attn::selfcheck::split_exact(static_cast<std::uint32_t>(bits)) ? 0 : 1;
		++result.checked;
	}
	return result;
}

[[nodiscard]] constexpr auto binade(std::uint32_t exponent) -> std::uint32_t {
	return exponent << 23U;
}

[[nodiscard]] auto edge_binades_split_exactly() -> bool {
	auto const subnormals = sweep_weights(0, binade(1), 1);
	auto const smallest = sweep_weights(binade(1), binade(4), 1);
	auto const largest = sweep_weights(binade(125), binade(127) + 1, 1);
	auto const total = subnormals.inexact + smallest.inexact + largest.inexact + subnormals.disagreements + smallest.disagreements + largest.disagreements;
	return total == 0 && subnormals.checked + smallest.checked + largest.checked == 6 * (std::uint64_t{1} << 23U) + 1;
}

[[nodiscard]] auto sampled_values_split_exactly() -> bool {
	auto const sample = sweep_weights(binade(4), binade(125), 997);
	return sample.inexact == 0 && sample.disagreements == 0 && sample.checked > 1'000'000;
}

/** Without the scale the split is not exact: the low part falls below bf16's least subnormal for tiny weights. */
[[nodiscard]] auto unscaled_split_is_not_exact() -> bool {
	auto inexact = 0;
	for (auto bits = std::uint32_t{1}; bits < binade(20); bits += 4099) {
		inexact += exact(std::bit_cast<float>(bits)) ? 0 : 1;
	}
	return inexact > 0;
}

/** The weights that occur: e^x * 2^40 for every x <= 0 where e^x is subnormal, and sampled elsewhere. */
[[nodiscard]] auto occurring_weights_split_exactly() -> bool {
	auto sound = true;
	auto checked = std::uint64_t{0};
	auto const check = [&](float x) {
		auto const scaled = engine::model::attention::weight(x, 0.0F);
		sound = sound && scaled >= 0.0F && scaled <= engine::model::attention::weight_scale && exact(scaled);
		++checked;
	};
	// |x| in [64, 128): e^x runs from the normal range down through every subnormal to zero.
	for (auto bits = 0x80000000U | binade(133); bits < (0x80000000U | binade(134)); ++bits) {
		check(std::bit_cast<float>(bits));
	}
	for (auto bits = 0x80000000U; bits <= 0xff800000U; bits += 101) {
		check(std::bit_cast<float>(bits));
	}
	check(0.0F);
	return sound && checked > (std::uint64_t{1} << 23U);
}

/** The one-part rounding U21 times against is a rounding: it differs from the exact split on most weights. */
[[nodiscard]] auto one_part_is_a_rounding() -> bool {
	auto random = engine::testing::generator(7);
	auto rounded = 0;
	for (auto trial = 0; trial < 1000; ++trial) {
		auto const weight = std::uniform_real_distribution<float>{0.0F, 1.0F}(random) * engine::model::attention::weight_scale;
		auto const one = attn::split(weight, attn::PvParts::One);
		rounded += static_cast<double>(prims::widen(one.part[0])) == static_cast<double>(weight) ? 0 : 1;
	}
	return rounded > 900;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every subnormal, the smallest normal binades and [1/4, 1] split exactly once scaled", .holds = edge_binades_split_exactly()},
	    engine::testing::Case{.name = "a stride-997 sample of the other binades splits exactly once scaled", .holds = sampled_values_split_exactly()},
	    engine::testing::Case{.name = "the unscaled split is inexact for tiny weights, which is why the scale exists", .holds = unscaled_split_is_not_exact()},
	    engine::testing::Case{.name = "the weights exp produces for x <= 0 split exactly", .holds = occurring_weights_split_exactly()},
	    engine::testing::Case{.name = "one part rounds the weight", .holds = one_part_is_a_rounding()},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The rope and the norms against double precision: the inverse frequencies are theta^(-j/32) to the nearest fp32;
// the shared sine and cosine are within the stated bound of the exact values at every angle the rope takes (every
// position, every pair); and a prepared query or key is the exact normalized, rotated row to within its bf16 rounding.
import std;
import engine.attn;
import engine.model;
import engine.prims;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace prims = engine::device::prims;

/** The absolute error the shared sine and cosine stay within, over every angle the rope takes. */
inline constexpr double sincos_bound = 0x1p-23;

[[nodiscard]] auto frequencies_are_nearest() -> bool {
	auto sound = true;
	for (auto pair = 0; pair < attn::rotary_pairs; ++pair) {
		auto const exact = std::pow(static_cast<double>(engine::model::rope_theta), -static_cast<double>(pair) / 32.0);
		auto const got = static_cast<double>(attn::inverse_frequency.value[pair]);
		auto const ulp = static_cast<double>(std::nextafter(static_cast<float>(got), 2.0F)) - got;
		sound = sound && std::abs(got - exact) <= 0.5 * ulp * (1.0 + 0x1p-20);
	}
	return sound;
}

struct Errors {
	double sine;
	double cosine;
};

/** The largest errors of the shared sine and cosine over every (position, pair) the model takes. */
[[nodiscard]] auto every_rope_angle() -> Errors {
	auto worst = Errors{.sine = 0.0, .cosine = 0.0};
	for (auto position = 0U; position < engine::model::max_positions; ++position) {
		for (auto pair = 0; pair < attn::rotary_pairs; ++pair) {
			auto const angle = attn::angle(position, pair);
			auto const turn = attn::sincos(angle);
			worst.sine = std::max(worst.sine, std::abs(static_cast<double>(turn.sine) - std::sin(static_cast<double>(angle))));
			worst.cosine = std::max(worst.cosine, std::abs(static_cast<double>(turn.cosine) - std::cos(static_cast<double>(angle))));
		}
	}
	return worst;
}

[[nodiscard]] auto sincos_within_bound() -> bool {
	auto const worst = every_rope_angle();
	return worst.sine <= sincos_bound && worst.cosine <= sincos_bound;
}

/** A prepared row against the exact norm and rotation in double (by the same fp32 angle), within bf16 rounding. */
[[nodiscard]] auto prepared_rows_are_exact_to_bf16(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 2000; ++trial) {
		auto const scale = std::ldexp(1.0F, std::uniform_int_distribution<int>{-12, 12}(random));
		auto projection = std::array<float, 256>{};
		auto weight = std::array<prims::Bf16, 256>{};
		for (auto index = std::size_t{0}; index < 256; ++index) {
			projection.at(index) = std::normal_distribution<float>{0.0F, scale}(random);
			weight.at(index) = prims::round_to_bf16(std::normal_distribution<float>{0.0F, 0.3F}(random));
		}
		auto const position = std::uniform_int_distribution<std::uint32_t>{0, static_cast<std::uint32_t>(engine::model::max_positions) - 1}(random);
		auto const row = engine::attn::prepare_rotated(projection, weight, position);
		auto squares = 0.0;
		for (auto const value : projection) {
			squares += static_cast<double>(value) * static_cast<double>(value);
		}
		auto const inverse = 1.0 / std::sqrt(squares / 256.0 + static_cast<double>(engine::model::rms_norm_eps));
		auto normal = std::array<double, 256>{};
		for (auto index = std::size_t{0}; index < 256; ++index) {
			normal.at(index) = static_cast<double>(projection.at(index)) * inverse * (1.0 + static_cast<double>(prims::widen(weight.at(index))));
		}
		auto exact = normal;
		for (auto pair = std::size_t{0}; pair < 32; ++pair) {
			auto const angle = static_cast<double>(attn::angle(position, static_cast<int>(pair)));
			exact.at(pair) = normal.at(pair) * std::cos(angle) - normal.at(pair + 32) * std::sin(angle);
			exact.at(pair + 32) = normal.at(pair + 32) * std::cos(angle) + normal.at(pair) * std::sin(angle);
		}
		auto largest = 0.0;
		for (auto const value : exact) {
			largest = std::max(largest, std::abs(value));
		}
		for (auto index = std::size_t{0}; index < 256; ++index) {
			auto const got = static_cast<double>(prims::widen(row.at(index)));
			// Half a bf16 ulp of the value, and the fp32 work's few ulps of the row's scale.
			auto const allowed = std::abs(exact.at(index)) * 0x1p-8 + largest * 0x1p-20;
			sound = sound && std::abs(got - exact.at(index)) <= allowed;
		}
	}
	return sound;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the inverse frequencies are the nearest fp32 of theta^(-j/32)", .holds = frequencies_are_nearest()},
	    engine::testing::Case{.name = "sine and cosine are within 2^-23 at every rope angle", .holds = sincos_within_bound()},
	    engine::testing::Case{.name = "prepared queries and keys are the exact rows to within bf16 rounding (seed 1)",
	                          .holds = prepared_rows_are_exact_to_bf16(1)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.prims;
import engine.testing;

namespace {

namespace device = engine::device::prims;
namespace ptx = device::ptx;
namespace decode = device::decode;

using device::Bf16;
using device::Bf16x2;
using device::E4m3;

[[nodiscard]] auto as_double(Bf16 value) -> double {
	return static_cast<double>(device::widen(value));
}

[[nodiscard]] auto bf16_of(double exact_bf16_value) -> Bf16 {
	return {.bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(static_cast<float>(exact_bf16_value)) >> 16U)};
}

[[nodiscard]] auto is_nan(Bf16 value) -> bool {
	return std::isnan(as_double(value));
}

/** The bf16 quantum at magnitude `value`: 2^(e - 7) in binade e, never finer than the subnormal step 2^-133. */
[[nodiscard]] auto quantum(double value) -> double {
	return std::ldexp(1.0, std::max(std::ilogb(value) - 7, -133));
}

/**
 * The oracle for one bf16 fma: the exact a * b + c held as an unevaluated double sum (two-sum), rounded to the
 * nearest bf16 with ties to even by scaling to the quantum and `nearbyint`.
 */
[[nodiscard]] auto oracle_fma(Bf16 a, Bf16 b, Bf16 c) -> Bf16 {
	constexpr auto canonical_nan = Bf16{.bits = 0x7fff};
	auto const x = as_double(a);
	auto const y = as_double(b);
	auto const z = as_double(c);
	if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
		return canonical_nan;
	}
	if ((std::isinf(x) && y == 0.0) || (std::isinf(y) && x == 0.0)) {
		return canonical_nan;
	}
	// a * b is exact in double: 16 significant bits, exponents within double's range.
	auto const product = x * y;
	if (std::isinf(product) && std::isinf(z) && std::signbit(product) != std::signbit(z)) {
		return canonical_nan;
	}
	if (std::isinf(product)) {
		return bf16_of(product);
	}
	if (std::isinf(z)) {
		return c;
	}
	if (product == 0.0 && z == 0.0) {
		return bf16_of(std::signbit(product) && std::signbit(z) ? -0.0 : 0.0);
	}
	auto const sum = product + z;
	auto const virtual_z = sum - product;
	auto const error = (product - (sum - virtual_z)) + (z - virtual_z);
	if (sum == 0.0) {
		return bf16_of(0.0);
	}
	auto const step = quantum(std::abs(sum));
	auto const scaled = sum / step;
	auto rounded = std::nearbyint(scaled);
	if (error != 0.0 && std::abs(scaled - std::trunc(scaled)) == 0.5) {
		// Exactly on a midpoint in the double sum: the lost error decides.
		rounded = (error > 0.0) == (sum > 0.0) ? std::trunc(scaled) + std::copysign(1.0, sum) : std::trunc(scaled);
	}
	auto const result = rounded * step;
	if (std::abs(result) >= 0x1p128) {
		return bf16_of(std::copysign(std::numeric_limits<double>::infinity(), sum));
	}
	return bf16_of(result == 0.0 ? std::copysign(0.0, sum) : result);
}

[[nodiscard]] auto same(Bf16 got, Bf16 expected) -> bool {
	return is_nan(expected) ? got.bits == 0x7fff : got == expected;
}

/** bf16 bits with exponents drawn near each other (where rounding happens) or anywhere (where only the sign does). */
struct Bf16Generator {
	std::mt19937_64 random;

	[[nodiscard]] auto any() -> Bf16 {
		return {.bits = static_cast<std::uint16_t>(random() & 0xffffU)};
	}

	[[nodiscard]] auto near(int exponent_field) -> Bf16 {
		auto const spread = static_cast<int>(random() % 21) - 10;
		auto const field = std::clamp(exponent_field + spread, 0, 254);
		auto const sign = static_cast<std::uint32_t>(random() & 1U) << 15U;
		auto const mantissa = static_cast<std::uint32_t>(random() & 0x7fU);
		return {.bits = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(field) << 7U) | mantissa)};
	}
};

[[nodiscard]] auto fma_model_agrees_with_the_oracle(std::uint64_t seed) -> bool {
	auto generator = Bf16Generator{.random = engine::testing::generator(seed)};
	auto holds = true;
	for (auto trial = 0; trial < 2'000'000; ++trial) {
		auto const a = generator.any();
		auto const b = trial % 2 == 0 ? generator.any() : generator.near(127);
		auto const product_field = (static_cast<int>((std::uint32_t{a.bits} >> 7U) & 0xffU) + static_cast<int>((std::uint32_t{b.bits} >> 7U) & 0xffU)) - 127;
		auto const c = trial % 3 == 0 ? generator.any() : generator.near(std::clamp(product_field, 0, 254));
		holds = holds && same(ptx::fma_rn(a, b, c), oracle_fma(a, b, c));
	}
	return holds;
}

/** The decode's own fma, over every bf16: x * 2^120 + (-0) and + (+0). */
[[nodiscard]] auto fma_model_agrees_on_every_scaling() -> bool {
	auto holds = true;
	for (auto bits = 0U; bits <= 0xffffU; ++bits) {
		auto const a = Bf16{.bits = static_cast<std::uint16_t>(bits)};
		for (auto const addend : {Bf16{.bits = 0x8000}, Bf16{.bits = 0}}) {
			auto const scale = device::first(decode::two_to_120);
			holds = holds && same(ptx::fma_rn(a, scale, addend), oracle_fma(a, scale, addend));
		}
	}
	return holds;
}

/** Each finite code's value, from its fields by an independent formula. */
[[nodiscard]] auto oracle_value(std::uint8_t code) -> double {
	auto const sign = (code & 0x80U) != 0 ? -1.0 : 1.0;
	auto const exponent = static_cast<int>((std::uint32_t{code} >> 3U) & 0xfU);
	auto const mantissa = static_cast<double>(code & 0x7U);
	return sign * (exponent == 0 ? std::ldexp(mantissa / 8.0, -6) : std::ldexp(1.0 + mantissa / 8.0, exponent - 7));
}

[[nodiscard]] auto every_finite_code_converts_exactly() -> bool {
	auto holds = true;
	auto finite = 0;
	for (auto code = 0U; code < 256; ++code) {
		auto const e4m3 = E4m3{.bits = static_cast<std::uint8_t>(code)};
		if (!device::finite(e4m3)) {
			continue;
		}
		++finite;
		auto const expected = oracle_value(static_cast<std::uint8_t>(code));
		holds = holds && static_cast<double>(device::value(e4m3)) == expected;
		holds = holds && as_double(device::to_bf16(e4m3)) == expected && std::signbit(as_double(device::to_bf16(e4m3))) == std::signbit(expected);
	}
	return holds && finite == 254;
}

/** Decoding through the instruction models gives the exact bf16 of every code, in every place of a word. */
[[nodiscard]] auto decode_is_exact_for_every_pair_of_codes() -> bool {
	auto holds = true;
	for (auto first = 0U; first < 256; ++first) {
		for (auto second = 0U; second < 256; ++second) {
			auto const x = E4m3{.bits = static_cast<std::uint8_t>(first)};
			auto const y = E4m3{.bits = static_cast<std::uint8_t>(second)};
			if (!device::finite(x) || !device::finite(y)) {
				continue;
			}
			auto const got = decode::model(decode::encode(x, y, y, x));
			holds = holds && got.top == device::pack(device::to_bf16(x), device::to_bf16(y));
			holds = holds && got.bottom == device::pack(device::to_bf16(y), device::to_bf16(x));
		}
	}
	return holds;
}

[[nodiscard]] auto decode_is_exact_for_random_words(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const draw = [&random] {
		for (;;) {
			auto const code = E4m3{.bits = static_cast<std::uint8_t>(random() & 0xffU)};
			if (device::finite(code)) {
				return code;
			}
		}
	};
	auto holds = true;
	for (auto trial = 0; trial < 1'000'000; ++trial) {
		auto const codes = std::array{draw(), draw(), draw(), draw()};
		auto const got = decode::model(decode::encode(codes[0], codes[1], codes[2], codes[3]));
		holds = holds && got.top == device::pack(device::to_bf16(codes[0]), device::to_bf16(codes[1]));
		holds = holds && got.bottom == device::pack(device::to_bf16(codes[2]), device::to_bf16(codes[3]));
	}
	return holds;
}

/** Folding 2^120 into the scale instead is not exact: the payload alone is 2^-120 times the value. */
[[nodiscard]] auto the_payload_is_the_value_scaled_by_two_to_minus_120() -> bool {
	auto holds = true;
	for (auto code = 0U; code < 256; ++code) {
		auto const e4m3 = E4m3{.bits = static_cast<std::uint8_t>(code)};
		if (device::finite(e4m3)) {
			auto const payload = Bf16{.bits = static_cast<std::uint16_t>(decode::payload(e4m3))};
			holds = holds && as_double(payload) == std::ldexp(oracle_value(static_cast<std::uint8_t>(code)), -120);
		}
	}
	return holds;
}

/** The nearest bf16 to a float by distance, ties to the even neighbour. */
[[nodiscard]] auto oracle_round(float value) -> std::uint16_t {
	auto const bits = std::bit_cast<std::uint32_t>(value);
	if (std::isnan(value)) {
		return 0x7fff;
	}
	auto const below = static_cast<std::uint16_t>(bits >> 16U);
	auto const above = static_cast<std::uint16_t>(below + 1U);
	auto const low = static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(below) << 16U));
	auto const high_float = std::bit_cast<float>(static_cast<std::uint32_t>(above) << 16U);
	// Past the largest finite bf16 the upper neighbour is infinity, at the distance the next binade would have.
	auto const high = std::isinf(high_float) ? std::copysign(0x1p128, static_cast<double>(value)) : static_cast<double>(high_float);
	auto const exact = static_cast<double>(value);
	auto const to_low = std::abs(exact - low);
	auto const to_high = std::abs(high - exact);
	if (to_low < to_high || (to_low == to_high && (below & 1U) == 0)) {
		return below;
	}
	return above;
}

[[nodiscard]] auto rounding_to_bf16_is_nearest_even(std::uint64_t seed) -> bool {
	auto holds = true;
	for (auto high = 0U; high <= 0xffffU; ++high) {
		for (auto const low : {0x0000U, 0x0001U, 0x7fffU, 0x8000U, 0x8001U, 0xffffU}) {
			auto const value = std::bit_cast<float>((high << 16U) | low);
			if (!std::isnan(value) && (high & 0x7fffU) != 0x7f80U) {
				holds = holds && device::round_to_bf16(value).bits == oracle_round(value);
			}
		}
	}
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 4'000'000; ++trial) {
		auto const value = std::bit_cast<float>(static_cast<std::uint32_t>(random()));
		if (!std::isnan(value) && !std::isinf(value)) {
			holds = holds && device::round_to_bf16(value).bits == oracle_round(value);
		}
	}
	holds = holds && device::round_to_bf16(std::numeric_limits<float>::quiet_NaN()).bits == 0x7fff;
	holds = holds && device::round_to_bf16(std::numeric_limits<float>::infinity()).bits == 0x7f80;
	return holds;
}

/** bf16 widening is exact and inverts rounding on every bf16. */
[[nodiscard]] auto widening_round_trips_every_bf16() -> bool {
	auto holds = true;
	for (auto bits = 0U; bits <= 0xffffU; ++bits) {
		auto const value = Bf16{.bits = static_cast<std::uint16_t>(bits)};
		auto const wide = device::widen(value);
		holds = holds && (std::isnan(wide) ? (bits & 0x7fffU) > 0x7f80U : device::round_to_bf16(wide) == value);
	}
	return holds;
}

/** The byte permutes the widening uses give exactly the bf16's fp32 bits, for both halves of every register. */
[[nodiscard]] auto permute_widening_matches_the_bits(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 1'000'000; ++trial) {
		auto const pair = Bf16x2{.bits = static_cast<std::uint32_t>(random())};
		holds = holds && device::widening::first_bits(pair) == static_cast<std::uint32_t>(device::first(pair).bits) << 16U;
		holds = holds && device::widening::second_bits(pair) == static_cast<std::uint32_t>(device::second(pair).bits) << 16U;
	}
	return holds;
}

/** `lop3` under the decode's table is the bitwise select, and every table is the boolean function it encodes. */
[[nodiscard]] auto lop3_tables_are_their_truth_tables(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 100'000; ++trial) {
		auto const a = static_cast<std::uint32_t>(random());
		auto const b = static_cast<std::uint32_t>(random());
		auto const c = static_cast<std::uint32_t>(random());
		holds = holds && ptx::lop3(a, b, c, decode::select_lut) == ((c & a) | (~c & b));
		holds = holds && ptx::lop3(a, b, c, 0xF0U) == a && ptx::lop3(a, b, c, 0xCCU) == b && ptx::lop3(a, b, c, 0xAAU) == c;
		holds = holds && ptx::lop3(a, b, c, 0x96U) == (a ^ b ^ c) && ptx::lop3(a, b, c, 0x80U) == (a & b & c);
	}
	return holds;
}

/** `prmt`'s sign-replicate mode fills a byte with its source byte's top bit. */
[[nodiscard]] auto prmt_selects_and_replicates_bytes(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 100'000; ++trial) {
		auto const a = static_cast<std::uint32_t>(random());
		auto const b = static_cast<std::uint32_t>(random());
		auto const source = static_cast<std::uint64_t>(a) | (static_cast<std::uint64_t>(b) << 32U);
		auto const selector = static_cast<std::uint32_t>(random() & 0xffffU);
		auto expected = std::uint32_t{0};
		for (auto byte = 0U; byte < 4; ++byte) {
			auto const nibble = (selector >> (4U * byte)) & 0xfU;
			auto const picked = static_cast<std::uint32_t>((source >> (8U * (nibble % 8U))) & 0xffU);
			auto const placed = nibble >= 8U ? ((picked >= 0x80U) ? 0xffU : 0U) : picked;
			expected |= placed << (8U * byte);
		}
		holds = holds && ptx::prmt(a, b, selector) == expected;
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every finite e4m3 code has its exact value and bf16", .holds = every_finite_code_converts_exactly()},
	    engine::testing::Case{.name = "a code's payload is exactly 2^-120 times its value", .holds = the_payload_is_the_value_scaled_by_two_to_minus_120()},
	    engine::testing::Case{.name = "the decode is exact for every pair of finite codes in every place",
	                          .holds = decode_is_exact_for_every_pair_of_codes()},
	    engine::testing::Case{.name = "the decode is exact for random words of four codes (seed 11)", .holds = decode_is_exact_for_random_words(11)},
	    engine::testing::Case{.name = "the fma model agrees with an exact oracle on every bf16 times 2^120 plus a zero",
	                          .holds = fma_model_agrees_on_every_scaling()},
	    engine::testing::Case{.name = "the fma model agrees with an exact oracle on random operands (seed 12)", .holds = fma_model_agrees_with_the_oracle(12)},
	    engine::testing::Case{.name = "the fma model agrees with an exact oracle on random operands (seed 13)", .holds = fma_model_agrees_with_the_oracle(13)},
	    engine::testing::Case{.name = "fp32 rounds to the nearest bf16, ties to even (seed 14)", .holds = rounding_to_bf16_is_nearest_even(14)},
	    engine::testing::Case{.name = "every bf16 widens exactly and rounds back to itself", .holds = widening_round_trips_every_bf16()},
	    engine::testing::Case{.name = "the widening byte permutes give a bf16's fp32 bits (seed 15)", .holds = permute_widening_matches_the_bits(15)},
	    engine::testing::Case{.name = "lop3 tables compute their boolean functions (seed 16)", .holds = lop3_tables_are_their_truth_tables(16)},
	    engine::testing::Case{.name = "prmt selects and sign-replicates bytes (seed 17)", .holds = prmt_selects_and_replicates_bytes(17)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The engine's own bf16 and e4m3 (no sdk headers), their exact conversions, and a bit-exact model of every ptx
// instruction the weight decode runs. Pure constexpr, compiled by the device and the host alike: the host tests
// prove the decode exact over every code through the very instruction semantics the device executes, and the
// device self-check compares its hardware decode against the same model.
#pragma once

namespace engine::device::prims {

/** A bfloat16 by its bits: sign, 8 exponent bits (bias 127), 7 mantissa bits. */
struct Bf16 {
	unsigned short bits;

	[[nodiscard]] constexpr auto operator==(Bf16 const&) const -> bool = default;
};

/** Two bf16 in one 32-bit register, as `mma` and `fma.bf16x2` take them: the first element in the low half. */
struct Bf16x2 {
	unsigned bits;

	[[nodiscard]] constexpr auto operator==(Bf16x2 const&) const -> bool = default;
};

/** An e4m3fn code: sign, 4 exponent bits (bias 7), 3 mantissa bits; no infinities, nan only at 0x7f and 0xff. */
struct E4m3 {
	unsigned char bits;

	[[nodiscard]] constexpr auto operator==(E4m3 const&) const -> bool = default;
};

[[nodiscard]] constexpr auto pack(Bf16 first, Bf16 second) -> Bf16x2 {
	return {.bits = static_cast<unsigned>(first.bits) | (static_cast<unsigned>(second.bits) << 16U)};
}

[[nodiscard]] constexpr auto first(Bf16x2 pair) -> Bf16 {
	return {.bits = static_cast<unsigned short>(static_cast<unsigned>(pair.bits) & 0xffffU)};
}

[[nodiscard]] constexpr auto second(Bf16x2 pair) -> Bf16 {
	return {.bits = static_cast<unsigned short>(static_cast<unsigned>(pair.bits) >> 16U)};
}

/** Whether `code` is a number; the baker refuses the two that are not, so the device never meets them. */
[[nodiscard]] constexpr auto finite(E4m3 code) -> bool {
	return (static_cast<unsigned>(code.bits) & 0x7fU) != 0x7fU;
}

/** bf16 widens to fp32 exactly: its bits are fp32's upper half. */
[[nodiscard]] constexpr auto widen(Bf16 value) -> float {
	return __builtin_bit_cast(float, static_cast<unsigned>(value.bits) << 16U);
}

/**
 * fp32 to bf16, rounded to nearest with ties to even, subnormals kept (the two share an exponent range, so dropping
 * 16 bits is the whole conversion); every nan becomes the canonical 0x7fff.
 */
[[nodiscard]] constexpr auto round_to_bf16(float value) -> Bf16 {
	unsigned const bits = __builtin_bit_cast(unsigned, value);
	if ((bits & 0x7fffffffU) > 0x7f800000U) {
		return {.bits = 0x7fffU};
	}
	unsigned const rounded = bits + 0x7fffU + ((bits >> 16U) & 1U);
	return {.bits = static_cast<unsigned short>(rounded >> 16U)};
}

/** The exact value of a finite code, built from its fields: m * 2^-9 when the exponent field is 0, else
 * (8 + m) * 2^(e - 10). */
[[nodiscard]] constexpr auto value(E4m3 code) -> float {
	unsigned const exponent = (static_cast<unsigned>(code.bits) >> 3U) & 0xfU;
	unsigned const mantissa = static_cast<unsigned>(code.bits) & 0x7U;
	float const magnitude = exponent == 0
	                            ? static_cast<float>(mantissa) * 0x1p-9F
	                            : __builtin_bit_cast(float, ((exponent + 120U) << 23U) | (mantissa << 20U));
	return (static_cast<unsigned>(code.bits) & 0x80U) != 0 ? -magnitude : magnitude;
}

/** A finite code's bf16: exact, because e4m3's exponents and 3 mantissa bits all fit in bf16. */
[[nodiscard]] constexpr auto to_bf16(E4m3 code) -> Bf16 {
	return {.bits = static_cast<unsigned short>(__builtin_bit_cast(unsigned, value(code)) >> 16U)};
}

// The semantics of each ptx instruction the decode runs, as the isa defines them on 32-bit registers.
namespace ptx {

/** `lop3.b32 d, a, b, c, lut`: bit i of d is bit (a_i * 4 + b_i * 2 + c_i) of lut. */
[[nodiscard]] constexpr auto lop3(unsigned a, unsigned b, unsigned c, unsigned lut) -> unsigned {
	unsigned result = 0;
	for (unsigned index = 0; index < 8; ++index) {
		if (((lut >> index) & 1U) != 0) {
			unsigned const term_a = (index & 4U) != 0 ? a : ~a;
			unsigned const term_b = (index & 2U) != 0 ? b : ~b;
			unsigned const term_c = (index & 1U) != 0 ? c : ~c;
			result |= term_a & term_b & term_c;
		}
	}
	return result;
}

/**
 * `prmt.b32 d, a, b, selector` in its default mode: bytes 0-3 of a and 4-7 of b form an 8-byte source; nibble i of
 * the selector picks d's byte i by its low 3 bits, and its high bit replicates that byte's sign instead.
 */
[[nodiscard]] constexpr auto prmt(unsigned a, unsigned b, unsigned selector) -> unsigned {
	unsigned long long const source = static_cast<unsigned long long>(a) | (static_cast<unsigned long long>(b) << 32U);
	unsigned result = 0;
	for (unsigned byte = 0; byte < 4; ++byte) {
		unsigned const nibble = (selector >> (4U * byte)) & 0xfU;
		auto picked = static_cast<unsigned>((source >> (8U * (nibble & 7U))) & 0xffU);
		if ((nibble & 8U) != 0) {
			picked = (picked & 0x80U) != 0 ? 0xffU : 0U;
		}
		result |= picked << (8U * byte);
	}
	return result;
}

/** A finite bf16 as sign, integer significand and binary exponent: value = (-1)^negative * significand * 2^exponent. */
struct Exact {
	bool negative;
	unsigned significand;
	int exponent;
};

[[nodiscard]] constexpr auto exact(Bf16 value) -> Exact {
	unsigned const field = (static_cast<unsigned>(value.bits) >> 7U) & 0xffU;
	unsigned const mantissa = static_cast<unsigned>(value.bits) & 0x7fU;
	bool const negative = (static_cast<unsigned>(value.bits) & 0x8000U) != 0;
	if (field == 0) {
		return {.negative = negative, .significand = mantissa, .exponent = -133};
	}
	return {.negative = negative, .significand = mantissa | 0x80U, .exponent = static_cast<int>(field) - 134};
}

[[nodiscard]] constexpr auto is_nan(Bf16 value) -> bool {
	return (static_cast<unsigned>(value.bits) & 0x7fffU) > 0x7f80U;
}

[[nodiscard]] constexpr auto is_infinite(Bf16 value) -> bool {
	return (static_cast<unsigned>(value.bits) & 0x7fffU) == 0x7f80U;
}

/**
 * The bf16 nearest (ties to even) to (-1)^negative * magnitude * 2^exponent, subnormals included and overflow to
 * infinity; `magnitude` is nonzero and below 2^63.
 */
[[nodiscard]] constexpr auto round_exact(bool negative, unsigned long long magnitude, int exponent) -> Bf16 {
	int top = 0;
	while ((magnitude >> static_cast<unsigned>(top + 1)) != 0) {
		++top;
	}
	// The result's quantum: 2^(e - 7) for a value in [2^e, 2^(e+1)), never finer than the subnormal step 2^-133.
	int const scale = top + exponent - 7 > -133 ? top + exponent - 7 : -133;
	int const shift = scale - exponent;
	unsigned long long kept = 0;
	if (shift <= 0) {
		kept = magnitude << static_cast<unsigned>(-shift);
	} else if (shift <= top + 1) {
		kept = magnitude >> static_cast<unsigned>(shift);
		unsigned long long const rest = magnitude - (kept << static_cast<unsigned>(shift));
		unsigned long long const half = 1ULL << static_cast<unsigned>(shift - 1);
		if (rest > half || (rest == half && (kept & 1U) != 0)) {
			++kept;
		}
	}
	int quantum = scale;
	if (kept == 0x100U) {
		kept = 0x80U;
		++quantum;
	}
	unsigned const sign = negative ? 0x8000U : 0U;
	if (kept < 0x80U) {
		return {.bits = static_cast<unsigned short>(sign | static_cast<unsigned>(kept))};
	}
	int const field = quantum + 134;
	if (field >= 0xff) {
		return {.bits = static_cast<unsigned short>(sign | 0x7f80U)};
	}
	return {.bits = static_cast<unsigned short>(sign | (static_cast<unsigned>(field) << 7U) | (static_cast<unsigned>(kept) & 0x7fU))};
}

/**
 * `fma.rn.bf16` on one lane: a * b + c computed exactly and rounded once to nearest even; subnormal inputs and
 * outputs are kept (bf16 fma has no flush mode); every nan result is the canonical 0x7fff.
 */
[[nodiscard]] constexpr auto fma_rn(Bf16 a, Bf16 b, Bf16 c) -> Bf16 {
	constexpr Bf16 canonical_nan = {.bits = 0x7fffU};
	if (is_nan(a) || is_nan(b) || is_nan(c)) {
		return canonical_nan;
	}
	bool const product_negative = ((static_cast<unsigned>(a.bits) ^ static_cast<unsigned>(b.bits)) & 0x8000U) != 0;
	bool const a_zero = (static_cast<unsigned>(a.bits) & 0x7fffU) == 0;
	bool const b_zero = (static_cast<unsigned>(b.bits) & 0x7fffU) == 0;
	if (is_infinite(a) || is_infinite(b)) {
		if (a_zero || b_zero || (is_infinite(c) && (((static_cast<unsigned>(c.bits) & 0x8000U) != 0) != product_negative))) {
			return canonical_nan;
		}
		return {.bits = static_cast<unsigned short>(product_negative ? 0xff80U : 0x7f80U)};
	}
	if (is_infinite(c)) {
		return c;
	}
	Exact const x = exact(a);
	Exact const y = exact(b);
	Exact const z = exact(c);
	unsigned const product = x.significand * y.significand;
	bool const c_zero = z.significand == 0;
	if (product == 0 || c_zero) {
		if (product == 0 && c_zero) {
			// An exact zero sum keeps the sign only when both terms carry it.
			bool const negative = product_negative && z.negative;
			return {.bits = static_cast<unsigned short>(negative ? 0x8000U : 0U)};
		}
		if (product == 0) {
			return c;
		}
		return round_exact(product_negative, product, x.exponent + y.exponent);
	}
	int product_exponent = x.exponent + y.exponent;
	int addend_exponent = z.exponent;
	unsigned long long product_significand = product;
	unsigned long long addend_significand = z.significand;
	// A term more than 40 binades below the other is below a 2^-15 fraction of the result's quantum, so only its
	// sign can matter (when the larger term sits on a rounding midpoint); it is replaced by the smallest term of its
	// sign there, which rounds identically and keeps the aligned sum below 2^57.
	if (product_exponent - addend_exponent > 40) {
		addend_significand = 1;
		addend_exponent = product_exponent - 40;
	} else if (addend_exponent - product_exponent > 40) {
		product_significand = 1;
		product_exponent = addend_exponent - 40;
	}
	int const base = product_exponent < addend_exponent ? product_exponent : addend_exponent;
	auto const product_aligned = static_cast<long long>(product_significand << static_cast<unsigned>(product_exponent - base));
	auto const addend_aligned = static_cast<long long>(addend_significand << static_cast<unsigned>(addend_exponent - base));
	long long const sum = (product_negative ? -product_aligned : product_aligned) + (z.negative ? -addend_aligned : addend_aligned);
	if (sum == 0) {
		return {.bits = 0};
	}
	return round_exact(sum < 0, static_cast<unsigned long long>(sum < 0 ? -sum : sum), base);
}

/** `fma.rn.bf16x2 d, a, b, c`: `fma_rn` on each half. */
[[nodiscard]] constexpr auto fma_rn(Bf16x2 a, Bf16x2 b, Bf16x2 c) -> Bf16x2 {
	return pack(fma_rn(first(a), first(b), first(c)), fma_rn(second(a), second(b), second(c)));
}

}

// The decode. The bake stores each 4-code word re-encoded so that one pair's payload is already at its bf16 bit
// positions (sign at 15, exponent-and-mantissa at 10:4, scaled by 2^-120) and the other pair's sits in the 16 bits
// left free, reachable by one left and one right shift of 4. Five integer instructions place both pairs; one
// `fma.rn.bf16x2` each multiplies by 2^120, which is exact for every finite code, subnormals included, and the -0
// addend keeps -0 as -0.
namespace decode {

inline constexpr unsigned placed_mask = 0x87F087F0U;
inline constexpr unsigned left_mask = 0x080F080FU;
inline constexpr unsigned right_mask = 0x07000700U;
// `lop3` with this table picks c ? a : b.
inline constexpr unsigned select_lut = 0xE4U;
inline constexpr Bf16x2 two_to_120 = {.bits = 0x7B807B80U};
inline constexpr Bf16x2 negative_zero = {.bits = 0x80008000U};

/** A word's two bf16x2 registers: rows g and g + 8 of an m16n8k16 a-fragment, each holding two adjacent k. */
struct Pairs {
	Bf16x2 top;
	Bf16x2 bottom;

	[[nodiscard]] constexpr auto operator==(Pairs const&) const -> bool = default;
};

/** The 16-bit payload of one code at bf16 positions: 2^-120 times its value, exactly. */
[[nodiscard]] constexpr auto payload(E4m3 code) -> unsigned {
	return ((static_cast<unsigned>(code.bits) & 0x80U) << 8U) | ((static_cast<unsigned>(code.bits) & 0x7fU) << 4U);
}

/**
 * The baker's re-encoding of four finite codes: (top_first, top_second) decode to the `top` register and
 * (bottom_first, bottom_second) to `bottom`. A bijection on the 32 bits, so it loses nothing.
 */
[[nodiscard]] constexpr auto encode(E4m3 top_first, E4m3 top_second, E4m3 bottom_first, E4m3 bottom_second) -> unsigned {
	unsigned const top = payload(top_first) | (payload(top_second) << 16U);
	unsigned const bottom = payload(bottom_first) | (payload(bottom_second) << 16U);
	return top | ((bottom & right_mask) << 4U) | ((bottom & (placed_mask & ~right_mask)) >> 4U);
}

/** The device's decode, instruction by instruction, through the instruction models. */
[[nodiscard]] constexpr auto model(unsigned word) -> Pairs {
	unsigned const top = word & placed_mask;
	unsigned const left = (word & left_mask) << 4U;
	unsigned const right = word >> 4U;
	unsigned const bottom = ptx::lop3(right, left, right_mask, select_lut);
	return {
	    .top = ptx::fma_rn(Bf16x2{.bits = top}, two_to_120, negative_zero),
	    .bottom = ptx::fma_rn(Bf16x2{.bits = bottom}, two_to_120, negative_zero),
	};
}

}

// A bf16x2 register widens to two fp32 with one byte permute each, on the integer pipe: the first element moves to
// the upper half over zeros, the second keeps its place over zeros.
namespace widening {
inline constexpr unsigned first_selector = 0x1044U;
inline constexpr unsigned second_selector = 0x3244U;

[[nodiscard]] constexpr auto first_bits(Bf16x2 pair) -> unsigned {
	return ptx::prmt(static_cast<unsigned>(pair.bits), 0U, first_selector);
}

[[nodiscard]] constexpr auto second_bits(Bf16x2 pair) -> unsigned {
	return ptx::prmt(static_cast<unsigned>(pair.bits), 0U, second_selector);
}
}

}

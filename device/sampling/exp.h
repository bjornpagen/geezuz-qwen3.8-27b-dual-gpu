// SPDX-License-Identifier: AGPL-3.0-only
//
// The exponential the sampling weights are made of, held once so the device and the host compute the same bits: the
// device has no libm (no CUDA SDK), and two exponentials that differ in the last bit would let the two sides pick
// different tokens from the same uniform.
//
// Why it is built from exact products and integers: every floating-point product below is exact (a power-of-two
// scaling, or a small integer times a short constant) and the series is evaluated in 64-bit fixed point, so
// contraction cannot change a bit: fused or not, each sum rounds once from the same exact operands.
#pragma once

namespace engine::device::sampling {

static_assert(sizeof(long long) == 8 && sizeof(double) == 8, "the exponential works in 64-bit words");

/**
 * (a * b) / 2^62 rounded to nearest, for fixed-point values with 62 fraction bits and |a|, |b| < 2^63: the 128-bit
 * product is assembled from four 32-bit partial products.
 */
[[nodiscard]] constexpr auto multiply_fixed62(long long a, long long b) -> long long {
	unsigned long long const magnitude_a = a < 0 ? 0ULL - static_cast<unsigned long long>(a) : static_cast<unsigned long long>(a);
	unsigned long long const magnitude_b = b < 0 ? 0ULL - static_cast<unsigned long long>(b) : static_cast<unsigned long long>(b);
	unsigned long long const a_low = magnitude_a & 0xffffffffULL;
	unsigned long long const a_high = magnitude_a >> 32U;
	unsigned long long const b_low = magnitude_b & 0xffffffffULL;
	unsigned long long const b_high = magnitude_b >> 32U;
	unsigned long long const low_low = a_low * b_low;
	unsigned long long const low_high = a_low * b_high;
	unsigned long long const high_low = a_high * b_low;
	unsigned long long const high_high = a_high * b_high;
	unsigned long long const middle = (low_low >> 32U) + (low_high & 0xffffffffULL) + (high_low & 0xffffffffULL);
	unsigned long long high = high_high + (low_high >> 32U) + (high_low >> 32U) + (middle >> 32U);
	unsigned long long low = (middle << 32U) | (low_low & 0xffffffffULL);
	constexpr unsigned long long half = 1ULL << 61U;
	high += low + half < low ? 1ULL : 0ULL;
	low += half;
	auto const magnitude = static_cast<long long>((high << 2U) | (low >> 62U));
	return (a < 0) != (b < 0) ? -magnitude : magnitude;
}

/**
 * e^x for x <= 0, within one ulp where it is at least 2^-1021 relative to the leading weight, and exactly 0 below
 * x = -708 (a weight under 2^-1021 is a probability far below the 2^-53 resolution of a uniform). Exactly 1 at
 * x = 0, and 0 for -infinity. x = k ln2 + r with |r| <= ln2 / 2, k ln2 subtracted in three exact products; e^r is
 * its Taylor series to the 22nd term in fixed point with 62 fraction bits (truncation error under 2^-90), rounded
 * once to binary64; 2^k is added to the exponent field.
 */
[[nodiscard]] constexpr auto exp_nonpositive(double x) -> double {
	constexpr double ln2 = 0x1.62e42fefa39efp-1;
	// ln2 = part1 + part2 + part3 to 2^-112; the parts have 33, 33 and 43 significant bits, so k * part is exact for
	// |k| <= 1021.
	constexpr double part1 = 0x1.62e42ffp-1;
	constexpr double part2 = -0x1.718432a2p-35;
	constexpr double part3 = 0x1.3c7673007e4p-69;
	constexpr double round_shift = 0x1.8p52;
	// Below -708 the result would leave the normal range; the negated comparison also sends -infinity there.
	if (!(x >= -708.0)) {
		return 0.0;
	}
	double const multiple = (x / ln2 + round_shift) - round_shift;
	double const reduced = ((x - multiple * part1) - multiple * part2) - multiple * part3;
	auto const fixed_reduced = static_cast<long long>(reduced * 0x1p62);
	auto term = static_cast<long long>(1ULL << 62U);
	long long sum = term;
	for (long long order = 1; order <= 22; ++order) {
		term = multiply_fixed62(term, fixed_reduced) / order;
		sum += term;
	}
	double const mantissa = static_cast<double>(static_cast<unsigned long long>(sum));
	auto const exponent = static_cast<long long>(multiple) - 62;
	return __builtin_bit_cast(double, __builtin_bit_cast(long long, mantissa) + exponent * static_cast<long long>(1ULL << 52U));
}

static_assert(exp_nonpositive(0.0) == 1.0 && exp_nonpositive(-0.0) == 1.0, "the leading weight is exactly one");
static_assert(exp_nonpositive(-708.5) == 0.0 && exp_nonpositive(-__builtin_inf()) == 0.0);
static_assert(exp_nonpositive(-708.0) > 0x1p-1022, "every nonzero weight is normal");

}

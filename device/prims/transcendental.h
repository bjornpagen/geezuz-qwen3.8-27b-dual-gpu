// SPDX-License-Identifier: AGPL-3.0-only
//
// The transcendentals of the engine, specified once: exp, log and rsqrt, and the sigmoid, silu and softplus built on
// them. Each is a fixed sequence of ieee fp32 operations rounded to nearest (add, mul, fma, div, sqrt), so the
// kernels and the host reference, which both compile this header, compute the same bits on every input, whatever
// llvm or the driver's jit would otherwise choose. Both sides compile with contraction off (the build graph sets it for
// host and device alike), so no product is fused into a sum the specification rounds separately; every fma is
// spelled.
//
// Error bounds, against the exact value over every fp32 input (the host test proves them exhaustively on each run,
// against a double-precision reference): exp and log within 1 ulp, rsqrt within 0.51 ulp, softplus within 2 ulp,
// sigmoid and silu within 2.5 ulp; in the subnormal range the ulp is the subnormal step. Every nan input gives a nan;
// infinities and zeros give the limits.
#pragma once

namespace engine::device::prims::math {

[[nodiscard]] constexpr auto bits(float value) -> unsigned {
	return __builtin_bit_cast(unsigned, value);
}

[[nodiscard]] constexpr auto from_bits(unsigned value) -> float {
	return __builtin_bit_cast(float, value);
}

[[nodiscard]] constexpr auto fma(float a, float b, float c) -> float {
	return __builtin_fmaf(a, b, c);
}

inline constexpr float infinity = __builtin_inff();
inline constexpr float quiet_nan = __builtin_bit_cast(float, 0x7fc00000U);

// ln 2 split so that k * ln2_hi is exact for every |k| < 2^9 (15 significant bits), with the rest in ln2_lo.
inline constexpr float ln2_hi = 0x1.62e400p-1F;
inline constexpr float ln2_lo = 0x1.7f7d1cp-20F;
inline constexpr float log2_e = 0x1.715476p+0F;

/** 2^k for k in [-126, 127], by its exponent field. */
[[nodiscard]] constexpr auto power_of_two(int k) -> float {
	return from_bits(static_cast<unsigned>(k + 127) << 23U);
}

/**
 * v * 2^k rounded once, for k in [-252, 254] and v whose product with 2^(k/2) stays normal: the first power-of-two
 * factor is exact, so only the second rounds, and a subnormal result is not rounded twice.
 */
[[nodiscard]] constexpr auto scale(float v, int k) -> float {
	int const k_low = k / 2;
	float const partial = v * power_of_two(k_low);
	return partial * power_of_two(k - k_low);
}

/** e^x as mantissa * 2^exponent, the mantissa within [e^(-ln2/2), e^(ln2/2)]. */
struct Exponential {
	float mantissa;
	int exponent;
};

/**
 * e^x split, for finite x in [-112, 89]: x = k ln 2 + r with |r| <= ln 2 / 2. k is x log2(e) rounded by one fma
 * against 1.5 * 2^23, r is taken in two fmas (the first exact), and e^r is the degree-7 Taylor polynomial
 * (truncation under 2^-27) as 1 + (r + (r^2 q(r) + r's rounding error)), so the one rounding at the scale of the
 * result is the last.
 */
[[nodiscard]] constexpr auto exp_split(float x) -> Exponential {
	constexpr float shifter = 0x1.8p23F;
	float const shifted = fma(x, log2_e, shifter);
	float const k_float = shifted - shifter;
	int const k = static_cast<int>(bits(shifted)) - static_cast<int>(bits(shifter));
	float const r_hi = fma(-k_float, ln2_hi, x);
	float const r = fma(-k_float, ln2_lo, r_hi);
	float q = 1.0F / 5040.0F;
	q = fma(q, r, 1.0F / 720.0F);
	q = fma(q, r, 1.0F / 120.0F);
	q = fma(q, r, 1.0F / 24.0F);
	q = fma(q, r, 1.0F / 6.0F);
	q = fma(q, r, 0.5F);
	// r's own rounding, exact by fma: r_hi - r is exact whenever it matters (r_hi and r within a factor of two).
	float const r_hi_less_r = r_hi - r;
	float const r_error = fma(-k_float, ln2_lo, r_hi_less_r);
	float const square = r * r;
	float const small = fma(q, square, r_error);
	float const tail = r + small;
	return {.mantissa = 1.0F + tail, .exponent = k};
}

/** e^x; beyond the range where it is a finite nonzero float, the limit. */
[[nodiscard]] constexpr auto exp(float x) -> float {
	if (x != x) {
		return x + x;
	}
	if (x > 89.0F) {
		return infinity;
	}
	if (x < -104.0F) {
		return 0.0F;
	}
	Exponential const split = exp_split(x);
	return scale(split.mantissa, split.exponent);
}

/**
 * log(2^k (1 + f) (1 + correction)) for |f| <= sqrt(2) - 1, with `correction` the first-order term of a rounding
 * the caller made. s = f / (2 + f) gives log(1 + f) = f - f^2/2 + s (f^2/2 + R(s^2)), where R is the atanh series'
 * tail (Taylor coefficients 2/3 .. 2/15, truncation under 2^-40); the small terms are summed first.
 */
[[nodiscard]] constexpr auto log_core(int k, float f, float correction) -> float {
	float const k_float = static_cast<float>(k);
	float const denominator = 2.0F + f;
	float const s = f / denominator;
	float const z = s * s;
	float r = 2.0F / 15.0F;
	r = fma(r, z, 2.0F / 13.0F);
	r = fma(r, z, 2.0F / 11.0F);
	r = fma(r, z, 2.0F / 9.0F);
	r = fma(r, z, 2.0F / 7.0F);
	r = fma(r, z, 2.0F / 5.0F);
	r = fma(r, z, 2.0F / 3.0F);
	r = r * z;
	float const half_square = 0.5F * f * f;
	float const tail = s * (half_square + r);
	float const low = fma(k_float, ln2_lo, correction);
	float const small = tail + low;
	float const difference = half_square - small;
	float const mantissa_log = difference - f;
	return fma(k_float, ln2_hi, -mantissa_log);
}

/** The natural logarithm: -inf at zero, nan below it, +inf at +inf. */
[[nodiscard]] constexpr auto log(float x) -> float {
	if (x != x) {
		return x + x;
	}
	if (x < 0.0F) {
		return quiet_nan;
	}
	if (x == 0.0F) {
		return -infinity;
	}
	if (x == infinity) {
		return infinity;
	}
	int exponent = 0;
	float normal = x;
	if (x < 0x1p-126F) {
		normal = x * 0x1p23F;
		exponent = -23;
	}
	unsigned const word = bits(normal);
	exponent += static_cast<int>(word >> 23U) - 127;
	unsigned mantissa_bits = (word & 0x007fffffU) | 0x3f800000U;
	// Keep the mantissa in [sqrt(1/2), sqrt(2)) so that f = m - 1 is small and exact.
	if (mantissa_bits > 0x3fb504f3U) {
		mantissa_bits -= 0x00800000U;
		exponent += 1;
	}
	float const f = from_bits(mantissa_bits) - 1.0F;
	return log_core(exponent, f, 0.0F);
}

/**
 * log(1 + u) for u in [0, 1]. w = 1 + u rounds, but the rounding error u - (w - 1) is exact, and enters as the
 * first-order correction (u - (w - 1)) / w.
 */
[[nodiscard]] constexpr auto log1p_unit(float u) -> float {
	float const w = 1.0F + u;
	float const w_less_one = w - 1.0F;
	float const lost = u - w_less_one;
	float const correction = lost / w;
	if (w > 0x1.6a09e6p+0F) {
		float const w_less_two = w - 2.0F;
		return log_core(1, w_less_two * 0.5F, correction);
	}
	return log_core(0, w_less_one, correction);
}

/**
 * 1 / sqrt(x) for x zero or at least 2^-100. r = 1 / s with s = sqrt(x) carries both roundings; the remainders 1 - r s and
 * x - s^2 are exact by fma, and correct r to first order: 1 / sqrt(x) = r (1 + (1 - r s) - (x - s^2) / 2x + O(2^-46)).
 */
[[nodiscard]] constexpr auto rsqrt_normal(float x) -> float {
	float const root = __builtin_sqrtf(x);
	float const reciprocal = 1.0F / root;
	// At zero, infinity and outside the domain the remainders are not finite; the plain quotient is the limit.
	if (!(x > 0.0F) || x == infinity) {
		return reciprocal;
	}
	float const division_remainder = fma(-reciprocal, root, 1.0F);
	float const root_remainder = fma(-root, root, x);
	float const relative = root_remainder / x;
	float const correction = fma(-0.5F, relative, division_remainder);
	return fma(reciprocal, correction, reciprocal);
}

/** 1 / sqrt(x): +inf at zero, nan below it, 0 at +inf. */
[[nodiscard]] constexpr auto rsqrt(float x) -> float {
	// Below 2^-100 the remainder x - s^2 can fall under the normal range and round; scaling by 2^24 keeps it exact.
	if (x > 0.0F && x < 0x1p-100F) {
		return rsqrt_normal(x * 0x1p24F) * 0x1p12F;
	}
	return rsqrt_normal(x);
}

// Down to -80 the logistic functions divide by 1 + e^-x, which is at least 1 so nothing cancels and e^-x stays
// finite; below it 1 + e^x rounds to 1, so they are e^x and x e^x outright.
inline constexpr float logistic_floor = -80.0F;

/** 1 / (1 + e^-x). */
[[nodiscard]] constexpr auto sigmoid(float x) -> float {
	if (x < logistic_floor) {
		return exp(x);
	}
	float const e = exp(-x);
	float const denominator = 1.0F + e;
	return 1.0F / denominator;
}

/** x / (1 + e^-x); below the floor x e^x, with the power of two applied last so a subnormal result rounds once. */
[[nodiscard]] constexpr auto silu(float x) -> float {
	if (x != x) {
		return x + x;
	}
	if (x >= logistic_floor) {
		float const e = exp(-x);
		float const denominator = 1.0F + e;
		return x / denominator;
	}
	// Below -112 the product is under half the least subnormal; its limit is -0.
	if (x < -112.0F) {
		return -0.0F;
	}
	Exponential const split = exp_split(x);
	float const product = x * split.mantissa;
	return scale(product, split.exponent);
}

/** log(1 + e^x) = max(x, 0) + log(1 + e^-|x|), which neither overflows nor loses the small side. */
[[nodiscard]] constexpr auto softplus(float x) -> float {
	if (x != x) {
		return x + x;
	}
	float const magnitude = x < 0.0F ? -x : x;
	float const tail = log1p_unit(exp(-magnitude));
	return x > 0.0F ? x + tail : tail;
}

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's arithmetic, specified once as fixed fp32 operations rounded to nearest: the q and k norms, the partial
// rope and its angles, the softmax weights and their exact split into bf16 parts, the online fold of a group's chunks,
// the merge of groups, and the gated output. The kernels and the host reference compile this header, both with
// contraction off, so they compute the same bits; only the tensor core's sums (the scores and the weighted values) lie
// outside it, and the reference takes those from the model of the mma.
//
// Activation points: q and k leave the projection in fp32, are normalized and rotated in fp32, and are rounded to
// bf16 once (k into the kv cache); v is rounded to bf16 once into the cache; the gated output is rounded to bf16 once,
// as o_proj's input. Everything between is fp32.
#pragma once

#include "attn/geometry.h"
#include "prims/numeric.h"
#include "prims/transcendental.h"

namespace engine::device::attn {

using prims::Bf16;
namespace math = prims::math;

// ---------------------------------------------------------------------------------------------------------------------
// A row in a warp: lane L holds features [8L, 8L + 8), the row's logical chunk L.

inline constexpr int lane_features = static_cast<int>(features_per_chunk);

static_assert(head_dim == 32 * features_per_chunk, "one warp holds one row, one chunk per lane");

/** The fp32 sum of 32 values as the warp's butterfly forms it: at distance 16, 8, 4, 2, 1 each adds its partner's. */
[[nodiscard]] constexpr auto butterfly_sum(float const (&values)[32]) -> float {
	float level[32] = {};
	for (unsigned lane = 0; lane < 32; ++lane) {
		level[lane] = values[lane];
	}
	for (unsigned distance = 16; distance != 0; distance /= 2) {
		float next[32] = {};
		for (unsigned lane = 0; lane < 32; ++lane) {
			next[lane] = level[lane] + level[lane ^ distance];
		}
		for (unsigned lane = 0; lane < 32; ++lane) {
			level[lane] = next[lane];
		}
	}
	return level[0];
}

// ---------------------------------------------------------------------------------------------------------------------
// RMS norm over a head, applying (1 + w): each lane sums its 8 squares in feature order, the warp adds the lanes by its
// butterfly, and the mean is that sum times 2^-8, which is exact.

/** One lane's share of a row's sum of squares. */
[[nodiscard]] constexpr auto lane_squares(float const (&values)[lane_features]) -> float {
	float sum = 0.0F;
	for (float const value : values) {
		float const square = value * value;
		sum = sum + square;
	}
	return sum;
}

inline constexpr float inverse_head_dim = 1.0F / static_cast<float>(head_dim);

static_assert(head_dim == 256, "the mean is a multiplication by an exact power of two");

/** 1 / sqrt(mean square + eps) from the row's sum of squares. */
[[nodiscard]] constexpr auto inverse_rms(float squares) -> float {
	float const mean = squares * inverse_head_dim;
	return math::rsqrt(mean + model::rms_norm_eps);
}

[[nodiscard]] constexpr auto normalized(float value, float inverse, Bf16 weight) -> float {
	float const scaled = value * inverse;
	float const gain = 1.0F + prims::widen(weight);
	return scaled * gain;
}

static_assert(model::rms_norm_scale == model::NormScale::OnePlusWeight, "q_norm and k_norm apply (1 + w)");

// ---------------------------------------------------------------------------------------------------------------------
// Partial rope: pair j (0 to 31) rotates feature j with feature j + 32 by the angle position * f_j, f_j = theta^(-j/32),
// in natural order. f_j is the fp32 nearest theta^(-j/32) evaluated in double at compile time; the angle is one fp32
// product (the position is exact); sine and cosine are a fixed fp32 routine.

inline constexpr int rotary_pairs = static_cast<int>(model::rotary_dim / 2);

static_assert(rotary_pairs == 32 && model::rotary_dim == 2 * features_per_chunk * 4, "the rotated features are lanes 0-7");

[[nodiscard]] consteval auto double_sqrt(double value) -> double {
	double root = value > 1.0 ? value : 1.0;
	for (int step = 0; step < 200; ++step) {
		double const next = 0.5 * (root + value / root);
		if (next == root) {
			break;
		}
		root = next;
	}
	return root;
}

struct Frequencies {
	float value[32];
};

[[nodiscard]] consteval auto inverse_frequencies() -> Frequencies {
	// theta^(1/32) by five square roots, then its negative powers by repeated division.
	double root = static_cast<double>(model::rope_theta);
	for (int step = 0; step < 5; ++step) {
		root = double_sqrt(root);
	}
	auto result = Frequencies{};
	double power = 1.0;
	for (int pair = 0; pair < rotary_pairs; ++pair) {
		result.value[pair] = static_cast<float>(1.0 / power);
		power = power * root;
	}
	return result;
}

inline constexpr Frequencies inverse_frequency = inverse_frequencies();

/** The angle pair `pair` rotates by at `position`. */
[[nodiscard]] constexpr auto angle(unsigned position, int pair) -> float {
	return static_cast<float>(position) * inverse_frequency.value[pair];
}

static_assert(model::max_positions <= (1U << 24U), "every position is exact in fp32");

struct SinCos {
	float sine;
	float cosine;
};

// pi / 2 as three fp32 parts, the first two exact multiples of their own ulp, from a double-double of pi / 2.
struct HalfPi {
	float high;
	float middle;
	float low;
};

[[nodiscard]] consteval auto half_pi_parts() -> HalfPi {
	constexpr double high_double = 0x1.921fb54442d18p+0;
	constexpr double low_double = 0x1.1a62633145c07p-54;
	auto const high = static_cast<float>(high_double);
	double const rest = high_double - static_cast<double>(high);
	auto const middle = static_cast<float>(rest + low_double);
	double const tail = (rest - static_cast<double>(middle)) + low_double;
	return {.high = high, .middle = middle, .low = static_cast<float>(tail)};
}

inline constexpr HalfPi half_pi = half_pi_parts();
inline constexpr float two_over_pi = 0x1.45f306p-1F;

/**
 * sin and cos of an angle in [0, 2^19): k = nearest(x 2 / pi) by one fma against 1.5 * 2^23; r = x - k pi/2 by three
 * fmas, the first exact; then Taylor polynomials of degree 9 and 10 on |r| <= pi/4 (truncation below 2^-28) and the
 * quadrant's signs and swap. The host test bounds the error over every angle the rope takes.
 */
[[nodiscard]] constexpr auto sincos(float x) -> SinCos {
	constexpr float shifter = 0x1.8p23F;
	float const shifted = math::fma(x, two_over_pi, shifter);
	float const k_float = shifted - shifter;
	auto const quadrant = (math::bits(shifted) - math::bits(shifter)) & 3U;
	float reduced = math::fma(-k_float, half_pi.high, x);
	reduced = math::fma(-k_float, half_pi.middle, reduced);
	reduced = math::fma(-k_float, half_pi.low, reduced);
	float const z = reduced * reduced;
	float sine_tail = 1.0F / 362880.0F;
	sine_tail = math::fma(sine_tail, z, -1.0F / 5040.0F);
	sine_tail = math::fma(sine_tail, z, 1.0F / 120.0F);
	sine_tail = math::fma(sine_tail, z, -1.0F / 6.0F);
	float const cube = reduced * z;
	float const sine = math::fma(cube, sine_tail, reduced);
	float cosine_tail = -1.0F / 3628800.0F;
	cosine_tail = math::fma(cosine_tail, z, 1.0F / 40320.0F);
	cosine_tail = math::fma(cosine_tail, z, -1.0F / 720.0F);
	cosine_tail = math::fma(cosine_tail, z, 1.0F / 24.0F);
	float const fourth = z * z;
	float const head = math::fma(z, -0.5F, 1.0F);
	float const cosine = math::fma(fourth, cosine_tail, head);
	switch (quadrant) {
	case 0:
		return {.sine = sine, .cosine = cosine};
	case 1:
		return {.sine = cosine, .cosine = -sine};
	case 2:
		return {.sine = -sine, .cosine = -cosine};
	default:
		return {.sine = -cosine, .cosine = sine};
	}
}

/** Feature j of a rotated pair: x_j cos - x_{j+32} sin. */
[[nodiscard]] constexpr auto rotate_first(float first, float second, SinCos turn) -> float {
	float const kept = first * turn.cosine;
	float const moved = second * turn.sine;
	return kept - moved;
}

/** Feature j + 32 of a rotated pair: x_{j+32} cos + x_j sin. */
[[nodiscard]] constexpr auto rotate_second(float first, float second, SinCos turn) -> float {
	float const kept = second * turn.cosine;
	float const moved = first * turn.sine;
	return kept + moved;
}

// ---------------------------------------------------------------------------------------------------------------------
// Scores and the softmax weights.
//
// A score is the mma's q.k sum (16 k16 steps in feature order) times 1/16 = 1/sqrt(256), exact. Within a group, chunk c
// raises the row's running maximum to m' = max(m, max of c's visible scores), rescales the running output and total by
// alpha = e^(m - m'), and weighs each visible position by P = e^(s - m') * 2^40. Scaling by 2^40 is exact and puts every
// bit of every P in [0, 2^40] at or above 2^-109, inside bf16's range, so P splits into three bf16 parts that sum to
// it exactly; the three PV mmas then take P's every bit. The total carries the same scale, so the ratio is unchanged.

inline constexpr float score_scale = 0.0625F;
inline constexpr float weight_scale = 0x1p40F;

static_assert(head_dim == 256, "1/16 is 1/sqrt(head_dim), an exact power of two");

[[nodiscard]] constexpr auto score(float dot) -> float {
	return dot * score_scale;
}

/** The larger of two maxima, neither a nan (an empty running maximum is -inf). */
[[nodiscard]] constexpr auto larger(float left, float right) -> float {
	return right > left ? right : left;
}

/** A chunk's step of the online softmax: the new running maximum and the rescale of what ran before. */
struct Advance {
	float maximum;
	float rescale;
};

[[nodiscard]] constexpr auto advance(float running_maximum, float chunk_maximum) -> Advance {
	float const maximum = larger(running_maximum, chunk_maximum);
	float const gap = running_maximum - maximum;
	return {.maximum = maximum, .rescale = math::exp(gap)};
}

/** The scaled softmax weight of a visible score under the running maximum. */
[[nodiscard]] constexpr auto weight(float score_value, float maximum) -> float {
	float const gap = score_value - maximum;
	return math::exp(gap) * weight_scale;
}

/** The running total after a chunk whose scaled weights sum (by the butterfly over its 32 positions) to `sum`. */
[[nodiscard]] constexpr auto running_total(float total, float rescale, float sum) -> float {
	float const kept = total * rescale;
	return kept + sum;
}

/** How many bf16 parts carry P into the PV mma: three in the engine. One exists only to measure what the split costs. */
enum class PvParts : unsigned char { Three, One };

[[nodiscard]] constexpr auto part_count(PvParts parts) -> int {
	return parts == PvParts::Three ? 3 : 1;
}

/** A scaled weight as bf16 parts, largest first. */
struct Parts {
	Bf16 part[3];
};

/** P = high + middle + low exactly: each part is the nearest bf16 of what the larger parts leave, which is exact. */
[[nodiscard]] constexpr auto split(float scaled) -> Parts {
	Bf16 const high = prims::round_to_bf16(scaled);
	float const after_high = scaled - prims::widen(high);
	Bf16 const middle = prims::round_to_bf16(after_high);
	float const after_middle = after_high - prims::widen(middle);
	return {.part = {high, middle, prims::round_to_bf16(after_middle)}};
}

/** P as `parts` parts: the exact split, or P rounded once. */
[[nodiscard]] constexpr auto split(float scaled, PvParts parts) -> Parts {
	if (parts == PvParts::One) {
		return {.part = {prims::round_to_bf16(scaled), Bf16{.bits = 0}, Bf16{.bits = 0}}};
	}
	return split(scaled);
}

// ---------------------------------------------------------------------------------------------------------------------
// Merging groups: the left group's output and total rescaled by e^(m_left - M), the right's by e^(m_right - M), summed,
// M the larger maximum. A row's result is the left fold of its groups in order, starting from the first group itself.

struct Factors {
	float maximum;
	float left;
	float right;
};

[[nodiscard]] constexpr auto factors(float left_maximum, float right_maximum) -> Factors {
	float const maximum = larger(left_maximum, right_maximum);
	float const left_gap = left_maximum - maximum;
	float const right_gap = right_maximum - maximum;
	return {.maximum = maximum, .left = math::exp(left_gap), .right = math::exp(right_gap)};
}

[[nodiscard]] constexpr auto combine(float left, float right, Factors const& weights) -> float {
	float const kept = left * weights.left;
	float const added = right * weights.right;
	return kept + added;
}

/** The gated output feature: (o / l) * sigmoid(gate), rounded to bf16 for o_proj. */
[[nodiscard]] constexpr auto finish(float output, float total, float gate) -> Bf16 {
	float const mean = output / total;
	float const opened = math::sigmoid(gate);
	return prims::round_to_bf16(mean * opened);
}

static_assert(model::attention_output_gate == model::Gate::Sigmoid, "the attention output gate is a sigmoid");

}

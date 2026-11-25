// SPDX-License-Identifier: AGPL-3.0-only
//
// The gated deltanet's arithmetic, one token at a time, as the checkpoint's model code defines it, operation for
// operation: the short causal convolution and its silu, the l2 normalization of queries and keys, the decay and beta
// gates, the delta rule over one column of the fp32 state, and the gated rms norm. Pure constexpr, compiled by the
// device and the host alike: the kernels, the host's fp32 emulation and the tests compute with these very definitions,
// and both sides build with contraction off, so every product here is rounded before the sum it enters.
//
// Where the model rounds to bf16 (its activations are bf16 tensors), so does this: the convolution's output and its
// silu, beta, the core's output before the norm, the normalized value and its product with the norm weight, and the
// gated output. Everything else is fp32: the state, the normalized queries and keys, the decay.
//
// Every 128-element sum (a column's projections onto k and q, the squares under each norm) is one fixed tree: `lanes`
// lanes each sum their consecutive `per_lane` products in order, then a butterfly at distances lanes / 2, ..., 1 adds
// each lane's partner to its own total. fp32 addition is commutative, so every lane ends with the same bits.
#pragma once

#include "model.h"
#include "prims/numeric.h"
#include "prims/transcendental.h"

namespace engine::device::deltanet {

using prims::Bf16;
using prims::round_to_bf16;
using prims::widen;

/** The length of every key, query and value vector, and so of a state column. */
inline constexpr int dim = 128;

static_assert(model::key_dim == dim && model::value_dim == dim, "one vector length for keys, queries, values and state");

/** One tree over a 128-element sum: `lanes` lanes, each holding `per_lane` consecutive elements. */
template<int lanes_>
struct Tree {
	static constexpr int lanes = lanes_;
	static constexpr int per_lane = dim / lanes;
	/** 16-byte chunks of fp32 a lane holds. */
	static constexpr int chunks = per_lane / 4;
	/** Columns one warp carries at once, one per group of `lanes` lanes. */
	static constexpr int columns_per_warp = 32 / lanes;

	static_assert(lanes >= 1 && lanes <= 32 && __builtin_popcount(static_cast<unsigned>(lanes)) == 1,
	              "a butterfly group is a power of two within a warp");
	static_assert(per_lane % 4 == 0, "a lane's elements are whole 16-byte chunks");
};

/**
 * The engine's tree: eight lanes per column, sixteen elements each. Shorter chains than a whole warp per column, and
 * four columns per warp keep a lane's share of the state at sixteen registers.
 */
using Column = Tree<8>;

/** The model's l2 normalization of queries and keys adds this under the square root. */
inline constexpr float l2_norm_eps = 1e-6F;

/** The model divides normalized queries by sqrt(key_dim), taken in fp32: 8 sqrt(2) rounded once. */
inline constexpr float query_divisor = 0x1.6a09e6p+3F;

static_assert(query_divisor * query_divisor > 127.99998F && query_divisor * query_divisor < 128.00002F);

/** fl(a[0] b[0]) + fl(a[1] b[1]) + ... in order: one lane's share of a dot product. */
template<int n>
[[nodiscard]] constexpr auto partial_dot(float const (&a)[n], float const (&b)[n]) -> float {
	float total = a[0] * b[0];
	for (int m = 1; m < n; ++m) {
		float const product = a[m] * b[m];
		total = total + product;
	}
	return total;
}

/** One butterfly step: a lane's total and its partner's. */
[[nodiscard]] constexpr auto combine(float own, float partner) -> float {
	return own + partner;
}

/** The tree's total of `lanes` partial sums: what every lane of the group holds after the butterfly. */
template<int lanes>
[[nodiscard]] constexpr auto tree_sum(float const (&partials)[lanes]) -> float {
	float values[lanes] = {};
#pragma unroll
	for (int lane = 0; lane < lanes; ++lane) {
		values[lane] = partials[lane];
	}
#pragma unroll
	for (unsigned distance = static_cast<unsigned>(lanes) / 2U; distance != 0; distance /= 2U) {
		float next[lanes] = {};
#pragma unroll
		for (unsigned lane = 0; lane < static_cast<unsigned>(lanes); ++lane) {
			next[lane] = combine(values[lane], values[lane ^ distance]);
		}
#pragma unroll
		for (int lane = 0; lane < lanes; ++lane) {
			values[lane] = next[lane];
		}
	}
	return values[0];
}

// ---------------------------------------------------------------------------------------------------------------------
// The convolution and the gates.

/** Four bf16 of one channel, oldest first: its convolution taps, or its four newest inputs. */
struct Four {
	Bf16 element[4];
};

/**
 * One channel's convolution output for the newest of four inputs, and its silu. The weight's tap t multiplies the
 * input t positions after the oldest. bf16 products are exact in fp32, so only the sum's order (oldest first) and the
 * two roundings to bf16 are choices. Taps and inputs come by value so that the device keeps them in registers.
 */
[[nodiscard]] constexpr auto convolve(Four taps, Four inputs) -> Bf16 {
	float total = widen(taps.element[0]) * widen(inputs.element[0]);
	for (int tap = 1; tap < 4; ++tap) {
		float const product = widen(taps.element[tap]) * widen(inputs.element[tap]);
		total = total + product;
	}
	Bf16 const convolved = round_to_bf16(total);
	return round_to_bf16(prims::math::silu(widen(convolved)));
}

static_assert(model::conv_kernel == 4, "the convolution reads its window of three and the new input");

/** What one token gives one value head's recurrence besides its vectors. */
struct Gates {
	/** e^g, g = -e^A_log softplus(a + dt_bias): how much of the state survives the token. */
	float decay;
	/** sigmoid(b), a bf16 activation widened: how much of the token's error the state takes in. */
	float beta;
};

[[nodiscard]] constexpr auto gates(Bf16 a, Bf16 b, Bf16 a_log, Bf16 dt_bias) -> Gates {
	float const shifted = widen(a) + widen(dt_bias);
	float const rate = prims::math::exp(widen(a_log));
	float const step = rate * prims::math::softplus(shifted);
	float const g = -step;
	return {.decay = prims::math::exp(g), .beta = widen(round_to_bf16(prims::math::sigmoid(widen(b))))};
}

// ---------------------------------------------------------------------------------------------------------------------
// The l2 normalization of queries and keys, by one lane of a tree.

/** A lane's share of the sum of squares of its widened bf16 elements. */
template<int n>
[[nodiscard]] constexpr auto partial_squares(float const (&x)[n]) -> float {
	return partial_dot(x, x);
}

/** 1 / sqrt(sum of squares + eps), the l2 norm's scale. */
[[nodiscard]] constexpr auto l2_scale(float sum_squares) -> float {
	float const shifted = sum_squares + l2_norm_eps;
	return prims::math::rsqrt(shifted);
}

/** Which vector of a token a normalization makes, which decides its last step. */
enum class Role : unsigned char {
	/** Normalized, then divided by sqrt(key_dim). */
	Query,
	/** Normalized. */
	Key,
};

template<int n>
constexpr void normalize(float (&x)[n], float scale, Role role) {
	for (int m = 0; m < n; ++m) {
		float const normalized = x[m] * scale;
		x[m] = role == Role::Query ? normalized / query_divisor : normalized;
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The delta rule over one state column, by one lane holding `n` of its rows. Per token, with d the decay:
//   S <- d S;  kv = S^T k;  delta = (v - kv) beta;  S <- S + k delta^T;  o = S^T q.
// In the model each product is its own tensor and each sum a reduction over the key dimension, so here each product
// rounds before its sum, and each reduction is the tree.

/** Decays the lane's rows and returns its share of the column's projection onto k. */
template<int n>
[[nodiscard]] constexpr auto decay_and_project(float (&state)[n], float const (&key)[n], float decay) -> float {
	for (int m = 0; m < n; ++m) {
		state[m] = state[m] * decay;
	}
	return partial_dot(state, key);
}

/** The column's error on the token's value, scaled by beta. */
[[nodiscard]] constexpr auto delta(float value, float projected, float beta) -> float {
	float const error = value - projected;
	return error * beta;
}

/** Adds k delta to the lane's rows. */
template<int n>
constexpr void update(float (&state)[n], float const (&key)[n], float column_delta) {
	for (int m = 0; m < n; ++m) {
		float const step = key[m] * column_delta;
		state[m] = state[m] + step;
	}
}

/** The lane's share of the column's output, its projection onto q. */
template<int n>
[[nodiscard]] constexpr auto read(float const (&state)[n], float const (&query)[n]) -> float {
	return partial_dot(state, query);
}

// ---------------------------------------------------------------------------------------------------------------------
// The gated rms norm over one head's output. The model rounds the core's output to bf16, normalizes it in fp32, rounds,
// multiplies by the stored weight (applied as w, not 1 + w) in bf16, and multiplies that by silu(z) in fp32.

static_assert(model::deltanet_norm_scale == model::NormScale::Weight, "the deltanet norm applies w itself");
static_assert(model::deltanet_output_gate == model::Gate::Silu, "the deltanet output gate is silu(z)");

/** 1 / sqrt(mean square + eps) over one head's 128 outputs, from the tree's sum of their squares. */
[[nodiscard]] constexpr auto norm_scale(float sum_squares) -> float {
	float const mean = sum_squares / static_cast<float>(dim);
	float const shifted = mean + model::rms_norm_eps;
	return prims::math::rsqrt(shifted);
}

/** One normalized output: the head's norm scale and the norm weight applied, rounded as the model rounds. */
[[nodiscard]] constexpr auto normalized(Bf16 output, float scale, Bf16 weight) -> Bf16 {
	Bf16 const scaled = round_to_bf16(widen(output) * scale);
	return round_to_bf16(widen(weight) * widen(scaled));
}

/** out_proj's input: the normalized output times silu(z). The output phase applies it while staging its input. */
[[nodiscard]] constexpr auto gated(Bf16 normalized_output, Bf16 z) -> Bf16 {
	return round_to_bf16(widen(normalized_output) * prims::math::silu(widen(z)));
}

}

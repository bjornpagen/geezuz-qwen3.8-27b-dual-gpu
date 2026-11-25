// SPDX-License-Identifier: AGPL-3.0-only
//
// The gated deltanet's arithmetic as the kernels lay it on their lanes. Every rounding point, sum order and fixed
// algorithm is the model description's (`model::deltanet`, `model::deltanet_vector`, the norms); this header only says
// which lane holds which elements, so the kernels, the host's emulation and the tests compute with the model's very
// definitions. Everything between in_proj_qkv's bf16 output and the gated output is fp32: the convolution and its
// silu, the normalized queries and keys, the gates, the state, the core's output and its norm.
//
// Every 128-element sum (a column's projections onto k and q, the squares under each norm) is the model's
// `deltanet_vector` tree: eight lanes each fold their sixteen consecutive products by fma, then a butterfly at
// distances 4, 2, 1 adds each lane's partner to its own total.
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

/** The engine's tree: the model's `deltanet_vector`. The probe also times a whole warp per column, never shipped. */
using Column = Tree<static_cast<int>(model::deltanet_vector.lanes)>;

static_assert(Column::lanes * Column::per_lane == static_cast<int>(model::deltanet_vector.width));

/** One lane's share of a dot product: the model's fold of its elements. */
template<int n>
[[nodiscard]] constexpr auto partial_dot(float const (&a)[n], float const (&b)[n]) -> float {
	return model::lane_fold(a, b, 0, n);
}

/** One butterfly step on the device: a lane's total and its partner's, as `model::butterfly` adds them. */
[[nodiscard]] constexpr auto combine(float own, float partner) -> float {
	return own + partner;
}

/** The tree's total of `lanes` partial sums: what every lane of the group holds after the butterfly. */
template<int lanes>
[[nodiscard]] constexpr auto tree_sum(float const (&partials)[lanes]) -> float {
	return model::butterfly<static_cast<model::Count>(lanes)>(partials);
}

// ---------------------------------------------------------------------------------------------------------------------
// The convolution and the gates.

/** Four bf16 of one channel, oldest first: its convolution taps, or its four newest inputs. */
struct Four {
	Bf16 element[4];
};

static_assert(model::conv_kernel == 4, "the convolution reads its window of three and the new input");

/** One channel's convolution output for the newest of four inputs, and its silu (`model::deltanet::convolved`). */
[[nodiscard]] constexpr auto convolve(Four taps, Four inputs) -> float {
	float const weights[4] = {widen(taps.element[0]), widen(taps.element[1]), widen(taps.element[2]), widen(taps.element[3])};
	float const values[4] = {widen(inputs.element[0]), widen(inputs.element[1]), widen(inputs.element[2]), widen(inputs.element[3])};
	return model::deltanet::convolved(weights, values);
}

/** What one token gives one value head's recurrence besides its vectors. */
struct Gates {
	/** `model::deltanet::decay`: how much of the state survives the token. */
	float decay;
	/** `model::deltanet::write_strength`: how much of the token's error the state takes in. */
	float beta;
};

/** The gates from in_proj_a's and in_proj_b's fp32 outputs and the head's stored A_log and dt_bias. */
[[nodiscard]] constexpr auto gates(float a, float b, Bf16 a_log, Bf16 dt_bias) -> Gates {
	return {.decay = model::deltanet::decay(a, widen(a_log), widen(dt_bias)), .beta = model::deltanet::write_strength(b)};
}

// ---------------------------------------------------------------------------------------------------------------------
// The l2 normalization of queries and keys, by one lane of a tree.

/** A lane's share of the sum of squares of its elements. */
template<int n>
[[nodiscard]] constexpr auto partial_squares(float const (&x)[n]) -> float {
	return partial_dot(x, x);
}

/** Which vector of a token a normalization makes, which decides its last step. */
enum class Role : unsigned char {
	/** Normalized, then divided by `model::deltanet::query_divisor`. */
	Query,
	/** Normalized. */
	Key,
};

/** The lane's elements normalized by the vector's l2 factor, as `role` finishes them. */
template<int n>
constexpr void normalize(float (&x)[n], float squares, Role role) {
	float const factor = model::deltanet::l2_factor(squares);
	for (int m = 0; m < n; ++m) {
		x[m] = role == Role::Query ? model::deltanet::query(x[m], factor) : model::deltanet::key(x[m], factor);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The delta rule over one state column, by one lane holding `n` of its rows (`model::Sum::DeltanetProducts`).

/** Decays the lane's rows and returns its share of the column's projection onto k. */
template<int n>
[[nodiscard]] constexpr auto decay_and_project(float (&state)[n], float const (&key)[n], float decay) -> float {
	for (int m = 0; m < n; ++m) {
		state[m] = model::deltanet::decayed(state[m], decay);
	}
	return partial_dot(state, key);
}

/** Adds k delta to the lane's rows, one fma each. */
template<int n>
constexpr void update(float (&state)[n], float const (&key)[n], float column_delta) {
	for (int m = 0; m < n; ++m) {
		state[m] = model::deltanet::updated(state[m], key[m], column_delta);
	}
}

/** The lane's share of the column's output, its projection onto q. */
template<int n>
[[nodiscard]] constexpr auto read(float const (&state)[n], float const (&query)[n]) -> float {
	return partial_dot(state, query);
}

using model::deltanet::delta;

// ---------------------------------------------------------------------------------------------------------------------
// The gated rms norm over one head's output (`model::Sum::ValueSquares`). The core phase leaves the normalized,
// weighted output in fp32; the output phase gates it by silu(z) and rounds once, as out_proj's input.

/** 1 / sqrt(mean square + eps) over one head's 128 outputs, from the tree's sum of their squares. */
[[nodiscard]] constexpr auto norm_scale(float squares) -> float {
	return model::rms_factor(squares, dim);
}

/** One normalized output: the head's norm factor and the norm weight applied (`model::deltanet::normed`). */
[[nodiscard]] constexpr auto normalized(float output, float scale, Bf16 weight) -> float {
	return model::deltanet::normed(output, scale, widen(weight));
}

/** out_proj's input: the normalized output times silu(z), rounded once. The output phase applies it while staging. */
[[nodiscard]] constexpr auto gated(float normed_output, float z) -> Bf16 {
	return round_to_bf16(model::deltanet::gated(normed_output, z));
}

static_assert(model::precision(model::Value::DeltanetNormed) == model::Precision::Fp32);
static_assert(model::precision(model::Value::DeltanetGated) == model::Precision::Bf16);

}

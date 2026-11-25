// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's arithmetic as the kernels lay it on their lanes: every rounding point, sum order and fixed algorithm is
// the model description's (`model::attention`, `model::rope`, the norms), and this header only says which lane holds
// what. A row is one warp, lane L holding features [8L, 8L + 8) (`model::attention_head`); the rope's sines and cosines
// come from a table the device fills with `model::rope::rotation`; the softmax weights are split by
// `model::attention::split_weight`. The kernels and the host emulation compile this header, both with contraction off.
#pragma once

#include "attn/geometry.h"
#include "model.h"
#include "prims/numeric.h"
#include "prims/transcendental.h"

namespace engine::device::attn {

using prims::Bf16;
namespace math = prims::math;

// ---------------------------------------------------------------------------------------------------------------------
// A row in a warp.

inline constexpr int lane_features = static_cast<int>(features_per_chunk);

static_assert(model::attention_head.lanes == 32 && model::per_lane(model::attention_head) == features_per_chunk,
              "one warp holds one row, one 16-byte chunk per lane");

/** One lane's share of a row's sum of squares: its eight features, folded as `model::attention_head` says. */
[[nodiscard]] constexpr auto lane_squares(float const (&values)[lane_features]) -> float {
	return model::lane_fold(values, values, 0, lane_features);
}

/** q_norm's or k_norm's factor from the row's sum of squares. */
[[nodiscard]] constexpr auto inverse_rms(float squares) -> float {
	return model::rms_factor(squares, head_dim);
}

[[nodiscard]] constexpr auto normalized(float value, float inverse, Bf16 weight) -> float {
	return model::one_plus_scaled(value, inverse, prims::widen(weight));
}

// ---------------------------------------------------------------------------------------------------------------------
// The rope table: `model::rope::rotation` of every (position, pair), [position][pair] as {cosine, sine}. Filled once on
// the device by `attn_rope_table`, which evaluates the very function in ieee double; it costs 64 MiB of each card's
// memory and no arithmetic per row, where evaluating the double routine per row would stall every job's preparation.

inline constexpr int rotary_pairs = static_cast<int>(model::rope::pairs);

static_assert(rotary_pairs == 32 && model::rotary_dim == 2 * features_per_chunk * 4, "the rotated features are lanes 0-7");
static_assert(model::max_positions <= (1U << 24U), "every position is exact in fp32");

using model::rope::Rotation;

static_assert(sizeof(Rotation) == 8);

inline constexpr Count rope_table_bytes = model::max_positions * model::rope::pairs * sizeof(Rotation);

/** The byte offset of (`position`, `pair`)'s rotation in the table. */
[[nodiscard]] constexpr auto rope_offset(Count position, Count pair) -> Count {
	return (position * model::rope::pairs + pair) * sizeof(Rotation);
}

// ---------------------------------------------------------------------------------------------------------------------
// The weights' bf16 parts.

/** How many bf16 parts carry P into the PV mma: three in the engine. One exists only to measure what the split costs. */
enum class PvParts : unsigned char { Three, One };

[[nodiscard]] constexpr auto part_count(PvParts parts) -> int {
	return parts == PvParts::Three ? 3 : 1;
}

/** A scaled weight as bf16 parts, largest first. */
struct Parts {
	Bf16 part[3];
};

/** P as `parts` parts: the model's exact split, or (measurement only) P rounded once. */
[[nodiscard]] constexpr auto split(float scaled, PvParts parts) -> Parts {
	if (parts == PvParts::One) {
		return {.part = {prims::round_to_bf16(scaled), Bf16{.bits = 0}, Bf16{.bits = 0}}};
	}
	model::attention::WeightParts const exact = model::attention::split_weight(scaled);
	return {.part = {Bf16{.bits = exact.high}, Bf16{.bits = exact.middle}, Bf16{.bits = exact.low}}};
}

/** The gated output feature, rounded once for o_proj. */
[[nodiscard]] constexpr auto finish(float output, float total, float gate) -> Bf16 {
	return prims::round_to_bf16(model::attention::output(output, total, gate));
}

static_assert(model::precision(model::Value::AttentionGated) == model::Precision::Bf16);
static_assert(model::precision(model::Value::Query) == model::Precision::Bf16);
static_assert(model::precision(model::Value::KeyValue) == model::Precision::Bf16);

}

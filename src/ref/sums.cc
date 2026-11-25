// SPDX-License-Identifier: AGPL-3.0-only
//
// The model description's fixed summation orders (`model::Sum`) and the norms on them, generic over their widths so a
// small description exercises the very code the model runs. Every order and rounding point here is the model's own
// function; this partition only names which tree each width takes.
export module engine.ref:sums;

import std;
import engine.model;
import engine.prims;
import :numeric;

namespace engine::ref {

namespace model = engine::model;

using model::Count;

/** Lanes of a whole warp: a hidden tile's and an attention head's tree. */
export inline constexpr Count lanes = model::attention_head.lanes;

/** Lanes of a deltanet vector's tree. */
export inline constexpr Count deltanet_lanes = model::deltanet_vector.lanes;

static_assert(model::hidden_tile.lanes == lanes);

/** The warp butterfly: at distance 16, 8, 4, 2, 1 every lane adds its partner's value; every lane ends equal. */
export [[nodiscard]] constexpr auto butterfly(std::array<float, lanes> const& values) -> float {
	return model::butterfly<lanes>(values);
}

/** sum_i x_i y_i on `Lanes` lanes: each lane folds its Width / Lanes consecutive products by fma from +0, then the butterfly. */
export template<Count Lanes, Count Width>
    requires(Width % Lanes == 0)
[[nodiscard]] constexpr auto lane_dot(std::span<float const, Width> x, std::span<float const, Width> y) -> float {
	return model::lane_sum<model::Lanes{.lanes = Lanes, .width = Width}>(x, y, 0);
}

export template<Count Lanes, Count Width>
[[nodiscard]] constexpr auto lane_squares(std::span<float const, Width> x) -> float {
	return lane_dot<Lanes, Width>(x, x);
}

/** An rms norm's sum of squares over a hidden-sized row: each 128-feature tile under `model::hidden_tile`, folded in order. */
export template<Count Width>
    requires(Width % model::hidden_tile.width == 0)
[[nodiscard]] constexpr auto tiled_squares(std::span<float const, Width> x) -> float {
	auto total = model::lane_sum<model::hidden_tile>(x, x, 0);
	for (auto tile = Count{1}; tile < Width / model::hidden_tile.width; ++tile) {
		total = total + model::lane_sum<model::hidden_tile>(x, x, tile * model::hidden_tile.width);
	}
	return total;
}

static_assert(model::hidden % model::hidden_tile.width == 0);

/** An rms norm over the hidden size with a zero-centred weight, rounded once: `Sum::HiddenSquares`. */
export template<Count Width>
[[nodiscard]] auto hidden_norm(std::span<float const, Width> x, std::span<Bf16 const, Width> weight) -> std::vector<Bf16> {
	auto const factor = model::rms_factor(tiled_squares<Width>(x), Width);
	auto out = std::vector<Bf16>(Width);
	for (auto index = Count{0}; index < Width; ++index) {
		out[index] = round(model::one_plus_scaled(x[index], factor, widen(weight[index])));
	}
	return out;
}

/** q_norm or k_norm over one head, in fp32 (rope follows before the rounding): `Sum::HeadSquares`. */
export template<Count Width>
[[nodiscard]] auto head_norm(std::span<float const, Width> x, std::span<Bf16 const, Width> weight) -> std::array<float, Width> {
	auto const factor = model::rms_factor(lane_squares<lanes, Width>(x), Width);
	auto out = std::array<float, Width>{};
	for (auto index = Count{0}; index < Width; ++index) {
		out.at(index) = model::one_plus_scaled(x[index], factor, widen(weight[index]));
	}
	return out;
}

/** Which deltanet vector an l2 norm makes, which decides its last step. */
export enum class Role : unsigned char {
	Query,
	Key,
};

/** The deltanet l2 norm, the query then divided by its divisor: `Sum::KeySquares`. */
export template<Count Width>
[[nodiscard]] auto l2_normalized(std::span<float const, Width> x, Role role) -> std::array<float, Width> {
	auto const factor = model::deltanet::l2_factor(lane_squares<deltanet_lanes, Width>(x));
	auto out = std::array<float, Width>{};
	for (auto index = Count{0}; index < Width; ++index) {
		out.at(index) = role == Role::Query ? model::deltanet::query(x[index], factor) : model::deltanet::key(x[index], factor);
	}
	return out;
}

/** The deltanet output norm, gated by silu(z) and rounded once: `Sum::ValueSquares`, then `model::deltanet::gated`. */
export template<Count Width>
[[nodiscard]] auto gated_norm(std::span<float const, Width> x, std::span<Bf16 const, Width> weight, std::span<float const, Width> gate)
    -> std::array<Bf16, Width> {
	auto const factor = model::rms_factor(lane_squares<deltanet_lanes, Width>(x), Width);
	auto out = std::array<Bf16, Width>{};
	for (auto index = Count{0}; index < Width; ++index) {
		out.at(index) = round(model::deltanet::gated(model::deltanet::normed(x[index], factor, widen(weight[index])), gate[index]));
	}
	return out;
}

}

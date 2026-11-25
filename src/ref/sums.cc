// SPDX-License-Identifier: AGPL-3.0-only
//
// The fixed summation orders of the model description's `Sum`, and the norms built on them, generic over their
// widths so a small description exercises the very code the model runs.
export module engine.ref:sums;

import std;
import engine.model;
import engine.prims;
import :numeric;

namespace engine::ref {

namespace math = engine::device::prims::math;
namespace model = engine::model;

using model::Count;

export inline constexpr Count lanes = 32;

/** The warp butterfly: at distance 16, 8, 4, 2, 1 every lane adds its partner's value; every lane ends equal. */
export [[nodiscard]] constexpr auto butterfly(std::array<float, lanes> values) -> float {
	for (auto distance = lanes / 2; distance != 0; distance /= 2) {
		auto next = values;
		for (auto lane = Count{0}; lane < lanes; ++lane) {
			next.at(lane) = values.at(lane) + values.at(lane ^ distance);
		}
		values = next;
	}
	return values[0];
}

/** "Lanes of w" for sum_i x_i y_i: lane l folds its w / 32 consecutive products by fma from +0, then the butterfly. */
export template<Count Width>
    requires(Width % lanes == 0)
[[nodiscard]] constexpr auto lane_dot(std::span<float const, Width> x, std::span<float const, Width> y) -> float {
	constexpr auto per_lane = Width / lanes;
	auto partials = std::array<float, lanes>{};
	for (auto lane = Count{0}; lane < lanes; ++lane) {
		auto partial = 0.0F;
		for (auto term = lane * per_lane; term < (lane + 1) * per_lane; ++term) {
			partial = std::fma(x[term], y[term], partial);
		}
		partials.at(lane) = partial;
	}
	return butterfly(partials);
}

export template<Count Width>
[[nodiscard]] constexpr auto lane_squares(std::span<float const, Width> x) -> float {
	return lane_dot<Width>(x, x);
}

/** An rms norm's sum of squares over the hidden size: lanes of 128 per 128-feature tile, the tiles folded in order. */
export template<Count Width>
    requires(Width % 128 == 0)
[[nodiscard]] constexpr auto tiled_squares(std::span<float const, Width> x) -> float {
	auto total = lane_squares<128>(x.template first<128>());
	for (auto tile = Count{1}; tile < Width / 128; ++tile) {
		total = total + lane_squares<128>(std::span<float const, 128>{x.subspan(tile * 128, 128)});
	}
	return total;
}

/** rsqrt(sum / width + eps): the factor an rms norm scales by. */
export template<Count Width>
[[nodiscard]] constexpr auto rms_factor(float sum) -> float {
	auto const mean = sum / static_cast<float>(Width);
	return math::rsqrt(mean + model::rms_norm_eps);
}

/** x * factor * (1 + w), rounded once: a zero-centred rms norm's bf16 output. */
export template<Count Width>
[[nodiscard]] auto one_plus_scaled(std::span<float const, Width> x, float factor, std::span<Bf16 const, Width> weight) -> std::vector<Bf16> {
	auto out = std::vector<Bf16>(Width);
	for (auto index = Count{0}; index < Width; ++index) {
		auto const scale = 1.0F + widen(weight[index]);
		out[index] = round((x[index] * factor) * scale);
	}
	return out;
}

/** An rms norm over the hidden size with a zero-centred weight: `Sum::HiddenSquares`. */
export template<Count Width>
[[nodiscard]] auto hidden_norm(std::span<float const, Width> x, std::span<Bf16 const, Width> weight) -> std::vector<Bf16> {
	return one_plus_scaled<Width>(x, rms_factor<Width>(tiled_squares<Width>(x)), weight);
}

/** q_norm or k_norm over one head, in fp32 (rope follows before the rounding): `Sum::HeadSquares`. */
export template<Count Width>
[[nodiscard]] auto head_norm(std::span<float const, Width> x, std::span<Bf16 const, Width> weight) -> std::array<float, Width> {
	auto const factor = rms_factor<Width>(lane_squares<Width>(x));
	auto out = std::array<float, Width>{};
	for (auto index = Count{0}; index < Width; ++index) {
		out.at(index) = (x[index] * factor) * (1.0F + widen(weight[index]));
	}
	return out;
}

/** The deltanet l2 norm, then `scale` (1 for a key, exactly a no-op): `Sum::KeySquares`. */
export template<Count Width>
[[nodiscard]] auto l2_normalized(std::span<float const, Width> x, float scale) -> std::array<float, Width> {
	auto const factor = math::rsqrt(lane_squares<Width>(x) + model::l2_norm_eps);
	auto out = std::array<float, Width>{};
	for (auto index = Count{0}; index < Width; ++index) {
		out.at(index) = (x[index] * factor) * scale;
	}
	return out;
}

/** The deltanet output norm, gated: ((x * factor) * w) * silu(z), rounded once: `Sum::ValueSquares`. */
export template<Count Width>
[[nodiscard]] auto gated_norm(std::span<float const, Width> x, std::span<Bf16 const, Width> weight, std::span<float const, Width> gate)
    -> std::array<Bf16, Width> {
	auto const factor = rms_factor<Width>(lane_squares<Width>(x));
	auto out = std::array<Bf16, Width>{};
	for (auto index = Count{0}; index < Width; ++index) {
		out.at(index) = round(((x[index] * factor) * widen(weight[index])) * math::silu(gate[index]));
	}
	return out;
}

}

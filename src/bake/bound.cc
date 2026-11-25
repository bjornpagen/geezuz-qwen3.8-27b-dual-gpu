// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention carries the softmax weights into the PV product losslessly by scaling P by 2^40 and splitting it into three
// bf16 parts; the fp32 accumulators then hold 2^40 * sum_t P_t v_t, at most 2^40 * max |v| because the P_t sum to one.
// That stays far below fp32's 2^128 at any context length exactly when |v| is bounded, so the bake proves a bound from
// the weights alone, for every input:
//
//   v_j = sum_i W_ji y_i with y_i = (1 + w_i) x_i, where x is the rms-normalized input and w the norm's stored weight.
//   rms normalization gives sum_i x_i^2 = n (n = hidden; the epsilon only shrinks it), so |x_i| <= sqrt(n) for each i;
//   hence |v_j| <= sqrt(n) * sum_i |W_ji| |1 + w_i|, and B = sqrt(n) * max_j sum_i |W_ji| |1 + w_i| bounds every |v|.
//   W_ji is the e4m3 value times its 128 x 128 block's scale, both exact.
//
// B is evaluated in double with every operation rounded up (round to nearest, then one ulp toward +infinity), so the
// computed B is never below the true one. The bake refuses B >= 2^70. The device rounds y to bf16, accumulates v in
// fp32 and stores it in bf16; each step changes a magnitude by a relative 2^-7 at most, so the device's |v| stays
// below 2^71 and its accumulators below 2^111.
export module engine.bake:bound;

import std;
import engine.core;
import engine.model;
import engine.prims;
import engine.schedule;
import :error;
import :plan;
import :source;

namespace engine::bake {

/** The proven bound on |v| over every attention layer and input, and the v_proj that attains it. */
export struct Bound {
	double value;
	std::string tensor;
};

/** The refusal threshold: below it the scaled accumulators cannot overflow. */
export inline constexpr double overflow_limit = 0x1p70;

/** `value` rounded to nearest, moved one ulp up: never below the exact result it came from. */
[[nodiscard]] auto up(double value) -> double {
	return std::nextafter(value, std::numeric_limits<double>::infinity());
}

[[nodiscard]] auto bf16_at(std::span<std::byte const> bytes, std::size_t index) -> double {
	auto const bits = std::to_integer<std::uint32_t>(bytes[2 * index]) | (std::to_integer<std::uint32_t>(bytes[(2 * index) + 1]) << 8U);
	return static_cast<double>(std::bit_cast<float>(bits << 16U));
}

/** |the factor the norm applies for stored weight `weight`|, rounded up. */
[[nodiscard]] auto gain(double weight) -> double {
	switch (model::rms_norm_scale) {
	case model::NormScale::OnePlusWeight:
		return up(std::fabs(1.0 + weight));
	case model::NormScale::Weight:
		return std::fabs(weight);
	}
	std::unreachable();
}

/** sum_i |W_ji| |1 + w_i| rounded up, maximized over the rows j of one v_proj. */
[[nodiscard]] auto widest_row(std::string const& name, std::span<std::byte const> codes, std::span<std::byte const> scales,
                              std::span<double const> gains, model::Shape const& shape) -> core::Result<double, BakeError> {
	auto const rows = shape.extents[0];
	auto const columns = shape.extents[1];
	auto const block = model::scale_block;
	auto const scale_columns = model::divide_up(columns, block);
	auto widest = 0.0;
	for (auto row = std::size_t{0}; row < rows; ++row) {
		auto total = 0.0;
		for (auto first = std::size_t{0}; first < columns; first += block) {
			auto sum = 0.0;
			for (auto column = first; column < std::min(first + block, columns); ++column) {
				auto const code = prims::FiniteE4m3::parse(std::to_integer<std::uint8_t>(codes[(row * columns) + column]));
				if (!code) {
					return refuse(NanCode{.tensor = name, .row = row, .column = column});
				}
				auto const magnitude = std::fabs(static_cast<double>(device::prims::value(code->code())));
				sum = up(sum + up(magnitude * gains[column]));
			}
			auto const scale = std::fabs(bf16_at(scales, ((row / block) * scale_columns) + (first / block)));
			total = up(total + up(sum * scale));
		}
		widest = std::max(widest, total);
	}
	return widest;
}

/** The bound over every attention layer `P` places; a bound at or past the limit is refused. */
export template<Plan P>
[[nodiscard]] auto attention_bound(Source const& source) -> core::Result<Bound, BakeError> {
	auto const& v = std::span{P::inventory}[P::v_proj];
	auto const& norm = std::span{P::inventory}[P::input_norm];
	auto const hidden = v.shape.extents[1];
	core::invariant(norm.shape.rank == 1 && norm.shape.extents[0] == hidden);
	auto const root = up(std::sqrt(static_cast<double>(hidden)));
	auto bound = Bound{.value = 0.0, .tensor = {}};
	// Every attention layer's v_proj is placed once per card, as tiles; card zero's placements name each layer once.
	for (auto const& placement : P::placements(model::Card::Zero)) {
		if (placement.read.part != schedule::Part::Tiles || placement.read.tensor[0] != P::v_proj) {
			continue;
		}
		auto const name = checkpoint_name(v, placement.instance);
		auto const weights = source.bytes(checkpoint_name(norm, placement.instance));
		auto gains = std::vector<double>(hidden);
		for (auto index = std::size_t{0}; index < hidden; ++index) {
			gains[index] = gain(bf16_at(weights, index));
		}
		auto const widest = widest_row(name, source.bytes(name), source.bytes(scale_name(name)), gains, v.shape);
		if (!widest) {
			return std::unexpected(widest.error());
		}
		auto const layer = up(root * *widest);
		if (layer >= bound.value) {
			bound = Bound{.value = layer, .tensor = name};
		}
	}
	if (bound.value >= overflow_limit) {
		return refuse(AttentionOverflow{.bound = bound.value, .tensor = bound.tensor});
	}
	return bound;
}

}

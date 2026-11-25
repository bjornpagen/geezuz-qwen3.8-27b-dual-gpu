// SPDX-License-Identifier: AGPL-3.0-only
//
// One gated deltanet layer's token step, from its projections to out_proj's input, generic over the layer's shape so
// a small shape runs the very code the model does.
export module engine.ref:deltanet;

import std;
import engine.core;
import engine.model;
import engine.prims;
import :numeric;
import :sums;

namespace engine::ref {

namespace model = engine::model;

using model::Count;

/** The dimensions a deltanet step reads. */
export template<class D>
concept DeltanetShape = requires {
	{ D::key_heads } -> std::convertible_to<Count>;
	{ D::value_heads } -> std::convertible_to<Count>;
	{ D::key_dim } -> std::convertible_to<Count>;
	{ D::value_dim } -> std::convertible_to<Count>;
	{ D::conv_kernel } -> std::convertible_to<Count>;
} && D::value_heads % D::key_heads == 0 && D::key_dim % 32 == 0 && D::value_dim % 32 == 0;

export template<DeltanetShape D>
inline constexpr Count conv_channels = (2 * D::key_heads * D::key_dim) + (D::value_heads * D::value_dim);

/**
 * One slot's state of one deltanet layer: the fp32 recurrent state, value head by value head, each column j of the
 * key_dim x value_dim matrix contiguous (the order the recurrence reads it), and the convolution's window of the last
 * conv_kernel - 1 bf16 inputs, oldest first. Both start at zero.
 */
export template<DeltanetShape D>
struct DeltanetState {
	std::vector<float> state = std::vector<float>(D::value_heads * D::value_dim * D::key_dim);
	std::vector<Bf16> window = std::vector<Bf16>((D::conv_kernel - 1) * conv_channels<D>);
};

/** A deltanet layer's own small weights, as stored (bf16). */
export struct DeltanetWeights {
	/** [channels][1][conv_kernel]. */
	std::span<Bf16 const> conv;
	std::span<Bf16 const> a_log;
	std::span<Bf16 const> dt_bias;
	std::span<Bf16 const> norm;
};

/** One token's projections into the layer. */
export struct DeltanetInputs {
	/** in_proj_qkv's output, rounded: q | k | v channels. */
	std::span<Bf16 const> projection;
	/** in_proj_z's output. */
	std::span<float const> gate;
	/** in_proj_a's and in_proj_b's outputs, one per value head. */
	std::span<float const> a;
	std::span<float const> b;
};

/**
 * The causal convolution of one token (`model::Value::DeltanetConvolved`): per channel, the window's conv_kernel - 1
 * inputs, oldest first, then the token's own, by `model::deltanet::convolved`.
 */
export template<DeltanetShape D>
[[nodiscard]] auto convolution(std::span<Bf16 const> window, std::span<Bf16 const> conv, std::span<Bf16 const> projection)
    -> std::vector<float> {
	constexpr auto channels = conv_channels<D>;
	constexpr auto taps = D::conv_kernel;
	core::invariant(window.size() == (taps - 1) * channels && conv.size() == channels * taps && projection.size() == channels);
	auto convolved = std::vector<float>(channels);
	for (auto channel = Count{0}; channel < channels; ++channel) {
		auto tap_weights = std::array<float, taps>{};
		auto inputs = std::array<float, taps>{};
		for (auto tap = Count{0}; tap < taps; ++tap) {
			tap_weights.at(tap) = widen(conv[(channel * taps) + tap]);
			inputs.at(tap) = widen(tap + 1 < taps ? window[(tap * channels) + channel] : projection[channel]);
		}
		convolved[channel] = model::deltanet::convolved(tap_weights, inputs);
	}
	return convolved;
}

/**
 * Advances `state` by one token and returns out_proj's input (`model::Value::DeltanetGated`): the causal convolution,
 * the l2 norms, the gates, the recurrence column by column and the gated norm, in the model description's order and on
 * its trees (`model::deltanet_vector` for every 128-term sum).
 */
export template<DeltanetShape D>
[[nodiscard]] auto deltanet_step(DeltanetState<D>& state, DeltanetWeights const& weights, DeltanetInputs const& in) -> std::vector<Bf16> {
	constexpr auto channels = conv_channels<D>;
	constexpr auto taps = D::conv_kernel;
	constexpr auto key_features = D::key_heads * D::key_dim;
	core::invariant(in.projection.size() == channels && weights.conv.size() == channels * taps);
	auto const convolved = convolution<D>(state.window, weights.conv, in.projection);
	std::ranges::copy(std::span{state.window}.subspan(channels), state.window.begin());
	std::ranges::copy(in.projection, std::span{state.window}.subspan((taps - 2) * channels).begin());

	auto queries = std::vector<std::array<float, D::key_dim>>{};
	auto keys = std::vector<std::array<float, D::key_dim>>{};
	for (auto head = Count{0}; head < D::key_heads; ++head) {
		auto const query = std::span<float const, D::key_dim>{std::span{convolved}.subspan(head * D::key_dim, D::key_dim)};
		auto const key = std::span<float const, D::key_dim>{std::span{convolved}.subspan(key_features + (head * D::key_dim), D::key_dim)};
		queries.push_back(l2_normalized<D::key_dim>(query, Role::Query));
		keys.push_back(l2_normalized<D::key_dim>(key, Role::Key));
	}

	auto out = std::vector<Bf16>{};
	out.reserve(D::value_heads * D::value_dim);
	constexpr auto group = D::value_heads / D::key_heads;
	for (auto head = Count{0}; head < D::value_heads; ++head) {
		auto const& query = queries[head / group];
		auto const& key = keys[head / group];
		auto const values = std::span{convolved}.subspan((2 * key_features) + (head * D::value_dim), D::value_dim);
		auto const strength = model::deltanet::write_strength(in.b[head]);
		auto const decay = model::deltanet::decay(in.a[head], widen(weights.a_log[head]), widen(weights.dt_bias[head]));
		auto core = std::array<float, D::value_dim>{};
		for (auto column = Count{0}; column < D::value_dim; ++column) {
			auto const held = std::span{state.state}.subspan(((head * D::value_dim) + column) * D::key_dim, D::key_dim);
			for (auto& element : held) {
				element = model::deltanet::decayed(element, decay);
			}
			auto const memory = lane_dot<deltanet_lanes, D::key_dim>(std::span<float const, D::key_dim>{held}, key);
			auto const delta = model::deltanet::delta(values[column], memory, strength);
			for (auto const [element, k] : std::views::zip(held, key)) {
				element = model::deltanet::updated(element, k, delta);
			}
			core.at(column) = lane_dot<deltanet_lanes, D::key_dim>(std::span<float const, D::key_dim>{held}, query);
		}
		auto const gate = std::span<float const, D::value_dim>{in.gate.subspan(head * D::value_dim, D::value_dim)};
		auto const norm = std::span<Bf16 const, D::value_dim>{weights.norm.first(D::value_dim)};
		auto const gated = gated_norm<D::value_dim>(core, norm, gate);
		out.insert(out.end(), gated.begin(), gated.end());
	}
	return out;
}

}

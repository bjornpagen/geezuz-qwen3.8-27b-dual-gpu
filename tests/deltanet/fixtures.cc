// SPDX-License-Identifier: AGPL-3.0-only
//
// Generated inputs for the deltanet tests: a layer's side data, a slot's memory of one layer, and token rows as the
// projections leave them, at the magnitudes the checkpoint's activations and parameters take. The generators keep
// only the shapes; every value is drawn, so nothing the kernels must handle is left out by construction.
export module engine.deltanet_fixtures;

import std;
import engine.deltanet;

namespace engine::deltanet_fixtures {

namespace dn = engine::deltanet;

export using Generator = std::mt19937_64;

export [[nodiscard]] auto normal(Generator& generator, float deviation) -> float {
	return std::normal_distribution<float>{0.0F, deviation}(generator);
}

export [[nodiscard]] auto uniform(Generator& generator, float low, float high) -> float {
	return std::uniform_real_distribution<float>{low, high}(generator);
}

export [[nodiscard]] auto bf16(float value) -> dn::Bf16 {
	return dn::round_to_bf16(value);
}

/** One deltanet layer's side data, owned. */
export struct WeightsStore {
	std::vector<dn::Bf16> conv;
	std::vector<dn::Bf16> a_log;
	std::vector<dn::Bf16> dt_bias;
	std::vector<dn::Bf16> norm;

	[[nodiscard]] auto view() const -> dn::LayerWeights {
		return {.conv = conv, .a_log = a_log, .dt_bias = dt_bias, .norm = norm};
	}
};

export [[nodiscard]] auto random_weights(Generator& generator) -> WeightsStore {
	auto store = WeightsStore{};
	for (auto index = 0; index < dn::channels * 4; ++index) {
		store.conv.push_back(bf16(normal(generator, 0.4F)));
	}
	for (auto head = 0; head < dn::value_heads; ++head) {
		store.a_log.push_back(bf16(std::log(uniform(generator, 0.5F, 16.0F))));
		store.dt_bias.push_back(bf16(uniform(generator, -3.0F, 3.0F)));
	}
	for (auto feature = 0; feature < dn::dim; ++feature) {
		store.norm.push_back(bf16(uniform(generator, 0.6F, 1.4F)));
	}
	return store;
}

/** A slot's memory of one layer: a state that has seen tokens, a window of activations, a ring of leftovers. */
export [[nodiscard]] auto random_layer(Generator& generator) -> dn::LayerBuffers {
	auto buffers = dn::LayerBuffers{};
	std::ranges::generate(buffers.state, [&] {
		return normal(generator, 0.05F);
	});
	std::ranges::generate(buffers.window, [&] {
		return bf16(normal(generator, 1.0F));
	});
	std::ranges::generate(buffers.ring, [&] {
		return bf16(normal(generator, 1.0F));
	});
	return buffers;
}

/** One token's projections: in_proj_qkv's channels, and a | b per value head. */
export struct Token {
	std::vector<dn::Bf16> qkv;
	std::vector<std::uint32_t> gates;
};

export [[nodiscard]] auto random_token(Generator& generator) -> Token {
	auto token = Token{};
	for (auto channel = 0; channel < dn::channels; ++channel) {
		token.qkv.push_back(bf16(normal(generator, 1.0F)));
	}
	for (auto head = 0; head < dn::value_heads; ++head) {
		auto const a = bf16(normal(generator, 1.5F));
		auto const b = bf16(normal(generator, 2.0F));
		token.gates.push_back(std::uint32_t{a.bits} | (std::uint32_t{b.bits} << 16U));
	}
	return token;
}

/** Activation rows, owned: tokens appended in row order, and room for their outputs. */
export struct RowsStore {
	std::vector<dn::Bf16> qkv;
	std::vector<std::uint32_t> gates;
	std::vector<dn::Bf16> outputs;

	void push(Token const& token) {
		qkv.insert(qkv.end(), token.qkv.begin(), token.qkv.end());
		gates.insert(gates.end(), token.gates.begin(), token.gates.end());
		outputs.resize(outputs.size() + static_cast<std::size_t>(dn::value_features));
	}

	[[nodiscard]] auto view() -> dn::Activations {
		return {.qkv = qkv, .gates = gates, .outputs = outputs};
	}

	/** Row `row`'s outputs. */
	[[nodiscard]] auto output(std::size_t row) const -> std::span<dn::Bf16 const> {
		return std::span{outputs}.subspan(row * static_cast<std::size_t>(dn::value_features), static_cast<std::size_t>(dn::value_features));
	}
};

export [[nodiscard]] auto same_bits(std::span<dn::Bf16 const> left, std::span<dn::Bf16 const> right) -> bool {
	return std::ranges::equal(left, right, [](dn::Bf16 a, dn::Bf16 b) {
		return a.bits == b.bits;
	});
}

/** Whether two slots' memories agree in their stored state and windows; the ring is scratch. */
export [[nodiscard]] auto same_stored(dn::LayerBuffers const& left, dn::LayerBuffers const& right) -> bool {
	auto const floats = std::ranges::equal(left.state, right.state, [](float a, float b) {
		return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
	});
	return floats && same_bits(left.window, right.window);
}

}

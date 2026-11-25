// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet kernels against the cpu reference, bit for bit, at one card's real layer shape: the host emulation of
// the decode core and the prefill (the device's lanes, its eight-lane trees, its layouts of state, window and replay
// ring) must equal the reference's token step on the same projections: the convolution, every row's gated output
// (the emulation's normalized output gated as the output phase stages it), the committed state and the window. Decode
// runs verify cycles of one to five rows with drafts accepted and replayed; prefill runs a prompt chunk.
import std;
import engine.deltanet;
import engine.model;
import engine.prims;
import engine.ref;
import engine.testing;

namespace {

namespace dn = engine::deltanet;
namespace model = engine::model;
namespace prims = engine::device::prims;
namespace ref = engine::ref;

using model::Count;

/** One card's deltanet layer: its key heads and their value heads. */
struct Card {
	static constexpr Count key_heads = dn::key_heads;
	static constexpr Count value_heads = dn::value_heads;
	static constexpr Count key_dim = model::key_dim;
	static constexpr Count value_dim = model::value_dim;
	static constexpr Count conv_kernel = model::conv_kernel;
};

static_assert(ref::conv_channels<Card> == dn::channels, "the card's channels are the reference's q | k | v");

[[nodiscard]] auto index(int value) -> std::size_t {
	return static_cast<std::size_t>(value);
}

/** One token's projections: in_proj_qkv's bf16 channels, a and b per value head, z per value feature. */
struct Token {
	std::vector<prims::Bf16> qkv;
	std::vector<float> a;
	std::vector<float> b;
	std::vector<float> z;
};

struct Layer {
	std::vector<prims::Bf16> conv;
	std::vector<prims::Bf16> a_log;
	std::vector<prims::Bf16> dt_bias;
	std::vector<prims::Bf16> norm;
};

[[nodiscard]] auto normal(std::mt19937_64& random, float deviation) -> float {
	return std::normal_distribution<float>{0.0F, deviation}(random);
}

[[nodiscard]] auto random_layer(std::mt19937_64& random) -> Layer {
	auto layer = Layer{};
	for (auto index = 0; index < dn::channels * 4; ++index) {
		layer.conv.push_back(prims::round_to_bf16(normal(random, 0.4F)));
	}
	for (auto head = 0; head < dn::value_heads; ++head) {
		layer.a_log.push_back(prims::round_to_bf16(std::log(std::uniform_real_distribution<float>{0.5F, 16.0F}(random))));
		layer.dt_bias.push_back(prims::round_to_bf16(std::uniform_real_distribution<float>{-3.0F, 3.0F}(random)));
	}
	for (auto feature = 0; feature < dn::dim; ++feature) {
		layer.norm.push_back(prims::round_to_bf16(std::uniform_real_distribution<float>{0.6F, 1.4F}(random)));
	}
	return layer;
}

[[nodiscard]] auto random_token(std::mt19937_64& random) -> Token {
	auto token = Token{};
	for (auto channel = 0; channel < dn::channels; ++channel) {
		token.qkv.push_back(prims::round_to_bf16(normal(random, 1.0F)));
	}
	for (auto head = 0; head < dn::value_heads; ++head) {
		token.a.push_back(normal(random, 1.5F));
		token.b.push_back(normal(random, 2.0F));
	}
	for (auto feature = 0; feature < dn::value_features; ++feature) {
		token.z.push_back(normal(random, 2.0F));
	}
	return token;
}

/** The same slot memory in both forms: the device layouts the emulation keeps, and the reference's natural ones. */
struct Memory {
	dn::LayerBuffers device;
	ref::DeltanetState<Card> reference;
};

/** A state that has seen tokens and a window of activations, laid out both ways. */
[[nodiscard]] auto random_memory(std::mt19937_64& random) -> Memory {
	auto memory = Memory{.device = {}, .reference = {}};
	for (auto head = 0; head < dn::value_heads; ++head) {
		for (auto column = 0; column < dn::dim; ++column) {
			for (auto row = 0; row < dn::dim; ++row) {
				auto const value = normal(random, 0.05F);
				memory.device.state.at(index(head * dn::head_floats + dn::State::element(row, column))) = value;
				memory.reference.state.at(index((head * dn::dim + column) * dn::dim + row)) = value;
			}
		}
	}
	for (auto key_head = 0; key_head < dn::key_heads; ++key_head) {
		for (auto row = 0; row < dn::window_rows; ++row) {
			for (auto local = 0; local < dn::group_channels; ++local) {
				auto const value = prims::round_to_bf16(normal(random, 1.0F));
				memory.device.window.at(index((key_head * dn::window_rows + row) * dn::group_channels + local)) = value;
				memory.reference.window.at(index(row * dn::channels + dn::card_channel(key_head, local))) = value;
			}
		}
	}
	return memory;
}

/** Whether the emulation's stored state and window are the reference's, element by element through the layouts. */
[[nodiscard]] auto same_memory(dn::LayerBuffers const& device, ref::DeltanetState<Card> const& reference) -> bool {
	auto holds = true;
	for (auto head = 0; head < dn::value_heads; ++head) {
		for (auto column = 0; column < dn::dim; ++column) {
			for (auto row = 0; row < dn::dim; ++row) {
				auto const got = device.state.at(index(head * dn::head_floats + dn::State::element(row, column)));
				auto const want = reference.state.at(index((head * dn::dim + column) * dn::dim + row));
				holds = holds && std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(want);
			}
		}
	}
	for (auto key_head = 0; key_head < dn::key_heads; ++key_head) {
		for (auto row = 0; row < dn::window_rows; ++row) {
			for (auto local = 0; local < dn::group_channels; ++local) {
				auto const got = device.window.at(index((key_head * dn::window_rows + row) * dn::group_channels + local));
				auto const want = reference.window.at(index(row * dn::channels + dn::card_channel(key_head, local)));
				holds = holds && got.bits == want.bits;
			}
		}
	}
	return holds;
}

/** Tokens as the activation rows the kernels read, and room for their outputs. */
struct Rows {
	std::vector<prims::Bf16> qkv;
	std::vector<float> gates;
	std::vector<float> outputs;

	explicit Rows(std::span<Token const> tokens) {
		for (auto const& token : tokens) {
			qkv.insert(qkv.end(), token.qkv.begin(), token.qkv.end());
			for (auto head = std::size_t{0}; head < token.a.size(); ++head) {
				gates.push_back(token.a.at(head));
				gates.push_back(token.b.at(head));
			}
		}
		outputs.resize(tokens.size() * index(dn::value_features));
	}

	[[nodiscard]] auto view() -> dn::Activations {
		return {.qkv = qkv, .gates = gates, .outputs = outputs};
	}
};

/** The reference's step over one token: its gated output. */
[[nodiscard]] auto reference_step(ref::DeltanetState<Card>& state, Layer const& layer, Token const& token) -> std::vector<prims::Bf16> {
	auto const weights = ref::DeltanetWeights{.conv = layer.conv, .a_log = layer.a_log, .dt_bias = layer.dt_bias, .norm = layer.norm};
	return ref::deltanet_step<Card>(state, weights, {.projection = token.qkv, .gate = token.z, .a = token.a, .b = token.b});
}

/** Whether the emulation's normalized output of `row`, gated by the token's z as the output phase stages it, is `want`. */
[[nodiscard]] auto same_output(Rows const& rows, std::size_t row, Token const& token, std::span<prims::Bf16 const> want) -> bool {
	auto holds = true;
	for (auto feature = std::size_t{0}; feature < index(dn::value_features); ++feature) {
		auto const got = dn::gated(rows.outputs.at(row * index(dn::value_features) + feature), token.z.at(feature));
		holds = holds && got.bits == want[feature].bits;
	}
	return holds;
}

/** The convolution of every card channel for one token: the device's lanes' `convolve` against the reference's. */
[[nodiscard]] auto convolution_agrees(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const layer = random_layer(random);
	auto holds = true;
	for (auto trial = 0; trial < 8; ++trial) {
		auto const memory = random_memory(random);
		auto const token = random_token(random);
		auto const reference = ref::convolution<Card>(memory.reference.window, layer.conv, token.qkv);
		for (auto key_head = 0; key_head < dn::key_heads; ++key_head) {
			for (auto local = 0; local < dn::group_channels; ++local) {
				auto const channel = dn::card_channel(key_head, local);
				auto taps = dn::Four{};
				auto inputs = dn::Four{};
				for (auto tap = 0; tap < 4; ++tap) {
					taps.element[tap] = layer.conv.at(index(channel * 4 + tap));
					inputs.element[tap] =
					    tap < dn::window_rows
					        ? memory.device.window.at(index((key_head * dn::window_rows + tap) * dn::group_channels + local))
					        : token.qkv.at(index(channel));
				}
				auto const got = dn::convolve(taps, inputs);
				holds = holds && std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(reference.at(index(channel)));
			}
		}
	}
	return holds;
}

/**
 * Verify cycles as the decode core runs them: each replays the drafts the last cycle accepted from the ring, runs its
 * rows from the stored state, and commits after its first row. Every row's output must be the reference's after the
 * same tokens, and the committed state and window the reference's after the cycle's replays and first row.
 */
[[nodiscard]] auto decode_agrees(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const layer = random_layer(random);
	auto memory = random_memory(random);
	auto const weights = dn::LayerWeights{.conv = layer.conv, .a_log = layer.a_log, .dt_bias = layer.dt_bias, .norm = layer.norm};
	// (rows, drafts accepted): one to five rows, the accepted drafts replayed by the next cycle.
	constexpr auto cycles = std::array<std::array<std::uint32_t, 2>, 6>{{{1, 0}, {3, 2}, {5, 4}, {2, 0}, {4, 1}, {5, 0}}};
	auto pending = std::vector<Token>{};
	auto holds = true;
	for (auto const [rows_count, accepted] : cycles) {
		auto tokens = std::vector<Token>{};
		for (auto row = std::uint32_t{0}; row < rows_count; ++row) {
			tokens.push_back(random_token(random));
		}
		auto rows = Rows{tokens};
		auto const replays = static_cast<std::uint32_t>(pending.size());
		dn::decode(weights, memory.device.view(), rows.view(), {.slot = 0, .replays = replays, .rows = rows_count, .first_row = 0});
		auto running = memory.reference;
		for (auto const& token : pending) {
			std::ignore = reference_step(running, layer, token);
		}
		for (auto row = std::size_t{0}; row < tokens.size(); ++row) {
			auto const want = reference_step(running, layer, tokens.at(row));
			holds = holds && same_output(rows, row, tokens.at(row), want);
			if (row == 0) {
				memory.reference = running;
			}
		}
		holds = holds && same_memory(memory.device, memory.reference);
		pending.assign(tokens.begin() + 1, tokens.begin() + 1 + accepted);
	}
	return holds;
}

/** A prompt chunk of `count` tokens as `deltanet_prefill` and its window move run it, against the reference's steps. */
[[nodiscard]] auto prefill_agrees(std::uint64_t seed, std::uint32_t count) -> bool {
	auto random = engine::testing::generator(seed);
	auto const layer = random_layer(random);
	auto memory = random_memory(random);
	auto const weights = dn::LayerWeights{.conv = layer.conv, .a_log = layer.a_log, .dt_bias = layer.dt_bias, .norm = layer.norm};
	auto tokens = std::vector<Token>{};
	for (auto token = std::uint32_t{0}; token < count; ++token) {
		tokens.push_back(random_token(random));
	}
	auto rows = Rows{tokens};
	dn::prefill(weights, memory.device.view(), rows.view(), {.slot = 0, .tokens = count, .first_row = 0});
	auto holds = true;
	for (auto row = std::size_t{0}; row < tokens.size(); ++row) {
		holds = holds && same_output(rows, row, tokens.at(row), reference_step(memory.reference, layer, tokens.at(row)));
	}
	return holds && same_memory(memory.device, memory.reference);
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the convolution of every card channel equals the reference's (seed 1)",
		                      .holds = convolution_agrees(1)},
	    engine::testing::Case{.name = "decode cycles of one to five rows with drafts replayed equal the reference's steps (seed 2)",
		                      .holds = decode_agrees(2)},
	    engine::testing::Case{.name = "a two-token prompt chunk, shorter than the window, equals the reference's steps (seed 3)",
		                      .holds = prefill_agrees(3, 2)},
	    engine::testing::Case{.name = "a 37-token prompt chunk across prefill blocks equals the reference's steps (seed 4)",
		                      .holds = prefill_agrees(4, 37)},
	};
	return engine::testing::report(cases);
}

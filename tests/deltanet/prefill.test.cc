// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill leaves exactly the state decode expects: a prompt chunk must produce, bit for bit, the outputs, stored state
// and windows of the same tokens decoded one per cycle, at every chunk length (shorter than the window, across a
// prefill block, several blocks) and however a prompt is cut into chunks.
import std;
import engine.deltanet;
import engine.deltanet_fixtures;
import engine.testing;

namespace {

namespace dn = engine::deltanet;
namespace fx = engine::deltanet_fixtures;

struct Run {
	dn::LayerBuffers memory;
	std::vector<float> outputs;
};

[[nodiscard]] auto decoded(fx::WeightsStore const& weights, dn::LayerBuffers memory, std::span<fx::Token const> tokens) -> Run {
	auto outputs = std::vector<float>{};
	for (auto const& token : tokens) {
		auto rows = fx::RowsStore{};
		rows.push(token);
		dn::decode(weights.view(), memory.view(), rows.view(), {.slot = 0, .replays = 0, .rows = 1, .first_row = 0});
		outputs.insert(outputs.end(), rows.outputs.begin(), rows.outputs.end());
	}
	return {.memory = std::move(memory), .outputs = std::move(outputs)};
}

/** The prompt cut into chunks of `lengths`, each placed after `offset` unrelated rows. */
[[nodiscard]] auto prefilled(fx::WeightsStore const& weights, dn::LayerBuffers memory, std::span<fx::Token const> tokens,
                             std::span<std::uint32_t const> lengths, fx::Generator& generator) -> Run {
	auto outputs = std::vector<float>{};
	auto committed = dn::Committed{.tokens = 0};
	auto taken = std::size_t{0};
	for (auto const length : lengths) {
		auto rows = fx::RowsStore{};
		auto const offset = std::uniform_int_distribution<std::uint32_t>{0, 3}(generator);
		for (auto row = std::uint32_t{0}; row < offset; ++row) {
			rows.push(fx::random_token(generator));
		}
		for (auto token = std::size_t{0}; token < length; ++token) {
			rows.push(tokens[taken + token]);
		}
		auto const chunk = dn::prompt_chunk(committed, length, 1, offset);
		dn::prefill(weights.view(), memory.view(), rows.view(), chunk.work);
		committed = chunk.after;
		auto const produced = std::span{rows.outputs}.subspan(std::size_t{offset} * static_cast<std::size_t>(dn::value_features));
		outputs.insert(outputs.end(), produced.begin(), produced.end());
		taken += length;
	}
	return {.memory = std::move(memory), .outputs = std::move(outputs)};
}

[[nodiscard]] auto prefill_equals_decode(std::uint64_t seed, std::vector<std::uint32_t> const& lengths) -> bool {
	auto generator = engine::testing::generator(seed);
	auto const weights = fx::random_weights(generator);
	auto const start = fx::random_layer(generator);
	auto const total = std::ranges::fold_left(lengths, std::uint32_t{0}, std::plus{});
	auto tokens = std::vector<fx::Token>{};
	for (auto token = std::uint32_t{0}; token < total; ++token) {
		tokens.push_back(fx::random_token(generator));
	}
	auto const reference = decoded(weights, start, tokens);
	auto const chunked = prefilled(weights, start, tokens, lengths, generator);
	return fx::same_stored(reference.memory, chunked.memory) && fx::same_bits(reference.outputs, chunked.outputs);
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "one token", .holds = prefill_equals_decode(11, {1})},
	    engine::testing::Case{.name = "two tokens, shorter than the window", .holds = prefill_equals_decode(12, {2})},
	    engine::testing::Case{.name = "three tokens, the window's length", .holds = prefill_equals_decode(13, {3})},
	    engine::testing::Case{.name = "one prefill block", .holds = prefill_equals_decode(14, {16})},
	    engine::testing::Case{.name = "one block and one token", .holds = prefill_equals_decode(15, {17})},
	    engine::testing::Case{.name = "several blocks", .holds = prefill_equals_decode(16, {41})},
	    engine::testing::Case{.name = "chunks shorter than the window", .holds = prefill_equals_decode(17, {1, 2, 1, 1})},
	    engine::testing::Case{.name = "chunks across blocks", .holds = prefill_equals_decode(18, {5, 16, 11, 1, 20})},
	};
	return engine::testing::report(cases);
}

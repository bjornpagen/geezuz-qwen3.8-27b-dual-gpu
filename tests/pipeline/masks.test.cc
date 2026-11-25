// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.grammar;
import engine.model;
import engine.pipeline;
import engine.testing;
import engine.tokenizer;
import engine.vocabulary;

namespace {

namespace grammar = engine::grammar;
namespace pipeline = engine::pipeline;
namespace model = engine::model;
namespace tokenizer = engine::tokenizer;
namespace vocabulary = engine::vocabulary;

/**
 * A row mask copied from a matcher allows exactly the tokens the matcher's mask allows, read through the vocabulary's
 * own map, and each card's half is that card's rows in its lm_head order with the padding rows clear: checked over
 * every token and every row, at positions along a walk of a small grammar.
 */
[[nodiscard]] auto row_masks_carry_the_matchers_mask(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto matcher = grammar::Matcher::make(grammar::compile_regex("(ab|cd|[a-z ]{1,3}x)+").value(), grammar::engine_layout);
	for (auto step = 0; step < 6; ++step) {
		auto const view = matcher.mask(vocabulary).value();
		auto const rows = pipeline::RowMask::of(view);
		auto allowed = std::vector<pipeline::Token>{};
		for (auto token = pipeline::Token{0}; token < model::sampled_vocabulary; ++token) {
			if (rows.allows(token) != view.allows(token)) {
				return false;
			}
			if (view.allows(token) && token < tokenizer::bpe_vocabulary) {
				allowed.push_back(token);
			}
		}
		for (auto const card : {model::Card::Zero, model::Card::One}) {
			auto const half = rows.card(card);
			for (auto row = std::size_t{0}; row < model::vocab_per_card; ++row) {
				auto const token = vocabulary::token(card, row);
				auto const set = ((half[row / 64] >> (row % 64)) & 1U) != 0;
				if (set != (token < model::sampled_vocabulary && view.allows(token))) {
					return false;
				}
			}
		}
		if (allowed.empty() || !matcher.advance(vocabulary, allowed[random() % allowed.size()])) {
			return false;
		}
	}
	return true;
}

/** The mask of a request whose grammar overran allows exactly the two tokens that end a turn. */
[[nodiscard]] auto the_ending_mask_allows_only_the_end_of_the_turn() -> bool {
	auto const ending = pipeline::RowMask::ending();
	auto const ends = std::array{tokenizer::id(tokenizer::Added::im_end), tokenizer::id(tokenizer::Added::endoftext)};
	for (auto token = pipeline::Token{0}; token < model::sampled_vocabulary; ++token) {
		if (ending.allows(token) != std::ranges::contains(ends, token)) {
			return false;
		}
	}
	auto set = std::size_t{0};
	for (auto const word : ending.words) {
		set += engine::core::narrow<std::size_t>(std::popcount(word));
	}
	return set == ends.size();
}

}

auto main() -> int {
	auto const loaded = tokenizer::load();
	if (!loaded) {
		return 1;
	}
	auto const vocabulary = grammar::engine_vocabulary(*loaded);
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "row masks carry the matcher's mask (seed 1)", .holds = row_masks_carry_the_matchers_mask(vocabulary, 1)},
	    {.name = "row masks carry the matcher's mask (seed 2)", .holds = row_masks_carry_the_matchers_mask(vocabulary, 2)},
	    {.name = "the ending mask allows only the end of the turn", .holds = the_ending_mask_allows_only_the_end_of_the_turn()},
	});
	return engine::testing::report(cases);
}

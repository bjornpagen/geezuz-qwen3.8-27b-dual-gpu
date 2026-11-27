// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:engine_tokens;

import std;
import engine.core;
import engine.model;
import engine.tokenizer;
import engine.vocabulary;
import :tokens;
import :matcher;

namespace engine::grammar {

/**
 * The engine's mask layout: token order over every token the tokenizer has, and row order over both cards' lm_head
 * rows, card c's row r at bit `c * vocab_per_card + r` (each card's half word-aligned, so a card reads its own words).
 */
export inline constexpr Layout engine_layout{.tokens = model::sampled_vocabulary, .rows = model::vocab};

static_assert(model::vocab_per_card % 64 == 0, "each card's rows start on a word");
static_assert(tokenizer::vocabulary == model::sampled_vocabulary, "the mask covers exactly the tokens the tokenizer has");

/** One card's words of a row-order mask, in its lm_head row order: the width the cards take their masks in. */
export [[nodiscard]] auto card_rows(MaskView const& mask, model::Card card) -> std::span<std::uint64_t const, model::card_mask_words> {
	return mask.rows.subspan(std::size_t{std::to_underlying(card)} * model::card_mask_words).first<model::card_mask_words>();
}

/**
 * The engine's vocabulary: every byte-pair token as its bytes, every ordinary added token (`<think>`, `<tool_call>`,
 * ...) as its spelling, which is what it decodes to and what the chat format reads; special tokens never, except
 * the two the generation config ends a turn with, which the grammar allows exactly where it accepts.
 */
export [[nodiscard]] auto engine_vocabulary(tokenizer::Tokenizer const& tokenizer) -> Vocabulary {
	auto const row = [](tokenizer::Id id) {
		auto const place = vocabulary::place(id);
		return core::narrow<std::uint32_t>((std::size_t{std::to_underlying(place.card)} * model::vocab_per_card) + place.row);
	};
	auto tokens = std::vector<TokenBytes>{};
	tokens.reserve(model::sampled_vocabulary);
	for (auto id = tokenizer::Id{0}; id < tokenizer::vocabulary; ++id) {
		if (auto const added = tokenizer::added(id); added && tokenizer::entry(*added).role == tokenizer::Role::Special) {
			continue;
		}
		tokens.push_back(TokenBytes{.token = id, .bytes = std::string{tokenizer.token(id).value()}, .row = row(id)});
	}
	auto const ends = std::array{
	    EndToken{.token = tokenizer::id(tokenizer::Added::im_end), .row = row(tokenizer::id(tokenizer::Added::im_end))},
	    EndToken{.token = tokenizer::id(tokenizer::Added::endoftext), .row = row(tokenizer::id(tokenizer::Added::endoftext))},
	};
	return Vocabulary::make(tokens, ends, engine_layout);
}

}

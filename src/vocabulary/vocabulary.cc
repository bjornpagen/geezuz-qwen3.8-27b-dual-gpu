// SPDX-License-Identifier: AGPL-3.0-only
//
// The vocabulary permutation: which token each card's lm_head row scores. The draft ids sit almost all in the lower
// half of the vocabulary, so no card could hold half of them inside a half of the ids; instead the whole vocabulary is
// permuted before it is halved. Card c's rows are the draft ids d[2i + c] for i below half the draft vocabulary, then
// the other ids n[2j + c], each in ascending order, where d is the draft list and n every id not in it. Each card
// thus holds half of the draft vocabulary as a contiguous prefix of its rows (the draft lm_head is a view of the
// lm_head), and half of the rest after it. The baker, the top-k merge and the reference all map rows to tokens here.
export module engine.vocabulary;

import std;
import engine.core;
import engine.draft_vocabulary;
import engine.model;
import engine.sha256;

namespace engine::vocabulary {

/** A token id: a row of the checkpoint's lm_head and embedding. */
export using Token = std::uint32_t;

/** The SHA-256 of data/draft_vocab_32k.bin: a changed list fails its test instead of silently moving rows. */
export inline constexpr sha256::Digest draft_vocabulary_digest =
    sha256::pinned("cad444c6a5e571479eb1dec71768c760c76f7c65f0ecfe00c860dca231fc9d24");

inline constexpr std::size_t id_bytes = 4;

static_assert(draft_vocabulary::bytes.size() == model::draft_vocab * id_bytes, "the list is exactly draft_vocab whole ids");

// The list is decoded by the build a stage per constant evaluation: all of it in one exceeds clang's step budget.
inline constexpr std::size_t stage_ids = 4096;

static_assert(model::draft_vocab % stage_ids == 0);

template<std::size_t stage>
inline constexpr std::array<Token, stage_ids> decoded_stage = [] {
	auto ids = std::array<Token, stage_ids>{};
	for (auto index = std::size_t{0}; index < stage_ids; ++index) {
		auto const at = (stage * stage_ids + index) * id_bytes;
		auto const bytes = draft_vocabulary::bytes.subspan(at, id_bytes);
		ids.at(index) = Token{bytes[0]} | (Token{bytes[1]} << 8U) | (Token{bytes[2]} << 16U) | (Token{bytes[3]} << 24U);
	}
	return ids;
}();

template<std::size_t... stages>
[[nodiscard]] consteval auto join(std::index_sequence<stages...> /*stages*/) -> std::array<Token, model::draft_vocab> {
	auto ids = std::array<Token, model::draft_vocab>{};
	auto out = ids.begin();
	((out = std::ranges::copy(decoded_stage<stages>, out).out), ...);
	return ids;
}

/** The draft vocabulary: the token ids the mtp head drafts over, strictly ascending. */
export inline constexpr std::array<Token, model::draft_vocab> draft = join(std::make_index_sequence<model::draft_vocab / stage_ids>{});

static_assert(std::ranges::adjacent_find(draft, std::ranges::greater_equal{}) == draft.end(), "draft ids strictly ascend");
static_assert(draft.back() < model::sampled_vocabulary, "every draft id is a token the tokenizer has");

inline constexpr std::size_t draft_rows = model::draft_vocab_per_card;
inline constexpr std::size_t other_rows = model::vocab_per_card - draft_rows;

static_assert(model::draft_vocab == model::cards * draft_rows && model::vocab - model::draft_vocab == model::cards * other_rows,
              "both cards hold equally many draft and other ids");

/** Draft ids below `token`. */
[[nodiscard]] constexpr auto drafts_below(Token token) -> std::size_t {
	return static_cast<std::size_t>(std::ranges::lower_bound(draft, token) - draft.begin());
}

/**
 * The `rank`-th id not in the draft list. It is rank + m, where m counts the draft ids below it: d[i] lies below it
 * exactly when d[i] - i <= rank, and d[i] - i never decreases, so m is a binary search.
 */
[[nodiscard]] constexpr auto other(std::size_t rank) -> Token {
	auto low = std::size_t{0};
	auto high = draft.size();
	while (low < high) {
		auto const middle = low + (high - low) / 2;
		if (draft.at(middle) - middle <= rank) {
			low = middle + 1;
		} else {
			high = middle;
		}
	}
	return core::narrow<Token>(rank + low);
}

/** The token card `card`'s lm_head row `row` scores, row below `model::vocab_per_card`. */
export [[nodiscard]] constexpr auto token(model::Card card, std::size_t row) -> Token {
	core::invariant(row < model::vocab_per_card);
	auto const parity = std::size_t{std::to_underlying(card)};
	if (row < draft_rows) {
		return draft.at((model::cards * row) + parity);
	}
	return other((model::cards * (row - draft_rows)) + parity);
}

/** Where a token's lm_head row lives. */
export struct Place {
	model::Card card;
	std::size_t row;

	[[nodiscard]] constexpr auto operator==(Place const&) const -> bool = default;
};

[[nodiscard]] constexpr auto card_of(std::size_t parity) -> model::Card {
	return parity == 0 ? model::Card::Zero : model::Card::One;
}

static_assert(model::cards == 2, "a parity names a card");

/** The card and row holding `token`'s lm_head row, token below `model::vocab`: the inverse of `token`. */
export [[nodiscard]] constexpr auto place(Token token) -> Place {
	core::invariant(token < model::vocab);
	auto const below = drafts_below(token);
	if (below < draft.size() && draft.at(below) == token) {
		return {.card = card_of(below % model::cards), .row = below / model::cards};
	}
	auto const rank = token - below;
	return {.card = card_of(rank % model::cards), .row = draft_rows + (rank / model::cards)};
}

/** Whether `token` is one the mtp head drafts. */
export [[nodiscard]] constexpr auto drafted(Token token) -> bool {
	return std::ranges::binary_search(draft, token);
}

/** Card `card`'s row-to-token table, row by row: the table the engine uploads to the card at start. */
export [[nodiscard]] auto row_tokens(model::Card card) -> std::vector<Token> {
	auto tokens = std::vector<Token>(model::vocab_per_card);
	for (auto row = std::size_t{0}; row < tokens.size(); ++row) {
		tokens[row] = token(card, row);
	}
	return tokens;
}

}

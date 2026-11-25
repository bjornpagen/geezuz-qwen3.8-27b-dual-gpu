// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline:masks;

import std;
import engine.core;
import engine.grammar;
import engine.model;
import engine.tokenizer;
import engine.vocabulary;

namespace engine::pipeline {

export using Token = tokenizer::Id;

/** Words of a row-order mask: both cards' lm_head rows, as the grammar lays them out. */
export inline constexpr std::size_t row_words = model::divide_up(grammar::engine_layout.rows, 64);

static_assert(row_words == model::cards * grammar::card_words, "the row layout is each card's half, word-aligned");

/** The bit of `token`'s lm_head row in the row layout: card c's row r is bit c * vocab_per_card + r. */
[[nodiscard]] auto row_bit(Token token) -> std::size_t {
	auto const place = vocabulary::place(token);
	return (std::size_t{std::to_underlying(place.card)} * model::vocab_per_card) + place.row;
}

/**
 * One verify row's mask as the cards read it, owned: the grammar's row-order mask copied out of the matcher whose
 * borrow it was, so it can travel to the cards. Each card reads its own word-aligned half; rows no token owns (the
 * vocabulary's padding) stay clear.
 */
export struct RowMask {
	std::array<std::uint64_t, row_words> words;

	[[nodiscard]] static auto of(grammar::MaskView const& view) -> RowMask {
		// The engine's matchers are all made with the engine's layout.
		core::invariant(view.rows.size() == row_words);
		auto mask = RowMask{};
		std::ranges::copy(view.rows, mask.words.begin());
		return mask;
	}

	/**
	 * Only the tokens that end the turn: what a request whose grammar outgrew its bounds is sampled under, so its next
	 * committed token ends it at once.
	 */
	[[nodiscard]] static auto ending() -> RowMask {
		auto mask = RowMask{};
		mask.words.fill(0);
		for (auto const end : {tokenizer::Added::im_end, tokenizer::Added::endoftext}) {
			auto const bit = row_bit(tokenizer::id(end));
			mask.words[bit / 64] |= std::uint64_t{1} << (bit % 64);
		}
		return mask;
	}

	/** Card `card`'s half, in its lm_head row order. */
	[[nodiscard]] auto card(model::Card which) const -> std::span<std::uint64_t const, grammar::card_words> {
		auto const offset = std::size_t{std::to_underlying(which)} * grammar::card_words;
		return std::span<std::uint64_t const, grammar::card_words>{std::span{words}.subspan(offset, grammar::card_words)};
	}

	/** Whether the row holding `token` is allowed. */
	[[nodiscard]] auto allows(Token token) const -> bool {
		auto const bit = row_bit(token);
		return ((words[bit / 64] >> (bit % 64)) & 1U) != 0;
	}

	[[nodiscard]] auto operator==(RowMask const&) const -> bool = default;
};

/**
 * The masks of one batch's verify rows, in the batch's order: row r's verify row v (its committed token, then each
 * draft) at r * (1 + drafts(depth)) + v. The buffer is reserved once for the row budget and travels to the cards and
 * back each cycle, so a cycle allocates no masks.
 */
export struct Masks {
	std::vector<RowMask> rows;
};

}

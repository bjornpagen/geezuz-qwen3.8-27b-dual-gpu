// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline:masks;

import std;
import engine.core;
import engine.model;
import engine.tokenizer;
import engine.vocabulary;

namespace engine::pipeline {

export using Token = tokenizer::Id;

inline constexpr std::size_t word_bits = 64;

/** Words of a mask over the tokenizer's tokens: token t is bit t % 64 of word t / 64. */
export inline constexpr std::size_t token_words = model::divide_up(model::sampled_vocabulary, word_bits);

/** Words of a mask over one card's lm_head rows: row r is bit r % 64 of word r / 64. */
export inline constexpr std::size_t card_words = model::divide_up(model::vocab_per_card, word_bits);

/**
 * A grammar's mask at one position, in token order: the tokens it allows set. Bits past the tokenizer's vocabulary
 * are clear: no padding row is ever sampleable.
 */
export template<class M>
concept TokenOrderMask = requires(M const& mask) {
	{ mask.words() } -> std::convertible_to<std::span<std::uint64_t const, token_words>>;
};

/**
 * One verify row's mask as the cards read it: each card's half in its own lm_head row order, so a card excludes
 * masked rows from its top-k without translating ids. Rows no token owns (the vocabulary's padding) stay clear.
 */
export struct RowMask {
	std::array<std::array<std::uint64_t, card_words>, model::cards> cards;

	/** Whether the row holding `token` is allowed. */
	[[nodiscard]] auto allows(Token token) const -> bool {
		auto const place = vocabulary::place(token);
		auto const& half = cards[std::to_underlying(place.card)];
		return ((half[place.row / word_bits] >> (place.row % word_bits)) & 1U) != 0;
	}

	[[nodiscard]] auto operator==(RowMask const&) const -> bool = default;
};

/**
 * Translates token-order masks into row order through `engine.vocabulary`, the one place the row-to-token map is
 * held. The map is inverted once into a table at construction; a mask costs a pass over its allowed tokens. A
 * grammar repeats its masks (free text allows the same tokens at every position, and a batch's rows sit in few
 * grammar states), so the last few translations are kept and a mask equal to one of them costs a comparison.
 *
 * Large (the table and the kept translations): the owner keeps it behind a `std::unique_ptr`.
 */
export class Rows {
public:
	Rows() : places_(model::sampled_vocabulary), kept_(kept_translations) {
		for (auto token = Token{0}; token < model::sampled_vocabulary; ++token) {
			auto const place = vocabulary::place(token);
			places_[token] = Place{.card = std::to_underlying(place.card), .row = core::narrow<std::uint32_t>(place.row)};
		}
	}

	/** The mask in row order: a borrow of this translator's own storage, valid until the next call. */
	template<TokenOrderMask M>
	[[nodiscard]] auto rows(M const& mask) -> RowMask const& {
		auto const words = std::span<std::uint64_t const, token_words>{mask.words()};
		auto const same = std::ranges::find_if(kept_, [&](std::optional<Translated> const& kept) {
			return kept && std::ranges::equal(words, kept->tokens);
		});
		if (same != kept_.end()) {
			return (*same)->rows;
		}
		// The oldest translation makes way.
		auto& kept = kept_[next_ % kept_.size()];
		next_ += 1;
		if (!kept) {
			kept.emplace();
		}
		std::ranges::copy(words, kept->tokens.begin());
		for (auto& half : kept->rows.cards) {
			half.fill(0);
		}
		for (auto const [index, word] : std::views::enumerate(words)) {
			for (auto bits = word; bits != 0; bits &= bits - 1) {
				auto const token = (core::narrow<std::size_t>(index) * word_bits) + core::narrow<std::size_t>(std::countr_zero(bits));
				// A grammar's bits past the vocabulary are clear; one that sets them names a token no row scores.
				core::invariant(token < model::sampled_vocabulary);
				auto const place = places_[token];
				kept->rows.cards[place.card][place.row / word_bits] |= std::uint64_t{1} << (place.row % word_bits);
			}
		}
		return kept->rows;
	}

private:
	/** As many as a batch has rows: the most distinct grammar states one batch's committed tokens can be in. */
	static constexpr std::size_t kept_translations = model::slots;

	struct Place {
		std::uint8_t card;
		std::uint32_t row;
	};

	struct Translated {
		std::array<std::uint64_t, token_words> tokens;
		RowMask rows;
	};

	std::vector<Place> places_;
	std::vector<std::optional<Translated>> kept_;
	std::size_t next_ = 0;
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

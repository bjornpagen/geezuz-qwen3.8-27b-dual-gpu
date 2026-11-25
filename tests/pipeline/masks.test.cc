// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.model;
import engine.pipeline;
import engine.testing;
import engine.vocabulary;

namespace {

namespace pipeline = engine::pipeline;
namespace model = engine::model;
namespace vocabulary = engine::vocabulary;

struct Mask {
	std::array<std::uint64_t, pipeline::token_words> bits;

	[[nodiscard]] auto words() const -> std::span<std::uint64_t const, pipeline::token_words> {
		return bits;
	}

	[[nodiscard]] auto allows(pipeline::Token token) const -> bool {
		return ((bits[token / 64] >> (token % 64)) & 1U) != 0;
	}
};

static_assert(pipeline::TokenOrderMask<Mask>);

/** A mask allowing each token with probability `density` / 64, never a bit past the vocabulary. */
[[nodiscard]] auto random_mask(std::mt19937_64& random, std::uint64_t density) -> Mask {
	auto mask = Mask{};
	for (auto& word : mask.bits) {
		word = 0;
		for (auto bit = 0U; bit < 64; ++bit) {
			word |= (random() % 64 < density ? std::uint64_t{1} : 0) << bit;
		}
	}
	auto const tail = model::sampled_vocabulary % 64;
	if (tail != 0) {
		mask.bits.back() &= (std::uint64_t{1} << tail) - 1;
	}
	return mask;
}

/** Bits the row mask sets on both cards together. */
[[nodiscard]] auto population(pipeline::RowMask const& rows) -> std::size_t {
	auto total = std::size_t{0};
	for (auto const& half : rows.cards) {
		for (auto const word : half) {
			total += static_cast<std::size_t>(std::popcount(word));
		}
	}
	return total;
}

/**
 * The row-order mask allows exactly the tokens the token-order mask allows, read through the vocabulary's own map,
 * and sets no other row: checked over every token, for masks from empty to full.
 */
[[nodiscard]] auto rows_allow_exactly_the_masked_tokens(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto rows = std::make_unique<pipeline::Rows>();
	for (auto const density : std::array<std::uint64_t, 4>{0, 1, 32, 64}) {
		auto const mask = random_mask(random, density);
		auto const& translated = rows->rows(mask);
		auto allowed = std::size_t{0};
		for (auto token = pipeline::Token{0}; token < model::sampled_vocabulary; ++token) {
			if (translated.allows(token) != mask.allows(token)) {
				return false;
			}
			allowed += mask.allows(token) ? 1 : 0;
		}
		if (population(translated) != allowed) {
			return false;
		}
	}
	return true;
}

/** Row r of card c is set exactly when the token the vocabulary puts there is allowed. */
[[nodiscard]] auto rows_follow_each_cards_order(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto rows = std::make_unique<pipeline::Rows>();
	auto const mask = random_mask(random, 20);
	auto const& translated = rows->rows(mask);
	for (auto const card : {model::Card::Zero, model::Card::One}) {
		auto const& half = translated.cards[std::to_underlying(card)];
		for (auto probe = 0; probe < 20'000; ++probe) {
			auto const row = random() % model::vocab_per_card;
			auto const token = vocabulary::token(card, row);
			auto const set = ((half[row / 64] >> (row % 64)) & 1U) != 0;
			if (set != (token < model::sampled_vocabulary && mask.allows(token))) {
				return false;
			}
		}
	}
	return true;
}

/** Translations kept for reuse answer exactly as fresh ones, across more distinct masks than are kept. */
[[nodiscard]] auto kept_translations_equal_fresh_ones(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto masks = std::vector<Mask>{};
	for (auto index = 0; index < 12; ++index) {
		masks.push_back(random_mask(random, 1 + (random() % 63)));
	}
	auto expected = std::vector<pipeline::RowMask>{};
	for (auto const& mask : masks) {
		auto fresh = std::make_unique<pipeline::Rows>();
		expected.push_back(fresh->rows(mask));
	}
	auto reused = std::make_unique<pipeline::Rows>();
	for (auto step = 0; step < 200; ++step) {
		auto const index = random() % masks.size();
		if (reused->rows(masks[index]) != expected[index]) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "rows allow exactly the masked tokens (seed 1)", .holds = rows_allow_exactly_the_masked_tokens(1)},
	    {.name = "rows follow each card's order (seed 2)", .holds = rows_follow_each_cards_order(2)},
	    {.name = "kept translations equal fresh ones (seed 3)", .holds = kept_translations_equal_fresh_ones(3)},
	});
	return engine::testing::report(cases);
}

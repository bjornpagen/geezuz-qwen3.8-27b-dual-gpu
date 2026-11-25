// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.draft_vocabulary;
import engine.model;
import engine.sha256;
import engine.testing;
import engine.vocabulary;

namespace {

namespace model = engine::model;
namespace vocabulary = engine::vocabulary;

using vocabulary::Token;

inline constexpr std::array cards = {model::Card::Zero, model::Card::One};

/**
 * The oracle: the permutation by its plain definition. Every id is drafted or not; the drafted ones, ascending, go to
 * the cards alternately, then the rest, ascending, alternately.
 */
[[nodiscard]] auto by_definition() -> std::array<std::vector<Token>, model::cards> {
	auto const& list = vocabulary::draft;
	auto drafted = std::vector<bool>(model::vocab, false);
	for (auto const id : list) {
		drafted.at(id) = true;
	}
	auto rows = std::array<std::vector<Token>, model::cards>{};
	for (auto index = std::size_t{0}; index < list.size(); ++index) {
		rows.at(index % model::cards).push_back(list.at(index));
	}
	auto rank = std::size_t{0};
	for (auto id = Token{0}; id < model::vocab; ++id) {
		if (!drafted.at(id)) {
			rows.at(rank % model::cards).push_back(id);
			++rank;
		}
	}
	return rows;
}

[[nodiscard]] auto rows_follow_the_definition(std::array<std::vector<Token>, model::cards> const& expected) -> bool {
	return std::ranges::all_of(cards, [&expected](model::Card card) {
		auto const& rows = expected.at(std::to_underlying(card));
		auto holds = rows.size() == model::vocab_per_card && vocabulary::row_tokens(card) == rows;
		for (auto row = std::size_t{0}; holds && row < rows.size(); ++row) {
			holds = vocabulary::token(card, row) == rows[row];
		}
		return holds;
	});
}

[[nodiscard]] auto place_inverts_token() -> bool {
	auto seen = std::vector<int>(model::vocab, 0);
	for (auto const card : cards) {
		for (auto row = std::size_t{0}; row < model::vocab_per_card; ++row) {
			auto const token = vocabulary::token(card, row);
			if (token >= model::vocab || vocabulary::place(token) != vocabulary::Place{.card = card, .row = row}) {
				return false;
			}
			++seen.at(token);
		}
	}
	return std::ranges::all_of(seen, [](int count) { return count == 1; });
}

[[nodiscard]] auto draft_rows_are_the_prefix() -> bool {
	return std::ranges::all_of(cards, [](model::Card card) {
		auto const rows = vocabulary::row_tokens(card);
		auto const prefix = std::span{rows}.first(model::draft_vocab_per_card);
		auto const rest = std::span{rows}.subspan(model::draft_vocab_per_card);
		return std::ranges::all_of(prefix, vocabulary::drafted) && std::ranges::none_of(rest, vocabulary::drafted) &&
		       std::ranges::is_sorted(prefix) && std::ranges::is_sorted(rest);
	});
}

[[nodiscard]] auto the_built_in_list_is_the_pinned_file() -> bool {
	auto const& bytes = engine::draft_vocabulary::bytes;
	auto as_bytes = std::vector<std::byte>(bytes.size());
	std::ranges::transform(bytes, as_bytes.begin(), [](unsigned char byte) { return std::byte{byte}; });
	return engine::sha256::digest(as_bytes) == vocabulary::draft_vocabulary_digest;
}

}

auto main() -> int {
	auto const expected = by_definition();
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the built-in draft list hashes to the pinned SHA-256 of data/draft_vocab_32k.bin",
	        .holds = the_built_in_list_is_the_pinned_file(),
	    },
	    engine::testing::Case{
	        .name = "every card's row tokens are the draft ids then the other ids, alternating between the cards",
	        .holds = rows_follow_the_definition(expected),
	    },
	    engine::testing::Case{
	        .name = "every token has exactly one card row, and place inverts token on all of them",
	        .holds = place_inverts_token(),
	    },
	    engine::testing::Case{
	        .name = "each card's draft rows are exactly its first draft_vocab_per_card rows, each part ascending",
	        .holds = draft_rows_are_the_prefix(),
	    },
	};
	return engine::testing::report(cases);
}

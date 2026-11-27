// SPDX-License-Identifier: AGPL-3.0-only
//
// The harness's prompt set and its tokens: any well-formed set written out parses back to itself; each malformed line
// is refused with its kind and line; a prompt's tokens are the tokenizer's plain-text encoding, special-token
// spellings read as characters.
import std;
import engine.ref;
import engine.testing;
import engine.tokenizer;

namespace {

namespace ref = engine::ref;
namespace tokenizer = engine::tokenizer;

[[nodiscard]] auto render(std::span<ref::Prompt const> prompts) -> std::string {
	auto text = std::string{};
	for (auto const& prompt : prompts) {
		text += std::format("{} {}\n", prompt.name, prompt.text);
	}
	return text;
}

/** Random sets: names of letters, digits and punctuation; texts of any printable character and space, unicode too. */
[[nodiscard]] auto random_set(std::mt19937_64& generator) -> std::vector<ref::Prompt> {
	constexpr auto name_letters = std::string_view{"abcdefghijklmnopqrstuvwxyz0123456789-_.:/"};
	constexpr auto text_pieces = std::array<std::string_view, 8>{" ", "a", "Z", "7", ",", "\t", "\xc3\xa9", "<|im_start|>"};
	auto count = std::uniform_int_distribution<std::size_t>{1, 12};
	auto length = std::uniform_int_distribution<std::size_t>{1, 20};
	auto letter = std::uniform_int_distribution<std::size_t>{0, name_letters.size() - 1};
	auto piece = std::uniform_int_distribution<std::size_t>{0, text_pieces.size() - 1};
	auto prompts = std::vector<ref::Prompt>{};
	auto const total = count(generator);
	for (auto index = std::size_t{0}; index < total; ++index) {
		auto name = std::format("p{}-", index);
		for (auto each = length(generator); each > 0; --each) {
			name.push_back(name_letters[letter(generator)]);
		}
		auto text = std::string{};
		for (auto each = length(generator); each > 0; --each) {
			text += text_pieces.at(piece(generator));
		}
		prompts.push_back({.name = std::move(name), .text = std::move(text)});
	}
	return prompts;
}

[[nodiscard]] auto sets_round_trip(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 500; ++trial) {
		auto const prompts = random_set(generator);
		auto const parsed = ref::parse_prompts(render(prompts));
		// Blank lines between prompts are skipped wherever they fall.
		auto const spaced = ref::parse_prompts("\n" + render(prompts) + "\n\n");
		if (!parsed || *parsed != prompts || !spaced || *spaced != prompts) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto refused(std::string_view text, ref::PromptsFailure failure, std::size_t line) -> bool {
	auto const parsed = ref::parse_prompts(text);
	return !parsed && parsed.error() == ref::PromptsError{.failure = failure, .line = line};
}

[[nodiscard]] auto malformed_sets_are_refused() -> bool {
	using enum ref::PromptsFailure;
	return refused("", Empty, 0) && refused("\n\n", Empty, 0) && refused("alpha hello\nbeta\n", Unnamed, 2) &&
	       refused("alpha hello\n gamma text\n", Unnamed, 2) && refused("alpha \n", Unnamed, 1) &&
	       refused("alpha hello\r\nbeta hi\r\n", Unreadable, 1) && refused("al\tpha hello\n", Unreadable, 1) &&
	       refused("alpha hello\n\nbeta hi\nalpha again\n", Duplicate, 4);
}

[[nodiscard]] auto tokens_are_the_plain_text_encoding() -> bool {
	auto const loaded = tokenizer::load();
	if (!loaded) {
		return false;
	}
	// The ids transformers' tokenizer gives this text (the checkpoint test's prompt).
	constexpr auto counting = std::array<ref::Token, 14>{16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11};
	auto const counted = ref::prompt_tokens(*loaded, "1, 2, 3, 4, 5,");
	auto const spelled = ref::prompt_tokens(*loaded, "<|im_start|>user");
	auto const plain = std::array{tokenizer::Piece{tokenizer::Text{.utf8 = "<|im_start|>user", .spelling = tokenizer::Spelling::Characters}}};
	auto const expected = loaded->encode(plain);
	auto const broken = ref::prompt_tokens(*loaded, "ab\xff");
	auto const empty = ref::prompt_tokens(*loaded, "");
	return counted && std::ranges::equal(*counted, counting) && spelled && expected && *spelled == *expected &&
	       std::ranges::none_of(*spelled, [](ref::Token id) { return id >= tokenizer::bpe_vocabulary; }) &&
	       broken.error() == ref::PromptTokensError{.failure = ref::PromptTokensFailure::NotUtf8, .offset = 2} &&
	       empty.error().failure == ref::PromptTokensFailure::Empty;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "any well-formed prompt set written out parses back to itself (seed 51)", .holds = sets_round_trip(51)},
	    engine::testing::Case{.name = "empty sets, unnamed lines, crlf, tabbed names and repeated names are refused at their line", .holds = malformed_sets_are_refused()},
	    engine::testing::Case{.name = "a prompt's tokens are the plain-text encoding, special spellings as characters", .holds = tokens_are_the_plain_text_encoding()},
	};
	return engine::testing::report(cases);
}

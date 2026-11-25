// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.sched;
import engine.testing;
import engine.tokenizer;

namespace {

namespace sched = engine::sched;
namespace tokenizer = engine::tokenizer;

using engine::core::narrow;
using sched::Count;
using sched::Token;
using Tokens = std::vector<Token>;

constexpr auto im_start = tokenizer::id(tokenizer::Added::im_start);
constexpr auto im_end = tokenizer::id(tokenizer::Added::im_end);
constexpr auto endoftext = tokenizer::id(tokenizer::Added::endoftext);

/** Builds token sequences piece by piece, so each test controls exactly where the token boundaries fall. */
class Encoder {
public:
	explicit Encoder(tokenizer::Tokenizer const& tokens) : tokens_{tokens} {}

	[[nodiscard]] auto operator()(std::string_view text) const -> Tokens {
		auto const piece = std::array{tokenizer::Piece{tokenizer::Text{.utf8 = text, .spelling = tokenizer::Spelling::Token}}};
		auto encoded = tokens_.encode(piece);
		engine::core::invariant(encoded.has_value());
		return std::move(*encoded);
	}

	[[nodiscard]] auto tokenizer() const -> tokenizer::Tokenizer const& {
		return tokens_;
	}

private:
	tokenizer::Tokenizer const& tokens_;
};

[[nodiscard]] auto join(std::initializer_list<Tokens> parts) -> Tokens {
	auto joined = Tokens{};
	for (auto const& part : parts) {
		joined.append_range(part);
	}
	return joined;
}

[[nodiscard]] auto word(Encoder const& text, int index) -> Tokens {
	return text(std::format("w{} ", index));
}

[[nodiscard]] auto user(Encoder const& text, int index) -> Tokens {
	return join({{im_start}, text("user\n"), word(text, index), {im_end}, text("\n")});
}

[[nodiscard]] auto generation(Encoder const& text) -> Tokens {
	return join({{im_start}, text("assistant\n<think>\n")});
}

constexpr auto none = sched::Marks{.system_end = std::nullopt, .last_user_start = std::nullopt, .generation_start = std::nullopt};

struct Upstream {
	bool replays_next;
	bool marks_follow;
	bool replays_later;
	bool edited_kept;
	bool unrelated_kept;
	bool met_once;
};

/** The cases upstream's turn replay was checked against, on real tokens. */
[[nodiscard]] auto upstream(Encoder const& text) -> Upstream {
	auto replay = sched::Replay{};
	auto filler = Tokens{};
	for (auto index = 20; index < 50; ++index) {
		filler.append_range(word(text, index));
	}
	auto const prompt = join({user(text, 10), filler, generation(text)});
	// The model wrote two newlines before </think>; the template renders one.
	auto const generated = join({word(text, 12), text("\n\n"), text("</think>\n\n"), text("<tool_call>\n"), word(text, 13), {im_end}});
	auto const rendered = join({word(text, 12), text("\n"), text("</think>\n\n"), text("<tool_call>\n"), word(text, 13), {im_end}});
	replay.finished(prompt, generated);
	auto const next = join({prompt, rendered, text("\n"), user(text, 14), generation(text)});
	auto const query = prompt.size() + rendered.size() + text("\n").size();
	auto const out = replay.apply(text.tokenizer(), next, {.system_end = std::nullopt, .last_user_start = query, .generation_start = std::nullopt});
	auto const expected = join({prompt, generated, text("\n"), user(text, 14), generation(text)});
	auto const moved = query + generated.size() - rendered.size();

	// The next turn: the first renders the same way again, and a second, which ended with the other end-of-generation
	// token, renders with one newline fewer: it replays only because a turn always closes with <|im_end|>.
	auto const second = join({word(text, 15), text("\n\n"), text("</think>\n\n"), word(text, 16)});
	auto const second_rendered = join({word(text, 15), text("\n"), text("</think>\n\n"), word(text, 16)});
	replay.finished(out.prompt, join({second, {endoftext}}));
	auto const later = join({next, second_rendered, {im_end}, text("\n"), user(text, 17), generation(text)});
	auto const later_out = replay.apply(text.tokenizer(), later, none);
	auto const later_expected = join({expected, second, {im_end}, text("\n"), user(text, 17), generation(text)});

	// A client that edits the turn's text keeps its rendering.
	replay.finished(prompt, generated);
	auto const edited = join({prompt, word(text, 12), text("\n"), text("</think>\n\n"), text("<tool_call>\n"), word(text, 18), {im_end},
	                          text("\n"), user(text, 14), generation(text)});
	auto const edited_out = replay.apply(text.tokenizer(), edited, none);

	auto const unrelated = join({user(text, 40), generation(text)});
	auto const unrelated_out = replay.apply(text.tokenizer(), unrelated, none);

	// A pending turn is met once: the second prompt finds it gone (the stored pair still replays the turn).
	replay.finished(prompt, generated);
	auto const pending = replay.pending();
	std::ignore = replay.apply(text.tokenizer(), next, none);
	auto const after_first = replay.pending();
	auto const again = replay.apply(text.tokenizer(), next, none);
	return {
	    .replays_next = out.prompt == expected && generated != rendered,
	    .marks_follow = out.marks.last_user_start == moved && out.prompt[moved] == im_start,
	    .replays_later = later_out.prompt == later_expected,
	    .edited_kept = edited_out.prompt == edited,
	    .unrelated_kept = unrelated_out.prompt == unrelated,
	    .met_once = after_first + 1 == pending && replay.pending() == after_first && again.prompt == expected,
	};
}

using Rng = std::mt19937_64;

/** A turn's words with random separators and markup that a parser normalizes; `edit` changes one word. */
[[nodiscard]] auto turn(Encoder const& text, Rng& rng, std::span<int const> words, std::optional<std::size_t> edit) -> Tokens {
	static constexpr auto separators = std::array{std::string_view{" "}, std::string_view{"\n"}, std::string_view{"\n\n"}, std::string_view{"  "}};
	auto tokens = Tokens{};
	for (auto const [index, each] : std::views::enumerate(words)) {
		auto const value = edit == narrow<std::size_t>(index) ? each + 1 : each;
		tokens.append_range(text(std::format("v{}", value)));
		tokens.append_range(text(separators.at(rng() % separators.size())));
		if (index == 0) {
			static constexpr auto markup = std::array{std::string_view{"</think>\n\n"}, std::string_view{"<tool_call>\n"}, std::string_view{"<think>\n"}};
			if (auto const pick = rng() % (markup.size() + 1); pick < markup.size()) {
				tokens.append_range(text(markup.at(pick)));
			}
		}
	}
	tokens.push_back(im_end);
	return tokens;
}

/**
 * For random turns: a rendering that differs from the generated turn in whitespace alone is replayed, one with a word
 * changed is not, and the marks after the turn move by the change in length. Seeds 1 to 60.
 */
[[nodiscard]] auto search(Encoder const& text) -> bool {
	for (auto seed = std::uint64_t{1}; seed <= 60; ++seed) {
		auto rng = Rng{seed};
		auto replay = sched::Replay{};
		auto words = std::vector<int>(1 + (rng() % 12));
		std::ranges::generate(words, [&] {
			return narrow<int>(rng() % 500);
		});
		auto const prompt = join({user(text, narrow<int>(seed)), generation(text)});
		auto const generated = turn(text, rng, words, std::nullopt);
		auto const edit = rng() % 3 == 0 ? std::optional{narrow<std::size_t>(rng() % words.size())} : std::nullopt;
		auto const rendered = turn(text, rng, words, edit);
		replay.finished(prompt, generated);
		auto const next = join({prompt, rendered, text("\n"), user(text, 7), generation(text)});
		auto const query = prompt.size() + rendered.size() + text("\n").size();
		auto const out = replay.apply(text.tokenizer(), next, {.system_end = 0, .last_user_start = query, .generation_start = next.size() - 1});
		auto const replayed = join({prompt, generated, text("\n"), user(text, 7), generation(text)});
		auto const expected = edit ? next : replayed;
		auto const moved = edit ? query : query + generated.size() - rendered.size();
		if (out.prompt != expected || out.marks.last_user_start != moved || out.marks.system_end != 0 ||
		    out.marks.generation_start != out.prompt.size() - 1 || replay.pending() != 0) {
			std::println("seed {}: replay disagrees with the whitespace rule", seed);
			return false;
		}
	}
	return true;
}

/** However many turns finish, at most `replay_recent` wait, the newest. */
[[nodiscard]] auto pending_bounded(Encoder const& text) -> bool {
	auto replay = sched::Replay{};
	auto const generated = join({word(text, 1), {im_end}});
	for (auto index = 0; index < 200; ++index) {
		replay.finished(join({user(text, index), generation(text)}), generated);
	}
	auto const oldest = join({user(text, 0), generation(text), word(text, 2), {im_end}, user(text, 3), generation(text)});
	auto const newest = join({user(text, 199), generation(text), word(text, 1), text("  "), {im_end}, user(text, 3), generation(text)});
	auto const pending = replay.pending();
	std::ignore = replay.apply(text.tokenizer(), oldest, none);
	auto const after_oldest = replay.pending();
	auto const out = replay.apply(text.tokenizer(), newest, none);
	return pending == sched::replay_recent && after_oldest == pending && replay.pending() == pending - 1 && out.prompt != newest;
}

}

auto main() -> int {
	auto const loaded = tokenizer::load();
	if (!loaded) {
		std::println("the tokenizer image does not load");
		return 1;
	}
	auto const text = Encoder{*loaded};
	auto const cases_of = upstream(text);
	auto const cases = std::array{
	    engine::testing::Case{.name = "the next prompt replays a turn whose rendering differs in whitespace", .holds = cases_of.replays_next},
	    engine::testing::Case{.name = "marks after a replayed turn move with it", .holds = cases_of.marks_follow},
	    engine::testing::Case{.name = "later prompts replay the stored turn; a turn ended by the other end token closes with <|im_end|>",
	                          .holds = cases_of.replays_later},
	    engine::testing::Case{.name = "a turn the client edited keeps its rendering", .holds = cases_of.edited_kept},
	    engine::testing::Case{.name = "an unrelated prompt passes unchanged", .holds = cases_of.unrelated_kept},
	    engine::testing::Case{.name = "a finished turn is met by one prompt only", .holds = cases_of.met_once},
	    engine::testing::Case{.name = "random turns: whitespace and markup differences replay, edits do not (seeds 1-60)", .holds = search(text)},
	    engine::testing::Case{.name = "at most the most recent finished turns wait for their next prompt", .holds = pending_bounded(text)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:split;

import std;
import engine.core;
import :unicode;

namespace engine::tokenizer {

/** What the pretokenizer's pattern can tell apart about one character. */
enum class Kind : std::uint8_t {
	Other,
	Letter,
	Mark,
	Number,
	/** U+0020, the one space the pattern names literally. */
	Space,
	/** CR or LF. */
	Newline,
	/** Any other `\s`. */
	Whitespace,
};

[[nodiscard]] auto kind_of(Unicode const& unicode, char32_t point) -> Kind {
	if (point == U' ') {
		return Kind::Space;
	}
	if (point == U'\r' || point == U'\n') {
		return Kind::Newline;
	}
	switch (class_of(unicode, point)) {
	case Class::Other:
		return Kind::Other;
	case Class::Letter:
		return Kind::Letter;
	case Class::Mark:
		return Kind::Mark;
	case Class::Number:
		return Kind::Number;
	case Class::Whitespace:
		return Kind::Whitespace;
	}
	return Kind::Other;
}

[[nodiscard]] constexpr auto lettered(Kind kind) -> bool {
	return kind == Kind::Letter || kind == Kind::Mark;
}

[[nodiscard]] constexpr auto spacing(Kind kind) -> bool {
	return kind == Kind::Space || kind == Kind::Newline || kind == Kind::Whitespace;
}

// The contraction alternative is case-insensitive, and the reference regex engine folds case over unicode: besides
// upper and lower ascii, U+017F (long s) matches `s`. Probing every codepoint against the reference found no other.
[[nodiscard]] constexpr auto folds_to(char32_t point, char32_t lower) -> bool {
	return point == lower || point == lower - (U'a' - U'A') || (lower == U's' && point == U'ſ');
}

inline constexpr auto contractions = std::array<std::u32string_view, 7>{U"s", U"t", U"re", U"ve", U"m", U"ll", U"d"};

/** The length of the contraction at the front of the text, if one is there. */
[[nodiscard]] constexpr auto contraction(std::span<char32_t const> text) -> std::optional<std::size_t> {
	if (text.empty() || text.front() != U'\'') {
		return std::nullopt;
	}
	for (auto const suffix : contractions) {
		if (text.size() > suffix.size() && std::ranges::equal(text.subspan(1, suffix.size()), suffix, folds_to)) {
			return suffix.size() + 1;
		}
	}
	return std::nullopt;
}

/** The pretokenizer's pattern, exactly as tokenizer.json states it. `split` is this pattern, matched by hand. */
export inline constexpr std::string_view pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

/**
 * Splits text into the pieces `pattern` matches, calling `emit(begin, end)` for each, in order; the pieces tile the
 * text. The alternatives are tried in the pattern's order, with the backtracking each one needs worked out once.
 * Every alternative extends over one run of a kind, and each character is examined a bounded number of times, so
 * the split is linear in the text.
 */
template<class Emit>
auto split(std::span<char32_t const> text, std::span<Kind const> kinds, Emit const& emit) -> void {
	core::invariant(text.size() == kinds.size());
	auto const size = kinds.size();
	auto const run = [&](std::size_t from, auto const& member) {
		while (from < size && member(kinds[from])) {
			++from;
		}
		return from;
	};
	auto at = std::size_t{0};
	while (at < size) {
		auto const here = kinds[at];
		auto const next = at + 1 < size ? std::optional{kinds[at + 1]} : std::nullopt;
		auto end = at;
		if (auto const length = contraction(text.subspan(at))) {
			end = at + *length;
		} else if (here != Kind::Newline && here != Kind::Letter && here != Kind::Number && next && lettered(*next)) {
			end = run(at + 1, lettered);
		} else if (lettered(here)) {
			end = run(at, lettered);
		} else if (here == Kind::Number) {
			end = at + 1;
		} else if (here == Kind::Other || (here == Kind::Space && next == Kind::Other)) {
			auto const others = run(here == Kind::Space ? at + 1 : at, [](Kind kind) {
				return kind == Kind::Other;
			});
			end = run(others, [](Kind kind) {
				return kind == Kind::Newline;
			});
		} else {
			// Whitespace: `\s*[\r\n]+` takes the run up to its last newline; failing that, `\s+(?!\S)` takes the run but
			// leaves its last character for the non-space after it, unless the run ends the text or is one character.
			auto const spaces = run(at, spacing);
			auto last_newline = std::optional<std::size_t>{};
			for (auto position = at; position < spaces; ++position) {
				if (kinds[position] == Kind::Newline) {
					last_newline = position;
				}
			}
			if (last_newline) {
				end = *last_newline + 1;
			} else if (spaces == size || spaces - at == 1) {
				end = spaces;
			} else {
				end = spaces - 1;
			}
		}
		emit(at, end);
		at = end;
	}
}

}

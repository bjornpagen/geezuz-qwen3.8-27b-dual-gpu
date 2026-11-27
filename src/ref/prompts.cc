// SPDX-License-Identifier: AGPL-3.0-only
//
// The determinism harness's prompt set and its tokens, held once so the reference and the engine hash the same token
// sequence under each prompt name. A prompts file is utf-8 text, one prompt per line: its name (one word: no space,
// tab or carriage return), one space, then its text to the end of the line. Blank lines are skipped. The text is plain
// text: a special token's spelling inside it is read as the characters it is made of, never as that token.
export module engine.ref:prompts;

import std;
import engine.core;
import engine.tokenizer;
import engine.vocabulary;

namespace engine::ref {

/** One named prompt of the harness's set. */
export struct Prompt {
	std::string name;
	std::string text;

	[[nodiscard]] auto operator==(Prompt const&) const -> bool = default;
};

/** Why text is not a prompt set; each kind names what the caller fixes in the file. */
export enum class PromptsFailure : std::uint8_t {
	/** No prompt at all: an empty file or blank lines only. */
	Empty,
	/** A line that is not a name, one space and a text: no space, a leading space, or nothing after the space. */
	Unnamed,
	/** A name holding a tab, or a line holding a carriage return (a file with crlf line ends). */
	Unreadable,
	/** A name used by an earlier line. */
	Duplicate,
};

export struct [[nodiscard]] PromptsError {
	PromptsFailure failure;
	/** The 1-based line the failure is on; zero for `Empty`. */
	std::size_t line;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case PromptsFailure::Empty:
		case PromptsFailure::Unnamed:
		case PromptsFailure::Unreadable:
		case PromptsFailure::Duplicate:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(PromptsError const&) const -> bool = default;
};

/** The prompts of a prompts file, in file order. */
export [[nodiscard]] auto parse_prompts(std::string_view text) -> core::Result<std::vector<Prompt>, PromptsError> {
	auto prompts = std::vector<Prompt>{};
	auto names = std::set<std::string, std::less<>>{};
	auto number = std::size_t{0};
	for (auto const piece : std::views::split(text, '\n')) {
		++number;
		auto const line = std::string_view{piece.begin(), piece.end()};
		if (line.empty()) {
			continue;
		}
		auto const fail = [number](PromptsFailure failure) {
			return std::unexpected(PromptsError{.failure = failure, .line = number});
		};
		auto const space = line.find(' ');
		if (space == std::string_view::npos || space == 0 || space + 1 == line.size()) {
			return fail(PromptsFailure::Unnamed);
		}
		auto const name = line.substr(0, space);
		if (line.contains('\r') || name.contains('\t')) {
			return fail(PromptsFailure::Unreadable);
		}
		if (!names.emplace(name).second) {
			return fail(PromptsFailure::Duplicate);
		}
		prompts.push_back({.name = std::string{name}, .text = std::string{line.substr(space + 1)}});
	}
	if (prompts.empty()) {
		return std::unexpected(PromptsError{.failure = PromptsFailure::Empty, .line = 0});
	}
	return prompts;
}

/** Why a prompt's text has no token sequence. */
export enum class PromptTokensFailure : std::uint8_t {
	/** The text is not utf-8. */
	NotUtf8,
	/** The text encodes to no token, so it has no row to hash. */
	Empty,
};

export struct [[nodiscard]] PromptTokensError {
	PromptTokensFailure failure;
	/** The byte offset of the first invalid sequence; zero for `Empty`. */
	std::size_t offset;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case PromptTokensFailure::NotUtf8:
		case PromptTokensFailure::Empty:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(PromptTokensError const&) const -> bool = default;
};

/** A prompt's text as the token sequence every side of the harness feeds and hashes. */
export [[nodiscard]] auto prompt_tokens(tokenizer::Tokenizer const& tokenizer, std::string_view text)
    -> core::Result<std::vector<vocabulary::Token>, PromptTokensError> {
	auto const pieces = std::array{tokenizer::Piece{tokenizer::Text{.utf8 = text, .spelling = tokenizer::Spelling::Characters}}};
	auto ids = tokenizer.encode(pieces);
	if (!ids) {
		return std::unexpected(PromptTokensError{.failure = PromptTokensFailure::NotUtf8, .offset = ids.error().offset});
	}
	if (ids->empty()) {
		return std::unexpected(PromptTokensError{.failure = PromptTokensFailure::Empty, .offset = 0});
	}
	return std::move(*ids);
}

}

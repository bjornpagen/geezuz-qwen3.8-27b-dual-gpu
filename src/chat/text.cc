// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:text;

import std;

namespace engine::chat {

/** One character at the front of utf-8 text. */
struct Character {
	char32_t value;
	std::size_t length;
};

[[nodiscard]] constexpr auto byte_at(std::string_view text, std::size_t at) -> std::uint32_t {
	return std::bit_cast<std::uint8_t>(text[at]);
}

/**
 * The character at the front of non-empty text, or nothing when the text ends inside it. A byte that cannot start
 * a character, or a lead byte without its continuations, reads as one character of its own that is no whitespace:
 * the engine's text is utf-8, so these only keep the reading total.
 */
[[nodiscard]] constexpr auto front(std::string_view text) -> std::optional<Character> {
	auto const lead = byte_at(text, 0);
	auto const length = lead < 0x80U    ? std::size_t{1}
	                    : lead >= 0xf0U ? std::size_t{4}
	                    : lead >= 0xe0U ? std::size_t{3}
	                    : lead >= 0xc0U ? std::size_t{2}
	                                    : std::size_t{1};
	if (length == 1) {
		return Character{.value = lead < 0x80U ? char32_t{lead} : char32_t{0xfffd}, .length = 1};
	}
	auto value = char32_t{lead & (0x7fU >> length)};
	for (auto at = std::size_t{1}; at < length; ++at) {
		if (at == text.size()) {
			return std::nullopt;
		}
		auto const next = byte_at(text, at);
		if ((next & 0xc0U) != 0x80U) {
			return Character{.value = 0xfffd, .length = 1};
		}
		value = (value << 6U) | (next & 0x3fU);
	}
	return Character{.value = value, .length = length};
}

/** Python's `str.isspace`: what `str.strip` removes, and so what the chat template's `trim` removes. */
[[nodiscard]] constexpr auto python_space(char32_t c) -> bool {
	return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
	       c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

/** The `regex` module's `\s`, which the response template's patterns use: Unicode White_Space. */
[[nodiscard]] constexpr auto regex_space(char32_t c) -> bool {
	// The information separators U+001C..U+001F are Python whitespace but not Unicode White_Space.
	return python_space(c) && (c < 0x1c || c > 0x1f);
}

/** Text without its leading and trailing Python whitespace, as `str.strip` and the template's `trim` leave it. */
[[nodiscard]] constexpr auto strip(std::string_view text) -> std::string_view {
	while (!text.empty()) {
		auto const first = front(text);
		if (!first || !python_space(first->value)) {
			break;
		}
		text.remove_prefix(first->length);
	}
	while (!text.empty()) {
		// The last character starts at the last byte that is not a continuation byte.
		auto start = text.size() - 1;
		while (start > 0 && text.size() - start < 4 && (byte_at(text, start) & 0xc0U) == 0x80U) {
			--start;
		}
		auto const last = front(text.substr(start));
		if (!last || start + last->length != text.size() || !python_space(last->value)) {
			break;
		}
		text.remove_suffix(last->length);
	}
	return text;
}

}

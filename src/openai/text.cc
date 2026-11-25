// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:text;

import std;

namespace engine::openai {

/** The width of the UTF-8 sequence `lead` begins; the text is well formed. */
[[nodiscard]] constexpr auto sequence_width(char lead) -> std::size_t {
	auto const byte = std::bit_cast<std::uint8_t>(lead);
	return byte < 0x80 ? 1 : byte < 0xe0 ? 2 : byte < 0xf0 ? 3 : 4;
}

[[nodiscard]] constexpr auto continuation(char byte) -> bool {
	return (std::bit_cast<std::uint8_t>(byte) & 0xc0U) == 0x80U;
}

/** The code point of the well-formed UTF-8 sequence that is `bytes`, whole. */
[[nodiscard]] constexpr auto code_point(std::string_view bytes) -> char32_t {
	auto const lead = std::bit_cast<std::uint8_t>(bytes.front());
	auto const payload = bytes.size() == 1 ? lead : bytes.size() == 2 ? (lead & 0x1fU) : bytes.size() == 3 ? (lead & 0x0fU) : (lead & 0x07U);
	auto point = char32_t{payload};
	for (auto const byte : bytes.substr(1)) {
		point = (point << 6U) | (std::bit_cast<std::uint8_t>(byte) & 0x3fU);
	}
	return point;
}

/**
 * The longest prefix of well-formed `text` no longer than `bytes` that ends on a character boundary, but never
 * shorter than the first character: a piece of text always makes progress.
 */
[[nodiscard]] constexpr auto character_prefix(std::string_view text, std::size_t bytes) -> std::size_t {
	if (bytes >= text.size()) {
		return text.size();
	}
	while (bytes > 0 && continuation(text[bytes])) {
		--bytes;
	}
	return bytes > 0 ? bytes : sequence_width(text.front());
}

}

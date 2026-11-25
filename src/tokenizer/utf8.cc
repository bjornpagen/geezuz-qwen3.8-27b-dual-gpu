// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:utf8;

import std;
import engine.core;

namespace engine::tokenizer {

/** What one lead byte allows: the sequence length, the payload bits it carries, and the range of the next byte. */
struct Lead {
	std::size_t length;
	std::uint8_t payload;
	std::uint8_t low;
	std::uint8_t high;
};

// The second byte's range depends on the lead: it is what excludes overlong forms, surrogates and values past
// U+10FFFF, and it is what decides where an invalid sequence's maximal subpart ends.
[[nodiscard]] constexpr auto lead(std::uint8_t first) -> std::optional<Lead> {
	if (first < 0x80) {
		return Lead{.length = 1, .payload = 0x7f, .low = 0, .high = 0};
	}
	if (first < 0xc2) {
		return std::nullopt;
	}
	if (first < 0xe0) {
		return Lead{.length = 2, .payload = 0x1f, .low = 0x80, .high = 0xbf};
	}
	if (first < 0xf0) {
		auto const low = first == 0xe0 ? std::uint8_t{0xa0} : std::uint8_t{0x80};
		auto const high = first == 0xed ? std::uint8_t{0x9f} : std::uint8_t{0xbf};
		return Lead{.length = 3, .payload = 0x0f, .low = low, .high = high};
	}
	if (first < 0xf5) {
		auto const low = first == 0xf0 ? std::uint8_t{0x90} : std::uint8_t{0x80};
		auto const high = first == 0xf4 ? std::uint8_t{0x8f} : std::uint8_t{0xbf};
		return Lead{.length = 4, .payload = 0x07, .low = low, .high = high};
	}
	return std::nullopt;
}

[[nodiscard]] constexpr auto byte_at(std::string_view bytes, std::size_t at) -> std::uint8_t {
	return std::bit_cast<std::uint8_t>(bytes[at]);
}

/** A whole scalar value at the front of the bytes. */
struct Scalar {
	char32_t value;
	std::size_t length;
};

/** The front bytes are no prefix of any scalar's encoding: the maximal subpart, at least one byte, is replaced. */
struct Invalid {
	std::size_t length;
};

/** Every byte left is a proper prefix of some scalar's encoding: more bytes decide. */
struct Incomplete {};

using Step = std::variant<Scalar, Invalid, Incomplete>;

/** Reads the front of non-empty bytes. */
[[nodiscard]] constexpr auto step(std::string_view bytes) -> Step {
	core::invariant(!bytes.empty());
	auto const first = lead(byte_at(bytes, 0));
	if (!first) {
		return Invalid{.length = 1};
	}
	auto value = char32_t{byte_at(bytes, 0)} & char32_t{first->payload};
	for (auto at = std::size_t{1}; at < first->length; ++at) {
		if (at == bytes.size()) {
			return Incomplete{};
		}
		auto const next = byte_at(bytes, at);
		auto const low = at == 1 ? first->low : std::uint8_t{0x80};
		auto const high = at == 1 ? first->high : std::uint8_t{0xbf};
		if (next < low || next > high) {
			return Invalid{.length = at};
		}
		value = (value << 6U) | (char32_t{next} & 0x3fU);
	}
	return Scalar{.value = value, .length = first->length};
}

/** One scalar's utf-8 encoding. */
struct Encoded {
	std::array<char, 4> bytes;
	std::size_t length;

	[[nodiscard]] constexpr auto view() const -> std::string_view {
		return {bytes.data(), length};
	}
};

[[nodiscard]] constexpr auto as_char(char32_t bits) -> char {
	return std::bit_cast<char>(core::narrow<std::uint8_t>(std::uint32_t{bits}));
}

/** Encodes a scalar value; callers hold only values `step` produced or the replacement character. */
[[nodiscard]] constexpr auto utf8(char32_t value) -> Encoded {
	if (value < 0x80) {
		return {.bytes = {as_char(value), 0, 0, 0}, .length = 1};
	}
	if (value < 0x800) {
		return {.bytes = {as_char(0xc0U | (value >> 6U)), as_char(0x80U | (value & 0x3fU)), 0, 0}, .length = 2};
	}
	if (value < 0x10000) {
		return {.bytes = {as_char(0xe0U | (value >> 12U)), as_char(0x80U | ((value >> 6U) & 0x3fU)), as_char(0x80U | (value & 0x3fU)), 0},
		        .length = 3};
	}
	return {.bytes = {as_char(0xf0U | (value >> 18U)), as_char(0x80U | ((value >> 12U) & 0x3fU)), as_char(0x80U | ((value >> 6U) & 0x3fU)),
	                  as_char(0x80U | (value & 0x3fU))},
	        .length = 4};
}

inline constexpr std::string_view replacement = "\xef\xbf\xbd";

/** Where the bytes stop being utf-8: the byte offset of the first invalid or truncated sequence. */
struct NotUtf8 {
	std::size_t offset;
};

/** Decodes valid utf-8 into scalar values, or names where it is not utf-8. */
[[nodiscard]] constexpr auto scalars(std::string_view bytes) -> std::expected<std::vector<char32_t>, NotUtf8> {
	auto out = std::vector<char32_t>{};
	out.reserve(bytes.size());
	auto at = std::size_t{0};
	while (at < bytes.size()) {
		auto const next = step(bytes.substr(at));
		if (!std::holds_alternative<Scalar>(next)) {
			return std::unexpected(NotUtf8{.offset = at});
		}
		auto const scalar = std::get<Scalar>(next);
		out.push_back(scalar.value);
		at += scalar.length;
	}
	return out;
}

}

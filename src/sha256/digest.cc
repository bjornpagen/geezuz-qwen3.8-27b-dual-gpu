// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sha256:digest;

import std;
import engine.core;

namespace engine::sha256 {

export inline constexpr std::size_t digest_bytes = 32;

/** A SHA-256 digest: its 32 bytes in the order the standard writes them. */
export struct Digest {
	std::array<std::byte, digest_bytes> bytes;

	[[nodiscard]] constexpr auto operator==(Digest const&) const -> bool = default;
};

/** Why text is not the hex spelling of a fixed number of bytes. */
export enum class HexFailure : std::uint8_t {
	/** Not exactly two digits per byte. */
	WrongLength,
	/** A character outside 0-9, a-f and A-F. */
	NotHex,
};

export struct [[nodiscard]] HexError {
	HexFailure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case HexFailure::WrongLength:
		case HexFailure::NotHex:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(HexError const&) const -> bool = default;
};

inline constexpr std::string_view lower_digits = "0123456789abcdef";
inline constexpr std::string_view upper_digits = "0123456789ABCDEF";

/** Lowercase hex, two digits per byte, the high nibble first: the spelling `sha256sum` and git print. */
export template<std::size_t N>
[[nodiscard]] constexpr auto hex(std::array<std::byte, N> const& bytes) -> std::array<char, 2 * N> {
	auto text = std::array<char, 2 * N>{};
	for (auto index = std::size_t{0}; index < N; ++index) {
		auto const value = std::to_integer<std::size_t>(bytes.at(index));
		text.at(2 * index) = lower_digits[value >> 4U];
		text.at((2 * index) + 1) = lower_digits[value & 0xfU];
	}
	return text;
}

[[nodiscard]] constexpr auto nibble(char digit) -> std::optional<std::uint8_t> {
	auto const lower = lower_digits.find(digit);
	auto const found = lower == std::string_view::npos ? upper_digits.find(digit) : lower;
	if (found == std::string_view::npos) {
		return std::nullopt;
	}
	return core::narrow<std::uint8_t>(found);
}

/** Decodes exactly `2 N` hex digits of either case into `N` bytes, the high nibble first. */
export template<std::size_t N>
[[nodiscard]] constexpr auto from_hex(std::string_view text) -> core::Result<std::array<std::byte, N>, HexError> {
	if (text.size() != 2 * N) {
		return std::unexpected(HexError{.failure = HexFailure::WrongLength});
	}
	auto bytes = std::array<std::byte, N>{};
	for (auto index = std::size_t{0}; index < N; ++index) {
		auto const high = nibble(text[2 * index]);
		auto const low = nibble(text[(2 * index) + 1]);
		if (!high || !low) {
			return std::unexpected(HexError{.failure = HexFailure::NotHex});
		}
		bytes.at(index) = std::byte{core::narrow<std::uint8_t>((std::uint32_t{*high} << 4U) | std::uint32_t{*low})};
	}
	return bytes;
}

/** Reads a digest spelled as 64 hex digits, as `sha256sum` and Hugging Face print it. */
export [[nodiscard]] constexpr auto parse(std::string_view text) -> core::Result<Digest, HexError> {
	return from_hex<digest_bytes>(text).transform([](std::array<std::byte, digest_bytes> const& bytes) { return Digest{.bytes = bytes}; });
}

/** A digest written in the source: malformed text does not compile. */
export [[nodiscard]] consteval auto pinned(std::string_view text) -> Digest {
	auto const read = parse(text);
	core::invariant(read.has_value());
	return *read;
}

}

/** Formats a digest as its 64 lowercase hex digits; it takes no format specification. */
export template<>
struct std::formatter<engine::sha256::Digest> {
	constexpr auto parse(std::format_parse_context& context) -> std::format_parse_context::iterator {
		return context.begin();
	}

	auto format(engine::sha256::Digest const& digest, std::format_context& context) const -> std::format_context::iterator {
		auto const text = engine::sha256::hex(digest.bytes);
		return std::ranges::copy(text, context.out()).out;
	}
};

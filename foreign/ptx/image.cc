// SPDX-License-Identifier: AGPL-3.0-only
export module engine.ptx:image;

import std;
import engine.core;

namespace engine::ptx {

/** What a ptx image's header promises the driver's jit: its isa (major * 10 + minor), target sm and address width. */
export struct Header {
	int isa;
	int sm;
	int address_bits;

	[[nodiscard]] constexpr auto operator==(Header const&) const -> bool = default;
};

/**
 * A ptx image embedded in the binary. `text` ends in its one NUL, because the driver's loader takes a NUL-terminated
 * string; everything before it is the ptx.
 */
export struct Image {
	std::span<char const> text;
	Header header;
};

[[nodiscard]] constexpr auto parse_decimal(std::string_view digits) -> std::optional<int> {
	if (digits.empty() || digits.size() > 4) {
		return std::nullopt;
	}
	auto value = 0;
	for (auto const digit : digits) {
		if (digit < '0' || digit > '9') {
			return std::nullopt;
		}
		value = value * 10 + (digit - '0');
	}
	return value;
}

[[nodiscard]] constexpr auto directive(std::string_view text, std::string_view name) -> std::optional<std::string_view> {
	for (auto const line : text | std::views::split('\n')) {
		auto const view = std::string_view{line.begin(), line.end()};
		if (view.starts_with(name)) {
			return view.substr(name.size());
		}
	}
	return std::nullopt;
}

/** Reads the header directives of ptx text; nullopt when any of them is missing or malformed. */
export [[nodiscard]] constexpr auto parse_header(std::string_view text) -> std::optional<Header> {
	auto const version = directive(text, ".version ");
	auto const target = directive(text, ".target sm_");
	auto const address = directive(text, ".address_size ");
	if (!version || !target || !address) {
		return std::nullopt;
	}
	auto const dot = version->find('.');
	if (dot == std::string_view::npos) {
		return std::nullopt;
	}
	auto const major = parse_decimal(version->substr(0, dot));
	auto const minor = parse_decimal(version->substr(dot + 1));
	auto const sm = parse_decimal(*target);
	auto const bits = parse_decimal(*address);
	if (!major || !minor || *minor > 9 || !sm || !bits) {
		return std::nullopt;
	}
	return Header{.isa = *major * 10 + *minor, .sm = *sm, .address_bits = *bits};
}

/**
 * The image of embedded ptx text, checked while compiling: a missing terminator or an unreadable header does not
 * build.
 */
export [[nodiscard]] consteval auto image(std::span<char const> text) -> Image {
	core::invariant(!text.empty() && text.back() == '\0');
	auto const body = std::string_view{text.data(), text.size() - 1};
	core::invariant(body.find('\0') == std::string_view::npos);
	return Image{.text = text, .header = parse_header(body).value()};
}

}

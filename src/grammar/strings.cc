// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:strings;

import std;
import engine.core;
import engine.regex;

namespace engine::grammar {

/**
 * The whitespace the engine lets JSON hold between its tokens: none, one space, or a newline and up to 48 spaces or
 * tabs of indentation. That admits compact JSON, Python's `", "` and `": "`, and pretty-printing up to 24 levels of
 * two-space indentation, while no run of whitespace is unbounded: a model cannot loop on it.
 */
export inline constexpr std::string_view json_whitespace = R"((?:\x20|\n[\x20\t]{0,48})?)";

[[nodiscard]] auto regex_of(regex::Store& store, std::string_view text) -> regex::Expr {
	auto parsed = regex::parse(store, text, regex::Matching::Whole);
	core::invariant(parsed.has_value());
	return *parsed;
}

[[nodiscard]] auto whitespace(regex::Store& store) -> regex::Expr {
	return regex_of(store, json_whitespace);
}

/** `ws text ws`, the shape of every JSON punctuation lexeme. */
[[nodiscard]] auto punctuation(regex::Store& store, std::string_view before, std::string_view text, std::string_view after) -> regex::Expr {
	auto const space = whitespace(store);
	return store.sequence(std::array{before.empty() ? regex::empty : space, store.literal(text), after.empty() ? regex::empty : space});
}

/** The characters JSON writes inside a string without an escape. */
[[nodiscard]] auto plain_characters() -> regex::CharClass {
	return regex::CharClass::all().minus(regex::CharClass::of(std::to_array<regex::Range>({
	    {.low = 0, .high = 0x1f},
	    {.low = '"', .high = '"'},
	    {.low = '\\', .high = '\\'},
	})));
}

/**
 * Every JSON string's text, quotes included: plain characters and escapes. A `\u` escape names a control character
 * only, the characters that must be escaped: a printable character is written as itself (as llguidance does), so a
 * value has one spelling and no escape can name a lone surrogate.
 */
[[nodiscard]] auto any_json_string(regex::Store& store) -> regex::Expr {
	auto const plain = regex::characters(store, plain_characters());
	auto const short_escape = store.concat(store.byte('\\'), regex_of(store, R"(["\\/bfnrt])"));
	auto const unicode_escape = store.concat(store.literal("\\u"), regex_of(store, "00[01][0-9a-fA-F]"));
	auto const content = store.repeat(store.any_of(std::array{plain, short_escape, unicode_escape}), 0, regex::unbounded);
	return store.sequence(std::array{store.byte('"'), content, store.byte('"')});
}

/**
 * The JSON text of strings whose values lie in `values` (a language over the value's utf-8), written with plain
 * characters only: values needing an escape are left out, so each value has one text and the text is the value.
 */
[[nodiscard]] auto plain_json_string(regex::Store& store, regex::Expr values) -> regex::Expr {
	auto const plain = store.repeat(regex::characters(store, plain_characters()), 0, regex::unbounded);
	return store.sequence(std::array{store.byte('"'), store.both(values, plain), store.byte('"')});
}

/** Texts not containing any word of `forbidden`. */
[[nodiscard]] auto avoiding(regex::Store& store, regex::Expr forbidden) -> regex::Expr {
	return store.complement(store.sequence(std::array{regex::everything, forbidden, regex::everything}));
}

/** The markers of a set, as one expression. */
[[nodiscard]] auto any_literal(regex::Store& store, std::span<std::string_view const> words) -> regex::Expr {
	auto choices = std::vector<regex::Expr>{};
	for (auto const word : words) {
		choices.push_back(store.literal(word));
	}
	return store.any_of(choices);
}

/** A string `format` the engine implements. */
export enum class Format : std::uint8_t {
	Date,
	Time,
	DateTime,
	Uuid,
	Ipv4,
	Email,
	Hostname,
};

[[nodiscard]] auto format_named(std::string_view name) -> std::optional<Format> {
	auto const names = std::to_array<std::pair<std::string_view, Format>>({
	    {"date", Format::Date},
	    {"time", Format::Time},
	    {"date-time", Format::DateTime},
	    {"uuid", Format::Uuid},
	    {"ipv4", Format::Ipv4},
	    {"email", Format::Email},
	    {"hostname", Format::Hostname},
	});
	auto const found = std::ranges::find(names, name, &std::pair<std::string_view, Format>::first);
	if (found == names.end()) {
		return std::nullopt;
	}
	return found->second;
}

/** RFC 3339 `full-date`, with each month's days and February's 29th only in leap years. */
inline constexpr std::string_view date_pattern =
    "(?:[0-9]{4}-(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])"
    "|[0-9]{4}-(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)"
    "|[0-9]{4}-02-(?:0[1-9]|1[0-9]|2[0-8])"
    "|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)";

/** RFC 3339 `full-time` without leap seconds: `HH:MM:SS[.frac]` and a `Z` or numeric offset. */
inline constexpr std::string_view time_pattern = "(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\\.[0-9]{1,9})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])";

/** One DNS label: letters, digits and inner hyphens, at most 63 characters. */
inline constexpr std::string_view label_pattern = "[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?";

/**
 * The language of a format. Each is a subset of what the format's specification admits that every common
 * validator accepts: no leap seconds, `T` and `Z` in capitals, a dot-atom email, an ascii hostname.
 */
[[nodiscard]] auto format_language(regex::Store& store, Format format) -> regex::Expr {
	auto const label = std::string{label_pattern};
	switch (format) {
	case Format::Date:
		return regex_of(store, date_pattern);
	case Format::Time:
		return regex_of(store, time_pattern);
	case Format::DateTime:
		return regex_of(store, std::string{date_pattern} + "T" + std::string{time_pattern});
	case Format::Uuid:
		return regex_of(store, "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");
	case Format::Ipv4:
		return regex_of(store, "(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])");
	case Format::Email: {
		// A local part of at most 64 characters, as SMTP bounds it, at a domain of two or three labels.
		auto const atom = std::string{"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"};
		auto const local = store.both(regex_of(store, atom + "(?:\\." + atom + ")*"), regex_of(store, "[^@]{1,64}"));
		return store.sequence(std::array{local, store.byte('@'), regex_of(store, label + "(?:\\." + label + "){1,2}")});
	}
	case Format::Hostname:
		// At most three labels of at most 63 characters: within the 253 a name may have, with no length to count.
		return regex_of(store, label + "(?:\\." + label + "){0,2}");
	}
	std::unreachable();
}

}

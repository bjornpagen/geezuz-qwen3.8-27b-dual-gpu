// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:cast;

import std;
import engine.core;
import engine.json;
import :text;
import :dumps;

namespace engine::chat {

/** A JSON-schema type the response template casts a tool-call argument to. A string needs no cast. */
export enum class Cast : std::uint8_t {
	Integer,
	Number,
	Boolean,
	Null,
	Object,
	Array,
};

/** The cast a schema type name asks for: none for `string`, nor for a name that is no JSON type. */
export [[nodiscard]] constexpr auto cast_named(std::string_view name) -> std::optional<Cast> {
	if (name == "integer") {
		return Cast::Integer;
	}
	if (name == "number") {
		return Cast::Number;
	}
	if (name == "boolean") {
		return Cast::Boolean;
	}
	if (name == "null") {
		return Cast::Null;
	}
	if (name == "object") {
		return Cast::Object;
	}
	if (name == "array") {
		return Cast::Array;
	}
	return std::nullopt;
}

/**
 * The casts a parameter's schema declares, in the order the response template tries them: its `type` (a name or a
 * list of names), then every `anyOf` and `oneOf` choice's, then `null` when `nullable` is truthy and no `null` came
 * before. Type names with no cast (`string`, unknown names) are dropped: trying them changes nothing.
 */
auto add_casts(std::vector<Cast>& casts, Json const& schema) -> void {
	if (!std::holds_alternative<Object>(schema.value)) {
		return;
	}
	auto const& object = std::get<Object>(schema.value);
	auto const add_named = [&casts](Json const& name) {
		if (std::holds_alternative<std::string>(name.value)) {
			if (auto const cast = cast_named(std::get<std::string>(name.value))) {
				casts.push_back(*cast);
			}
		}
	};
	if (auto const declared = object.find("type")) {
		if (std::holds_alternative<Array>(declared->value)) {
			std::ranges::for_each(std::get<Array>(declared->value), add_named);
		} else {
			add_named(*declared);
		}
	}
	for (auto const union_name : {std::string_view{"anyOf"}, std::string_view{"oneOf"}}) {
		// The reference iterates a union that is not a list (a mapping's keys, a string's characters), which names
		// no schema; a list is the only union that adds casts.
		if (auto const choices = object.find(union_name); choices && std::holds_alternative<Array>(choices->value)) {
			for (auto const& choice : std::get<Array>(choices->value)) {
				add_casts(casts, choice);
			}
		}
	}
	if (auto const nullable = object.find("nullable"); nullable && truthy(*nullable) && !std::ranges::contains(casts, Cast::Null)) {
		casts.push_back(Cast::Null);
	}
}

/** A parameter the response template casts, and its casts in the order it tries them. */
export struct ParameterCasts {
	std::string name;
	std::vector<Cast> casts;

	[[nodiscard]] auto operator==(ParameterCasts const&) const -> bool = default;
};

/**
 * The casts of a function tool's parameters, as the response template casts a call's arguments: for each property of
 * the `parameters` schema `schema`, in the order written, the casts its schema declares; a property with none (a
 * string, or no type) is left out. `reader` reads the schema; a schema that does not read back is its error.
 */
export template<json::Bounds B>
[[nodiscard]] auto parameter_casts(json::Reader<B>& reader, std::string_view schema) -> core::Result<std::vector<ParameterCasts>, json::ReadError> {
	auto const loaded = load(reader, schema);
	if (!loaded) {
		return std::unexpected(loaded.error());
	}
	auto out = std::vector<ParameterCasts>{};
	auto const properties = std::holds_alternative<Object>(loaded->value) ? std::get<Object>(loaded->value).find("properties") : std::nullopt;
	if (!properties || !std::holds_alternative<Object>(properties->value)) {
		return out;
	}
	for (auto const& property : std::get<Object>(properties->value).members) {
		auto casts = std::vector<Cast>{};
		add_casts(casts, property.value);
		if (!casts.empty()) {
			out.push_back(ParameterCasts{.name = property.name, .casts = std::move(casts)});
		}
	}
	return out;
}

/** The first code point of each run of ten Unicode decimal digits past ASCII (Unicode 15.1, as the reference's Python). */
inline constexpr auto decimal_zeros = std::to_array<char32_t>({
    0x660,   0x6f0,   0x7c0,   0x966,   0x9e6,   0xa66,   0xae6,   0xb66,   0xbe6,   0xc66,   0xce6,   0xd66,   0xde6,   0xe50,
    0xed0,   0xf20,   0x1040,  0x1090,  0x17e0,  0x1810,  0x1946,  0x19d0,  0x1a80,  0x1a90,  0x1b50,  0x1bb0,  0x1c40,  0x1c50,
    0xa620,  0xa8d0,  0xa900,  0xa9d0,  0xa9f0,  0xaa50,  0xabf0,  0xff10,  0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0,
    0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0,
});

static_assert(std::ranges::is_sorted(decimal_zeros));
static_assert(std::ranges::adjacent_find(decimal_zeros, [](char32_t a, char32_t b) {
	              return b - a < 10;
              }) == decimal_zeros.end());

[[nodiscard]] constexpr auto decimal_digit(char32_t c) -> std::optional<char> {
	auto const after = std::ranges::upper_bound(decimal_zeros, c);
	if (after == decimal_zeros.begin() || c - *std::prev(after) >= 10) {
		return std::nullopt;
	}
	return static_cast<char>('0' + (c - *std::prev(after)));
}

/**
 * The text Python's `int()` and `float()` parse: ASCII as itself, other whitespace as a space, other decimal digits
 * as their ASCII digit. Any other character makes the text no number at all.
 */
[[nodiscard]] auto number_text(std::string_view raw) -> std::optional<std::string> {
	auto out = std::string{};
	while (!raw.empty()) {
		auto const c = front(raw);
		if (!c) {
			return std::nullopt;
		}
		if (c->value < 0x7f) {
			out.push_back(static_cast<char>(c->value));
		} else if (python_space(c->value)) {
			out.push_back(' ');
		} else if (auto const digit = decimal_digit(c->value)) {
			out.push_back(*digit);
		} else {
			return std::nullopt;
		}
		raw.remove_prefix(c->length);
	}
	return out;
}

/** The whitespace C's `isspace` names, which Python's number parsers skip around a number. */
[[nodiscard]] constexpr auto ascii_space(char c) -> bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

[[nodiscard]] constexpr auto ascii_digit(char c) -> bool {
	return c >= '0' && c <= '9';
}

[[nodiscard]] constexpr auto trim_ascii(std::string_view text) -> std::string_view {
	while (!text.empty() && ascii_space(text.front())) {
		text.remove_prefix(1);
	}
	while (!text.empty() && ascii_space(text.back())) {
		text.remove_suffix(1);
	}
	return text;
}

/** Python refuses to read integers longer than this many digits (`sys.get_int_max_str_digits()`). */
inline constexpr std::size_t max_int_digits = 4300;

/** Python's `int(text)` written as `json.dumps` writes it, or nothing where Python raises. */
[[nodiscard]] auto python_int(std::string_view text) -> std::optional<std::string> {
	text = trim_ascii(text);
	auto const negative = text.starts_with('-');
	if (text.starts_with('-') || text.starts_with('+')) {
		text.remove_prefix(1);
	}
	if (text.empty() || !ascii_digit(text.front()) || !ascii_digit(text.back())) {
		return std::nullopt;
	}
	auto digits = std::string{};
	for (auto const [index, c] : std::views::enumerate(text)) {
		if (ascii_digit(c)) {
			digits.push_back(c);
		} else if (c != '_' || !ascii_digit(text[core::narrow<std::size_t>(index) + 1])) {
			// An underscore only ever stands between two digits; the first and last characters are digits.
			return std::nullopt;
		}
	}
	if (digits.size() > max_int_digits) {
		return std::nullopt;
	}
	auto const significant = std::min(digits.find_first_not_of('0'), digits.size() - 1);
	auto const magnitude = std::string_view{digits}.substr(significant);
	return negative && magnitude != "0" ? "-" + std::string{magnitude} : std::string{magnitude};
}

/** Whether ASCII text is a decimal the way Python's float parser reads one: digits with an optional point and exponent. */
[[nodiscard]] auto decimal_grammar(std::string_view text) -> bool {
	auto at = std::size_t{0};
	auto const digits = [&] {
		auto const begin = at;
		while (at < text.size() && ascii_digit(text[at])) {
			++at;
		}
		return at - begin;
	};
	auto mantissa = digits();
	if (at < text.size() && text[at] == '.') {
		++at;
		mantissa += digits();
	}
	if (mantissa == 0) {
		return false;
	}
	if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
		++at;
		if (at < text.size() && (text[at] == '+' || text[at] == '-')) {
			++at;
		}
		if (digits() == 0) {
			return false;
		}
	}
	return at == text.size();
}

[[nodiscard]] constexpr auto ascii_lower(char c) -> char {
	return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

[[nodiscard]] auto equal_folded(std::string_view text, std::string_view lower) -> bool {
	return std::ranges::equal(text, lower, {}, ascii_lower);
}

/** Python's `float(text)`, infinities and NaN included, or nothing where Python raises. */
[[nodiscard]] auto python_float_of(std::string_view text) -> std::optional<double> {
	text = trim_ascii(text);
	auto plain = std::string{};
	if (text.contains('_')) {
		// An underscore stands only between two digits, and is dropped.
		for (auto const [index, c] : std::views::enumerate(text)) {
			auto const at = core::narrow<std::size_t>(index);
			if (c != '_') {
				plain.push_back(c);
			} else if (at == 0 || at + 1 == text.size() || !ascii_digit(text[at - 1]) || !ascii_digit(text[at + 1])) {
				return std::nullopt;
			}
		}
		text = plain;
	}
	auto const negative = text.starts_with('-');
	if (text.starts_with('-') || text.starts_with('+')) {
		text.remove_prefix(1);
	}
	auto constexpr infinity = std::numeric_limits<double>::infinity();
	if (equal_folded(text, "inf") || equal_folded(text, "infinity")) {
		return negative ? -infinity : infinity;
	}
	if (equal_folded(text, "nan")) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (!decimal_grammar(text)) {
		return std::nullopt;
	}
	auto value = 0.0;
	auto const read = std::from_chars(text.data(), std::to_address(text.end()), value);
	if (read.ec == std::errc::result_out_of_range) {
		value = overflows(text) ? infinity : 0.0;
	} else {
		core::invariant(!std::make_error_code(read.ec) && read.ptr == std::to_address(text.end()));
	}
	return negative ? -value : value;
}

/** An integral double as the exact integer Python's `int()` makes of it. */
[[nodiscard]] auto exact_integer(double value) -> std::string {
	core::invariant(std::isfinite(value) && value == std::trunc(value));
	if (value == 0.0) {
		return "0";
	}
	auto buffer = std::array<char, 400>{};
	auto const written = std::to_chars(buffer.data(), std::to_address(buffer.end()), value, std::chars_format::fixed, 0);
	core::invariant(!std::make_error_code(written.ec));
	return std::string{buffer.data(), written.ptr};
}

[[nodiscard]] auto integral(double value) -> bool {
	return std::isfinite(value) && value == std::trunc(value);
}

/** One cast as the response template applies it to an argument's text: its JSON, or nothing when it does not apply. */
template<json::Bounds B>
    requires json::Enforceable<B>
[[nodiscard]] auto apply(Cast cast, std::string_view raw, json::Reader<B>& reader) -> std::optional<std::string> {
	switch (cast) {
	case Cast::Integer: {
		auto const text = number_text(raw);
		if (!text) {
			return std::nullopt;
		}
		if (auto integer = python_int(*text)) {
			return integer;
		}
		// The schema's integers include whole numbers written as reals ("3.0", "1e3").
		auto const real = python_float_of(*text);
		if (!real || !integral(*real)) {
			return std::nullopt;
		}
		return exact_integer(*real);
	}
	case Cast::Number: {
		auto const text = number_text(raw);
		auto const real = text ? python_float_of(*text) : std::nullopt;
		if (!real || !std::isfinite(*real)) {
			return std::nullopt;
		}
		// A whole number written without a point reads as an integer, as the reference keeps it.
		if (integral(*real) && !raw.contains('.')) {
			return exact_integer(*real);
		}
		return python_float(*real);
	}
	case Cast::Boolean: {
		auto const word = strip(raw);
		if (equal_folded(word, "true") || word == "1") {
			return "true";
		}
		if (equal_folded(word, "false") || word == "0") {
			return "false";
		}
		return std::nullopt;
	}
	case Cast::Null: {
		auto const word = strip(raw);
		if (word == "null" || word == "None") {
			return "null";
		}
		return std::nullopt;
	}
	case Cast::Object:
	case Cast::Array: {
		auto const value = load(reader, raw);
		if (!value) {
			return std::nullopt;
		}
		auto const wanted =
		    cast == Cast::Object ? std::holds_alternative<Object>(value->value) : std::holds_alternative<Array>(value->value);
		if (!wanted) {
			return std::nullopt;
		}
		return dumps(*value);
	}
	}
	std::unreachable();
}

/**
 * An argument's JSON after the response template's casts: the first of `casts` that applies, else the text as a
 * string. Object and array casts read the text as strict JSON, where the reference's `json.loads` also reads NaN,
 * infinities and repeated keys; such a text stays a string here, so the arguments stay JSON.
 */
template<json::Bounds B>
    requires json::Enforceable<B>
[[nodiscard]] auto cast_argument(std::string_view raw, std::span<Cast const> casts, json::Reader<B>& reader) -> std::string {
	for (auto const cast : casts) {
		if (auto json = apply(cast, raw, reader)) {
			return std::move(*json);
		}
	}
	auto out = std::string{};
	put_string(out, raw);
	return out;
}

}

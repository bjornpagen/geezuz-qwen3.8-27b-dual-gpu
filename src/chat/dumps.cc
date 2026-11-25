// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:dumps;

import std;
import engine.core;
import engine.json;

namespace engine::chat {

struct Json;
struct Member;

struct Null {
	[[nodiscard]] auto operator==(Null const&) const -> bool = default;
};

/** A number as Python's `json.dumps` writes the value `json.loads` reads from its literal. */
struct Number {
	std::string text;
};

using Array = std::vector<Json>;

struct Object {
	std::vector<Member> members;

	[[nodiscard]] auto find(std::string_view name) const -> std::optional<Json const&>;
};

/**
 * A JSON value as Python holds it after `json.loads`: what the chat template's `tojson` and the response template's
 * casts see. Objects keep their members in the order written.
 */
struct Json {
	std::variant<Null, bool, Number, std::string, Array, Object> value;
};

struct Member {
	std::string name;
	Json value;
};

auto Object::find(std::string_view name) const -> std::optional<Json const&> {
	auto const found = std::ranges::find(members, name, &Member::name);
	if (found == members.end()) {
		return std::nullopt;
	}
	return found->value;
}

/**
 * Appends `text` as Python's `json.dumps(..., ensure_ascii=False)` writes a string, which is how the chat template's
 * `tojson` writes one: only the quote, the backslash and the controls below U+0020 are escaped, with Python's short
 * forms where it has them and lowercase `\u00xx` otherwise.
 */
auto put_string(std::string& out, std::string_view text) -> void {
	out.push_back('"');
	for (auto const byte : text) {
		switch (byte) {
		case '"':
			out.append("\\\"");
			break;
		case '\\':
			out.append("\\\\");
			break;
		case '\n':
			out.append("\\n");
			break;
		case '\r':
			out.append("\\r");
			break;
		case '\t':
			out.append("\\t");
			break;
		case '\b':
			out.append("\\b");
			break;
		case '\f':
			out.append("\\f");
			break;
		default:
			if (auto const code = std::bit_cast<std::uint8_t>(byte); code < 0x20U) {
				std::format_to(std::back_inserter(out), "\\u{:04x}", code);
			} else {
				out.push_back(byte);
			}
		}
	}
	out.push_back('"');
}

/** A finite double as Python's `repr` writes it: the shortest digits that read back, fixed from 1e-4 to below 1e16. */
[[nodiscard]] auto python_float(double value) -> std::string {
	core::invariant(std::isfinite(value));
	if (value == 0.0) {
		return std::signbit(value) ? "-0.0" : "0.0";
	}
	auto buffer = std::array<char, 32>{};
	auto const written = std::to_chars(buffer.data(), std::to_address(buffer.end()), value, std::chars_format::scientific);
	core::invariant(!std::make_error_code(written.ec));
	auto const text = std::string_view{buffer.data(), written.ptr};
	auto const negative = text.starts_with('-');
	auto const body = text.substr(negative ? 1 : 0);
	auto const e = body.find('e');
	auto digits = std::string{};
	std::ranges::copy_if(body.substr(0, e), std::back_inserter(digits), [](char c) {
		return c != '.';
	});
	auto const exponent_text = body.substr(e + 1);
	// The exponent is signed: '+' or '-', then its digits.
	auto const exponent_digits = exponent_text.substr(1);
	auto magnitude = 0;
	auto const read = std::from_chars(exponent_digits.data(), std::to_address(exponent_digits.end()), magnitude);
	core::invariant(!std::make_error_code(read.ec));
	auto const exponent = exponent_text.starts_with('-') ? -magnitude : magnitude;
	// The value is 0.d1d2... times 10^point; Python writes it fixed when -4 < point <= 16.
	auto const point = exponent + 1;
	auto const count = core::narrow<int>(digits.size());
	auto out = std::string{negative ? "-" : ""};
	if (point > -4 && point <= 16) {
		if (point <= 0) {
			out.append("0.");
			out.append(core::narrow<std::size_t>(-point), '0');
			out.append(digits);
		} else if (point >= count) {
			out.append(digits);
			out.append(core::narrow<std::size_t>(point - count), '0');
			out.append(".0");
		} else {
			out.append(digits, 0, core::narrow<std::size_t>(point));
			out.push_back('.');
			out.append(digits, core::narrow<std::size_t>(point));
		}
		return out;
	}
	out.push_back(digits.front());
	if (count > 1) {
		out.push_back('.');
		out.append(digits, 1);
	}
	std::format_to(std::back_inserter(out), "e{}{:02}", exponent < 0 ? '-' : '+', exponent < 0 ? -exponent : exponent);
	return out;
}

/**
 * Whether a decimal literal that `from_chars` found out of the double range is too large rather than too small:
 * whether its leading nonzero digit stands at or above the units place. Python reads the first as an infinity and
 * the second as zero.
 */
[[nodiscard]] auto overflows(std::string_view decimal) -> bool {
	auto const e = decimal.find_first_of("eE");
	auto const mantissa = decimal.substr(0, e);
	auto const point = std::min(mantissa.find('.'), mantissa.size());
	auto const first = mantissa.find_first_of("123456789");
	core::invariant(first != std::string_view::npos);
	// The place of the leading digit: zero for units, positive above, negative below.
	auto place = first < point ? std::int64_t{core::narrow<std::int64_t>(point - first) - 1} : -core::narrow<std::int64_t>(first - point);
	if (e != std::string_view::npos) {
		auto exponent_text = decimal.substr(e + 1);
		auto const negative = exponent_text.starts_with('-');
		if (exponent_text.starts_with('-') || exponent_text.starts_with('+')) {
			exponent_text.remove_prefix(1);
		}
		auto magnitude = std::int64_t{0};
		auto const read = std::from_chars(exponent_text.data(), std::to_address(exponent_text.end()), magnitude);
		// An exponent past 64 bits dwarfs every place a literal within bounds can hold.
		if (read.ec == std::errc::result_out_of_range) {
			return !negative;
		}
		place += negative ? -magnitude : magnitude;
	}
	return place >= 0;
}

/** Whether text is a JSON number literal. */
[[nodiscard]] auto json_number(std::string_view literal) -> bool {
	auto at = std::size_t{0};
	auto const digits = [&] {
		auto const begin = at;
		while (at < literal.size() && literal[at] >= '0' && literal[at] <= '9') {
			++at;
		}
		return at - begin;
	};
	if (at < literal.size() && literal[at] == '-') {
		++at;
	}
	auto const whole_begin = at;
	auto const whole = digits();
	if (whole == 0 || (whole > 1 && literal[whole_begin] == '0')) {
		return false;
	}
	if (at < literal.size() && literal[at] == '.') {
		++at;
		if (digits() == 0) {
			return false;
		}
	}
	if (at < literal.size() && (literal[at] == 'e' || literal[at] == 'E')) {
		++at;
		if (at < literal.size() && (literal[at] == '+' || literal[at] == '-')) {
			++at;
		}
		if (digits() == 0) {
			return false;
		}
	}
	return at == literal.size();
}

/**
 * Python's `json.dumps(json.loads(literal))` for a JSON number literal: an integer literal reads as an exact integer
 * of any size, any other as the nearest double. Nothing when the text is no literal, or its value is beyond the
 * double range: Python reads that as an infinity and writes `Infinity`, which is not JSON.
 */
[[nodiscard]] auto python_number(std::string_view literal) -> std::optional<std::string> {
	if (!json_number(literal)) {
		return std::nullopt;
	}
	if (literal.find_first_of(".eE") == std::string_view::npos) {
		return literal == "-0" ? std::string{"0"} : std::string{literal};
	}
	auto value = 0.0;
	auto const read = std::from_chars(literal.data(), std::to_address(literal.end()), value);
	if (read.ec == std::errc::result_out_of_range) {
		if (overflows(literal)) {
			return std::nullopt;
		}
		return literal.starts_with('-') ? std::string{"-0.0"} : std::string{"0.0"};
	}
	core::invariant(!std::make_error_code(read.ec) && read.ptr == std::to_address(literal.end()));
	return python_float(value);
}

[[nodiscard]] inline auto refused(json::ParseFault fault) -> std::unexpected<json::ReadError> {
	return std::unexpected(json::ReadError{.cause = json::ParseError{.fault = fault}});
}

/** Reads a whole value, checking every part, into the value Python's `json.loads` makes of it. */
template<json::Bounds B>
    requires json::Enforceable<B>
[[nodiscard]] auto read_json(json::Value<B> value) -> core::Result<Json, json::ReadError> {
	auto const kind = value.kind();
	if (!kind) {
		return std::unexpected(kind.error());
	}
	switch (*kind) {
	case json::Kind::Null:
		return value.as_null().transform([] {
			return Json{.value = Null{}};
		});
	case json::Kind::Boolean:
		return value.as_boolean().transform([](bool read) {
			return Json{.value = read};
		});
	case json::Kind::Integer:
	case json::Kind::Real: {
		// The literal, not the reader's number: Python keeps integers of any size exactly.
		auto const literal = value.raw();
		if (!literal) {
			return std::unexpected(literal.error());
		}
		if (literal->size() > B.number_chars) {
			return refused(json::ParseFault::NumberTooLong);
		}
		auto text = python_number(*literal);
		if (!text) {
			return refused(json::ParseFault::InvalidNumber);
		}
		return Json{.value = Number{.text = std::move(*text)}};
	}
	case json::Kind::String:
		return value.as_string().transform([](std::string_view read) {
			return Json{.value = std::string{read}};
		});
	case json::Kind::Array: {
		auto array = value.as_array();
		if (!array) {
			return std::unexpected(array.error());
		}
		auto elements = Array{};
		while (true) {
			auto element = array->next();
			if (!element) {
				return std::unexpected(element.error());
			}
			if (!*element) {
				return Json{.value = std::move(elements)};
			}
			auto read = read_json(std::move(**element));
			if (!read) {
				return read;
			}
			elements.push_back(std::move(*read));
		}
	}
	case json::Kind::Object: {
		auto object = value.as_object();
		if (!object) {
			return std::unexpected(object.error());
		}
		auto members = Object{};
		while (true) {
			auto member = object->next();
			if (!member) {
				return std::unexpected(member.error());
			}
			if (!*member) {
				return Json{.value = std::move(members)};
			}
			auto name = std::string{(*member)->name};
			auto read = read_json(std::move((*member)->value));
			if (!read) {
				return read;
			}
			members.members.push_back(Member{.name = std::move(name), .value = std::move(*read)});
		}
	}
	}
	std::unreachable();
}

/** Reads a whole text with `reader` into the value Python's `json.loads` makes of it. */
template<json::Bounds B>
    requires json::Enforceable<B>
[[nodiscard]] auto load(json::Reader<B>& reader, std::string_view text) -> core::Result<Json, json::ReadError> {
	auto root = reader.read_copy(text);
	if (!root) {
		return std::unexpected(root.error());
	}
	auto value = read_json(std::move(*root));
	if (!value) {
		return value;
	}
	if (auto const finished = reader.finish(); !finished) {
		return std::unexpected(finished.error());
	}
	return value;
}

/** Appends a value as Python's `json.dumps(..., ensure_ascii=False)` writes it: `", "` and `": "` between parts. */
auto dump(std::string& out, Json const& value) -> void {
	auto const& held = value.value;
	if (std::holds_alternative<Null>(held)) {
		out.append("null");
	} else if (std::holds_alternative<bool>(held)) {
		out.append(std::get<bool>(held) ? "true" : "false");
	} else if (std::holds_alternative<Number>(held)) {
		out.append(std::get<Number>(held).text);
	} else if (std::holds_alternative<std::string>(held)) {
		put_string(out, std::get<std::string>(held));
	} else if (std::holds_alternative<Array>(held)) {
		out.push_back('[');
		for (auto const [index, element] : std::views::enumerate(std::get<Array>(held))) {
			out.append(index == 0 ? "" : ", ");
			dump(out, element);
		}
		out.push_back(']');
	} else {
		out.push_back('{');
		for (auto const [index, member] : std::views::enumerate(std::get<Object>(held).members)) {
			out.append(index == 0 ? "" : ", ");
			put_string(out, member.name);
			out.append(": ");
			dump(out, member.value);
		}
		out.push_back('}');
	}
}

[[nodiscard]] auto dumps(Json const& value) -> std::string {
	auto out = std::string{};
	dump(out, value);
	return out;
}

/** Python's truth value of a JSON value: false for null, false, zero, and empty strings, arrays and objects. */
[[nodiscard]] auto truthy(Json const& value) -> bool {
	auto const& held = value.value;
	if (std::holds_alternative<bool>(held)) {
		return std::get<bool>(held);
	}
	if (std::holds_alternative<Number>(held)) {
		auto const& text = std::get<Number>(held).text;
		return text != "0" && text != "0.0" && text != "-0.0";
	}
	if (std::holds_alternative<std::string>(held)) {
		return !std::get<std::string>(held).empty();
	}
	if (std::holds_alternative<Array>(held)) {
		return !std::get<Array>(held).empty();
	}
	if (std::holds_alternative<Object>(held)) {
		return !std::get<Object>(held).members.empty();
	}
	return false;
}

}

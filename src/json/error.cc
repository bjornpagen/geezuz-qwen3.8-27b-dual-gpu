// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:error;

import std;
import engine.core;
import engine.simdjson;

namespace engine::json {

export using simdjson::Kind;
export using simdjson::ParseError;
export using simdjson::ParseFault;

/** A value was read as one kind but is another. */
export struct [[nodiscard]] KindMismatch {
	Kind expected;
	Kind found;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(KindMismatch const&) const -> bool = default;
};

/** An object has no member of the name a reader requires. */
export struct [[nodiscard]] MissingMember {
	std::string name;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto operator==(MissingMember const&) const -> bool = default;
};

/** Why a read failed: the text is not a document within bounds, or it is not shaped as the reader requires. */
export struct [[nodiscard]] ReadError {
	std::variant<ParseError, KindMismatch, MissingMember> cause;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return std::visit([](auto const& error) { return error.retry(); }, cause);
	}

	[[nodiscard]] auto operator==(ReadError const&) const -> bool = default;
};

/** A short English phrase for a fault, for error messages. */
export [[nodiscard]] constexpr auto describe(ParseFault fault) -> std::string_view {
	switch (fault) {
	case ParseFault::Empty:
		return "no JSON value";
	case ParseFault::Truncated:
		return "the text ends inside a value";
	case ParseFault::Malformed:
		return "malformed JSON";
	case ParseFault::TrailingContent:
		return "content after the JSON value";
	case ParseFault::InvalidLiteral:
		return "invalid literal";
	case ParseFault::InvalidNumber:
		return "invalid number";
	case ParseFault::InvalidString:
		return "invalid string escape or unpaired surrogate";
	case ParseFault::ControlCharacter:
		return "unescaped control character in a string";
	case ParseFault::InvalidUtf8:
		return "invalid UTF-8";
	case ParseFault::DuplicateKey:
		return "duplicate member name";
	case ParseFault::TooLarge:
		return "document too large";
	case ParseFault::TooDeep:
		return "nesting too deep";
	case ParseFault::StringTooLong:
		return "string too long";
	case ParseFault::TooManyMembers:
		return "too many array elements or object members";
	case ParseFault::NumberTooLong:
		return "number literal too long";
	case ParseFault::OutOfMemory:
		return "out of memory";
	}
	std::unreachable();
}

export [[nodiscard]] constexpr auto describe(Kind kind) -> std::string_view {
	switch (kind) {
	case Kind::Null:
		return "null";
	case Kind::Boolean:
		return "a boolean";
	case Kind::Integer:
		return "an integer";
	case Kind::Real:
		return "a number";
	case Kind::String:
		return "a string";
	case Kind::Array:
		return "an array";
	case Kind::Object:
		return "an object";
	}
	std::unreachable();
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:value;

import std;
import engine.core;
import engine.simdjson;
import :bounds;
import :error;
import :text;

namespace engine::json {

/**
 * One value of a document being read under bounds `B`. A borrowed, forward-only view: each value is read at most
 * once, in document order, while its reader stays in place and reads nothing else. A typed read of the wrong kind
 * is a `KindMismatch` naming both kinds; a fault in the text is found when the part holding it is read.
 */
export template<Bounds B>
    requires Enforceable<B>
class Value {
public:
	[[nodiscard]] auto kind() -> core::Result<Kind, ReadError> {
		auto const kind = value_.kind();
		if (!kind) {
			return refuse(kind.error());
		}
		return *kind;
	}

	[[nodiscard]] auto as_null() -> core::Result<void, ReadError> {
		if (auto const is = expect(Kind::Null); !is) {
			return is;
		}
		return lift(value_.null());
	}

	[[nodiscard]] auto as_boolean() -> core::Result<bool, ReadError> {
		if (auto const is = expect(Kind::Boolean); !is) {
			return std::unexpected(is.error());
		}
		return lift(value_.boolean());
	}

	[[nodiscard]] auto as_integer() -> core::Result<std::int64_t, ReadError> {
		if (auto const is = expect(Kind::Integer); !is) {
			return std::unexpected(is.error());
		}
		if (auto const literal = bounded_number(); !literal) {
			return std::unexpected(literal.error());
		}
		return lift(value_.integer());
	}

	/**
	 * A number as a double: a real as read, or an integer the double format holds exactly (a config writes
	 * `10000000` where it means a real). An integer a double would round is a mismatch, never a silent rounding.
	 */
	[[nodiscard]] auto as_real() -> core::Result<double, ReadError> {
		auto const found = kind();
		if (!found) {
			return std::unexpected(found.error());
		}
		if (*found != Kind::Real && *found != Kind::Integer) {
			return mismatch(Kind::Real, *found);
		}
		if (auto const literal = bounded_number(); !literal) {
			return std::unexpected(literal.error());
		}
		if (*found == Kind::Real) {
			return lift(value_.real());
		}
		auto const integer = value_.integer();
		if (!integer) {
			return refuse(integer.error());
		}
		auto const rounded = static_cast<double>(*integer);
		// 2^63 is the first double past the int64 range; below it the conversion back is defined.
		if (rounded >= 0x1p63 || static_cast<std::int64_t>(rounded) != *integer) {
			return mismatch(Kind::Real, Kind::Integer);
		}
		return rounded;
	}

	/** A string, unescaped; the text lives until the reader reads its next document. */
	[[nodiscard]] auto as_string() -> core::Result<std::string_view, ReadError> {
		if (auto const is = expect(Kind::String); !is) {
			return std::unexpected(is.error());
		}
		auto const text = value_.string();
		if (!text) {
			return refuse(text.error());
		}
		if (text->size() > B.string_bytes) {
			return refuse(ParseFault::StringTooLong);
		}
		return *text;
	}

	[[nodiscard]] auto as_object() -> core::Result<Object<B>, ReadError> {
		if (auto const is = expect(Kind::Object); !is) {
			return std::unexpected(is.error());
		}
		if (depth_ + 1 > B.depth) {
			return refuse(ParseFault::TooDeep);
		}
		auto cursor = value_.object();
		if (!cursor) {
			return refuse(cursor.error());
		}
		return Object<B>{std::move(*cursor), reader_, depth_ + 1};
	}

	[[nodiscard]] auto as_array() -> core::Result<Array<B>, ReadError> {
		if (auto const is = expect(Kind::Array); !is) {
			return std::unexpected(is.error());
		}
		if (depth_ + 1 > B.depth) {
			return refuse(ParseFault::TooDeep);
		}
		auto cursor = value_.array();
		if (!cursor) {
			return refuse(cursor.error());
		}
		return Array<B>{std::move(*cursor), reader_, depth_ + 1};
	}

	/**
	 * The exact text of the value as written, without surrounding whitespace. Only its structure is checked: a
	 * reader that keeps the text and needs it valid checks it whole (`check_text`).
	 */
	[[nodiscard]] auto raw() -> core::Result<std::string_view, ReadError> {
		return lift(value_.raw());
	}

	/** Reads the whole value, checking every part, and keeps its exact text. */
	[[nodiscard]] auto capture() -> core::Result<JsonText, ReadError> {
		auto const text = reader_.text();
		auto const begin = value_.offset(text);
		if (!begin) {
			return refuse(begin.error());
		}
		if (auto const checked = check(*this); !checked) {
			return std::unexpected(checked.error());
		}
		auto const end = reader_.offset();
		core::invariant(*begin <= end && end <= text.size());
		auto const written = text.substr(*begin, end - *begin);
		return JsonText{written.substr(0, written.find_last_not_of(" \t\n\r") + 1)};
	}

private:
	friend class Reader<B>;
	friend class Object<B>;
	friend class Array<B>;

	Value(simdjson::Value value, Reader<B>& reader, std::size_t depth) : value_{std::move(value)}, reader_{reader}, depth_{depth} {}

	template<class T>
	[[nodiscard]] static auto lift(core::Result<T, ParseError> read) -> core::Result<T, ReadError> {
		if (!read) {
			return refuse(read.error());
		}
		if constexpr (std::is_void_v<T>) {
			return {};
		} else {
			return *std::move(read);
		}
	}

	[[nodiscard]] auto expect(Kind expected) -> core::Result<void, ReadError> {
		auto const found = kind();
		if (!found) {
			return std::unexpected(found.error());
		}
		if (*found != expected) {
			return mismatch(expected, *found);
		}
		return {};
	}

	[[nodiscard]] auto bounded_number() -> core::Result<void, ReadError> {
		if (value_.token().size() > B.number_chars) {
			return refuse(ParseFault::NumberTooLong);
		}
		return {};
	}

	simdjson::Value value_;
	Reader<B>& reader_;
	/** The containers around this value; the root's is zero. */
	std::size_t depth_;
};

export template<Bounds B>
    requires Enforceable<B>
struct Member {
	/** The member's name, unescaped; it lives until the reader reads its next document. */
	std::string_view name;
	Value<B> value;
};

}

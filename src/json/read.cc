// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:read;

import std;
import engine.core;
import engine.simdjson;
import :bounds;
import :error;

namespace engine::json {

/** The bytes a reader may read past the end of a text: an input buffer reserves them to be read in place. */
export using simdjson::padding;
export using simdjson::PaddedView;

[[nodiscard]] inline auto refuse(ParseFault fault) -> std::unexpected<ReadError> {
	return std::unexpected(ReadError{.cause = ParseError{.fault = fault}});
}

[[nodiscard]] inline auto refuse(ParseError error) -> std::unexpected<ReadError> {
	return std::unexpected(ReadError{.cause = error});
}

[[nodiscard]] inline auto mismatch(Kind expected, Kind found) -> std::unexpected<ReadError> {
	return std::unexpected(ReadError{.cause = KindMismatch{.expected = expected, .found = found}});
}

export template<Bounds B>
    requires Enforceable<B>
class Reader;

export template<Bounds B>
    requires Enforceable<B>
class Value;

/**
 * One complete JSON value as text, exactly as it was written, checked whole: a tool's definition, its parameter
 * schema, a tool-call argument. Holding it as text keeps every byte the client sent; holding it as this type keeps
 * the proof that the text is one valid value within the bounds it was read under, so a writer embeds it without
 * looking again. Only a reader makes one.
 */
export class JsonText {
public:
	/** Reads `text` whole with `reader`, checking every part; keeps it without surrounding whitespace. */
	template<Bounds B>
	    requires Enforceable<B>
	[[nodiscard]] static auto from(Reader<B>& reader, std::string_view text) -> core::Result<JsonText, ReadError>;

	[[nodiscard]] auto text() const -> std::string_view {
		return text_;
	}

	[[nodiscard]] auto operator==(JsonText const&) const -> bool = default;

private:
	template<Bounds B>
	    requires Enforceable<B>
	friend class Value;

	explicit JsonText(std::string_view text) : text_{text} {}

	std::string text_;
};

export template<Bounds B>
    requires Enforceable<B>
class Object;

export template<Bounds B>
    requires Enforceable<B>
[[nodiscard]] auto check(Value<B> value) -> core::Result<void, ReadError>;

export template<Bounds B>
    requires Enforceable<B>
class Array;

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

/**
 * The members of one object, read in document order under bounds `B`. Iterating checks every name: its length,
 * the member count, and that no name repeats. `find` and `member` look a name up from the object's start, compare
 * names unescaped, and start iteration over; they check no names, so a reader that needs every name checked
 * iterates.
 */
export template<Bounds B>
    requires Enforceable<B>
class Object {
public:
	/** The next member, or nothing at the end; the previous member's value is skipped where it was not read. */
	[[nodiscard]] auto next() -> core::Result<std::optional<Member<B>>, ReadError> {
		auto const field = cursor_.next();
		if (!field) {
			return refuse(field.error());
		}
		if (!*field) {
			return std::nullopt;
		}
		if (++count_ > B.members) {
			return refuse(ParseFault::TooManyMembers);
		}
		auto& [name, value] = **field;
		if (name.size() > B.string_bytes) {
			return refuse(ParseFault::StringTooLong);
		}
		if (!names().insert(name).second) {
			return refuse(ParseFault::DuplicateKey);
		}
		return Member<B>{.name = name, .value = Value<B>{std::move(value), reader_, depth_}};
	}

	/** The value of the member named `name`, if there is one. */
	[[nodiscard]] auto find(std::string_view name) -> core::Result<std::optional<Value<B>>, ReadError> {
		names().clear();
		count_ = 0;
		auto found = cursor_.find(name);
		if (!found) {
			return refuse(found.error());
		}
		if (!*found) {
			return std::nullopt;
		}
		return Value<B>{std::move(**found), reader_, depth_};
	}

	/** The value of the member named `name`, which the reader requires. */
	[[nodiscard]] auto member(std::string_view name) -> core::Result<Value<B>, ReadError> {
		auto found = find(name);
		if (!found) {
			return std::unexpected(found.error());
		}
		if (!*found) {
			return std::unexpected(ReadError{.cause = MissingMember{.name = std::string{name}}});
		}
		return std::move(**found);
	}

private:
	friend class Value<B>;

	Object(simdjson::ObjectCursor cursor, Reader<B>& reader, std::size_t depth) : cursor_{std::move(cursor)}, reader_{reader}, depth_{depth} {
		names().clear();
	}

	[[nodiscard]] auto names() -> std::unordered_set<std::string_view>&;

	simdjson::ObjectCursor cursor_;
	Reader<B>& reader_;
	/** The containers around this object's members, this one included. */
	std::size_t depth_;
	std::size_t count_ = 0;
};

/** The elements of one array, read in document order under bounds `B`; as `Object`. */
export template<Bounds B>
    requires Enforceable<B>
class Array {
public:
	/** The number of elements, counted ahead without reading them; only before the first `next`. */
	[[nodiscard]] auto size() -> core::Result<std::size_t, ReadError> {
		auto const count = cursor_.count();
		if (!count) {
			return refuse(count.error());
		}
		if (*count > B.members) {
			return refuse(ParseFault::TooManyMembers);
		}
		return *count;
	}

	/** The next element, or nothing at the end. */
	[[nodiscard]] auto next() -> core::Result<std::optional<Value<B>>, ReadError> {
		auto element = cursor_.next();
		if (!element) {
			return refuse(element.error());
		}
		if (!*element) {
			return std::nullopt;
		}
		if (++count_ > B.members) {
			return refuse(ParseFault::TooManyMembers);
		}
		return Value<B>{std::move(**element), reader_, depth_};
	}

private:
	friend class Value<B>;

	Array(simdjson::ArrayCursor cursor, Reader<B>& reader, std::size_t depth) : cursor_{std::move(cursor)}, reader_{reader}, depth_{depth} {}

	simdjson::ArrayCursor cursor_;
	Reader<B>& reader_;
	std::size_t depth_;
	std::size_t count_ = 0;
};

/**
 * Reads JSON documents under bounds `B`, one at a time, with simdjson's On-Demand parser: lazily and forward only,
 * never materializing a tree. The parser is allocated once for `B.document_bytes`. Starting a document ends the last
 * one and every value, cursor and string read from it. A reader stays in place while it reads (values refer to it),
 * so its owner keeps it where it made it.
 */
export template<Bounds B>
    requires Enforceable<B>
class Reader {
public:
	[[nodiscard]] static auto make() -> core::Result<Reader, ReadError> {
		auto parser = simdjson::Parser::make(B.document_bytes);
		if (!parser) {
			return refuse(parser.error());
		}
		return Reader{std::move(*parser)};
	}

	/** Reads a text whose storage extends `padding` bytes past its end, in place. */
	[[nodiscard]] auto read(PaddedView text) -> core::Result<Value<B>, ReadError> {
		if (text.text().size() > B.document_bytes) {
			return refuse(ParseFault::TooLarge);
		}
		text_ = text.text();
		auto root = parser_->iterate(text);
		if (!root) {
			return refuse(root.error());
		}
		return Value<B>{std::move(*root), *this, 0};
	}

	/** Copies a text into the reader's own padded buffer, then reads it. */
	[[nodiscard]] auto read_copy(std::string_view text) -> core::Result<Value<B>, ReadError> {
		if (text.size() > B.document_bytes) {
			return refuse(ParseFault::TooLarge);
		}
		copy_.assign(text);
		copy_.resize(text.size() + padding);
		return read(PaddedView::over(copy_, text.size()));
	}

	/** After the root has been read to its end: refuses anything but whitespace after it. */
	[[nodiscard]] auto finish() -> core::Result<void, ReadError> {
		if (auto const done = parser_->finish(); !done) {
			return refuse(done.error());
		}
		return {};
	}

private:
	friend class Object<B>;
	friend class Value<B>;

	[[nodiscard]] auto text() const -> std::string_view {
		return text_;
	}

	[[nodiscard]] auto offset() const -> std::size_t {
		return parser_->offset();
	}

	explicit Reader(std::unique_ptr<simdjson::Parser> parser) : parser_{std::move(parser)}, names_(B.depth + 1) {}

	std::unique_ptr<simdjson::Parser> parser_;
	/** The text of the current document. */
	std::string_view text_;
	/** A padded copy of the current text, when it was not read in place; at most `B.document_bytes` plus padding. */
	std::string copy_;
	/** The names seen so far in the object open at each depth: siblings at one depth are read one after another. */
	std::vector<std::unordered_set<std::string_view>> names_;
};

template<Bounds B>
    requires Enforceable<B>
auto Object<B>::names() -> std::unordered_set<std::string_view>& {
	return reader_.names_[depth_];
}

/** Reads a whole value, checking every part of it against the grammar and `B`. */
template<Bounds B>
    requires Enforceable<B>
auto check(Value<B> value) -> core::Result<void, ReadError> {
	auto const kind = value.kind();
	if (!kind) {
		return std::unexpected(kind.error());
	}
	auto const done = [](auto const& read) -> core::Result<void, ReadError> {
		if (!read) {
			return std::unexpected(read.error());
		}
		return {};
	};
	switch (*kind) {
	case Kind::Null:
		return value.as_null();
	case Kind::Boolean:
		return done(value.as_boolean());
	case Kind::Integer:
		return done(value.as_integer());
	case Kind::Real:
		return done(value.as_real());
	case Kind::String:
		return done(value.as_string());
	case Kind::Array: {
		auto array = value.as_array();
		if (!array) {
			return std::unexpected(array.error());
		}
		while (true) {
			auto element = array->next();
			if (!element) {
				return std::unexpected(element.error());
			}
			if (!*element) {
				return {};
			}
			if (auto const checked = check(std::move(**element)); !checked) {
				return checked;
			}
		}
	}
	case Kind::Object: {
		auto object = value.as_object();
		if (!object) {
			return std::unexpected(object.error());
		}
		while (true) {
			auto member = object->next();
			if (!member) {
				return std::unexpected(member.error());
			}
			if (!*member) {
				return {};
			}
			if (auto const checked = check(std::move((*member)->value)); !checked) {
				return checked;
			}
		}
	}
	}
	std::unreachable();
}

/** Reads a whole text and checks every part of it: the strict reading for a text that is kept or passed on. */
export template<Bounds B>
    requires Enforceable<B>
[[nodiscard]] auto check_text(Reader<B>& reader, std::string_view text) -> core::Result<void, ReadError> {
	auto root = reader.read_copy(text);
	if (!root) {
		return std::unexpected(root.error());
	}
	if (auto const checked = check(std::move(*root)); !checked) {
		return checked;
	}
	return reader.finish();
}

}

namespace engine::json {

template<Bounds B>
    requires Enforceable<B>
auto JsonText::from(Reader<B>& reader, std::string_view text) -> core::Result<JsonText, ReadError> {
	auto root = reader.read_copy(text);
	if (!root) {
		return std::unexpected(root.error());
	}
	auto captured = root->capture();
	if (!captured) {
		return captured;
	}
	if (auto const finished = reader.finish(); !finished) {
		return std::unexpected(finished.error());
	}
	return captured;
}

}

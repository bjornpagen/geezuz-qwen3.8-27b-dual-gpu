// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:reader;

import std;
import engine.core;
import engine.simdjson;
import :bounds;
import :error;
import :text;
import :value;
import :containers;

namespace engine::json {

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

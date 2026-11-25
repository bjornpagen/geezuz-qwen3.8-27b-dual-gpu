// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:text;

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
	[[nodiscard]] static auto from(Reader<B>& reader, std::string_view text) -> core::Result<JsonText, ReadError> {
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

}

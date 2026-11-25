// SPDX-License-Identifier: AGPL-3.0-only
module;

// simdjson's one amalgamated header: the On-Demand parser, error codes and padding contract.
#include "simdjson.h"

#if SIMDJSON_EXCEPTIONS
#error "simdjson is built without exceptions: its error codes map onto engine results"
#endif
#if (SIMDJSON_IMPLEMENTATION_ICELAKE + SIMDJSON_IMPLEMENTATION_HASWELL + SIMDJSON_IMPLEMENTATION_WESTMERE +           \
     SIMDJSON_IMPLEMENTATION_ARM64 + SIMDJSON_IMPLEMENTATION_PPC64 + SIMDJSON_IMPLEMENTATION_LSX +                    \
     SIMDJSON_IMPLEMENTATION_LASX + SIMDJSON_IMPLEMENTATION_RVV_VLS + SIMDJSON_IMPLEMENTATION_FALLBACK) != 1
#error "exactly one simdjson kernel is compiled, so nothing is chosen at run time"
#endif
#if defined(SIMDJSON_IS_ARM64) && !(SIMDJSON_IMPLEMENTATION_ARM64 && SIMDJSON_CAN_ALWAYS_RUN_ARM64 && SIMDJSON_BUILTIN_IMPLEMENTATION_IS(arm64))
#error "darwin arm64 runs the NEON kernel"
#endif
#if defined(SIMDJSON_IS_X86_64) && !(SIMDJSON_IMPLEMENTATION_HASWELL && SIMDJSON_CAN_ALWAYS_RUN_HASWELL && SIMDJSON_BUILTIN_IMPLEMENTATION_IS(haswell))
#error "the linux EPYC runs the AVX2 kernel"
#endif
#if !(defined(SIMDJSON_IS_ARM64) || defined(SIMDJSON_IS_X86_64))
#error "the engine builds for darwin arm64 and linux x86_64 only"
#endif

export module engine.simdjson;

import std;
import engine.core;

namespace od = ::simdjson::ondemand;

/**
 * The simdjson boundary: a narrow, typed surface over simdjson's On-Demand parser. Nothing above this unit sees a
 * simdjson type or error code. Values are read lazily, forward only, in document order: a value is read before the
 * next one is reached, and a reader that moves past a value skips what it did not read. Only what is read is
 * checked beyond structure and UTF-8; a reader that needs a whole text checked walks all of it.
 */
namespace engine::simdjson {

/** The bytes simdjson may read past the end of an input: every input's storage extends at least this far. */
export inline constexpr std::size_t padding = 64;

static_assert(padding == ::simdjson::SIMDJSON_PADDING, "simdjson's padding contract, restated where callers can see it");

/** Why a text is not a document the engine accepts, or could not be read. */
export enum class ParseFault : std::uint8_t {
	/** The text holds no value. */
	Empty,
	/** The text ends inside a string, array or object. */
	Truncated,
	/** The structure is wrong: a missing or extra comma, colon or bracket, a comment, a bare word, a missing name. */
	Malformed,
	/** A complete value followed by more than whitespace. */
	TrailingContent,
	/** A misspelled `true`, `false` or `null`. */
	InvalidLiteral,
	/** A number outside the JSON grammar, or beyond the range of a double. */
	InvalidNumber,
	/**
	 * A string with an invalid escape, or a `\u` surrogate escape that is not half of a high-low pair: a lone
	 * surrogate names no character, and every string the engine reads is UTF-8, so it is refused, never replaced.
	 */
	InvalidString,
	/** An unescaped byte below 0x20 inside a string. */
	ControlCharacter,
	/** Bytes that are not well-formed UTF-8. */
	InvalidUtf8,
	/** A member name that appears twice in one object. */
	DuplicateKey,
	/** The text is longer than its bounds allow. */
	TooLarge,
	/** Containers nest deeper than their bounds allow. */
	TooDeep,
	/** A string unescapes to more bytes than its bounds allow. */
	StringTooLong,
	/** An array or object holds more elements or members than its bounds allow. */
	TooManyMembers,
	/** A number literal is longer than its bounds allow. */
	NumberTooLong,
	/** The parser's buffers could not be allocated. */
	OutOfMemory,
};

/** A text could not be read. Every fault is the text's, except running out of memory, which may pass. */
export struct [[nodiscard]] ParseError {
	ParseFault fault;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (fault) {
		case ParseFault::OutOfMemory:
			return core::Retry::Transient;
		case ParseFault::Empty:
		case ParseFault::Truncated:
		case ParseFault::Malformed:
		case ParseFault::TrailingContent:
		case ParseFault::InvalidLiteral:
		case ParseFault::InvalidNumber:
		case ParseFault::InvalidString:
		case ParseFault::ControlCharacter:
		case ParseFault::InvalidUtf8:
		case ParseFault::DuplicateKey:
		case ParseFault::TooLarge:
		case ParseFault::TooDeep:
		case ParseFault::StringTooLong:
		case ParseFault::TooManyMembers:
		case ParseFault::NumberTooLong:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(ParseError const&) const -> bool = default;
};

/**
 * What a value is. A number literal without fraction or exponent that fits 64 signed bits is an integer, read
 * exactly; every other number is a real, the double nearest its literal.
 */
export enum class Kind : std::uint8_t {
	Null,
	Boolean,
	Integer,
	Real,
	String,
	Array,
	Object,
};

[[nodiscard]] auto fault_of(::simdjson::error_code code) -> ParseFault {
	switch (code) {
	case ::simdjson::EMPTY:
		return ParseFault::Empty;
	case ::simdjson::INCOMPLETE_ARRAY_OR_OBJECT:
	case ::simdjson::UNCLOSED_STRING:
		return ParseFault::Truncated;
	case ::simdjson::TRAILING_CONTENT:
		return ParseFault::TrailingContent;
	case ::simdjson::T_ATOM_ERROR:
	case ::simdjson::F_ATOM_ERROR:
	case ::simdjson::N_ATOM_ERROR:
		return ParseFault::InvalidLiteral;
	case ::simdjson::NUMBER_ERROR:
	case ::simdjson::NUMBER_OUT_OF_RANGE:
		return ParseFault::InvalidNumber;
	case ::simdjson::STRING_ERROR:
		return ParseFault::InvalidString;
	case ::simdjson::UNESCAPED_CHARS:
		return ParseFault::ControlCharacter;
	case ::simdjson::UTF8_ERROR:
		return ParseFault::InvalidUtf8;
	case ::simdjson::CAPACITY:
		return ParseFault::TooLarge;
	case ::simdjson::DEPTH_ERROR:
		return ParseFault::TooDeep;
	case ::simdjson::MEMALLOC:
		return ParseFault::OutOfMemory;
	case ::simdjson::OUT_OF_ORDER_ITERATION:
		// Reading a value after moving past it is a bug in the reader, not a property of the text.
		core::invariant(false);
		return ParseFault::Malformed;
	default:
		// Every other code reports a structure simdjson could not follow (a tape error, a token of the wrong
		// kind where a value must start); none of them is caused by anything but the text.
		return ParseFault::Malformed;
	}
}

[[nodiscard]] auto failure(::simdjson::error_code code) -> std::unexpected<ParseError> {
	return std::unexpected(ParseError{.fault = fault_of(code)});
}

/** The index just past the string starting at `at`, or nothing when the text ends inside it. */
[[nodiscard]] auto past_string(std::string_view text, std::size_t at) -> std::optional<std::size_t> {
	for (++at; at < text.size(); ++at) {
		if (text[at] == '\\') {
			++at;
		} else if (text[at] == '"') {
			return at + 1;
		}
	}
	return std::nullopt;
}

/** The index just past the root value starting at `at`, by its brackets and strings, or nothing when it is open. */
[[nodiscard]] auto past_root(std::string_view text, std::size_t at) -> std::optional<std::size_t> {
	if (text[at] == '"') {
		return past_string(text, at);
	}
	if (text[at] != '{' && text[at] != '[') {
		auto const end = text.find_first_of(" \t\n\r{}[],:\"", at);
		return end == std::string_view::npos ? std::nullopt : std::optional<std::size_t>{end};
	}
	auto depth = std::size_t{0};
	while (at < text.size()) {
		auto const byte = text[at];
		if (byte == '"') {
			auto const after = past_string(text, at);
			if (!after) {
				return std::nullopt;
			}
			at = *after;
			continue;
		}
		if (byte == '{' || byte == '[') {
			++depth;
		} else if (byte == '}' || byte == ']') {
			--depth;
			if (depth == 0) {
				return at + 1;
			}
		}
		++at;
	}
	return std::nullopt;
}

/**
 * Whether the text's root value ends before the text does, with more than whitespace after it. simdjson checks a
 * root container by its last structural byte, and scans strings before it reads any value, so a complete root
 * followed by content reads to it as an unclosed container or string; this walk tells the two apart.
 */
[[nodiscard]] auto root_closes_early(std::string_view text) -> bool {
	auto constexpr whitespace = std::string_view{" \t\n\r"};
	auto const at = text.find_first_not_of(whitespace);
	if (at == std::string_view::npos) {
		return false;
	}
	auto const end = past_root(text, at);
	return end && text.find_first_not_of(whitespace, *end) != std::string_view::npos;
}

/** An unclosed-container or unclosed-string code, told apart from content after a closed root. */
[[nodiscard]] auto truncated_or_trailing(std::string_view text, ::simdjson::error_code code) -> ::simdjson::error_code {
	auto const unclosed = code == ::simdjson::INCOMPLETE_ARRAY_OR_OBJECT || code == ::simdjson::UNCLOSED_STRING;
	return unclosed && root_closes_early(text) ? ::simdjson::TRAILING_CONTENT : code;
}

/** Whether bytes are well-formed UTF-8, by the same kernel the parser uses. */
export [[nodiscard]] auto valid_utf8(std::string_view text) -> bool {
	return ::simdjson::validate_utf8(text.data(), text.size());
}

/**
 * A text whose storage extends `padding` readable bytes past its end, borrowed for one read. The bytes past the
 * end may hold anything; simdjson reads them but never lets them change a result.
 */
export class PaddedView {
public:
	/** The first `length` bytes of `storage` as the text; the storage must extend `padding` bytes further. */
	[[nodiscard]] static auto over(std::span<char const> storage, std::size_t length) -> PaddedView {
		core::invariant(length <= storage.size() && storage.size() - length >= padding);
		return PaddedView{std::string_view{storage.data(), length}};
	}

	[[nodiscard]] auto text() const -> std::string_view {
		return text_;
	}

private:
	explicit PaddedView(std::string_view text) : text_{text} {}

	std::string_view text_;
};

export class Value;
export struct Field;

/** The members of one object, read in document order. A borrowed cursor: it lives no longer than its document. */
export class ObjectCursor {
public:
	/** The next member, or nothing at the end. The previous member's value is skipped where it was not read. */
	[[nodiscard]] auto next() -> core::Result<std::optional<Field>, ParseError>;

	/** The value of the first member named `name`, searching from the object's start; iteration starts over. */
	[[nodiscard]] auto find(std::string_view name) -> core::Result<std::optional<Value>, ParseError>;

private:
	friend class Value;

	explicit ObjectCursor(od::object object) : object_{std::move(object)} {}

	od::object object_;
	std::optional<od::object_iterator> at_;
	od::object_iterator end_;
};

/** The elements of one array, read in document order; as `ObjectCursor`. */
export class ArrayCursor {
public:
	/** The number of elements, counted ahead without reading them; only before the first `next`. */
	[[nodiscard]] auto count() -> core::Result<std::size_t, ParseError>;

	/** The next element, or nothing at the end. */
	[[nodiscard]] auto next() -> core::Result<std::optional<Value>, ParseError>;

private:
	friend class Value;

	explicit ArrayCursor(od::array array) : array_{std::move(array)} {}

	od::array array_;
	std::optional<od::array_iterator> at_;
	od::array_iterator end_;
};

/**
 * One value of a document, read at most once and in document order. The document's root and every nested value
 * have the same surface: simdjson spells a scalar root differently from a nested value, and this type hides it.
 */
export class Value {
public:
	[[nodiscard]] auto kind() -> core::Result<Kind, ParseError>;
	/** The literal `null`, read; anything else spelled from an `n` is a fault. */
	[[nodiscard]] auto null() -> core::Result<void, ParseError>;
	[[nodiscard]] auto boolean() -> core::Result<bool, ParseError>;
	[[nodiscard]] auto integer() -> core::Result<std::int64_t, ParseError>;
	/** A number as the double nearest its literal, integers beyond 64 bits included. */
	[[nodiscard]] auto real() -> core::Result<double, ParseError>;
	/** The literal text of a scalar, without the whitespace after it. */
	[[nodiscard]] auto token() -> std::string_view;
	/** A string, unescaped; the text lives as long as the parser's current document. */
	[[nodiscard]] auto string() -> core::Result<std::string_view, ParseError>;
	[[nodiscard]] auto object() -> core::Result<ObjectCursor, ParseError>;
	[[nodiscard]] auto array() -> core::Result<ArrayCursor, ParseError>;
	/** The exact text of the value, read to its end without checking more than its structure. */
	[[nodiscard]] auto raw() -> core::Result<std::string_view, ParseError>;
	/** Where the value starts in its text, before it is read. */
	[[nodiscard]] auto offset(std::string_view text) -> core::Result<std::size_t, ParseError>;

private:
	friend class Parser;
	friend class ObjectCursor;
	friend class ArrayCursor;

	explicit Value(od::value value) : handle_{value} {}
	Value(od::document_reference root, std::string_view text) : handle_{root}, root_text_{text} {}

	/** A container read failed as unclosed: the root's text tells a truncation from content after a closed root. */
	[[nodiscard]] auto unclosed(::simdjson::error_code code) const -> std::unexpected<ParseError>;

	std::variant<od::value, od::document_reference> handle_;
	/** The whole text, for the root only. */
	std::optional<std::string_view> root_text_;
};

export struct Field {
	/** The member's name, unescaped; it lives as long as the parser's current document. */
	std::string_view name;
	Value value;
};

/** Runs `read` on whichever simdjson handle the value holds; both spell the same calls. */
template<class F>
[[nodiscard]] auto on_handle(std::variant<od::value, od::document_reference>& handle, F const& read) {
	return std::visit([&read](auto& held) { return read(held); }, handle);
}

auto Value::kind() -> core::Result<Kind, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<Kind, ParseError> {
		auto type = od::json_type{};
		if (auto const code = held.type().get(type)) {
			return failure(code);
		}
		switch (type) {
		case od::json_type::null:
			return Kind::Null;
		case od::json_type::boolean:
			return Kind::Boolean;
		case od::json_type::string:
			return Kind::String;
		case od::json_type::array:
			return Kind::Array;
		case od::json_type::object:
			return Kind::Object;
		case od::json_type::number: {
			auto number = od::number_type::floating_point_number;
			if (auto const code = held.get_number_type().get(number)) {
				return failure(code);
			}
			return number == od::number_type::signed_integer ? Kind::Integer : Kind::Real;
		}
		case od::json_type::unknown:
			return failure(::simdjson::TAPE_ERROR);
		}
		std::unreachable();
	});
}

auto Value::null() -> core::Result<void, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<void, ParseError> {
		auto is_null = false;
		if (auto const code = held.is_null().get(is_null)) {
			return failure(code);
		}
		if (!is_null) {
			return failure(::simdjson::N_ATOM_ERROR);
		}
		return {};
	});
}

auto Value::boolean() -> core::Result<bool, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<bool, ParseError> {
		auto value = false;
		if (auto const code = held.get_bool().get(value)) {
			// A token that starts like a boolean and is not one is a misspelled literal.
			return failure(code == ::simdjson::INCORRECT_TYPE ? ::simdjson::T_ATOM_ERROR : code);
		}
		return value;
	});
}

auto Value::integer() -> core::Result<std::int64_t, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<std::int64_t, ParseError> {
		auto value = std::int64_t{0};
		if (auto const code = held.get_int64().get(value)) {
			return failure(code);
		}
		return value;
	});
}

auto Value::token() -> std::string_view {
	auto const text = on_handle(handle_, [](auto& held) -> std::string_view {
		if constexpr (requires { held.raw_json_token().value_unsafe(); }) {
			auto token = std::string_view{};
			std::ignore = held.raw_json_token().get(token);
			return token;
		} else {
			return held.raw_json_token();
		}
	});
	auto const end = text.find_last_not_of(" \t\n\r");
	return end == std::string_view::npos ? std::string_view{} : text.substr(0, end + 1);
}

auto Value::real() -> core::Result<double, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<double, ParseError> {
		auto value = 0.0;
		if (auto const code = held.get_double().get(value)) {
			return failure(code);
		}
		return value;
	});
}

auto Value::string() -> core::Result<std::string_view, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<std::string_view, ParseError> {
		auto text = std::string_view{};
		if (auto const code = held.get_string(false).get(text)) {
			return failure(code);
		}
		return text;
	});
}

auto Value::object() -> core::Result<ObjectCursor, ParseError> {
	return on_handle(handle_, [this](auto& held) -> core::Result<ObjectCursor, ParseError> {
		auto object = od::object{};
		if (auto const code = held.get_object().get(object)) {
			return unclosed(code);
		}
		return ObjectCursor{object};
	});
}

auto Value::array() -> core::Result<ArrayCursor, ParseError> {
	return on_handle(handle_, [this](auto& held) -> core::Result<ArrayCursor, ParseError> {
		auto array = od::array{};
		if (auto const code = held.get_array().get(array)) {
			return unclosed(code);
		}
		return ArrayCursor{array};
	});
}

auto Value::raw() -> core::Result<std::string_view, ParseError> {
	return on_handle(handle_, [](auto& held) -> core::Result<std::string_view, ParseError> {
		auto text = std::string_view{};
		if (auto const code = held.raw_json().get(text)) {
			return failure(code);
		}
		auto const end = text.find_last_not_of(" \t\n\r");
		return end == std::string_view::npos ? std::string_view{} : text.substr(0, end + 1);
	});
}

auto Value::unclosed(::simdjson::error_code code) const -> std::unexpected<ParseError> {
	return failure(root_text_ ? truncated_or_trailing(*root_text_, code) : code);
}

auto Value::offset(std::string_view text) -> core::Result<std::size_t, ParseError> {
	return on_handle(handle_, [text](auto& held) -> core::Result<std::size_t, ParseError> {
		auto location = static_cast<char const*>(nullptr);
		if (auto const code = held.current_location().get(location)) {
			return failure(code);
		}
		/* SAFETY: simdjson's location points into the text it was given, at or before its end. */
		return static_cast<std::size_t>(location - text.data());
	});
}

auto ObjectCursor::next() -> core::Result<std::optional<Field>, ParseError> {
	if (!at_) {
		auto begin = od::object_iterator{};
		if (auto const code = object_.begin().get(begin)) {
			return failure(code);
		}
		if (auto const code = object_.end().get(end_)) {
			return failure(code);
		}
		at_ = begin;
	} else {
		++*at_;
	}
	if (!(*at_ != end_)) {
		return std::nullopt;
	}
	auto field = od::field{};
	if (auto const code = (**at_).get(field)) {
		return failure(code);
	}
	auto name = std::string_view{};
	if (auto const code = field.unescaped_key(false).get(name)) {
		return failure(code);
	}
	return Field{.name = name, .value = Value{field.value()}};
}

auto ObjectCursor::find(std::string_view name) -> core::Result<std::optional<Value>, ParseError> {
	// simdjson's own lookup compares names as written, escapes and all; names are compared unescaped here, so a
	// lookup reads from the object's start.
	auto rewound = false;
	if (auto const code = object_.reset().get(rewound)) {
		return failure(code);
	}
	at_.reset();
	while (true) {
		auto field = next();
		if (!field) {
			return std::unexpected(field.error());
		}
		if (!*field) {
			return std::nullopt;
		}
		if ((*field)->name == name) {
			return std::move((*field)->value);
		}
	}
}

auto ArrayCursor::count() -> core::Result<std::size_t, ParseError> {
	core::invariant(!at_.has_value());
	auto count = std::size_t{0};
	if (auto const code = array_.count_elements().get(count)) {
		return failure(code);
	}
	return count;
}

auto ArrayCursor::next() -> core::Result<std::optional<Value>, ParseError> {
	if (!at_) {
		auto begin = od::array_iterator{};
		if (auto const code = array_.begin().get(begin)) {
			return failure(code);
		}
		if (auto const code = array_.end().get(end_)) {
			return failure(code);
		}
		at_ = begin;
	} else {
		++*at_;
	}
	if (!(*at_ != end_)) {
		return std::nullopt;
	}
	auto element = od::value{};
	if (auto const code = (**at_).get(element)) {
		return failure(code);
	}
	return Value{element};
}

/**
 * simdjson's On-Demand parser and its current document. A parser reads one document at a time; starting the next
 * ends the last, and every value, cursor and string read from it. The parser does not move while a document is
 * read: its owner keeps it in place (in practice, behind a `unique_ptr`).
 */
export class Parser {
public:
	/** A parser whose buffers hold documents of up to `capacity` bytes, allocated now. */
	[[nodiscard]] static auto make(std::size_t capacity) -> core::Result<std::unique_ptr<Parser>, ParseError> {
		auto parser = std::unique_ptr<Parser>{new Parser{}};
		parser->parser_.set_max_capacity(capacity);
		if (auto const code = parser->parser_.allocate(capacity)) {
			return failure(code);
		}
		return parser;
	}

	/** Starts reading `text`; returns its root value. */
	[[nodiscard]] auto iterate(PaddedView text) -> core::Result<Value, ParseError> {
		auto const bytes = text.text();
		auto const view = ::simdjson::padded_string_view{bytes.data(), bytes.size(), bytes.size() + padding};
		text_ = bytes;
		if (auto const code = parser_.iterate(view).get(document_)) {
			return failure(truncated_or_trailing(bytes, code));
		}
		return Value{od::document_reference{document_}, bytes};
	}

	/**
	 * How far into the text reading has gone: the start of the next token after everything read so far, or the
	 * end of the text once the root has been read.
	 */
	[[nodiscard]] auto offset() const -> std::size_t {
		auto location = static_cast<char const*>(nullptr);
		if (document_.current_location().get(location) != ::simdjson::SUCCESS) {
			return text_.size();
		}
		/* SAFETY: simdjson's location points into the text it was given, at or before its end. */
		return static_cast<std::size_t>(location - text_.data());
	}

	/** After the root has been read to its end: refuses anything but whitespace after it. */
	[[nodiscard]] auto finish() -> core::Result<void, ParseError> {
		if (!document_.at_end()) {
			return failure(::simdjson::TRAILING_CONTENT);
		}
		return {};
	}

private:
	Parser() = default;

	od::parser parser_{0};
	od::document document_;
	std::string_view text_;
};

}

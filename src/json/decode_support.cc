// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_support;

import std;
import engine.core;
import :bounds;
import :error;
import :text;
import :value;
import :containers;
import :reader;
import :decode_error;

namespace engine::json {

/**
 * The bounds of a JSON text a request carries inside itself: a function call's arguments, sent as a string. It is a
 * string of the request, so no such text is longer than one.
 */
export inline constexpr Bounds fragment_bounds{
    .document_bytes = request_bounds.string_bytes,
    .depth = request_bounds.depth,
    .string_bytes = request_bounds.string_bytes,
    .members = request_bounds.members,
    .number_chars = request_bounds.number_chars,
};

static_assert(Enforceable<fragment_bounds>);

using Body = Value<request_bounds>;
using Fragments = Reader<fragment_bounds>;

template<class T>
using Decoded = core::Result<T, DecodeError>;

[[nodiscard]] auto refuse(DecodeFault fault) -> std::unexpected<DecodeError> {
	return std::unexpected(DecodeError{.fault = std::move(fault), .path = {}});
}

/** Refuses the member `name` of the object being decoded. */
[[nodiscard]] auto refuse_at(std::string_view name, DecodeFault fault) -> std::unexpected<DecodeError> {
	return std::unexpected(DecodeError{.fault = std::move(fault), .path = {std::string{name}}});
}

[[nodiscard]] constexpr auto expected_of(Kind kind) -> Expected {
	switch (kind) {
	case Kind::Boolean:
		return Expected::Boolean;
	case Kind::Integer:
		return Expected::Integer;
	case Kind::Real:
		return Expected::Number;
	case Kind::String:
		return Expected::String;
	case Kind::Array:
		return Expected::Array;
	case Kind::Null:
	case Kind::Object:
		return Expected::Object;
	}
	std::unreachable();
}

/** A failed read in the decoder's terms. */
[[nodiscard]] auto decode_error(ReadError const& error) -> DecodeError {
	auto const& cause = error.cause;
	if (std::holds_alternative<KindMismatch>(cause)) {
		auto const& mismatch = std::get<KindMismatch>(cause);
		return DecodeError{.fault = WrongShape{.expected = expected_of(mismatch.expected), .found = mismatch.found}, .path = {}};
	}
	if (std::holds_alternative<MissingMember>(cause)) {
		return DecodeError{.fault = Violation::MissingField, .path = {std::get<MissingMember>(cause).name}};
	}
	return DecodeError{.fault = std::get<ParseError>(cause), .path = {}};
}

/** Lifts a read into the decoder's result. */
template<class T>
[[nodiscard]] auto lift(core::Result<T, ReadError> read) -> Decoded<T> {
	if (!read) {
		return std::unexpected(decode_error(read.error()));
	}
	if constexpr (std::is_void_v<T>) {
		return {};
	} else {
		return *std::move(read);
	}
}

/** Paths are collected innermost first while an error unwinds, and reversed once at the top. */
[[nodiscard]] auto within(PathStep step) {
	return [outer = std::move(step)](DecodeError error) {
		error.path.push_back(outer);
		return error;
	};
}

template<Bounds B>
[[nodiscard]] auto kind_of(Value<B>& value) -> Decoded<Kind> {
	return lift(value.kind());
}

template<Bounds B>
[[nodiscard]] auto text(Value<B>& value) -> Decoded<std::string> {
	return lift(value.as_string()).transform([](std::string_view read) { return std::string{read}; });
}

/** Reads `value` with `read`, or nothing when it is `null`: every optional member may be sent as `null`. */
template<class T, Bounds B, class F>
[[nodiscard]] auto nullable(Value<B>& value, F const& read) -> Decoded<std::optional<T>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null()).transform([] { return std::optional<T>{}; });
	}
	return read(value).transform([](T decoded) { return std::optional<T>{std::move(decoded)}; });
}

/** Visits every member of an object in document order; an error from a member is placed at its name. */
template<Bounds B, class F>
[[nodiscard]] auto each_member(Value<B>& value, F const& visit) -> Decoded<void> {
	auto object = lift(value.as_object());
	if (!object) {
		return std::unexpected(object.error());
	}
	while (true) {
		auto member = lift(object->next());
		if (!member) {
			return std::unexpected(member.error());
		}
		if (!*member) {
			return {};
		}
		auto& [name, inner] = **member;
		if (auto const visited = visit(name, inner); !visited) {
			return std::unexpected(within(std::string{name})(visited.error()));
		}
	}
}

/** Visits every element of an array in order; an error from an element is placed at its index. */
template<Bounds B, class F>
[[nodiscard]] auto each_element(Value<B>& value, F const& visit) -> Decoded<void> {
	auto array = lift(value.as_array());
	if (!array) {
		return std::unexpected(array.error());
	}
	for (auto index = std::size_t{0};; ++index) {
		auto element = lift(array->next());
		if (!element) {
			return std::unexpected(element.error());
		}
		if (!*element) {
			return {};
		}
		if (auto const visited = visit(index, **element); !visited) {
			return std::unexpected(within(index)(visited.error()));
		}
	}
}

/** A JSON schema member: kept as its exact text, checked whole; `null` leaves it absent. */
template<Bounds B>
[[nodiscard]] auto schema_of(Value<B>& value) -> Decoded<std::optional<JsonText>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null()).transform([] { return std::optional<JsonText>{}; });
	}
	if (*kind != Kind::Object) {
		return refuse(WrongShape{.expected = Expected::Object, .found = *kind});
	}
	return lift(value.capture()).transform([](JsonText schema) { return std::optional<JsonText>{std::move(schema)}; });
}

/** A boolean member the engine accepts at both values without acting on it (`strict`: everything is enforced). */
template<Bounds B>
[[nodiscard]] auto any_boolean(Value<B>& value) -> Decoded<void> {
	return nullable<bool>(value, [](Value<B>& flag) { return lift(flag.as_boolean()); }).transform([](std::optional<bool>) {});
}

}

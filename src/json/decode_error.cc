// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_error;

import std;
import engine.core;
import :error;

namespace engine::json {

/** What a refused member is wrong about, beyond its JSON shape. */
export enum class Violation : std::uint8_t {
	/** A required member is absent. */
	MissingField,
	/** The member is not one this object defines: the request names a parameter the api does not have. */
	UnknownField,
	/**
	 * The member is an api parameter (or the value of an api enumeration) the engine does not implement, sent with a
	 * value that would change the output: `background: true`, an image input, a `web_search` tool.
	 */
	Unsupported,
	/** The member is one the sdk marks deprecated or being replaced; it is refused, never translated. */
	Deprecated,
	/** A string outside the vocabulary its member defines: a role, an item type, a tool choice, an effort. */
	UnknownValue,
};

/** The shapes the decoder expects, by name: the JSON kinds, and the unions some members accept. */
export enum class Expected : std::uint8_t {
	Boolean,
	Integer,
	Number,
	String,
	Array,
	Object,
	/** A string, or an array of input items. */
	Input,
	/** A string, or an array of content parts. */
	Content,
	/** `"none"`, `"auto"`, `"required"`, or an object naming a tool. */
	ToolChoice,
};

export struct WrongShape {
	Expected expected;
	Kind found;

	[[nodiscard]] auto operator==(WrongShape const&) const -> bool = default;
};

/** The member conflicts with `other`, named by its path from the request. */
export struct Conflict {
	std::string other;

	[[nodiscard]] auto operator==(Conflict const&) const -> bool = default;
};

/** The member holds more than `limit` items. */
export struct TooMany {
	std::size_t limit;

	[[nodiscard]] auto operator==(TooMany const&) const -> bool = default;
};

/** The member's text is longer than `limit` bytes. */
export struct TooLong {
	std::size_t limit;

	[[nodiscard]] auto operator==(TooLong const&) const -> bool = default;
};

export using DecodeFault = std::variant<ParseError, Violation, WrongShape, Conflict, TooMany, TooLong>;

/** A member name or an array index on the way from the request to the refused member. */
export using PathStep = std::variant<std::string, std::size_t>;

/**
 * A request the engine refuses: what is wrong, and where. The path runs from the request's root to the member at
 * fault; it is empty when the body as a whole is not a JSON document.
 */
export struct [[nodiscard]] DecodeError {
	DecodeFault fault;
	std::vector<PathStep> path;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		if (std::holds_alternative<ParseError>(fault)) {
			return std::get<ParseError>(fault).retry();
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto operator==(DecodeError const&) const -> bool = default;
};

/** The path in OpenAI's `param` spelling: `input[2].content[0].type`. */
export [[nodiscard]] auto param(DecodeError const& error) -> std::string {
	auto text = std::string{};
	for (auto const& step : error.path) {
		if (std::holds_alternative<std::size_t>(step)) {
			text += std::format("[{}]", std::get<std::size_t>(step));
		} else {
			text += text.empty() ? std::get<std::string>(step) : "." + std::get<std::string>(step);
		}
	}
	return text;
}

[[nodiscard]] constexpr auto describe(Expected expected) -> std::string_view {
	switch (expected) {
	case Expected::Boolean:
		return "a boolean";
	case Expected::Integer:
		return "an integer";
	case Expected::Number:
		return "a number";
	case Expected::String:
		return "a string";
	case Expected::Array:
		return "an array";
	case Expected::Object:
		return "an object";
	case Expected::Input:
		return "a string or an array of input items";
	case Expected::Content:
		return "a string or an array of content parts";
	case Expected::ToolChoice:
		return R"("none", "auto", "required" or a tool to call)";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto describe(Violation violation) -> std::string_view {
	switch (violation) {
	case Violation::MissingField:
		return "is required";
	case Violation::UnknownField:
		return "is not a parameter of this api";
	case Violation::Unsupported:
		return "is not supported with this value";
	case Violation::Deprecated:
		return "is deprecated and not accepted";
	case Violation::UnknownValue:
		return "has a value outside its vocabulary";
	}
	std::unreachable();
}

/** An English message for a refused request, naming the member at fault. */
export [[nodiscard]] auto describe(DecodeError const& error) -> std::string {
	auto const where = param(error);
	auto const subject = where.empty() ? std::string{"the request body"} : std::format("'{}'", where);
	auto const& fault = error.fault;
	if (std::holds_alternative<ParseError>(fault)) {
		return std::format("{}: {}", subject, describe(std::get<ParseError>(fault).fault));
	}
	if (std::holds_alternative<Violation>(fault)) {
		return std::format("{} {}", subject, describe(std::get<Violation>(fault)));
	}
	if (std::holds_alternative<WrongShape>(fault)) {
		auto const& shape = std::get<WrongShape>(fault);
		return std::format("{} must be {}, not {}", subject, describe(shape.expected), describe(shape.found));
	}
	if (std::holds_alternative<Conflict>(fault)) {
		return std::format("{} conflicts with '{}'", subject, std::get<Conflict>(fault).other);
	}
	if (std::holds_alternative<TooMany>(fault)) {
		return std::format("{} holds more than {} items", subject, std::get<TooMany>(fault).limit);
	}
	return std::format("{} is longer than {} bytes", subject, std::get<TooLong>(fault).limit);
}

}

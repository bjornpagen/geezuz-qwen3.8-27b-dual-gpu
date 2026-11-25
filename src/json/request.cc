// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:request;

import std;
import engine.core;
import :error;
import :read;

/**
 * The OpenAI `/v1/chat/completions` request as the engine accepts it: the shapes a request decodes into. What a
 * value means beyond its shape (a temperature's range, whether a tool choice names a defined tool, where the system
 * message may stand) is the endpoint's to decide; every state these types hold is one the wire can express and the
 * engine can serve.
 */
namespace engine::json {

/** `chat_template_kwargs.enable_thinking`. */
export enum class Thinking : std::uint8_t {
	Enabled,
	Disabled,
};

/** `chat_template_kwargs.reasoning_effort`: the values the model's chat template defines. */
export enum class ReasoningEffort : std::uint8_t {
	Low,
	Medium,
	XHigh,
};

/** `chat_template_kwargs.preserve_thinking`: whether past assistant turns keep their reasoning in the prompt. */
export enum class PastThinking : std::uint8_t {
	Preserved,
	Dropped,
};

/** The chat template's variables a request may set; each is absent when the request leaves it to the template. */
export struct TemplateOptions {
	std::optional<Thinking> thinking;
	std::optional<ReasoningEffort> reasoning_effort;
	std::optional<PastThinking> past_thinking;

	[[nodiscard]] auto operator==(TemplateOptions const&) const -> bool = default;
};

/**
 * How the response is delivered: `stream` and `stream_options.include_usage` together, as the three states they
 * can be in (usage options exist only on a stream).
 */
export enum class Delivery : std::uint8_t {
	Whole,
	Stream,
	StreamWithUsage,
};

export enum class ToolChoiceMode : std::uint8_t {
	None,
	Auto,
	Required,
};

/** `tool_choice` naming one function the model must call. */
export struct ForcedFunction {
	std::string name;

	[[nodiscard]] auto operator==(ForcedFunction const&) const -> bool = default;
};

export using ToolChoice = std::variant<ToolChoiceMode, ForcedFunction>;

/**
 * A function tool. The chat template renders the tool as the client wrote it, so `definition` keeps the whole tools
 * element; `parameters` is its JSON schema, for reading tool-call arguments back by their declared types.
 */
export struct Tool {
	std::string name;
	std::optional<JsonText> parameters;
	JsonText definition;

	[[nodiscard]] auto operator==(Tool const&) const -> bool = default;
};

/** An argument the chat template renders as its text when it is a string, and as JSON otherwise. */
export using ArgumentValue = std::variant<std::string, JsonText>;

export struct Argument {
	std::string name;
	ArgumentValue value;

	[[nodiscard]] auto operator==(Argument const&) const -> bool = default;
};

/**
 * A tool call of a past assistant turn. Its arguments arrive as an object, as a string holding one, or as an empty
 * string for none; all three decode to the named arguments in the order written.
 */
export struct ToolCall {
	std::string id;
	std::string name;
	std::vector<Argument> arguments;

	[[nodiscard]] auto operator==(ToolCall const&) const -> bool = default;
};

/**
 * Message content is text: a string, or an array of text parts, which reads as their concatenation (as the chat
 * template renders it). A missing or `null` content is empty; harnesses resend tool-call-only and reasoning-only
 * assistant turns that way.
 */
export struct SystemMessage {
	std::string content;

	[[nodiscard]] auto operator==(SystemMessage const&) const -> bool = default;
};

export struct UserMessage {
	std::string content;

	[[nodiscard]] auto operator==(UserMessage const&) const -> bool = default;
};

export struct AssistantMessage {
	std::string content;
	/** The turn's reasoning as resent; empty when absent, which the chat template renders alike. */
	std::string reasoning_content;
	std::vector<ToolCall> tool_calls;

	[[nodiscard]] auto operator==(AssistantMessage const&) const -> bool = default;
};

export struct ToolMessage {
	std::string tool_call_id;
	std::string content;

	[[nodiscard]] auto operator==(ToolMessage const&) const -> bool = default;
};

/** One message, by role: `developer` is the newer name of `system` and decodes as one. */
export using Message = std::variant<SystemMessage, UserMessage, AssistantMessage, ToolMessage>;

export struct ChatRequest {
	std::optional<std::string> model;
	std::vector<Message> messages;
	std::vector<Tool> tools;
	std::optional<ToolChoice> tool_choice;
	Delivery delivery;
	/** `max_completion_tokens`, or its older name `max_tokens`; a request may send one of them. */
	std::optional<std::int64_t> max_tokens;
	std::optional<double> temperature;
	std::optional<double> top_p;
	std::optional<std::int64_t> top_k;
	std::optional<double> min_p;
	std::optional<std::int64_t> seed;
	/** Stop sequences; a lone string is a list of one. */
	std::vector<std::string> stop;
	TemplateOptions template_options;

	[[nodiscard]] auto operator==(ChatRequest const&) const -> bool = default;
};

/** The most stop sequences a request may send. */
export inline constexpr std::size_t max_stop_sequences = 16;

/** What a refused member is wrong about, beyond its JSON shape. */
export enum class Violation : std::uint8_t {
	/** A required member is absent. */
	MissingField,
	/** The member is not one this object defines: the request names a parameter the endpoint does not know. */
	UnknownField,
	/**
	 * The member is an OpenAI parameter (or the value an OpenAI enumeration) the engine does not implement, sent with
	 * a value that would change the output: `n: 2`, `logprobs: true`, an image content part, a `custom` tool.
	 */
	Unsupported,
	/** A string outside the vocabulary its member defines: a role, a part type, a tool choice, an effort. */
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
	/** A string, an array of text parts, or null. */
	Content,
	/** A string or an array of strings. */
	Stop,
	/** `"none"`, `"auto"`, `"required"`, or an object naming a function. */
	ToolChoice,
	/** An object, or a string holding one. */
	Arguments,
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

export using DecodeFault = std::variant<ParseError, Violation, WrongShape, Conflict, TooMany>;

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

/** The path in OpenAI's `param` spelling: `messages[2].content[0].type`. */
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
	case Expected::Content:
		return "a string, an array of text parts, or null";
	case Expected::Stop:
		return "a string or an array of strings";
	case Expected::ToolChoice:
		return R"("none", "auto", "required" or a function to call)";
	case Expected::Arguments:
		return "an object or a string holding one";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto describe(Violation violation) -> std::string_view {
	switch (violation) {
	case Violation::MissingField:
		return "is required";
	case Violation::UnknownField:
		return "is not a parameter of this endpoint";
	case Violation::Unsupported:
		return "is not supported with this value";
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
	return std::format("{} holds more than {} items", subject, std::get<TooMany>(fault).limit);
}

}

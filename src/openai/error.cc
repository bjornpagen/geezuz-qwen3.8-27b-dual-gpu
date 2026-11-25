// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:error;

import std;
import engine.core;
import engine.http;
import engine.json;
import engine.sampling;

namespace engine::openai {

/**
 * Every refusal this endpoint answers, one case per condition a client can tell apart. The wire spelling is the
 * error's `code`, stable across releases: clients branch on it, never on the message.
 */
export enum class Code : std::uint8_t {
	JsonEmpty,
	JsonTruncated,
	JsonMalformed,
	JsonTrailingContent,
	JsonInvalidLiteral,
	JsonInvalidNumber,
	JsonInvalidString,
	JsonControlCharacter,
	JsonInvalidUtf8,
	JsonDuplicateKey,
	JsonTooLarge,
	JsonTooDeep,
	JsonStringTooLong,
	JsonTooManyMembers,
	JsonNumberTooLong,
	/** The parser could not get memory for the body: the server is short, not the request wrong. */
	OutOfMemory,
	MissingField,
	UnknownParameter,
	UnsupportedParameter,
	UnknownValue,
	WrongType,
	ConflictingParameters,
	TooManyItems,
	/** No messages: the chat template has nothing to render. */
	MessagesEmpty,
	/** A system (or developer) message after the first: the template places the system turn only at the start. */
	SystemNotFirst,
	/** No user message that is not wholly a tool response: the template finds no query to answer. */
	NoUserQuery,
	/** Two tools share a function name, so a tool call could not say which it means. */
	DuplicateTool,
	/** A function name that is empty or longer than max_tool_name_bytes. */
	InvalidToolName,
	/** `tool_choice` requires a call (`required` or a named function) but the request defines no tools. */
	ToolChoiceWithoutTools,
	/** `tool_choice` names a function the request does not define. */
	UnknownTool,
	MaxTokensBelowOne,
	/** `max_tokens` is more than the whole context could hold. */
	MaxTokensAboveContext,
	StopEmpty,
	StopTooLong,
	TemperatureOutOfRange,
	TopKBelowOne,
	TopKAboveBound,
	TopPOutOfRange,
	MinPOutOfRange,
	/** The rendered prompt and the requested completion do not fit the context. */
	ContextLengthExceeded,
	/** The engine is still loading the model; the same request succeeds once it serves. */
	EngineLoading,
	/** A whole completion grew past what one response may carry. */
	ResponseTooLarge,
	/** A streaming client read so much slower than the model wrote that the unsent text reached its bound. */
	StreamBacklog,
};

export [[nodiscard]] constexpr auto spelling(Code code) -> std::string_view {
	switch (code) {
	case Code::JsonEmpty:
		return "json_empty";
	case Code::JsonTruncated:
		return "json_truncated";
	case Code::JsonMalformed:
		return "json_malformed";
	case Code::JsonTrailingContent:
		return "json_trailing_content";
	case Code::JsonInvalidLiteral:
		return "json_invalid_literal";
	case Code::JsonInvalidNumber:
		return "json_invalid_number";
	case Code::JsonInvalidString:
		return "json_invalid_string";
	case Code::JsonControlCharacter:
		return "json_control_character";
	case Code::JsonInvalidUtf8:
		return "json_invalid_utf8";
	case Code::JsonDuplicateKey:
		return "json_duplicate_key";
	case Code::JsonTooLarge:
		return "json_too_large";
	case Code::JsonTooDeep:
		return "json_too_deep";
	case Code::JsonStringTooLong:
		return "json_string_too_long";
	case Code::JsonTooManyMembers:
		return "json_too_many_members";
	case Code::JsonNumberTooLong:
		return "json_number_too_long";
	case Code::OutOfMemory:
		return "out_of_memory";
	case Code::MissingField:
		return "missing_required_parameter";
	case Code::UnknownParameter:
		return "unknown_parameter";
	case Code::UnsupportedParameter:
		return "unsupported_parameter";
	case Code::UnknownValue:
		return "invalid_value";
	case Code::WrongType:
		return "invalid_type";
	case Code::ConflictingParameters:
		return "conflicting_parameters";
	case Code::TooManyItems:
		return "too_many_items";
	case Code::MessagesEmpty:
		return "messages_empty";
	case Code::SystemNotFirst:
		return "system_message_not_first";
	case Code::NoUserQuery:
		return "no_user_query";
	case Code::DuplicateTool:
		return "duplicate_tool_name";
	case Code::InvalidToolName:
		return "invalid_tool_name";
	case Code::ToolChoiceWithoutTools:
		return "tool_choice_without_tools";
	case Code::UnknownTool:
		return "unknown_tool";
	case Code::MaxTokensBelowOne:
		return "max_tokens_below_one";
	case Code::MaxTokensAboveContext:
		return "max_tokens_above_context";
	case Code::StopEmpty:
		return "stop_sequence_empty";
	case Code::StopTooLong:
		return "stop_sequence_too_long";
	case Code::TemperatureOutOfRange:
		return "temperature_out_of_range";
	case Code::TopKBelowOne:
		return "top_k_below_one";
	case Code::TopKAboveBound:
		return "top_k_above_bound";
	case Code::TopPOutOfRange:
		return "top_p_out_of_range";
	case Code::MinPOutOfRange:
		return "min_p_out_of_range";
	case Code::ContextLengthExceeded:
		return "context_length_exceeded";
	case Code::EngineLoading:
		return "engine_loading";
	case Code::ResponseTooLarge:
		return "response_too_large";
	case Code::StreamBacklog:
		return "stream_backlog";
	}
	std::unreachable();
}

/** Who must change for a retry to succeed: the client's request, or the server's state. */
enum class Fault : std::uint8_t {
	Request,
	ServerBusy,
	ServerLimit,
};

[[nodiscard]] constexpr auto fault_of(Code code) -> Fault {
	switch (code) {
	case Code::OutOfMemory:
	case Code::EngineLoading:
		return Fault::ServerBusy;
	case Code::ResponseTooLarge:
	case Code::StreamBacklog:
		return Fault::ServerLimit;
	case Code::JsonEmpty:
	case Code::JsonTruncated:
	case Code::JsonMalformed:
	case Code::JsonTrailingContent:
	case Code::JsonInvalidLiteral:
	case Code::JsonInvalidNumber:
	case Code::JsonInvalidString:
	case Code::JsonControlCharacter:
	case Code::JsonInvalidUtf8:
	case Code::JsonDuplicateKey:
	case Code::JsonTooLarge:
	case Code::JsonTooDeep:
	case Code::JsonStringTooLong:
	case Code::JsonTooManyMembers:
	case Code::JsonNumberTooLong:
	case Code::MissingField:
	case Code::UnknownParameter:
	case Code::UnsupportedParameter:
	case Code::UnknownValue:
	case Code::WrongType:
	case Code::ConflictingParameters:
	case Code::TooManyItems:
	case Code::MessagesEmpty:
	case Code::SystemNotFirst:
	case Code::NoUserQuery:
	case Code::DuplicateTool:
	case Code::InvalidToolName:
	case Code::ToolChoiceWithoutTools:
	case Code::UnknownTool:
	case Code::MaxTokensBelowOne:
	case Code::MaxTokensAboveContext:
	case Code::StopEmpty:
	case Code::StopTooLong:
	case Code::TemperatureOutOfRange:
	case Code::TopKBelowOne:
	case Code::TopKAboveBound:
	case Code::TopPOutOfRange:
	case Code::MinPOutOfRange:
	case Code::ContextLengthExceeded:
		return Fault::Request;
	}
	std::unreachable();
}

export [[nodiscard]] constexpr auto status_of(Code code) -> http::StatusCode {
	switch (fault_of(code)) {
	case Fault::Request:
		return http::StatusCode::BadRequest;
	case Fault::ServerBusy:
		return http::StatusCode::ServiceUnavailable;
	case Fault::ServerLimit:
		return http::StatusCode::InternalServerError;
	}
	std::unreachable();
}

/** OpenAI's error `type`: the request's fault, or the server's. */
[[nodiscard]] constexpr auto type_of(Code code) -> std::string_view {
	return fault_of(code) == Fault::Request ? "invalid_request_error" : "server_error";
}

/** A refusal: its code, a sentence for a person, and the request member at fault when one is. */
export struct [[nodiscard]] ApiError {
	Code code;
	std::string message;
	std::optional<std::string> param;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return fault_of(code) == Fault::ServerBusy ? core::Retry::Transient : core::Retry::Permanent;
	}

	[[nodiscard]] auto operator==(ApiError const&) const -> bool = default;
};

[[nodiscard]] constexpr auto code_of(json::ParseFault fault) -> Code {
	switch (fault) {
	case json::ParseFault::Empty:
		return Code::JsonEmpty;
	case json::ParseFault::Truncated:
		return Code::JsonTruncated;
	case json::ParseFault::Malformed:
		return Code::JsonMalformed;
	case json::ParseFault::TrailingContent:
		return Code::JsonTrailingContent;
	case json::ParseFault::InvalidLiteral:
		return Code::JsonInvalidLiteral;
	case json::ParseFault::InvalidNumber:
		return Code::JsonInvalidNumber;
	case json::ParseFault::InvalidString:
		return Code::JsonInvalidString;
	case json::ParseFault::ControlCharacter:
		return Code::JsonControlCharacter;
	case json::ParseFault::InvalidUtf8:
		return Code::JsonInvalidUtf8;
	case json::ParseFault::DuplicateKey:
		return Code::JsonDuplicateKey;
	case json::ParseFault::TooLarge:
		return Code::JsonTooLarge;
	case json::ParseFault::TooDeep:
		return Code::JsonTooDeep;
	case json::ParseFault::StringTooLong:
		return Code::JsonStringTooLong;
	case json::ParseFault::TooManyMembers:
		return Code::JsonTooManyMembers;
	case json::ParseFault::NumberTooLong:
		return Code::JsonNumberTooLong;
	case json::ParseFault::OutOfMemory:
		return Code::OutOfMemory;
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto code_of(json::Violation violation) -> Code {
	switch (violation) {
	case json::Violation::MissingField:
		return Code::MissingField;
	case json::Violation::UnknownField:
		return Code::UnknownParameter;
	case json::Violation::Unsupported:
		return Code::UnsupportedParameter;
	case json::Violation::UnknownValue:
		return Code::UnknownValue;
	}
	std::unreachable();
}

struct FaultCode {
	[[nodiscard]] auto operator()(json::ParseError const& error) const -> Code {
		return code_of(error.fault);
	}

	[[nodiscard]] auto operator()(json::Violation violation) const -> Code {
		return code_of(violation);
	}

	[[nodiscard]] auto operator()(json::WrongShape const& /*shape*/) const -> Code {
		return Code::WrongType;
	}

	[[nodiscard]] auto operator()(json::Conflict const& /*conflict*/) const -> Code {
		return Code::ConflictingParameters;
	}

	[[nodiscard]] auto operator()(json::TooMany const& /*many*/) const -> Code {
		return Code::TooManyItems;
	}
};

/** A path that names a member is the error's `param`; a body that is not a document has none. */
[[nodiscard]] auto param_of(std::string path) -> std::optional<std::string> {
	if (path.empty()) {
		return std::nullopt;
	}
	return path;
}

export [[nodiscard]] auto refusal(json::DecodeError const& error) -> ApiError {
	return ApiError{.code = std::visit(FaultCode{}, error.fault), .message = json::describe(error), .param = param_of(json::param(error))};
}

struct Refused {
	Code code;
	std::string_view param;
	std::string_view message;
};

[[nodiscard]] constexpr auto refused(sampling::ParamErrorKind kind) -> Refused {
	switch (kind) {
	case sampling::ParamErrorKind::TemperatureOutOfRange:
		return {.code = Code::TemperatureOutOfRange, .param = "temperature", .message = "'temperature' must be within [0, 2]"};
	case sampling::ParamErrorKind::TopKBelowOne:
		return {.code = Code::TopKBelowOne, .param = "top_k", .message = "'top_k' must be at least 1"};
	case sampling::ParamErrorKind::TopKAboveBound:
		return {.code = Code::TopKAboveBound, .param = "top_k", .message = "'top_k' is above the most candidates the engine samples from"};
	case sampling::ParamErrorKind::TopPOutOfRange:
		return {.code = Code::TopPOutOfRange, .param = "top_p", .message = "'top_p' must be within [0, 1]"};
	case sampling::ParamErrorKind::MinPOutOfRange:
		return {.code = Code::MinPOutOfRange, .param = "min_p", .message = "'min_p' must be within [0, 1]"};
	}
	std::unreachable();
}

export [[nodiscard]] auto refusal(sampling::ParamError error) -> ApiError {
	auto const named = refused(error.kind);
	return ApiError{.code = named.code, .message = std::string{named.message}, .param = std::string{named.param}};
}

template<json::Sink O>
auto put_error(json::Members<O>& members, ApiError const& error) -> void {
	members.object("error", [&error](json::Members<O>& inner) {
		inner.string("message", error.message);
		inner.string("type", type_of(error.code));
		if (error.param) {
			inner.string("param", *error.param);
		} else {
			inner.null("param");
		}
		inner.string("code", spelling(error.code));
	});
}

/** OpenAI's error object: `{"error": {"message", "type", "param", "code"}}`. */
export [[nodiscard]] auto error_body(ApiError const& error) -> std::string {
	return json::encode([&error](auto& out) { json::write_object(out, [&error](auto& members) { put_error(members, error); }); });
}

export [[nodiscard]] auto error_response(ApiError const& error) -> http::Response {
	return http::Response{.status = status_of(error.code), .content_type = http::ContentType::Json, .body = error_body(error)};
}

}

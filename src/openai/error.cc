// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:error;

import std;
import engine.core;
import engine.http;
import engine.json;

namespace engine::openai {

/**
 * Every refusal the api answers, one case per condition a client can tell apart. The wire spelling is the error's
 * `code`, stable across releases: clients branch on it, never on the message.
 */
export enum class Code : std::uint8_t {
	/** The body is not one JSON document within the request's bounds. */
	InvalidJson,
	/** The parser could not get memory for the body: the server is short, not the request wrong. */
	OutOfMemory,
	MissingRequiredParameter,
	UnknownParameter,
	/** A parameter or value the sdk defines that the engine does not serve; never approximated. */
	UnsupportedParameter,
	DeprecatedParameter,
	InvalidValue,
	InvalidType,
	ConflictingParameters,
	TooManyItems,
	StringTooLong,
	ModelNotFound,
	InvalidToolName,
	DuplicateToolName,
	/** A tool's definition is longer than one JSON text the chat format reads back may be. */
	ToolTooLarge,
	UnknownTool,
	ToolChoiceWithoutTools,
	MaxOutputTokensBelowOne,
	MaxOutputTokensAboveContext,
	TemperatureOutOfRange,
	TopPOutOfRange,
	SystemMessageNotFirst,
	NoUserQuery,
	ResponseNotFound,
	PreviousResponseIncomplete,
	PreviousResponseMismatch,
	/** The rendered prompt and the requested output do not fit the context. */
	ContextLengthExceeded,
	/** The request's grammar outgrew the engine's bounds while the reply was decoded; the reply ended there. */
	GrammarTooLarge,
	/** The engine is still loading; the same request succeeds once it serves. */
	EngineLoading,
	/** The engine failed and serves nothing more; a restart is the operator's call. */
	EngineFailed,
	/** The engine failed while it answered this request. */
	ServerError,
	/** The engine holds as many requests as its intake bound allows; the same request succeeds once some end. */
	EngineBusy,
	/** A response grew past what one body or one event may carry. */
	ResponseTooLarge,
	/** A streaming client read so much slower than the model wrote that the unsent text reached its bound. */
	StreamBacklog,
};

export [[nodiscard]] constexpr auto spelling(Code code) -> std::string_view {
	switch (code) {
	case Code::InvalidJson:
		return "invalid_json";
	case Code::OutOfMemory:
		return "out_of_memory";
	case Code::MissingRequiredParameter:
		return "missing_required_parameter";
	case Code::UnknownParameter:
		return "unknown_parameter";
	case Code::UnsupportedParameter:
		return "unsupported_parameter";
	case Code::DeprecatedParameter:
		return "deprecated_parameter";
	case Code::InvalidValue:
		return "invalid_value";
	case Code::InvalidType:
		return "invalid_type";
	case Code::ConflictingParameters:
		return "conflicting_parameters";
	case Code::TooManyItems:
		return "too_many_items";
	case Code::StringTooLong:
		return "string_too_long";
	case Code::ModelNotFound:
		return "model_not_found";
	case Code::InvalidToolName:
		return "invalid_tool_name";
	case Code::DuplicateToolName:
		return "duplicate_tool_name";
	case Code::ToolTooLarge:
		return "tool_too_large";
	case Code::UnknownTool:
		return "unknown_tool";
	case Code::ToolChoiceWithoutTools:
		return "tool_choice_without_tools";
	case Code::MaxOutputTokensBelowOne:
		return "max_output_tokens_below_one";
	case Code::MaxOutputTokensAboveContext:
		return "max_output_tokens_above_context";
	case Code::TemperatureOutOfRange:
		return "temperature_out_of_range";
	case Code::TopPOutOfRange:
		return "top_p_out_of_range";
	case Code::SystemMessageNotFirst:
		return "system_message_not_first";
	case Code::NoUserQuery:
		return "no_user_query";
	case Code::ResponseNotFound:
		return "response_not_found";
	case Code::PreviousResponseIncomplete:
		return "previous_response_incomplete";
	case Code::PreviousResponseMismatch:
		return "previous_response_mismatch";
	case Code::ContextLengthExceeded:
		return "context_length_exceeded";
	case Code::GrammarTooLarge:
		return "grammar_too_large";
	case Code::EngineLoading:
		return "engine_loading";
	case Code::EngineFailed:
		return "engine_failed";
	case Code::ServerError:
		return "server_error";
	case Code::EngineBusy:
		return "engine_busy";
	case Code::ResponseTooLarge:
		return "response_too_large";
	case Code::StreamBacklog:
		return "stream_backlog";
	}
	std::unreachable();
}

/** The status a whole refusal is answered with. */
export [[nodiscard]] constexpr auto status_of(Code code) -> http::StatusCode {
	switch (code) {
	case Code::ModelNotFound:
	case Code::ResponseNotFound:
		return http::StatusCode::NotFound;
	case Code::OutOfMemory:
	case Code::EngineLoading:
	case Code::EngineFailed:
	case Code::EngineBusy:
		return http::StatusCode::ServiceUnavailable;
	case Code::ServerError:
	case Code::ResponseTooLarge:
		return http::StatusCode::InternalServerError;
	case Code::InvalidJson:
	case Code::MissingRequiredParameter:
	case Code::UnknownParameter:
	case Code::UnsupportedParameter:
	case Code::DeprecatedParameter:
	case Code::InvalidValue:
	case Code::InvalidType:
	case Code::ConflictingParameters:
	case Code::TooManyItems:
	case Code::StringTooLong:
	case Code::InvalidToolName:
	case Code::DuplicateToolName:
	case Code::ToolTooLarge:
	case Code::UnknownTool:
	case Code::ToolChoiceWithoutTools:
	case Code::MaxOutputTokensBelowOne:
	case Code::MaxOutputTokensAboveContext:
	case Code::TemperatureOutOfRange:
	case Code::TopPOutOfRange:
	case Code::SystemMessageNotFirst:
	case Code::NoUserQuery:
	case Code::PreviousResponseIncomplete:
	case Code::PreviousResponseMismatch:
	case Code::ContextLengthExceeded:
	case Code::GrammarTooLarge:
	case Code::StreamBacklog:
		return http::StatusCode::BadRequest;
	}
	std::unreachable();
}

/** OpenAI's error `type`: the client's fault, or the server's. */
[[nodiscard]] constexpr auto type_of(Code code) -> std::string_view {
	auto const status = status_of(code);
	return status == http::StatusCode::ServiceUnavailable || status == http::StatusCode::InternalServerError ? "server_error"
	                                                                                                         : "invalid_request_error";
}

/** Whether the same request, sent again unchanged, could be served: only when the server was short or not yet up. */
export [[nodiscard]] constexpr auto retry_of(Code code) -> core::Retry {
	switch (code) {
	case Code::OutOfMemory:
	case Code::EngineLoading:
	case Code::EngineBusy:
		return core::Retry::Transient;
	case Code::InvalidJson:
	case Code::MissingRequiredParameter:
	case Code::UnknownParameter:
	case Code::UnsupportedParameter:
	case Code::DeprecatedParameter:
	case Code::InvalidValue:
	case Code::InvalidType:
	case Code::ConflictingParameters:
	case Code::TooManyItems:
	case Code::StringTooLong:
	case Code::ModelNotFound:
	case Code::InvalidToolName:
	case Code::DuplicateToolName:
	case Code::ToolTooLarge:
	case Code::UnknownTool:
	case Code::ToolChoiceWithoutTools:
	case Code::MaxOutputTokensBelowOne:
	case Code::MaxOutputTokensAboveContext:
	case Code::TemperatureOutOfRange:
	case Code::TopPOutOfRange:
	case Code::SystemMessageNotFirst:
	case Code::NoUserQuery:
	case Code::ResponseNotFound:
	case Code::PreviousResponseIncomplete:
	case Code::PreviousResponseMismatch:
	case Code::ContextLengthExceeded:
	case Code::GrammarTooLarge:
	case Code::EngineFailed:
	case Code::ServerError:
	case Code::ResponseTooLarge:
	case Code::StreamBacklog:
		return core::Retry::Permanent;
	}
	std::unreachable();
}

/** A refusal: its code, a sentence a person reads, and the request member at fault in OpenAI's `param` spelling. */
export struct [[nodiscard]] ApiError {
	Code code;
	std::string message;
	std::optional<std::string> param;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return retry_of(code);
	}

	[[nodiscard]] auto operator==(ApiError const&) const -> bool = default;
};

/** OpenAI's error object, `{"error": {"message", "type", "param", "code"}}`. */
export [[nodiscard]] auto error_body(ApiError const& error) -> std::string {
	return json::encode([&error](auto& out) {
		json::write_object(out, [&error]<class O>(json::Members<O>& document) {
			document.object("error", [&error](json::Members<O>& inner) {
				inner.string("message", error.message);
				inner.string("type", type_of(error.code));
				if (error.param) {
					inner.string("param", *error.param);
				} else {
					inner.null("param");
				}
				inner.string("code", spelling(error.code));
			});
		});
	});
}

export [[nodiscard]] auto error_response(ApiError const& error) -> http::Response {
	return http::Response{.status = status_of(error.code), .content_type = http::ContentType::Json, .body = error_body(error)};
}

[[nodiscard]] constexpr auto code_of(json::Violation violation) -> Code {
	switch (violation) {
	case json::Violation::MissingField:
		return Code::MissingRequiredParameter;
	case json::Violation::UnknownField:
		return Code::UnknownParameter;
	case json::Violation::Unsupported:
		return Code::UnsupportedParameter;
	case json::Violation::Deprecated:
		return Code::DeprecatedParameter;
	case json::Violation::UnknownValue:
		return Code::InvalidValue;
	}
	std::unreachable();
}

struct FaultCode {
	[[nodiscard]] auto operator()(json::ParseError const& error) const -> Code {
		// Memory the parser could not get is the server's shortage; every other fault is the text's.
		return error.fault == json::ParseFault::OutOfMemory ? Code::OutOfMemory : Code::InvalidJson;
	}

	[[nodiscard]] auto operator()(json::Violation violation) const -> Code {
		return code_of(violation);
	}

	[[nodiscard]] auto operator()(json::WrongShape const& /*shape*/) const -> Code {
		return Code::InvalidType;
	}

	[[nodiscard]] auto operator()(json::Conflict const& /*conflict*/) const -> Code {
		return Code::ConflictingParameters;
	}

	[[nodiscard]] auto operator()(json::TooMany const& /*many*/) const -> Code {
		return Code::TooManyItems;
	}

	[[nodiscard]] auto operator()(json::TooLong const& /*long_text*/) const -> Code {
		return Code::StringTooLong;
	}

	[[nodiscard]] auto operator()(json::UnsupportedValue const& /*value*/) const -> Code {
		return Code::UnsupportedParameter;
	}
};

/** A request the decoder refused, as the api answers it. */
export [[nodiscard]] auto refusal(json::DecodeError const& error) -> ApiError {
	auto param = json::param(error);
	return ApiError{
	    .code = std::visit(FaultCode{}, error.fault),
	    .message = json::describe(error),
	    .param = param.empty() ? std::nullopt : std::optional{std::move(param)},
	};
}

}

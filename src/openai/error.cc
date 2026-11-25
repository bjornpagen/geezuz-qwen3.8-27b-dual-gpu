// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:error;

import std;
import engine.core;
import engine.grammar;
import engine.http;
import engine.json;
import engine.regex;

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
	/** A continuation asked of a caller that keeps no stored responses (the cli). */
	PreviousResponseUnavailable,
	/** The rendered prompt and the requested output do not fit the context. */
	ContextLengthExceeded,
	/** The request's grammar outgrew the engine's bounds while the reply was decoded; the reply ended there. */
	GrammarOverrun,
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
	case Code::PreviousResponseUnavailable:
		return "previous_response_unavailable";
	case Code::ContextLengthExceeded:
		return "context_length_exceeded";
	case Code::GrammarOverrun:
		return "grammar_overrun";
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
	case Code::PreviousResponseUnavailable:
	case Code::ContextLengthExceeded:
	case Code::GrammarOverrun:
	case Code::StreamBacklog:
		return http::StatusCode::BadRequest;
	}
	std::unreachable();
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
	case Code::PreviousResponseUnavailable:
	case Code::ContextLengthExceeded:
	case Code::GrammarOverrun:
	case Code::EngineFailed:
	case Code::ServerError:
	case Code::ResponseTooLarge:
	case Code::StreamBacklog:
		return core::Retry::Permanent;
	}
	std::unreachable();
}

/** A grammar the request's tools or format compile to, refused: each case its own code. */
export [[nodiscard]] constexpr auto spelling(grammar::Refusal refusal) -> std::string_view {
	switch (refusal) {
	case grammar::Refusal::SchemaShape:
		return "schema_shape";
	case grammar::Refusal::SchemaKeyword:
		return "schema_keyword";
	case grammar::Refusal::SchemaValue:
		return "schema_value";
	case grammar::Refusal::SchemaType:
		return "schema_type";
	case grammar::Refusal::SchemaReference:
		return "schema_reference";
	case grammar::Refusal::SchemaFormat:
		return "schema_format";
	case grammar::Refusal::SchemaOneOf:
		return "schema_one_of";
	case grammar::Refusal::SchemaMultipleOf:
		return "schema_multiple_of";
	case grammar::Refusal::SchemaBound:
		return "schema_bound";
	case grammar::Refusal::Empty:
		return "grammar_empty";
	case grammar::Refusal::SchemaOpenArgument:
		return "schema_open_argument";
	case grammar::Refusal::SchemaStringUnion:
		return "schema_string_union";
	case grammar::Refusal::SchemaRecursiveArgument:
		return "schema_recursive_argument";
	case grammar::Refusal::ToolParameters:
		return "tool_parameters";
	case grammar::Refusal::ToolName:
		return "tool_name";
	case grammar::Refusal::ToolDuplicate:
		return "tool_duplicate";
	case grammar::Refusal::ToolUnknown:
		return "tool_unknown";
	case grammar::Refusal::LarkSyntax:
		return "lark_syntax";
	case grammar::Refusal::LarkUndefined:
		return "lark_undefined";
	case grammar::Refusal::LarkRecursiveTerminal:
		return "lark_recursive_terminal";
	case grammar::Refusal::LarkNoStart:
		return "lark_no_start";
	case grammar::Refusal::LarkPriority:
		return "lark_priority";
	case grammar::Refusal::LarkTemplate:
		return "lark_template";
	case grammar::Refusal::LarkDirective:
		return "lark_directive";
	case grammar::Refusal::LarkToken:
		return "lark_token";
	case grammar::Refusal::LarkOption:
		return "lark_option";
	case grammar::Refusal::LarkOperator:
		return "lark_operator";
	case grammar::Refusal::Regex:
		return "regex";
	case grammar::Refusal::SchemaTooLarge:
		return "schema_too_large";
	case grammar::Refusal::SchemaTooMany:
		return "schema_too_many";
	case grammar::Refusal::LarkTooLarge:
		return "lark_too_large";
	case grammar::Refusal::AutomatonTooLarge:
		return "automaton_too_large";
	case grammar::Refusal::GrammarTooLarge:
		return "grammar_too_large";
	}
	std::unreachable();
}

/** A custom tool's regular expression, refused: each fault its own code. */
export [[nodiscard]] constexpr auto spelling(regex::Fault fault) -> std::string_view {
	switch (fault) {
	case regex::Fault::UnexpectedEnd:
		return "regex_unexpected_end";
	case regex::Fault::UnbalancedGroup:
		return "regex_unbalanced_group";
	case regex::Fault::NothingToRepeat:
		return "regex_nothing_to_repeat";
	case regex::Fault::BadRepetition:
		return "regex_bad_repetition";
	case regex::Fault::RepetitionTooLarge:
		return "regex_repetition_too_large";
	case regex::Fault::BadEscape:
		return "regex_bad_escape";
	case regex::Fault::BadClass:
		return "regex_bad_class";
	case regex::Fault::BadCharacter:
		return "regex_bad_character";
	case regex::Fault::TooDeep:
		return "regex_too_deep";
	case regex::Fault::TooLarge:
		return "regex_too_large";
	case regex::Fault::Lookaround:
		return "regex_lookaround";
	case regex::Fault::LazyQuantifier:
		return "regex_lazy_quantifier";
	case regex::Fault::Anchor:
		return "regex_anchor";
	case regex::Fault::WordBoundary:
		return "regex_word_boundary";
	case regex::Fault::Backreference:
		return "regex_backreference";
	case regex::Fault::UnicodeProperty:
		return "regex_unicode_property";
	case regex::Fault::UnicodeCase:
		return "regex_unicode_case";
	case regex::Fault::ClassSyntax:
		return "regex_class_syntax";
	case regex::Fault::GroupSyntax:
		return "regex_group_syntax";
	}
	std::unreachable();
}

/** A grammar refusal is the request as written: a 400, refused again if sent again. */
[[nodiscard]] constexpr auto status_of(grammar::Refusal /*refusal*/) -> http::StatusCode {
	return http::StatusCode::BadRequest;
}

[[nodiscard]] constexpr auto status_of(regex::Fault /*fault*/) -> http::StatusCode {
	return http::StatusCode::BadRequest;
}

/**
 * A refusal's stable `code`: the api's own, or one per grammar refusal and per regular expression fault, so a client
 * can tell every condition apart.
 */
export using ErrorCode = std::variant<Code, grammar::Refusal, regex::Fault>;

export [[nodiscard]] constexpr auto spelling(ErrorCode const& code) -> std::string_view {
	return std::visit(
	    [](auto chosen) {
		    return spelling(chosen);
	    },
	    code);
}

export [[nodiscard]] constexpr auto status_of(ErrorCode const& code) -> http::StatusCode {
	return std::visit(
	    [](auto chosen) {
		    return status_of(chosen);
	    },
	    code);
}

[[nodiscard]] constexpr auto type_of(ErrorCode const& code) -> std::string_view {
	auto const status = status_of(code);
	return status == http::StatusCode::ServiceUnavailable || status == http::StatusCode::InternalServerError ? "server_error"
	                                                                                                         : "invalid_request_error";
}

/** A refusal: its code, a sentence a person reads, and the request member at fault in OpenAI's `param` spelling. */
export struct [[nodiscard]] ApiError {
	ErrorCode code;
	std::string message;
	std::optional<std::string> param;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		// A grammar or regex refusal is of the request as written; the api's own codes say for themselves.
		auto const own = core::alternative<Code>(code);
		return own ? retry_of(*own) : core::Retry::Permanent;
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

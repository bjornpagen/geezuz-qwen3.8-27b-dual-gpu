// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:compile;

import std;
import engine.core;
import engine.grammar;
import engine.json;
import engine.regex;
import :error;

/**
 * A request's output grammar, compiled at the boundary: its tools, tool choice, format and reasoning mapped to the
 * grammar's request, and every refusal of the compiler answered by name before the engine sees the request.
 */
namespace engine::openai {

[[nodiscard]] auto custom_input(json::CustomTool const& tool) -> grammar::CustomInput {
	if (!tool.grammar) {
		return grammar::TextInput{};
	}
	switch (tool.grammar->syntax) {
	case json::GrammarSyntax::Lark:
		return grammar::LarkInput{.definition = tool.grammar->definition};
	case json::GrammarSyntax::Regex:
		return grammar::RegexInput{.definition = tool.grammar->definition};
	}
	std::unreachable();
}

[[nodiscard]] auto grammar_tool(json::Tool const& tool) -> grammar::Tool {
	if (auto const function = core::alternative<json::FunctionTool>(tool.input)) {
		auto parameters = function->parameters.transform([](json::JsonText const& schema) {
			return std::string{schema.text()};
		});
		return grammar::FunctionTool{.name = tool.name, .parameters = std::move(parameters)};
	}
	return grammar::CustomTool{.name = tool.name, .input = custom_input(std::get<json::CustomTool>(tool.input))};
}

[[nodiscard]] constexpr auto calls_of(json::ToolChoiceMode mode) -> grammar::Calls {
	switch (mode) {
	case json::ToolChoiceMode::None:
		return grammar::Calls::None;
	case json::ToolChoiceMode::Auto:
		return grammar::Calls::Auto;
	case json::ToolChoiceMode::Required:
		return grammar::Calls::Required;
	}
	std::unreachable();
}

[[nodiscard]] auto choice_of(json::ToolChoice const& choice) -> grammar::ToolChoice {
	if (auto const mode = core::alternative<json::ToolChoiceMode>(choice)) {
		return calls_of(*mode);
	}
	if (auto const function = core::alternative<json::ForcedFunction>(choice)) {
		return grammar::ForcedTool{.name = function->name};
	}
	return grammar::ForcedTool{.name = std::get<json::ForcedCustom>(choice).name};
}

[[nodiscard]] auto format_of(json::TextFormat const& format) -> grammar::ContentFormat {
	if (auto const mode = core::alternative<json::TextFormatMode>(format)) {
		switch (*mode) {
		case json::TextFormatMode::Text:
			return grammar::PlainText{};
		case json::TextFormatMode::JsonObject:
			return grammar::JsonObject{};
		}
		std::unreachable();
	}
	return grammar::JsonSchema{.schema = std::string{std::get<json::JsonSchemaFormat>(format).schema.text()}};
}

/** What a request asks of its reply's shape, as the grammar takes it. Parallel calls are the api's neutral `true`. */
export [[nodiscard]] auto output_request(std::span<json::Tool const> tools, json::ToolChoice const& choice, json::TextFormat const& format,
                                         json::Effort effort) -> grammar::OutputRequest {
	return grammar::OutputRequest{
	    .reasoning = effort == json::Effort::None ? grammar::Reasoning::Direct : grammar::Reasoning::Thinking,
	    .tools = tools | std::views::transform(grammar_tool) | std::ranges::to<std::vector<grammar::Tool>>(),
	    .tool_choice = choice_of(choice),
	    .parallel = grammar::Parallel::Many,
	    .format = format_of(format),
	};
}

/** The digits at the front of `text`, as a number, and what follows them. */
struct Leading {
	std::size_t number;
	std::string_view rest;
};

[[nodiscard]] auto leading_number(std::string_view text) -> std::optional<Leading> {
	auto number = std::size_t{0};
	auto const read = std::from_chars(text.data(), std::to_address(text.end()), number);
	if (std::make_error_code(read.ec)) {
		return std::nullopt;
	}
	return Leading{.number = number, .rest = text.substr(core::narrow<std::size_t>(read.ptr - text.data()))};
}

/**
 * The request member a refusal names, in OpenAI's `param` spelling. The compiler states where it refused as a
 * location in its own request (`tools/<i>/parameters/...`, `format/schema/...`), a lark line, or a tool's name; the
 * member is read off that location's head, and the whole location stays in the message.
 */
[[nodiscard]] auto param_of(grammar::GrammarError const& error) -> std::optional<std::string> {
	// These two name a tool, not a location.
	if (error.refusal == grammar::Refusal::ToolDuplicate) {
		return "tools";
	}
	if (error.refusal == grammar::Refusal::ToolUnknown) {
		return "tool_choice";
	}
	auto const at = std::string_view{error.at};
	if (at.starts_with("tools/")) {
		auto const index = leading_number(at.substr(6));
		if (!index) {
			return "tools";
		}
		if (index->rest.starts_with("/parameters")) {
			return std::format("tools[{}].parameters", index->number);
		}
		return error.refusal == grammar::Refusal::ToolName ? std::format("tools[{}].name", index->number)
		                                                   : std::format("tools[{}].format", index->number);
	}
	if (at.starts_with("format")) {
		return at.starts_with("format/schema") ? "text.format.schema" : "text.format";
	}
	// A lark grammar's faults name a line of it: they come from a custom tool's lark format.
	if (at.starts_with("line")) {
		return "tools";
	}
	return std::nullopt;
}

/** What each refusal means, for the message a person reads. */
[[nodiscard]] constexpr auto meaning(grammar::Refusal refusal) -> std::string_view {
	switch (refusal) {
	case grammar::Refusal::SchemaShape:
		return "a schema is neither an object nor a boolean";
	case grammar::Refusal::SchemaKeyword:
		return "a schema keyword the engine does not implement";
	case grammar::Refusal::SchemaValue:
		return "a schema keyword's value has the wrong shape";
	case grammar::Refusal::SchemaType:
		return "a schema type names no JSON type";
	case grammar::Refusal::SchemaReference:
		return "a $ref that is not '#', '#/$defs/<name>' or '#/definitions/<name>', or names no definition";
	case grammar::Refusal::SchemaFormat:
		return "a string format the engine does not implement";
	case grammar::Refusal::SchemaOneOf:
		return "oneOf whose choices the engine cannot prove disjoint";
	case grammar::Refusal::SchemaMultipleOf:
		return "multipleOf, which the engine does not implement";
	case grammar::Refusal::SchemaBound:
		return "a numeric bound the engine cannot hold exactly";
	case grammar::Refusal::Empty:
		return "the grammar admits no output at all";
	case grammar::Refusal::SchemaOpenArgument:
		return "a tool argument may hold an object with additional properties, whose repeated keys the chat format refuses";
	case grammar::Refusal::SchemaStringUnion:
		return "a tool argument's type list mixes string with a type the chat format would read the text as";
	case grammar::Refusal::SchemaRecursiveArgument:
		return "a tool argument's schema is recursive";
	case grammar::Refusal::ToolParameters:
		return "a function tool's parameters is not an object schema";
	case grammar::Refusal::ToolName:
		return "a tool or parameter name the chat format cannot write";
	case grammar::Refusal::ToolDuplicate:
		return "two tools share a name";
	case grammar::Refusal::ToolUnknown:
		return "the tool choice names no tool the request defines, or requires a call no tool may make";
	case grammar::Refusal::LarkSyntax:
		return "lark text that does not parse";
	case grammar::Refusal::LarkUndefined:
		return "a lark rule or terminal used but never defined, or defined twice";
	case grammar::Refusal::LarkRecursiveTerminal:
		return "a lark terminal defined in terms of itself";
	case grammar::Refusal::LarkNoStart:
		return "a lark grammar without a start rule";
	case grammar::Refusal::LarkPriority:
		return "lark priorities, which the engine does not implement";
	case grammar::Refusal::LarkTemplate:
		return "lark templates, which the engine does not implement";
	case grammar::Refusal::LarkDirective:
		return "a lark directive the engine does not implement";
	case grammar::Refusal::LarkToken:
		return "a lark special-token or grammar reference";
	case grammar::Refusal::LarkOption:
		return "a lark lexeme option the engine does not implement";
	case grammar::Refusal::LarkOperator:
		return "a lark '&' or '~' outside a terminal definition";
	case grammar::Refusal::Regex:
		return "a regular expression refused";
	case grammar::Refusal::SchemaTooLarge:
		return "a schema too long or too deep, or a string bound past what the engine's automata hold";
	case grammar::Refusal::SchemaTooMany:
		return "more properties, enum values or definitions than the engine's bounds";
	case grammar::Refusal::LarkTooLarge:
		return "a lark grammar or regular expression longer than the engine's bounds";
	case grammar::Refusal::AutomatonTooLarge:
		return "a pattern whose automaton grows past the engine's bounds";
	case grammar::Refusal::GrammarTooLarge:
		return "a grammar with more lexemes, rules or symbols than the engine's bounds";
	}
	std::unreachable();
}

/** A compiler refusal as the api answers it: its own code (a regex's fault its own), the location, the member. */
export [[nodiscard]] auto grammar_refusal(grammar::GrammarError const& error) -> ApiError {
	auto const where = error.at.empty() ? std::string{} : std::format(" (at {})", error.at);
	if (error.refusal == grammar::Refusal::Regex && error.regex) {
		return ApiError{
		    .code = error.regex->fault,
		    .message = std::format("a regular expression is refused at byte {}{}", error.regex->offset, where),
		    .param = param_of(error),
		};
	}
	return ApiError{.code = error.refusal, .message = std::format("{}{}", meaning(error.refusal), where), .param = param_of(error)};
}

/** The request's output grammar, compiled, or the compiler's refusal by name. */
export [[nodiscard]] auto compile_output(grammar::OutputRequest const& request) -> core::Result<grammar::Grammar, ApiError> {
	auto compiled = grammar::compile(request);
	if (!compiled) {
		return std::unexpected(grammar_refusal(compiled.error()));
	}
	return *std::move(compiled);
}

}

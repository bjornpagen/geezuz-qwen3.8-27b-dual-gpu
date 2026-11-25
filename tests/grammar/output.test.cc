// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.chat;
import engine.grammar;
import engine.grammar_support;
import engine.json;
import engine.testing;
import engine.tokenizer;

namespace {

namespace grammar = engine::grammar;
namespace support = engine::grammar_support;
namespace json = engine::json;

/** A request's shape, and a name for the case. */
struct Scenario {
	std::string_view name;
	grammar::OutputRequest request;
};

inline constexpr std::string_view weather_schema =
    R"({"type": "object", "properties": {"city": {"type": "string", "minLength": 1}, "days": {"type": "integer", "minimum": 1, "maximum": 14}, "units": {"enum": ["celsius", "fahrenheit"]}}, "required": ["city", "days"], "additionalProperties": false})";

inline constexpr std::string_view mixed_schema = R"({
  "type": "object",
  "properties": {
    "count": {"type": "integer", "minimum": -5, "maximum": 300},
    "ratio": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
    "flag": {"type": "boolean"},
    "note": {"type": ["string", "null"]},
    "choice": {"type": ["boolean", "string"]},
    "level": {"anyOf": [{"type": "boolean"}, {"type": "integer", "minimum": 0, "maximum": 3}]},
    "tags": {"type": "array", "items": {"type": "string", "pattern": "^[a-z]+$"}, "maxItems": 3},
    "when": {"type": "string", "format": "date"},
    "spot": {"type": "object", "properties": {"lat": {"type": "number"}, "name": {"type": "string"}}, "required": ["lat"], "additionalProperties": false},
    "items": {"type": "array", "items": {"$ref": "#/$defs/item"}},
    "extra": {"type": "string"}
  },
  "required": ["count", "flag", "level"],
  "additionalProperties": false,
  "$defs": {"item": {"type": "object", "properties": {"id": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/leaf"}, "maxItems": 2}}, "required": ["id"], "additionalProperties": false}, "leaf": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false}}
})";

inline constexpr std::string_view answer_schema =
    R"({"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "sources": {"type": "array", "items": {"type": "string", "format": "uri-free"}}}, "required": ["answer", "confidence"], "additionalProperties": false})";

inline constexpr std::string_view answer_schema_supported =
    R"({"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "id": {"type": "string", "format": "uuid"}}, "required": ["answer", "confidence"], "additionalProperties": false})";

inline constexpr std::string_view calc_lark = "start: expr\nexpr: term ((\"+\" | \"-\") term)*\nterm: NUMBER | \"(\" expr \")\"\nNUMBER: /[0-9]+/\n%ignore \" \"\n";
inline constexpr std::string_view code_lark = "start: line+\nline: /[a-z =]+/ \"\\n\"\n";

[[nodiscard]] auto tools() -> std::vector<grammar::Tool> {
	return {
	    grammar::FunctionTool{.name = "get_weather", .parameters = std::string{weather_schema}},
	    grammar::FunctionTool{.name = "record", .parameters = std::string{mixed_schema}},
	    grammar::FunctionTool{.name = "ping", .parameters = std::nullopt},
	    grammar::CustomTool{.name = "calc", .input = grammar::LarkInput{.definition = std::string{calc_lark}}},
	    grammar::CustomTool{.name = "code", .input = grammar::LarkInput{.definition = std::string{code_lark}}},
	    grammar::CustomTool{.name = "stamp", .input = grammar::RegexInput{.definition = "[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"}},
	    grammar::CustomTool{.name = "note", .input = grammar::TextInput{}},
	};
}

[[nodiscard]] auto scenarios() -> std::vector<Scenario> {
	return {
	    {.name = "thinking, any tool, free content",
	     .request = {.reasoning = grammar::Reasoning::Thinking, .tools = tools(), .tool_choice = grammar::Calls::Auto, .parallel = grammar::Parallel::Many, .format = grammar::PlainText{}}},
	    {.name = "a call required, typed arguments",
	     .request = {.reasoning = grammar::Reasoning::Direct,
	                 .tools = tools(),
	                 .tool_choice = grammar::AllowedTools{.calls = grammar::Calls::Required, .names = {"record", "get_weather"}},
	                 .parallel = grammar::Parallel::Many,
	                 .format = grammar::PlainText{}}},
	    {.name = "one forced custom lark call",
	     .request = {.reasoning = grammar::Reasoning::Thinking, .tools = tools(), .tool_choice = grammar::ForcedTool{.name = "calc"}, .parallel = grammar::Parallel::One, .format = grammar::PlainText{}}},
	    {.name = "forced custom calls: lark lines, regex, text",
	     .request = {.reasoning = grammar::Reasoning::Direct,
	                 .tools = tools(),
	                 .tool_choice = grammar::AllowedTools{.calls = grammar::Calls::Required, .names = {"code", "stamp", "note"}},
	                 .parallel = grammar::Parallel::Many,
	                 .format = grammar::PlainText{}}},
	    {.name = "no tools may be called, schema content",
	     .request = {.reasoning = grammar::Reasoning::Thinking,
	                 .tools = tools(),
	                 .tool_choice = grammar::Calls::None,
	                 .parallel = grammar::Parallel::Many,
	                 .format = grammar::JsonSchema{.schema = std::string{answer_schema_supported}}}},
	    {.name = "a JSON object, or calls",
	     .request = {.reasoning = grammar::Reasoning::Direct, .tools = tools(), .tool_choice = grammar::Calls::Auto, .parallel = grammar::Parallel::Many, .format = grammar::JsonObject{}}},
	};
}

/** The tools as the chat format's parser reads them. */
[[nodiscard]] auto chat_tools(std::vector<grammar::Tool> const& tools) -> std::vector<json::Tool> {
	auto reader = json::Reader<json::fragment_bounds>::make();
	engine::core::invariant(reader.has_value());
	auto out = std::vector<json::Tool>{};
	for (auto const& tool : tools) {
		if (auto const function = engine::core::alternative<grammar::FunctionTool>(tool)) {
			auto parameters = std::optional<json::JsonText>{};
			if (function->parameters) {
				auto text = json::JsonText::from(*reader, *function->parameters);
				engine::core::invariant(text.has_value());
				parameters = std::move(*text);
			}
			out.push_back(json::Tool{.name = function->name, .description = std::nullopt, .input = json::FunctionTool{.parameters = std::move(parameters)}});
			continue;
		}
		auto const& custom = std::get<grammar::CustomTool>(tool);
		auto input = json::CustomTool{.grammar = std::nullopt};
		if (auto const lark = engine::core::alternative<grammar::LarkInput>(custom.input)) {
			input.grammar = json::Grammar{.syntax = json::GrammarSyntax::Lark, .definition = lark->definition};
		} else if (auto const pattern = engine::core::alternative<grammar::RegexInput>(custom.input)) {
			input.grammar = json::Grammar{.syntax = json::GrammarSyntax::Regex, .definition = pattern->definition};
		}
		out.push_back(json::Tool{.name = custom.name, .description = std::nullopt, .input = std::move(input)});
	}
	return out;
}

[[nodiscard]] auto tool_named(std::vector<grammar::Tool> const& tools, std::string_view name) -> std::optional<grammar::Tool const&> {
	auto const found = std::ranges::find(tools, name, [](grammar::Tool const& tool) { return std::visit([](auto const& held) -> std::string const& { return held.name; }, tool); });
	if (found == tools.end()) {
		return std::nullopt;
	}
	return *found;
}

/** The names a choice lets the model call. */
[[nodiscard]] auto callable(grammar::OutputRequest const& request) -> std::vector<std::string> {
	auto out = std::vector<std::string>{};
	if (auto const calls = engine::core::alternative<grammar::Calls>(request.tool_choice)) {
		if (*calls != grammar::Calls::None) {
			for (auto const& tool : request.tools) {
				out.push_back(std::visit([](auto const& held) { return held.name; }, tool));
			}
		}
	} else if (auto const forced = engine::core::alternative<grammar::ForcedTool>(request.tool_choice)) {
		out.push_back(forced->name);
	} else {
		out = std::get<grammar::AllowedTools>(request.tool_choice).names;
	}
	return out;
}

[[nodiscard]] auto must_call(grammar::OutputRequest const& request) -> bool {
	if (auto const calls = engine::core::alternative<grammar::Calls>(request.tool_choice)) {
		return *calls == grammar::Calls::Required;
	}
	if (engine::core::alternative<grammar::ForcedTool>(request.tool_choice)) {
		return true;
	}
	return std::get<grammar::AllowedTools>(request.tool_choice).calls == grammar::Calls::Required;
}

/** Counts of what the walks of a scenario produced, so a test that reached nothing fails instead of passing idly. */
struct Coverage {
	std::size_t walks = 0;
	std::size_t calls = 0;
	std::size_t custom_calls = 0;
	std::size_t typed_arguments = 0;
};

/**
 * The oracle for a whole reply: the chat format's own parser reads it, and what it reads is what the request
 * asked for. Every call names a tool the choice allows, as many as it allows; a function call's arguments satisfy
 * its schema (the independent validator); a custom call's input is in its grammar's language (judged byte by byte
 * by the standalone grammar); JSON content satisfies its format.
 */
[[nodiscard]] auto reply_is_valid(Scenario const& scenario, std::string const& text, Coverage& coverage) -> bool {
	auto const& request = scenario.request;
	auto const thinking = request.reasoning == grammar::Reasoning::Thinking ? engine::chat::Thinking::Enabled : engine::chat::Thinking::Disabled;
	auto const tools = chat_tools(request.tools);
	auto const reply = engine::chat::parse(text, tools, thinking);
	if (!reply) {
		std::println("  chat could not parse {:?}", text);
		return false;
	}
	auto const names = callable(request);
	auto const fail = [&text](std::string_view why) {
		std::println("  {}: {:?}", why, text);
		return false;
	};
	if (must_call(request) && reply->calls.empty()) {
		return fail("a required call is missing");
	}
	if (request.parallel == grammar::Parallel::One && reply->calls.size() > 1) {
		return fail("more than one call");
	}
	for (auto const& call : reply->calls) {
		if (!std::ranges::contains(names, call.name)) {
			return fail("a call names a tool the choice does not allow");
		}
		auto const arguments = support::parse_json(call.arguments);
		if (!arguments) {
			return fail("arguments are not JSON");
		}
		auto const tool = tool_named(request.tools, call.name);
		if (auto const function = engine::core::alternative<grammar::FunctionTool>(tool.value())) {
			auto const schema = support::parse_json(function->parameters.value_or(R"({"type": "object", "additionalProperties": false})"));
			if (!schema || !support::validates(*schema, *schema, *arguments)) {
				std::println("  arguments {} of {}", call.arguments, call.name);
				return fail("function arguments do not satisfy the schema");
			}
			coverage.typed_arguments += function->parameters ? 1 : 0;
		} else {
			auto const input = arguments->find("input");
			auto const input_text = input ? engine::core::alternative<std::string>(input->held) : std::nullopt;
			if (!input_text || std::get<std::vector<support::JsonMember>>(arguments->held).size() != 1) {
				return fail("a custom call's arguments are not one input string");
			}
			auto const& custom = std::get<grammar::CustomTool>(tool.value());
			auto standalone = std::optional<grammar::Grammar>{};
			if (auto const lark = engine::core::alternative<grammar::LarkInput>(custom.input)) {
				standalone = grammar::compile_lark(lark->definition).value();
			} else if (auto const pattern = engine::core::alternative<grammar::RegexInput>(custom.input)) {
				standalone = grammar::compile_regex(pattern->definition).value();
			}
			if (standalone && support::judge(*standalone, support::byte_vocabulary(), *input_text) != support::Verdict::Accepted) {
				std::println("  input {:?}", *input_text);
				return fail("a custom call's input is not in its grammar's language");
			}
			++coverage.custom_calls;
		}
		++coverage.calls;
	}
	if (auto const format = engine::core::alternative<grammar::JsonSchema>(request.format)) {
		auto const content = support::parse_json(reply->content);
		auto const schema = support::parse_json(format->schema);
		if (!reply->calls.empty() && reply->content.empty()) {
			return true;
		}
		if (!content || !schema || !support::validates(*schema, *schema, *content)) {
			std::println("  content {:?}", reply->content);
			return fail("content does not satisfy its schema");
		}
	}
	if (std::holds_alternative<grammar::JsonObject>(request.format) && (reply->calls.empty() || !reply->content.empty())) {
		auto const content = support::parse_json(reply->content);
		if (!content || !std::holds_alternative<std::vector<support::JsonMember>>(content->held)) {
			return fail("content is not a JSON object");
		}
	}
	++coverage.walks;
	return true;
}

/** Random walks of a scenario all end within the limit, and each reply is valid. */
[[nodiscard]] auto walks_are_valid(Scenario const& scenario, grammar::Vocabulary const& vocabulary, support::Walker const& walker, std::uint64_t seed, std::size_t walks) -> bool {
	auto compiled = grammar::compile(scenario.request);
	if (!compiled) {
		std::println("  refused: {} at {}", std::to_underlying(compiled.error().refusal), compiled.error().at);
		return false;
	}
	auto coverage = Coverage{};
	for (auto index = std::size_t{0}; index < walks; ++index) {
		auto const taken = support::walk(*compiled, vocabulary, walker, (seed * 1000) + index, 100 + (index * 25), 700);
		if (!taken) {
			return false;
		}
		if (!taken->ended) {
			std::println("  did not end: {:?}", taken->text);
			return false;
		}
		if (!reply_is_valid(scenario, taken->text, coverage)) {
			return false;
		}
	}
	std::println("  {}: {} replies, {} calls ({} custom, {} with schemas)", scenario.name, coverage.walks, coverage.calls, coverage.custom_calls, coverage.typed_arguments);
	auto const calls_allowed = !callable(scenario.request).empty();
	return coverage.walks == walks && (!calls_allowed || coverage.calls > 0);
}

[[nodiscard]] auto refused(grammar::OutputRequest const& request, grammar::Refusal refusal) -> bool {
	auto const compiled = grammar::compile(request);
	if (compiled || compiled.error().refusal != refusal) {
		std::println("  expected refusal {}, got {}", std::to_underlying(refusal), compiled ? -1 : std::to_underlying(compiled.error().refusal));
		return false;
	}
	return true;
}

[[nodiscard]] auto with_tool(grammar::Tool tool) -> grammar::OutputRequest {
	return {.reasoning = grammar::Reasoning::Direct, .tools = {std::move(tool)}, .tool_choice = grammar::Calls::Auto, .parallel = grammar::Parallel::Many, .format = grammar::PlainText{}};
}

[[nodiscard]] auto function(std::string_view schema) -> grammar::Tool {
	return grammar::FunctionTool{.name = "f", .parameters = std::string{schema}};
}

}

auto main() -> int {
	auto const tokenizer = engine::tokenizer::load().value();
	auto const vocabulary = grammar::engine_vocabulary(tokenizer);
	auto const walker = support::make_walker(tokenizer);
	auto const all = scenarios();
	auto cases = std::vector<engine::testing::Case>{};
	for (auto const [index, scenario] : std::views::enumerate(all)) {
		cases.push_back({.name = scenario.name, .holds = walks_are_valid(scenario, vocabulary, walker, static_cast<std::uint64_t>(index) + 1, 4)});
	}
	auto const refusals = std::to_array<engine::testing::Case>({
	    {.name = "a tool named twice is refused",
	     .holds = refused({.reasoning = grammar::Reasoning::Direct, .tools = {function("{}"), function("{}")}, .tool_choice = grammar::Calls::Auto, .parallel = grammar::Parallel::Many, .format = grammar::PlainText{}},
	                      grammar::Refusal::ToolDuplicate)},
	    {.name = "a forced tool that does not exist is refused",
	     .holds = refused({.reasoning = grammar::Reasoning::Direct, .tools = {function("{}")}, .tool_choice = grammar::ForcedTool{.name = "g"}, .parallel = grammar::Parallel::Many, .format = grammar::PlainText{}},
	                      grammar::Refusal::ToolUnknown)},
	    {.name = "a required call with no tools is refused",
	     .holds = refused({.reasoning = grammar::Reasoning::Direct, .tools = {}, .tool_choice = grammar::Calls::Required, .parallel = grammar::Parallel::Many, .format = grammar::PlainText{}},
	                      grammar::Refusal::ToolUnknown)},
	    {.name = "a tool name the format cannot write is refused", .holds = refused(with_tool(grammar::FunctionTool{.name = "a>b", .parameters = std::nullopt}), grammar::Refusal::ToolName)},
	    {.name = "a parameter name the format cannot write is refused",
	     .holds = refused(with_tool(function(R"({"type": "object", "properties": {"a\nb": {"type": "string"}}})")), grammar::Refusal::ToolName)},
	    {.name = "parameters that are not an object schema are refused", .holds = refused(with_tool(function(R"({"type": "array"})")), grammar::Refusal::ToolParameters)},
	    {.name = "a string union the format would read as a number is refused",
	     .holds = refused(with_tool(function(R"({"type": "object", "properties": {"a": {"type": ["string", "integer"]}}})")), grammar::Refusal::SchemaStringUnion)},
	    {.name = "an open object inside an argument is refused",
	     .holds = refused(with_tool(function(R"({"type": "object", "properties": {"a": {"type": "object"}}})")), grammar::Refusal::SchemaOpenArgument)},
	    {.name = "a recursive argument schema is refused",
	     .holds = refused(with_tool(function(R"({"type": "object", "properties": {"t": {"type": "object", "properties": {"c": {"type": "array", "items": {"$ref": "#/$defs/n"}}}, "additionalProperties": false}}, "$defs": {"n": {"type": "object", "properties": {"c": {"type": "array", "items": {"$ref": "#/$defs/n"}}}, "additionalProperties": false}}})")), grammar::Refusal::SchemaRecursiveArgument)},
	    {.name = "an argument no cast can write is refused",
	     .holds = refused(with_tool(function(R"({"type": "object", "properties": {"a": {"enum": [1, 2]}}})")), grammar::Refusal::Empty)},
	    {.name = "an unsupported format is refused",
	     .holds = refused({.reasoning = grammar::Reasoning::Direct, .tools = {}, .tool_choice = grammar::Calls::None, .parallel = grammar::Parallel::Many, .format = grammar::JsonSchema{.schema = std::string{answer_schema}}},
	                      grammar::Refusal::SchemaFormat)},
	    {.name = "a custom tool's unsupported lark is refused",
	     .holds = refused(with_tool(grammar::CustomTool{.name = "c", .input = grammar::LarkInput{.definition = "start: A\nA.2: \"a\"\n"}}), grammar::Refusal::LarkPriority)},
	    {.name = "a custom tool's unsupported regex is refused",
	     .holds = refused(with_tool(grammar::CustomTool{.name = "c", .input = grammar::RegexInput{.definition = "a(?=b)"}}), grammar::Refusal::Regex)},
	});
	std::ranges::copy(refusals, std::back_inserter(cases));
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:output;

import std;
import engine.core;
import engine.regex;
import :error;
import :builder;
import :compiled;
import :document;
import :strings;
import :schema;
import :arguments;
import :lark;

namespace engine::grammar {

/** A function tool: its name and its parameters' JSON schema text, if it has parameters. */
export struct FunctionTool {
	std::string name;
	std::optional<std::string> parameters;

	[[nodiscard]] auto operator==(FunctionTool const&) const -> bool = default;
};

/** A custom tool's input is free text. */
export struct TextInput {
	[[nodiscard]] auto operator==(TextInput const&) const -> bool = default;
};

/** A custom tool's input is the language of a lark grammar (the llguidance dialect OpenAI's custom tools use). */
export struct LarkInput {
	std::string definition;

	[[nodiscard]] auto operator==(LarkInput const&) const -> bool = default;
};

/** A custom tool's input is the language of a regular expression (Rust `regex` syntax), matched whole. */
export struct RegexInput {
	std::string definition;

	[[nodiscard]] auto operator==(RegexInput const&) const -> bool = default;
};

export using CustomInput = std::variant<TextInput, LarkInput, RegexInput>;

/**
 * A custom tool. It reaches the model as a template tool with one string parameter, `custom_parameter`, whose text
 * is the input; the call's `input` is that parameter's value.
 */
export struct CustomTool {
	std::string name;
	CustomInput input;

	[[nodiscard]] auto operator==(CustomTool const&) const -> bool = default;
};

/** The parameter a custom tool's input is written in. */
export inline constexpr std::string_view custom_parameter = "input";

export using Tool = std::variant<FunctionTool, CustomTool>;

/** Whether the model may, must or must not call a tool. */
export enum class Calls : std::uint8_t {
	None,
	Auto,
	Required,
};

/** Calls of one named tool only, at least one. */
export struct ForcedTool {
	std::string name;

	[[nodiscard]] auto operator==(ForcedTool const&) const -> bool = default;
};

/** Calls of the named tools only, optional (`Calls::Auto`) or at least one (`Calls::Required`). */
export struct AllowedTools {
	Calls calls;
	std::vector<std::string> names;

	[[nodiscard]] auto operator==(AllowedTools const&) const -> bool = default;
};

export using ToolChoice = std::variant<Calls, ForcedTool, AllowedTools>;

/** How many calls one reply may make. */
export enum class Parallel : std::uint8_t {
	Many,
	One,
};

/** The reply's content is free text. */
export struct PlainText {
	[[nodiscard]] auto operator==(PlainText const&) const -> bool = default;
};

/** The reply's content is one JSON object. */
export struct JsonObject {
	[[nodiscard]] auto operator==(JsonObject const&) const -> bool = default;
};

/** The reply's content is one JSON value its schema admits. */
export struct JsonSchema {
	std::string schema;

	[[nodiscard]] auto operator==(JsonSchema const&) const -> bool = default;
};

export using ContentFormat = std::variant<PlainText, JsonObject, JsonSchema>;

/** Whether the generation prompt opened a reasoning region (`<think>\n`) or closed an empty one. */
export enum class Reasoning : std::uint8_t {
	Thinking,
	Direct,
};

/** What a request asks of the reply's shape: the input the output grammar is composed from. */
export struct OutputRequest {
	Reasoning reasoning;
	std::vector<Tool> tools;
	ToolChoice tool_choice;
	Parallel parallel;
	ContentFormat format;

	[[nodiscard]] auto operator==(OutputRequest const&) const -> bool = default;
};

/** What ends a reasoning region, as the template writes it. */
inline constexpr std::string_view think_close = "</think>";
/** What opens a tool call. */
inline constexpr std::string_view call_open = "<tool_call>";
/** The words the chat format reads as markers in content. */
inline constexpr auto content_markers = std::to_array<std::string_view>({"<think>", "<|im_end|>", "<|endoftext|>", "<tool_call>"});

[[nodiscard]] auto tool_name(Tool const& tool) -> std::string const& {
	return std::visit([](auto const& held) -> std::string const& { return held.name; }, tool);
}

/** The tools a choice lets the model call, by index, and whether it must call one. */
struct Callable {
	std::vector<std::size_t> tools;
	Calls calls;
};

[[nodiscard]] auto callable(OutputRequest const& request) -> core::Result<Callable, GrammarError> {
	auto const index_of = [&request](std::string const& name) -> std::optional<std::size_t> {
		auto const found = std::ranges::find(request.tools, name, tool_name);
		if (found == request.tools.end()) {
			return std::nullopt;
		}
		return core::narrow<std::size_t>(found - request.tools.begin());
	};
	auto out = Callable{.tools = {}, .calls = Calls::None};
	if (auto const calls = core::alternative<Calls>(request.tool_choice)) {
		out.calls = *calls;
		if (*calls != Calls::None) {
			for (auto index = std::size_t{0}; index < request.tools.size(); ++index) {
				out.tools.push_back(index);
			}
		}
	} else if (auto const forced = core::alternative<ForcedTool>(request.tool_choice)) {
		auto const index = index_of(forced->name);
		if (!index) {
			return refuse(Refusal::ToolUnknown, forced->name);
		}
		out = Callable{.tools = {*index}, .calls = Calls::Required};
	} else {
		auto const& allowed = std::get<AllowedTools>(request.tool_choice);
		out.calls = allowed.calls;
		for (auto const& name : allowed.names) {
			auto const index = index_of(name);
			if (!index) {
				return refuse(Refusal::ToolUnknown, name);
			}
			out.tools.push_back(*index);
		}
	}
	if (out.calls == Calls::Required && out.tools.empty()) {
		return refuse(Refusal::ToolUnknown, "tool_choice");
	}
	if (out.calls == Calls::None) {
		out.tools.clear();
	}
	return out;
}

/** The text from a call's `<tool_call>` to its `</tool_call>`, for one tool. */
[[nodiscard]] auto call(Builder& builder, Tool const& tool, std::size_t index, Usage& usage) -> core::Result<Symbol, GrammarError> {
	auto const& name = tool_name(tool);
	auto const at = std::format("tools/{}", index);
	if (!writable_name(name)) {
		return refuse(Refusal::ToolName, at);
	}
	auto& store = builder.store();
	auto const open = builder.literal(std::format("\n<function={}>\n", name));
	auto const close = builder.literal("</function>\n</tool_call>");
	auto const out = builder.nonterminal();
	if (auto const function = core::alternative<FunctionTool>(tool)) {
		if (!function->parameters) {
			builder.rule(out, {open, close});
			return out;
		}
		auto document = read_document(*function->parameters, at + "/parameters");
		if (!document) {
			return std::unexpected(document.error());
		}
		auto body = parameters(builder, *document, at + "/parameters", usage);
		if (!body) {
			return body;
		}
		builder.rule(out, {open, *body, close});
		return out;
	}
	auto const& custom = std::get<CustomTool>(tool);
	auto const parameter = builder.literal(std::format("<parameter={}>\n", custom_parameter));
	auto const forbidden = any_literal(store, call_markers);
	auto const closed_text = [&](regex::Expr text) {
		return builder.lexeme(store.concat(store.both(text, avoiding(store, forbidden)), store.literal(parameter_close)), Mode::Lazy, std::nullopt);
	};
	if (std::holds_alternative<TextInput>(custom.input)) {
		builder.rule(out, {open, parameter, closed_text(regex::any_text(store)), close});
		return out;
	}
	if (auto const pattern = core::alternative<RegexInput>(custom.input)) {
		if (pattern->definition.size() > builder.bounds().regex_bytes) {
			return refuse(Refusal::LarkTooLarge, at);
		}
		auto parsed = regex::parse(store, pattern->definition, regex::Matching::Whole);
		if (!parsed) {
			return refuse_regex(parsed.error(), at);
		}
		builder.rule(out, {open, parameter, closed_text(*parsed), close});
		return out;
	}
	auto const region = builder.region(forbidden, parameter_close);
	auto const setting = lark::Setting{.region = region, .json = Place{.markers = call_markers, .readback = Readback::Client, .top_numbers_excluded = regex::nothing}};
	auto input = lark::compile(builder, std::get<LarkInput>(custom.input).definition, setting, usage);
	if (!input) {
		return std::unexpected(input.error());
	}
	builder.rule(out, {open, parameter, *input, builder.region_close(region), close});
	return out;
}

/**
 * The output grammar of a request, in the chat format's syntax. With reasoning, the reply opens with free text
 * through `</think>` (free of `<tool_call>`, which would end the region unannounced); then content, free text or
 * the JSON its format names, free of the words the format reads as markers; then, as the tool choice allows, calls,
 * each `<tool_call>\n<function=NAME>\n` then its parameters then `</function>\n</tool_call>`, with up to four
 * whitespace characters between calls and after the last. The end token is allowed exactly where this grammar
 * accepts.
 */
export [[nodiscard]] auto compile(OutputRequest const& request) -> core::Result<Grammar, GrammarError> {
	auto builder = Builder{request_bounds};
	auto& store = builder.store();
	auto usage = Usage{};
	for (auto const& tool : request.tools) {
		if (std::ranges::count(request.tools, tool_name(tool), tool_name) > 1) {
			return refuse(Refusal::ToolDuplicate, tool_name(tool));
		}
	}
	auto const allowed = callable(request);
	if (!allowed) {
		return std::unexpected(allowed.error());
	}
	auto const content_forbidden = any_literal(store, content_markers);
	auto const content_text = store.both(regex::any_text(store), avoiding(store, content_forbidden));
	auto const space = [&store](std::uint32_t low) { return store.repeat(regex_of(store, R"([ \t\r\n])"), low, 4); };
	// Calls: one of the callable tools' bodies after each `<tool_call>`.
	auto const any_call = builder.nonterminal();
	for (auto const index : allowed->tools) {
		auto body = call(builder, request.tools[index], index, usage);
		if (!body) {
			return std::unexpected(body.error());
		}
		builder.rule(any_call, {*body});
	}
	auto const between = builder.lexeme(store.concat(space(0), store.literal(call_open)), Mode::Greedy, std::nullopt);
	auto const trailing = builder.optional(builder.lexeme(space(1), Mode::Greedy, std::nullopt));
	// After the first call's `<tool_call>`: its body, more calls when allowed, trailing whitespace.
	auto const calls = builder.nonterminal();
	if (request.parallel == Parallel::Many) {
		auto const another = builder.nonterminal();
		builder.rule(another, {between, any_call});
		builder.rule(calls, {any_call, builder.star(another), trailing});
	} else {
		builder.rule(calls, {any_call, trailing});
	}
	auto const may_call = !allowed->tools.empty();
	auto const must_call = may_call && allowed->calls == Calls::Required;
	auto const body = builder.nonterminal();
	if (std::holds_alternative<PlainText>(request.format)) {
		if (!must_call) {
			builder.rule(body, {builder.lexeme(content_text, Mode::Greedy, std::nullopt)});
		}
		if (may_call) {
			builder.rule(body, {builder.lexeme(store.concat(content_text, store.literal(call_open)), Mode::Lazy, std::nullopt), calls});
		}
	} else {
		auto document = std::optional<Value>{};
		if (auto const schema = core::alternative<JsonSchema>(request.format)) {
			auto read = read_document(schema->schema, "format/schema");
			if (!read) {
				return std::unexpected(read.error());
			}
			document = std::move(*read);
		}
		auto const root = document ? *document : Value{.held = true};
		auto compiler = SchemaCompiler{builder, root, "",
		                               Place{.markers = content_markers, .readback = Readback::Client, .top_numbers_excluded = regex::nothing}, usage};
		auto json = document ? compiler.value(*document, "format/schema", Types::all(), 0) : compiler.any(Types::only(Type::Object), "format");
		if (!json) {
			return std::unexpected(json.error());
		}
		auto const leading = builder.optional(builder.lexeme(space(1), Mode::Greedy, std::nullopt));
		if (!must_call) {
			builder.rule(body, {leading, *json, trailing});
		}
		if (may_call) {
			builder.rule(body, {leading, *json, between, calls});
			builder.rule(body, {between, calls});
		}
	}
	auto const start = builder.nonterminal();
	if (request.reasoning == Reasoning::Thinking) {
		auto const reasoning = store.both(regex::any_text(store), avoiding(store, store.literal(call_open)));
		builder.rule(start, {builder.lexeme(store.concat(reasoning, store.literal(think_close)), Mode::Lazy, std::nullopt), body});
	} else {
		builder.rule(start, {body});
	}
	return Grammar::compile(std::move(builder), start);
}

/** A standalone lark grammar's language: for tools and tests that constrain a whole text by one. */
export [[nodiscard]] auto compile_lark(std::string_view text) -> core::Result<Grammar, GrammarError> {
	auto builder = Builder{request_bounds};
	auto usage = Usage{};
	auto const setting = lark::Setting{.region = std::nullopt, .json = Place{.markers = {}, .readback = Readback::Client, .top_numbers_excluded = regex::nothing}};
	auto start = lark::compile(builder, text, setting, usage);
	if (!start) {
		return std::unexpected(start.error());
	}
	return Grammar::compile(std::move(builder), *start);
}

/** A standalone regular expression's language, matched whole. */
export [[nodiscard]] auto compile_regex(std::string_view text) -> core::Result<Grammar, GrammarError> {
	auto builder = Builder{request_bounds};
	if (text.size() > builder.bounds().regex_bytes) {
		return refuse(Refusal::LarkTooLarge, "");
	}
	auto parsed = regex::parse(builder.store(), text, regex::Matching::Whole);
	if (!parsed) {
		return refuse_regex(parsed.error(), "");
	}
	auto const start = builder.nonterminal();
	builder.rule(start, {builder.lexeme(*parsed, Mode::Greedy, std::nullopt)});
	return Grammar::compile(std::move(builder), start);
}

/** A standalone JSON schema's language: the JSON texts of the values it admits. */
export [[nodiscard]] auto compile_schema(std::string_view text) -> core::Result<Grammar, GrammarError> {
	auto builder = Builder{request_bounds};
	auto usage = Usage{};
	auto document = read_document(text, "");
	if (!document) {
		return std::unexpected(document.error());
	}
	auto compiler = SchemaCompiler{builder, *document, "", Place{.markers = {}, .readback = Readback::Client, .top_numbers_excluded = regex::nothing}, usage};
	auto json = compiler.value(*document, "", Types::all(), 0);
	if (!json) {
		return std::unexpected(json.error());
	}
	auto const start = builder.nonterminal();
	builder.rule(start, {*json});
	return Grammar::compile(std::move(builder), start);
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:request;

import std;
import engine.core;
import :text;

/**
 * The openai Responses create request as the engine accepts it: the shapes a request decodes into. What a value
 * means beyond its shape (a temperature's range, whether a tool choice names a defined tool, whether a continuation
 * matches its stored response) is the endpoint's to decide; every state these types hold is one the wire can express
 * and the engine can serve.
 */
namespace engine::json {

/** A message's role. `developer` and `system` are distinct on the wire and both render as the template's system. */
export enum class Role : std::uint8_t {
	User,
	Assistant,
	System,
	Developer,
};

/** A message: its text is its content string, or the concatenation of its text parts. */
export struct MessageItem {
	Role role;
	std::string text;

	[[nodiscard]] auto operator==(MessageItem const&) const -> bool = default;
};

/** A past function call; its arguments are one JSON object, held as the client wrote it. */
export struct FunctionCallItem {
	std::string call_id;
	std::string name;
	JsonText arguments;

	[[nodiscard]] auto operator==(FunctionCallItem const&) const -> bool = default;
};

/** A function call's result: its output string, or the concatenation of its text parts. */
export struct FunctionOutputItem {
	std::string call_id;
	std::string output;

	[[nodiscard]] auto operator==(FunctionOutputItem const&) const -> bool = default;
};

/** A past custom tool call: free text its tool's format constrained. */
export struct CustomCallItem {
	std::string call_id;
	std::string name;
	std::string input;

	[[nodiscard]] auto operator==(CustomCallItem const&) const -> bool = default;
};

export struct CustomOutputItem {
	std::string call_id;
	std::string output;

	[[nodiscard]] auto operator==(CustomOutputItem const&) const -> bool = default;
};

/** A past turn's reasoning: the concatenation of its `reasoning_text` parts. */
export struct ReasoningItem {
	std::string text;

	[[nodiscard]] auto operator==(ReasoningItem const&) const -> bool = default;
};

export using InputItem = std::variant<MessageItem, FunctionCallItem, FunctionOutputItem, CustomCallItem, CustomOutputItem, ReasoningItem>;

/** A function tool's input: an arguments object matching its parameter schema, or an empty object without one. */
export struct FunctionTool {
	std::optional<JsonText> parameters;

	[[nodiscard]] auto operator==(FunctionTool const&) const -> bool = default;
};

/** The grammar dialects a custom tool's input may be given in: the lark dialect openai documents, and a regex. */
export enum class GrammarSyntax : std::uint8_t {
	Lark,
	Regex,
};

/** A grammar as the client wrote it; compiling it is the grammar's own concern. */
export struct Grammar {
	GrammarSyntax syntax;
	std::string definition;

	[[nodiscard]] auto operator==(Grammar const&) const -> bool = default;
};

/** A custom tool's input: free text, or text its grammar accepts. */
export struct CustomTool {
	std::optional<Grammar> grammar;

	[[nodiscard]] auto operator==(CustomTool const&) const -> bool = default;
};

export using ToolInput = std::variant<FunctionTool, CustomTool>;

/** A tool. Every function tool is strict: `strict` is accepted at both values and changes nothing. */
export struct Tool {
	std::string name;
	std::optional<std::string> description;
	ToolInput input;

	[[nodiscard]] auto operator==(Tool const&) const -> bool = default;
};

export enum class ToolChoiceMode : std::uint8_t {
	None,
	Auto,
	Required,
};

/** `tool_choice` naming one function tool the model must call. */
export struct ForcedFunction {
	std::string name;

	[[nodiscard]] auto operator==(ForcedFunction const&) const -> bool = default;
};

/** `tool_choice` naming one custom tool the model must call. */
export struct ForcedCustom {
	std::string name;

	[[nodiscard]] auto operator==(ForcedCustom const&) const -> bool = default;
};

/** Absent, it is `auto`, as the sdk defaults it. */
export using ToolChoice = std::variant<ToolChoiceMode, ForcedFunction, ForcedCustom>;

/** `text.format` without a schema: free text, or any one JSON object. */
export enum class TextFormatMode : std::uint8_t {
	Text,
	JsonObject,
};

/** `text.format` of type `json_schema`: the output text is JSON matching `schema`. Every schema is enforced. */
export struct JsonSchemaFormat {
	std::string name;
	std::optional<std::string> description;
	JsonText schema;

	[[nodiscard]] auto operator==(JsonSchemaFormat const&) const -> bool = default;
};

/** Absent, it is free text. */
export using TextFormat = std::variant<TextFormatMode, JsonSchemaFormat>;

/**
 * `reasoning.effort` as the model's chat template has it: `none` turns thinking off, the rest are the template's
 * three efforts. Absent, it is `xhigh`, as the template defaults a thinking turn.
 */
export enum class Effort : std::uint8_t {
	None,
	Low,
	Medium,
	XHigh,
};

export enum class Delivery : std::uint8_t {
	Whole,
	Stream,
};

/** `store`: absent, the response is stored, as the sdk defaults it. */
export enum class Storage : std::uint8_t {
	Stored,
	Unstored,
};

/** The sdk's `metadata` bounds: pairs, and the bytes of each key and value. */
export inline constexpr std::size_t max_metadata_pairs = 16;
export inline constexpr std::size_t max_metadata_key_bytes = 64;
export inline constexpr std::size_t max_metadata_value_bytes = 512;

export struct MetadataPair {
	std::string key;
	std::string value;

	[[nodiscard]] auto operator==(MetadataPair const&) const -> bool = default;
};

/** `POST /v1/responses`. */
export struct CreateRequest {
	/** Absent, as the sdk allows; present, it must name the served model, which the endpoint checks. */
	std::optional<std::string> model;
	/** A string input decodes as one user message. */
	std::vector<InputItem> input;
	std::optional<std::string> instructions;
	std::vector<Tool> tools;
	ToolChoice tool_choice;
	TextFormat format;
	Effort effort;
	std::optional<std::int64_t> max_output_tokens;
	std::optional<double> temperature;
	std::optional<double> top_p;
	Delivery delivery;
	Storage storage;
	std::optional<std::string> previous_response_id;
	std::vector<MetadataPair> metadata;

	[[nodiscard]] auto operator==(CreateRequest const&) const -> bool = default;
};

}

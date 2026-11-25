// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:response;

import std;
import engine.core;
import :text;
import :write;
import :request;

/**
 * The openai Responses objects as the engine writes them: one whole `response`, the events of a streamed one, the
 * input items of a stored one, and a deletion. Each is appended to a caller's bounded buffer whole or not at all,
 * or encoded into a string of exactly its size; the http layer frames events.
 */
namespace engine::json {

export enum class ItemStatus : std::uint8_t {
	InProgress,
	Completed,
	Incomplete,
};

[[nodiscard]] constexpr auto spelling(ItemStatus status) -> std::string_view {
	switch (status) {
	case ItemStatus::InProgress:
		return "in_progress";
	case ItemStatus::Completed:
		return "completed";
	case ItemStatus::Incomplete:
		return "incomplete";
	}
	std::unreachable();
}

export struct ReasoningOutput {
	std::string id;
	std::string text;
	ItemStatus status;

	[[nodiscard]] auto operator==(ReasoningOutput const&) const -> bool = default;
};

export struct MessageOutput {
	std::string id;
	std::string text;
	ItemStatus status;

	[[nodiscard]] auto operator==(MessageOutput const&) const -> bool = default;
};

export struct FunctionCallOutput {
	std::string id;
	std::string call_id;
	std::string name;
	std::string arguments;
	ItemStatus status;

	[[nodiscard]] auto operator==(FunctionCallOutput const&) const -> bool = default;
};

/** The sdk's custom tool call has no status. */
export struct CustomCallOutput {
	std::string id;
	std::string call_id;
	std::string name;
	std::string input;

	[[nodiscard]] auto operator==(CustomCallOutput const&) const -> bool = default;
};

export using OutputItem = std::variant<ReasoningOutput, MessageOutput, FunctionCallOutput, CustomCallOutput>;

/** Token counts; the total is written, never held. Cached tokens are part of the input's, reasoning of the output's. */
export struct Usage {
	std::uint64_t input_tokens;
	std::uint64_t cached_tokens;
	std::uint64_t output_tokens;
	std::uint64_t reasoning_tokens;

	[[nodiscard]] auto operator==(Usage const&) const -> bool = default;
};

/** `in_progress` only while a stream opens; a finished response ran to its end or out of budget. */
export enum class ResponseStatus : std::uint8_t {
	InProgress,
	Completed,
	Incomplete,
};

/** A `response`, and the request members it echoes. */
export struct Response {
	std::string id;
	/** Unix seconds. */
	std::int64_t created_at;
	std::optional<std::int64_t> completed_at;
	ResponseStatus status;
	std::optional<std::string> instructions;
	std::vector<MetadataPair> metadata;
	std::string model;
	std::vector<OutputItem> output;
	std::vector<Tool> tools;
	ToolChoice tool_choice;
	double temperature;
	double top_p;
	TextFormat format;
	Effort effort;
	std::optional<std::int64_t> max_output_tokens;
	std::optional<std::string> previous_response_id;
	Storage storage;
	/** Absent until the response finishes. */
	std::optional<Usage> usage;

	[[nodiscard]] auto operator==(Response const&) const -> bool = default;
};

/** An input item as a stored response keeps it: with the id the input items list shows. */
export struct StoredInput {
	std::string id;
	InputItem item;

	[[nodiscard]] auto operator==(StoredInput const&) const -> bool = default;
};

[[nodiscard]] constexpr auto spelling(Effort effort) -> std::string_view {
	switch (effort) {
	case Effort::None:
		return "none";
	case Effort::Low:
		return "low";
	case Effort::Medium:
		return "medium";
	case Effort::XHigh:
		return "xhigh";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto spelling(ResponseStatus status) -> std::string_view {
	switch (status) {
	case ResponseStatus::InProgress:
		return "in_progress";
	case ResponseStatus::Completed:
		return "completed";
	case ResponseStatus::Incomplete:
		return "incomplete";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto spelling(Role role) -> std::string_view {
	switch (role) {
	case Role::User:
		return "user";
	case Role::Assistant:
		return "assistant";
	case Role::System:
		return "system";
	case Role::Developer:
		return "developer";
	}
	std::unreachable();
}

/** How much of an item a write shows: an item just added has no content yet, as the sdk's `added` events show it. */
enum class Shown : std::uint8_t {
	Opening,
	Whole,
};

template<Sink O>
struct WriteItem {
	Members<O>& item;
	Shown shown;

	auto status(ItemStatus status) const -> void {
		item.string("status", shown == Shown::Opening ? spelling(ItemStatus::InProgress) : spelling(status));
	}

	auto operator()(ReasoningOutput const& reasoning) const -> void {
		item.string("id", reasoning.id);
		item.string("type", "reasoning");
		item.array("summary", [](Elements<O>&) {});
		item.array("content", [&](Elements<O>& parts) {
			if (shown == Shown::Whole) {
				parts.object([&](Members<O>& part) {
					part.string("type", "reasoning_text");
					part.string("text", reasoning.text);
				});
			}
		});
		status(reasoning.status);
	}

	auto operator()(MessageOutput const& message) const -> void {
		item.string("id", message.id);
		item.string("type", "message");
		item.string("role", "assistant");
		status(message.status);
		item.array("content", [&](Elements<O>& parts) {
			if (shown == Shown::Whole) {
				parts.object([&](Members<O>& part) {
					part.string("type", "output_text");
					part.string("text", message.text);
					part.array("annotations", [](Elements<O>&) {});
					part.array("logprobs", [](Elements<O>&) {});
				});
			}
		});
	}

	auto operator()(FunctionCallOutput const& call) const -> void {
		item.string("id", call.id);
		item.string("type", "function_call");
		item.string("call_id", call.call_id);
		item.string("name", call.name);
		item.string("arguments", shown == Shown::Opening ? std::string_view{} : std::string_view{call.arguments});
		status(call.status);
	}

	auto operator()(CustomCallOutput const& call) const -> void {
		item.string("id", call.id);
		item.string("type", "custom_tool_call");
		item.string("call_id", call.call_id);
		item.string("name", call.name);
		item.string("input", shown == Shown::Opening ? std::string_view{} : std::string_view{call.input});
	}
};

template<Sink O>
auto put_tool(Elements<O>& tools, Tool const& tool) -> void {
	tools.object([&tool](Members<O>& written) {
		if (auto const function = core::alternative<FunctionTool>(tool.input)) {
			written.string("type", "function");
			written.string("name", tool.name);
			if (tool.description) {
				written.string("description", *tool.description);
			} else {
				written.null("description");
			}
			if (function->parameters) {
				written.json("parameters", *function->parameters);
			} else {
				written.null("parameters");
			}
			// Every function tool is enforced, so the effective strictness is always true.
			written.boolean("strict", true);
			return;
		}
		auto const& custom = std::get<CustomTool>(tool.input);
		written.string("type", "custom");
		written.string("name", tool.name);
		if (tool.description) {
			written.string("description", *tool.description);
		}
		written.object("format", [&custom](Members<O>& format) {
			if (!custom.grammar) {
				format.string("type", "text");
				return;
			}
			format.string("type", "grammar");
			format.string("syntax", custom.grammar->syntax == GrammarSyntax::Lark ? "lark" : "regex");
			format.string("definition", custom.grammar->definition);
		});
	});
}

template<Sink O>
auto put_tool_choice(Members<O>& members, ToolChoice const& choice) -> void {
	if (auto const mode = core::alternative<ToolChoiceMode>(choice)) {
		auto constexpr modes = std::array<std::string_view, 3>{"none", "auto", "required"};
		members.string("tool_choice", modes[std::to_underlying(*mode)]);
		return;
	}
	members.object("tool_choice", [&choice](Members<O>& forced) {
		if (auto const function = core::alternative<ForcedFunction>(choice)) {
			forced.string("type", "function");
			forced.string("name", function->name);
		} else {
			forced.string("type", "custom");
			forced.string("name", std::get<ForcedCustom>(choice).name);
		}
	});
}

template<Sink O>
auto put_format(Members<O>& text, TextFormat const& format) -> void {
	text.object("format", [&format](Members<O>& written) {
		if (auto const mode = core::alternative<TextFormatMode>(format)) {
			written.string("type", *mode == TextFormatMode::Text ? "text" : "json_object");
			return;
		}
		auto const& schema = std::get<JsonSchemaFormat>(format);
		written.string("type", "json_schema");
		written.string("name", schema.name);
		if (schema.description) {
			written.string("description", *schema.description);
		}
		written.json("schema", schema.schema);
		written.boolean("strict", true);
	});
	text.string("verbosity", "medium");
}

template<Sink O>
auto put_usage(Members<O>& members, Usage const& usage) -> void {
	core::invariant(usage.cached_tokens <= usage.input_tokens && usage.reasoning_tokens <= usage.output_tokens);
	core::invariant(usage.output_tokens <= std::numeric_limits<std::uint64_t>::max() - usage.input_tokens);
	members.object("usage", [&usage](Members<O>& inner) {
		inner.integer("input_tokens", usage.input_tokens);
		inner.object("input_tokens_details", [&usage](Members<O>& details) {
			details.integer("cached_tokens", usage.cached_tokens);
			// There are no explicit cache breakpoints, so nothing is written to a cache on the client's request.
			details.integer("cache_write_tokens", 0);
		});
		inner.integer("output_tokens", usage.output_tokens);
		inner.object("output_tokens_details", [&usage](Members<O>& details) { details.integer("reasoning_tokens", usage.reasoning_tokens); });
		inner.integer("total_tokens", usage.input_tokens + usage.output_tokens);
	});
}

template<Sink O>
auto put_response(Members<O>& members, Response const& response) -> void {
	members.string("id", response.id);
	members.string("object", "response");
	members.integer("created_at", response.created_at);
	if (response.completed_at) {
		members.integer("completed_at", *response.completed_at);
	} else {
		members.null("completed_at");
	}
	members.string("status", spelling(response.status));
	members.null("error");
	if (response.status == ResponseStatus::Incomplete) {
		members.object("incomplete_details", [](Members<O>& details) { details.string("reason", "max_output_tokens"); });
	} else {
		members.null("incomplete_details");
	}
	if (response.instructions) {
		members.string("instructions", *response.instructions);
	} else {
		members.null("instructions");
	}
	members.object("metadata", [&response](Members<O>& pairs) {
		for (auto const& pair : response.metadata) {
			pairs.string(pair.key, pair.value);
		}
	});
	members.string("model", response.model);
	members.array("output", [&response](Elements<O>& items) {
		for (auto const& item : response.output) {
			items.object([&item](Members<O>& written) { std::visit(WriteItem<O>{.item = written, .shown = Shown::Whole}, item); });
		}
	});
	members.boolean("parallel_tool_calls", true);
	members.real("temperature", response.temperature);
	put_tool_choice(members, response.tool_choice);
	members.array("tools", [&response](Elements<O>& tools) {
		for (auto const& tool : response.tools) {
			put_tool(tools, tool);
		}
	});
	members.real("top_p", response.top_p);
	members.boolean("background", false);
	if (response.max_output_tokens) {
		members.integer("max_output_tokens", *response.max_output_tokens);
	} else {
		members.null("max_output_tokens");
	}
	if (response.previous_response_id) {
		members.string("previous_response_id", *response.previous_response_id);
	} else {
		members.null("previous_response_id");
	}
	members.object("reasoning", [&response](Members<O>& reasoning) {
		reasoning.string("effort", spelling(response.effort));
		reasoning.null("summary");
	});
	members.string("service_tier", "default");
	members.boolean("store", response.storage == Storage::Stored);
	members.object("text", [&response](Members<O>& text) { put_format(text, response.format); });
	members.integer("top_logprobs", 0);
	members.string("truncation", "disabled");
	if (response.usage) {
		put_usage(members, *response.usage);
	} else {
		members.null("usage");
	}
}

/** Appends a whole response's JSON. */
export [[nodiscard]] auto append(Buffer& buffer, Response const& response) -> core::Result<void, Overflow> {
	return buffer.append([&response](auto& out) { write_object(out, [&response](auto& members) { put_response(members, response); }); });
}

/** A whole response's JSON, in a string of exactly its size. */
export [[nodiscard]] auto encode(Response const& response) -> std::string {
	return encode([&response](auto& out) { write_object(out, [&response](auto& members) { put_response(members, response); }); });
}

export enum class Lifecycle : std::uint8_t {
	Created,
	InProgress,
	Completed,
	Incomplete,
};

/** `response.created`, `.in_progress`, `.completed` or `.incomplete`, carrying the response as it stands. */
export struct ResponseEvent {
	Lifecycle lifecycle;
	Response const& response;
};

/** `response.output_item.added`: the item at `output_index`, before any of its content. */
export struct ItemAdded {
	std::uint32_t output_index;
	OutputItem const& item;
};

/** `response.output_item.done`: the item whole. */
export struct ItemDone {
	std::uint32_t output_index;
	OutputItem const& item;
};

/** The content part a message or a reasoning item holds (always its first, `content_index` 0). */
export enum class PartKind : std::uint8_t {
	OutputText,
	ReasoningText,
};

/** `response.content_part.added`: the part, still empty. */
export struct PartAdded {
	std::string_view item_id;
	std::uint32_t output_index;
	PartKind kind;
};

/** `response.content_part.done`: the part whole. */
export struct PartDone {
	std::string_view item_id;
	std::uint32_t output_index;
	PartKind kind;
	std::string_view text;
};

/** The text an item grows by: a message's, a reasoning item's, a function call's arguments, a custom call's input. */
export enum class Stream : std::uint8_t {
	OutputText,
	ReasoningText,
	Arguments,
	CustomInput,
};

/** The `.delta` event of a stream: the next piece of an item's text. */
export struct TextDelta {
	Stream stream;
	std::string_view item_id;
	std::uint32_t output_index;
	std::string_view delta;
};

/** The `.done` event of a stream: the item's whole text. */
export struct TextDone {
	Stream stream;
	std::string_view item_id;
	std::uint32_t output_index;
	std::string_view text;
};

/** The `error` event: the stream ends without its response. */
export struct ErrorEvent {
	std::string_view code;
	std::string_view message;
	std::optional<std::string_view> param;
};

/** One event of a streamed response. Its members are views, alive for the one call that writes it. */
export using StreamEvent = std::variant<ResponseEvent, ItemAdded, ItemDone, PartAdded, PartDone, TextDelta, TextDone, ErrorEvent>;

[[nodiscard]] constexpr auto event_type(Lifecycle lifecycle) -> std::string_view {
	switch (lifecycle) {
	case Lifecycle::Created:
		return "response.created";
	case Lifecycle::InProgress:
		return "response.in_progress";
	case Lifecycle::Completed:
		return "response.completed";
	case Lifecycle::Incomplete:
		return "response.incomplete";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto delta_type(Stream stream) -> std::string_view {
	switch (stream) {
	case Stream::OutputText:
		return "response.output_text.delta";
	case Stream::ReasoningText:
		return "response.reasoning_text.delta";
	case Stream::Arguments:
		return "response.function_call_arguments.delta";
	case Stream::CustomInput:
		return "response.custom_tool_call_input.delta";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto done_type(Stream stream) -> std::string_view {
	switch (stream) {
	case Stream::OutputText:
		return "response.output_text.done";
	case Stream::ReasoningText:
		return "response.reasoning_text.done";
	case Stream::Arguments:
		return "response.function_call_arguments.done";
	case Stream::CustomInput:
		return "response.custom_tool_call_input.done";
	}
	std::unreachable();
}

/** The member a `.done` event names its whole text by. */
[[nodiscard]] constexpr auto done_member(Stream stream) -> std::string_view {
	switch (stream) {
	case Stream::OutputText:
	case Stream::ReasoningText:
		return "text";
	case Stream::Arguments:
		return "arguments";
	case Stream::CustomInput:
		return "input";
	}
	std::unreachable();
}

/** Whether a stream's events carry a content index: text lives in a content part, calls do not. */
[[nodiscard]] constexpr auto in_part(Stream stream) -> bool {
	return stream == Stream::OutputText || stream == Stream::ReasoningText;
}

template<Sink O>
auto put_part(Members<O>& members, PartKind kind, std::string_view text) -> void {
	members.object("part", [kind, text](Members<O>& part) {
		if (kind == PartKind::OutputText) {
			part.string("type", "output_text");
			part.string("text", text);
			part.array("annotations", [](Elements<O>&) {});
			part.array("logprobs", [](Elements<O>&) {});
		} else {
			part.string("type", "reasoning_text");
			part.string("text", text);
		}
	});
}

template<Sink O>
struct WriteEvent {
	Members<O>& members;

	auto operator()(ResponseEvent const& event) const -> void {
		members.string("type", event_type(event.lifecycle));
		members.object("response", [&event](Members<O>& response) { put_response(response, event.response); });
	}

	auto operator()(ItemAdded const& event) const -> void {
		members.string("type", "response.output_item.added");
		members.integer("output_index", event.output_index);
		members.object("item", [&event](Members<O>& item) { std::visit(WriteItem<O>{.item = item, .shown = Shown::Opening}, event.item); });
	}

	auto operator()(ItemDone const& event) const -> void {
		members.string("type", "response.output_item.done");
		members.integer("output_index", event.output_index);
		members.object("item", [&event](Members<O>& item) { std::visit(WriteItem<O>{.item = item, .shown = Shown::Whole}, event.item); });
	}

	auto operator()(PartAdded const& event) const -> void {
		members.string("type", "response.content_part.added");
		members.string("item_id", event.item_id);
		members.integer("output_index", event.output_index);
		members.integer("content_index", 0);
		put_part(members, event.kind, {});
	}

	auto operator()(PartDone const& event) const -> void {
		members.string("type", "response.content_part.done");
		members.string("item_id", event.item_id);
		members.integer("output_index", event.output_index);
		members.integer("content_index", 0);
		put_part(members, event.kind, event.text);
	}

	auto operator()(TextDelta const& event) const -> void {
		members.string("type", delta_type(event.stream));
		members.string("item_id", event.item_id);
		members.integer("output_index", event.output_index);
		if (in_part(event.stream)) {
			members.integer("content_index", 0);
		}
		members.string("delta", event.delta);
		if (event.stream == Stream::OutputText) {
			members.array("logprobs", [](Elements<O>&) {});
		}
	}

	auto operator()(TextDone const& event) const -> void {
		members.string("type", done_type(event.stream));
		members.string("item_id", event.item_id);
		members.integer("output_index", event.output_index);
		if (in_part(event.stream)) {
			members.integer("content_index", 0);
		}
		members.string(done_member(event.stream), event.text);
		if (event.stream == Stream::OutputText) {
			members.array("logprobs", [](Elements<O>&) {});
		}
	}

	auto operator()(ErrorEvent const& event) const -> void {
		members.string("type", "error");
		members.string("code", event.code);
		members.string("message", event.message);
		if (event.param) {
			members.string("param", *event.param);
		} else {
			members.null("param");
		}
	}
};

/** Appends one event's JSON, numbered `sequence` in its stream. */
export [[nodiscard]] auto append(Buffer& buffer, StreamEvent const& event, std::uint64_t sequence) -> core::Result<void, Overflow> {
	return buffer.append([&](auto& out) {
		write_object(out, [&]<class O>(Members<O>& members) {
			std::visit(WriteEvent<O>{.members = members}, event);
			members.integer("sequence_number", sequence);
		});
	});
}

/** The bytes one event takes, as `append` would write it. */
export [[nodiscard]] auto measure(StreamEvent const& event, std::uint64_t sequence) -> std::size_t {
	auto measured = Measure{};
	write_object(measured, [&](Members<Measure>& members) {
		std::visit(WriteEvent<Measure>{.members = members}, event);
		members.integer("sequence_number", sequence);
	});
	return measured.bytes();
}

template<Sink O>
struct WriteInput {
	Members<O>& item;

	auto operator()(MessageItem const& message) const -> void {
		item.string("type", "message");
		item.string("role", spelling(message.role));
		item.string("status", "completed");
		item.array("content", [&message](Elements<O>& parts) {
			parts.object([&message](Members<O>& part) {
				if (message.role == Role::Assistant) {
					part.string("type", "output_text");
					part.string("text", message.text);
					part.array("annotations", [](Elements<O>&) {});
				} else {
					part.string("type", "input_text");
					part.string("text", message.text);
				}
			});
		});
	}

	auto operator()(FunctionCallItem const& call) const -> void {
		item.string("type", "function_call");
		item.string("call_id", call.call_id);
		item.string("name", call.name);
		item.string("arguments", call.arguments.text());
		item.string("status", "completed");
	}

	auto operator()(FunctionOutputItem const& output) const -> void {
		item.string("type", "function_call_output");
		item.string("call_id", output.call_id);
		item.string("output", output.output);
		item.string("status", "completed");
	}

	auto operator()(CustomCallItem const& call) const -> void {
		item.string("type", "custom_tool_call");
		item.string("call_id", call.call_id);
		item.string("name", call.name);
		item.string("input", call.input);
	}

	auto operator()(CustomOutputItem const& output) const -> void {
		item.string("type", "custom_tool_call_output");
		item.string("call_id", output.call_id);
		item.string("output", output.output);
	}

	auto operator()(ReasoningItem const& reasoning) const -> void {
		item.string("type", "reasoning");
		item.array("summary", [](Elements<O>&) {});
		item.array("content", [&reasoning](Elements<O>& parts) {
			parts.object([&reasoning](Members<O>& part) {
				part.string("type", "reasoning_text");
				part.string("text", reasoning.text);
			});
		});
	}
};

/** `GET /v1/responses/{id}/input_items`: one page of a stored response's input items. */
export [[nodiscard]] auto encode_input_items(std::span<StoredInput const> page, bool more) -> std::string {
	return encode([&](auto& out) {
		write_object(out, [&]<class O>(Members<O>& list) {
			list.string("object", "list");
			list.array("data", [&page](Elements<O>& items) {
				for (auto const& stored : page) {
					items.object([&stored](Members<O>& item) {
						item.string("id", stored.id);
						std::visit(WriteInput<O>{.item = item}, stored.item);
					});
				}
			});
			if (page.empty()) {
				list.null("first_id");
				list.null("last_id");
			} else {
				list.string("first_id", page.front().id);
				list.string("last_id", page.back().id);
			}
			list.boolean("has_more", more);
		});
	});
}

/** `DELETE /v1/responses/{id}`. */
export [[nodiscard]] auto encode_deleted(std::string_view id) -> std::string {
	return encode([id](auto& out) {
		write_object(out, [id](auto& deleted) {
			deleted.string("id", id);
			deleted.string("object", "response.deleted");
			deleted.boolean("deleted", true);
		});
	});
}

}

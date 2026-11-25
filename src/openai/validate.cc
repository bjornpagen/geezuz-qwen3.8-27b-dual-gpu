// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:validate;

import std;
import engine.core;
import engine.http;
import engine.json;
import engine.model;
import engine.sampling;
import :engine;
import :error;
import :stop;
import :text;

namespace engine::openai {

/** A request the service can hand the engine, and how its answer is delivered. */
export struct Admitted {
	Submission submission;
	json::Delivery delivery;
};

/** The code points Python's `str.strip()` removes, which the chat template's `trim` filter applies. */
[[nodiscard]] constexpr auto stripped(char32_t point) -> bool {
	return (point >= 0x09 && point <= 0x0d) || (point >= 0x1c && point <= 0x20) || point == 0x85 || point == 0xa0 || point == 0x1680 ||
	       (point >= 0x2000 && point <= 0x200a) || point == 0x2028 || point == 0x2029 || point == 0x202f || point == 0x205f ||
	       point == 0x3000;
}

/** `text` as the template's `trim` leaves it; the decoder delivers only valid UTF-8. */
[[nodiscard]] constexpr auto trim(std::string_view text) -> std::string_view {
	while (!text.empty()) {
		auto const width = sequence_width(text.front());
		if (!stripped(code_point(text.substr(0, width)))) {
			break;
		}
		text.remove_prefix(width);
	}
	while (!text.empty()) {
		auto start = text.size() - 1;
		while (start > 0 && continuation(text[start])) {
			--start;
		}
		if (!stripped(code_point(text.substr(start)))) {
			break;
		}
		text.remove_suffix(text.size() - start);
	}
	return text;
}

/** Whether a user message is a query, not a turn of tool results, as the template decides it. */
[[nodiscard]] constexpr auto is_query(json::UserMessage const& message) -> bool {
	auto const content = trim(message.content);
	return !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"));
}

[[nodiscard]] auto refuse(Code code, std::string message, std::string param) -> std::unexpected<ApiError> {
	return std::unexpected(ApiError{.code = code, .message = std::move(message), .param = std::move(param)});
}

/** The rules the chat template raises on, so that every admitted request renders. */
[[nodiscard]] auto check_messages(std::span<json::Message const> messages) -> core::Result<void, ApiError> {
	if (messages.empty()) {
		return refuse(Code::MessagesEmpty, "'messages' must hold at least one message", "messages");
	}
	for (auto index = std::size_t{1}; index < messages.size(); ++index) {
		if (std::holds_alternative<json::SystemMessage>(messages[index])) {
			return refuse(Code::SystemNotFirst, "a system message may only be the first message", std::format("messages[{}]", index));
		}
	}
	auto const queries = std::ranges::any_of(messages, [](json::Message const& message) {
		auto const user = core::alternative<json::UserMessage>(message);
		return user.has_value() && is_query(*user);
	});
	if (!queries) {
		return refuse(Code::NoUserQuery, "'messages' holds no user message other than tool responses", "messages");
	}
	return {};
}

[[nodiscard]] auto check_tools(std::span<json::Tool const> tools) -> core::Result<void, ApiError> {
	auto names = std::unordered_set<std::string_view>{};
	names.reserve(tools.size());
	for (auto index = std::size_t{0}; index < tools.size(); ++index) {
		auto const& name = tools[index].name;
		auto param = std::format("tools[{}].function.name", index);
		if (name.empty() || name.size() > max_tool_name_bytes) {
			return refuse(Code::InvalidToolName, std::format("a function name is 1 to {} bytes", max_tool_name_bytes), std::move(param));
		}
		if (!names.insert(name).second) {
			return refuse(Code::DuplicateTool, std::format("function '{}' is defined twice", name), std::move(param));
		}
	}
	return {};
}

[[nodiscard]] auto resolve_tool_choice(std::optional<json::ToolChoice> choice, std::span<json::Tool const> tools)
    -> core::Result<json::ToolChoice, ApiError> {
	if (!choice) {
		return json::ToolChoice{tools.empty() ? json::ToolChoiceMode::None : json::ToolChoiceMode::Auto};
	}
	auto const mode = core::alternative<json::ToolChoiceMode>(std::as_const(*choice));
	auto const forces = !mode || *mode == json::ToolChoiceMode::Required;
	if (forces && tools.empty()) {
		return refuse(Code::ToolChoiceWithoutTools, "'tool_choice' requires a tool call but no tools are defined", "tool_choice");
	}
	if (auto const forced = core::alternative<json::ForcedFunction>(*choice)) {
		if (!std::ranges::contains(tools, forced->name, &json::Tool::name)) {
			return refuse(Code::UnknownTool, std::format("'tool_choice' names function '{}', which no tool defines", forced->name),
			              "tool_choice.function.name");
		}
	}
	return *std::move(choice);
}

[[nodiscard]] auto resolve_budget(std::optional<std::int64_t> max_tokens) -> core::Result<Budget, ApiError> {
	if (!max_tokens) {
		return Budget{UntilContextFull{}};
	}
	if (*max_tokens < 1) {
		return refuse(Code::MaxTokensBelowOne, "'max_tokens' must be at least 1", "max_tokens");
	}
	if (std::cmp_greater(*max_tokens, model::max_positions)) {
		return refuse(Code::MaxTokensAboveContext, std::format("'max_tokens' exceeds the context of {} tokens", model::max_positions),
		              "max_tokens");
	}
	return Budget{AtMost{.tokens = core::narrow<std::uint32_t>(*max_tokens)}};
}

[[nodiscard]] auto resolve_stop(std::vector<std::string> stop) -> core::Result<StopSequences, ApiError> {
	auto parsed = StopSequences::parse(std::move(stop));
	if (!parsed) {
		auto const index = parsed.error().index;
		switch (parsed.error().fault) {
		case StopFault::Empty:
			return refuse(Code::StopEmpty, "a stop sequence must not be empty", std::format("stop[{}]", index));
		case StopFault::TooLong:
			return refuse(Code::StopTooLong, std::format("a stop sequence is at most {} bytes", max_stop_bytes), std::format("stop[{}]", index));
		}
		std::unreachable();
	}
	return *std::move(parsed);
}

/**
 * The sampling a turn gets when the request leaves a field out follows its thinking mode, which the template
 * defaults to on: a turn thinks unless `enable_thinking` is false.
 */
[[nodiscard]] constexpr auto defaults_for(json::TemplateOptions const& options) -> sampling::Defaults {
	switch (options.thinking.value_or(json::Thinking::Enabled)) {
	case json::Thinking::Enabled:
		return sampling::thinking_defaults;
	case json::Thinking::Disabled:
		return sampling::instruct_defaults;
	}
	std::unreachable();
}

/**
 * Checks a decoded request against everything the decoder cannot see and the template and engine require, in the
 * order messages, tools, tool choice, budget, stop, sampling; the first failure is the refusal.
 */
export [[nodiscard]] auto admit(json::ChatRequest request, http::RequestId id, std::uint64_t fresh_seed) -> core::Result<Admitted, ApiError> {
	if (auto checked = check_messages(request.messages); !checked) {
		return std::unexpected(std::move(checked).error());
	}
	if (auto checked = check_tools(request.tools); !checked) {
		return std::unexpected(std::move(checked).error());
	}
	auto tool_choice = resolve_tool_choice(std::move(request.tool_choice), request.tools);
	if (!tool_choice) {
		return std::unexpected(std::move(tool_choice).error());
	}
	auto budget = resolve_budget(request.max_tokens);
	if (!budget) {
		return std::unexpected(std::move(budget).error());
	}
	auto stop = resolve_stop(std::move(request.stop));
	if (!stop) {
		return std::unexpected(std::move(stop).error());
	}
	auto const fields = sampling::Request{
	    .temperature = request.temperature,
	    .top_k = request.top_k,
	    .top_p = request.top_p,
	    .min_p = request.min_p,
	    .seed = request.seed,
	};
	auto params = sampling::parse(fields, defaults_for(request.template_options), fresh_seed);
	if (!params) {
		return std::unexpected(refusal(params.error()));
	}
	return Admitted{
	    .submission =
	        Submission{
	            .id = std::move(id),
	            .messages = std::move(request.messages),
	            .tools = std::move(request.tools),
	            .tool_choice = *std::move(tool_choice),
	            .template_options = request.template_options,
	            .sampling = *params,
	            .budget = *budget,
	            .stop = *std::move(stop),
	        },
	    .delivery = request.delivery,
	};
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:validate;

import std;
import engine.chat;
import engine.core;
import engine.json;
import engine.model;
import engine.sampling;
import engine.sched;
import engine.store;
import :boundary;
import :error;

/**
 * A decoded request checked whole, once, at the boundary: what the decoder's shapes cannot say (the served model,
 * tool names, the tool choice against the tools, the budget, the sampling ranges, the chat template's rules, the
 * stored response a continuation names) is refused here by name, and what passes becomes the typed submission and
 * the response it will be answered with.
 */
namespace engine::openai {

/** The longest tool name the sdk allows. */
inline constexpr std::size_t max_tool_name_bytes = 64;

/** A request the engine can run, and everything the service answers it with. */
export struct Admitted {
	Submission submission;
	/** The response as it opens: in progress, its output empty, every member it echoes set. */
	json::Response response;
	/** The request's input items with the ids its input items list shows. */
	std::vector<json::StoredInput> input;
	json::Delivery delivery;
};

[[nodiscard]] auto refuse(Code code, std::string message, std::optional<std::string> param) -> std::unexpected<ApiError> {
	return std::unexpected(ApiError{.code = code, .message = std::move(message), .param = std::move(param)});
}

[[nodiscard]] constexpr auto tool_name_char(char c) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

/** Tool names are the sdk's `[a-zA-Z0-9_-]{1,64}` and distinct; each definition reads back as one JSON fragment. */
[[nodiscard]] auto check_tools(std::span<json::Tool const> tools) -> core::Result<void, ApiError> {
	for (auto const [index, tool] : std::views::enumerate(tools)) {
		auto const param = std::format("tools[{}].name", index);
		if (tool.name.empty() || tool.name.size() > max_tool_name_bytes || !std::ranges::all_of(tool.name, tool_name_char)) {
			return refuse(Code::InvalidToolName, std::format("tool names are 1 to {} letters, digits, '_' or '-'", max_tool_name_bytes),
			              param);
		}
		if (std::ranges::contains(tools.first(core::narrow<std::size_t>(index)), tool.name, &json::Tool::name)) {
			return refuse(Code::DuplicateToolName, std::format("the tool name '{}' is defined twice", tool.name), param);
		}
		// The chat format reads each definition back as one JSON fragment; within the request's other bounds, only
		// its length can exceed that fragment's.
		if (chat::definition(tool).size() > json::fragment_bounds.document_bytes) {
			return refuse(Code::ToolTooLarge,
			              std::format("the tool's definition is longer than {} bytes", json::fragment_bounds.document_bytes),
			              std::format("tools[{}]", index));
		}
	}
	return {};
}

/** A tool choice that demands a call names a tool the request defines, of the kind it says. */
[[nodiscard]] auto check_choice(json::ToolChoice const& choice, std::span<json::Tool const> tools) -> core::Result<void, ApiError> {
	if (auto const mode = core::alternative<json::ToolChoiceMode>(choice)) {
		if (*mode == json::ToolChoiceMode::Required && tools.empty()) {
			return refuse(Code::ToolChoiceWithoutTools, "tool_choice 'required' needs tools to call", "tool_choice");
		}
		return {};
	}
	if (tools.empty()) {
		return refuse(Code::ToolChoiceWithoutTools, "tool_choice names a tool, but the request defines no tools", "tool_choice");
	}
	auto const function = core::alternative<json::ForcedFunction>(choice);
	auto const name = function ? std::string_view{function->name} : std::string_view{std::get<json::ForcedCustom>(choice).name};
	auto const named = std::ranges::any_of(tools, [&](json::Tool const& tool) {
		return tool.name == name && std::holds_alternative<json::FunctionTool>(tool.input) == function.has_value();
	});
	if (!named) {
		return refuse(Code::UnknownTool,
		              std::format("tool_choice names '{}', which is no {} tool of the request", name, function ? "function" : "custom"),
		              "tool_choice.name");
	}
	return {};
}

/** `max_output_tokens`, as the engine's limit: absent, the reply may run until the context is full. */
[[nodiscard]] auto limit_of(std::optional<std::int64_t> tokens) -> core::Result<sched::Limit, ApiError> {
	if (!tokens) {
		return sched::Limit{sched::UntilContextFull{}};
	}
	if (*tokens < 1) {
		return refuse(Code::MaxOutputTokensBelowOne, "max_output_tokens must be at least 1", "max_output_tokens");
	}
	if (std::cmp_greater(*tokens, model::max_positions)) {
		return refuse(Code::MaxOutputTokensAboveContext,
		              std::format("max_output_tokens must be at most the context, {}", model::max_positions), "max_output_tokens");
	}
	return sched::Limit{sched::AtMost{.tokens = core::narrow<sched::Count>(*tokens)}};
}

/** The model's recommended sampling for a reasoning effort: `none` turns thinking off. */
[[nodiscard]] constexpr auto defaults_of(json::Effort effort) -> sampling::Defaults {
	switch (effort) {
	case json::Effort::None:
		return sampling::instruct_defaults;
	case json::Effort::Low:
	case json::Effort::Medium:
	case json::Effort::XHigh:
		return sampling::thinking_defaults;
	}
	std::unreachable();
}

[[nodiscard]] auto sampling_of(json::CreateRequest const& request, std::uint64_t seed) -> core::Result<sampling::Params, ApiError> {
	auto const params = sampling::parse({.temperature = request.temperature, .top_p = request.top_p}, defaults_of(request.effort), seed);
	if (params) {
		return *params;
	}
	switch (params.error().kind) {
	case sampling::ParamErrorKind::TemperatureOutOfRange:
		return refuse(Code::TemperatureOutOfRange, std::format("temperature must be from 0 to {}", sampling::temperature_maximum),
		              "temperature");
	case sampling::ParamErrorKind::TopPOutOfRange:
		return refuse(Code::TopPOutOfRange, "top_p must be from 0 to 1", "top_p");
	}
	std::unreachable();
}

/** A chat the template would refuse, as the api answers it: the item at fault, when one is. */
[[nodiscard]] auto shape_refusal(chat::ShapeError const& error) -> ApiError {
	auto const param = error.item ? std::format("input[{}]", *error.item) : std::string{"input"};
	switch (error.fault) {
	case chat::ShapeFault::MisplacedSystem:
		return ApiError{
		    .code = Code::SystemMessageNotFirst,
		    .message = "a system or developer message may only open the conversation, before every other item",
		    .param = param,
		};
	case chat::ShapeFault::NoUserQuery:
		return ApiError{.code = Code::NoUserQuery, .message = "the input holds no user message to answer", .param = param};
	}
	std::unreachable();
}

/**
 * The prompt of a continuation: the stored response's exact sequence, if it can be continued as this request asks.
 * The stored tokens already hold its instructions, tools and effort, so a request that changes any of them would be
 * a re-render, which a continuation never is.
 */
[[nodiscard]] auto continued(json::CreateRequest const& request, std::optional<store::Stored const&> previous)
    -> core::Result<Prompt, ApiError> {
	auto const& named = request.previous_response_id.value();
	if (!previous) {
		return refuse(Code::ResponseNotFound, std::format("no stored response has the id '{}'", named), "previous_response_id");
	}
	auto const& stored = previous->response;
	if (stored.status != json::ResponseStatus::Completed) {
		return refuse(Code::PreviousResponseIncomplete, "the previous response ended incomplete, so its turn cannot be continued",
		              "previous_response_id");
	}
	auto const mismatch = [](std::string_view member) {
		return refuse(Code::PreviousResponseMismatch, std::format("'{}' must equal the previous response's", member), std::string{member});
	};
	if (request.instructions != stored.instructions) {
		return mismatch("instructions");
	}
	if (request.tools != stored.tools) {
		return mismatch("tools");
	}
	if (request.effort != stored.effort) {
		return mismatch("reasoning.effort");
	}
	auto turns = chat::continuation(request.input);
	if (!turns) {
		return std::unexpected(shape_refusal(turns.error()));
	}
	return Prompt{Continued{.sequence = previous->sequence, .turns = *std::move(turns), .options = chat::options_of(request.effort)}};
}

[[nodiscard]] auto fresh(json::CreateRequest const& request) -> core::Result<Prompt, ApiError> {
	auto const instructions = request.instructions ? std::optional<std::string_view>{*request.instructions} : std::nullopt;
	auto conversation =
	    chat::conversation({.instructions = instructions, .items = request.input, .tools = request.tools, .effort = request.effort});
	if (!conversation) {
		return std::unexpected(shape_refusal(conversation.error()));
	}
	return Prompt{Fresh{.conversation = *std::move(conversation)}};
}

/** The prefix an input item's id carries, by its type. */
struct InputPrefix {
	[[nodiscard]] auto operator()(json::MessageItem const& /*item*/) const -> std::string_view {
		return "msg";
	}

	[[nodiscard]] auto operator()(json::FunctionCallItem const& /*item*/) const -> std::string_view {
		return "fc";
	}

	[[nodiscard]] auto operator()(json::FunctionOutputItem const& /*item*/) const -> std::string_view {
		return "fco";
	}

	[[nodiscard]] auto operator()(json::CustomCallItem const& /*item*/) const -> std::string_view {
		return "ctc";
	}

	[[nodiscard]] auto operator()(json::CustomOutputItem const& /*item*/) const -> std::string_view {
		return "ctco";
	}

	[[nodiscard]] auto operator()(json::ReasoningItem const& /*item*/) const -> std::string_view {
		return "rs";
	}
};

/**
 * Checks a decoded request in the api's order (model, tools, tool choice, budget, sampling, then the chat or the
 * continuation) and admits it, or refuses it with the first fault. `previous` finds a stored response by id.
 */
export template<Sources S, class Find>
    requires std::invocable<Find&, store::ResponseId> &&
	         std::same_as<std::invoke_result_t<Find&, store::ResponseId>, std::optional<store::Stored const&>>
[[nodiscard]] auto admit(json::CreateRequest request, std::string_view tag, S& sources, Find previous) -> core::Result<Admitted, ApiError> {
	if (request.model && *request.model != model_id) {
		return refuse(Code::ModelNotFound, std::format("the model '{}' does not exist; this server serves '{}'", *request.model, model_id),
		              "model");
	}
	if (auto const tools = check_tools(request.tools); !tools) {
		return std::unexpected(tools.error());
	}
	if (auto const choice = check_choice(request.tool_choice, request.tools); !choice) {
		return std::unexpected(choice.error());
	}
	auto const limit = limit_of(request.max_output_tokens);
	if (!limit) {
		return std::unexpected(limit.error());
	}
	auto const sampled = sampling_of(request, sources.draw());
	if (!sampled) {
		return std::unexpected(sampled.error());
	}
	auto prompt = [&] -> core::Result<Prompt, ApiError> {
		if (!request.previous_response_id) {
			return fresh(request);
		}
		auto const id = store::ResponseId::parse(*request.previous_response_id);
		return continued(request, id ? previous(*id) : std::nullopt);
	}();
	if (!prompt) {
		return std::unexpected(prompt.error());
	}
	auto const defaults = defaults_of(request.effort);
	auto response = json::Response{
	    .id = store::ResponseId{.high = sources.draw(), .low = sources.draw()}.text(),
	    .created_at = sources.unix_seconds(),
	    .completed_at = std::nullopt,
	    .status = json::ResponseStatus::InProgress,
	    .instructions = request.instructions,
	    .metadata = std::move(request.metadata),
	    .model = std::string{model_id},
	    .output = {},
	    .tools = request.tools,
	    .tool_choice = request.tool_choice,
	    .temperature = request.temperature.value_or(defaults.temperature),
	    .top_p = request.top_p.value_or(defaults.top_p),
	    .format = request.format,
	    .effort = request.effort,
	    .max_output_tokens = request.max_output_tokens,
	    .previous_response_id = request.previous_response_id,
	    .storage = request.storage,
	    .usage = std::nullopt,
	};
	auto input = std::vector<json::StoredInput>{};
	input.reserve(request.input.size());
	for (auto& item : request.input) {
		auto id = fresh_id(sources, std::visit(InputPrefix{}, item));
		input.push_back(json::StoredInput{.id = std::move(id), .item = std::move(item)});
	}
	return Admitted{
	    .submission =
	        Submission{
	            .tag = std::string{tag},
	            .prompt = *std::move(prompt),
	            .constraints =
	                Constraints{
	                    .tools = std::move(request.tools),
	                    .tool_choice = std::move(request.tool_choice),
	                    .format = std::move(request.format),
	                    .thinking = chat::options_of(request.effort).thinking,
	                },
	            .sampling = *sampled,
	            .limit = *limit,
	        },
	    .response = std::move(response),
	    .input = std::move(input),
	    .delivery = request.delivery,
	};
}

}

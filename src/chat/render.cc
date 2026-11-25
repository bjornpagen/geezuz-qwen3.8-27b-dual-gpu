// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:render;

import std;
import engine.core;
import engine.json;
import engine.tokenizer;
import :text;
import :dumps;

namespace engine::chat {

/** Why a chat cannot be rendered: each is a case where the chat template raises. */
export enum class RenderFault : std::uint8_t {
	/** Every user message is a tool response written as user text, or there is none. */
	NoUserQuery,
	/** A system message stands after the first message. */
	MisplacedSystem,
	/** Message text is not utf-8. */
	NotUtf8,
};

export struct [[nodiscard]] RenderError {
	/** The template's refusal, or a tool definition or argument that does not read back as JSON within bounds. */
	std::variant<RenderFault, json::ReadError> cause;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		if (std::holds_alternative<json::ReadError>(cause)) {
			return std::get<json::ReadError>(cause).retry();
		}
		switch (std::get<RenderFault>(cause)) {
		case RenderFault::NoUserQuery:
		case RenderFault::MisplacedSystem:
		case RenderFault::NotUtf8:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator==(RenderError const&) const -> bool = default;
};

/**
 * A rendered chat: the prompt's token ids, ending with the generation prompt, and where in them the blocks a prefix
 * cache snapshots before begin. Each mark is the offset of an `<|im_start|>`.
 */
export struct Prompt {
	std::vector<tokenizer::Id> ids;
	/** The first block after the system block (the system block holds the tools and the reasoning instructions). */
	std::size_t first_turn;
	/** The block of the last user message that is not a tool response: the one later turns are asked after. */
	std::size_t last_query;
	/** The generation prompt, which a later turn renders differently once the reply is in it. */
	std::size_t generation;

	[[nodiscard]] auto operator==(Prompt const&) const -> bool = default;
};

inline constexpr std::string_view extra_high_instructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
    "alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

inline constexpr std::string_view low_instructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary "
    "elaboration.";

inline constexpr std::string_view tools_heading = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

// The template's own text: its added-token spellings are read as tokens, as the model was trained on them.
inline constexpr std::string_view tools_instructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but "
    "NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>";

/** The thinking the chat asks for: on unless the request turns it off, as the template defaults it. */
[[nodiscard]] constexpr auto thinking_on(json::TemplateOptions const& options) -> bool {
	return options.thinking != json::Thinking::Disabled;
}

/** The system block's reasoning instructions; medium effort has none. */
[[nodiscard]] constexpr auto reasoning_instructions(json::TemplateOptions const& options) -> std::string_view {
	if (!thinking_on(options)) {
		return {};
	}
	switch (options.reasoning_effort.value_or(json::ReasoningEffort::XHigh)) {
	case json::ReasoningEffort::XHigh:
		return extra_high_instructions;
	case json::ReasoningEffort::Medium:
		return {};
	case json::ReasoningEffort::Low:
		return low_instructions;
	}
	std::unreachable();
}

/** A user message the template takes for a tool response: one whose trimmed text is wrapped in the response tags. */
[[nodiscard]] auto tool_response_text(std::string_view content) -> bool {
	auto const text = strip(content);
	return text.starts_with("<tool_response>") && text.ends_with("</tool_response>");
}

/**
 * The pieces of a prompt. Template text is spelled as the template spells it, so its added-token spellings read as
 * tokens; client text reads as characters, so no message can forge a control token; the template's control tokens
 * are placed by name.
 */
class Pieces {
public:
	auto token(tokenizer::Added added) -> void {
		if (added == tokenizer::Added::im_start) {
			++blocks_;
		}
		pieces_.emplace_back(added);
	}

	auto literal(std::string_view text) -> void {
		if (!text.empty()) {
			pieces_.emplace_back(tokenizer::Text{.utf8 = text, .spelling = tokenizer::Spelling::Token});
		}
	}

	auto client(std::string_view text) -> void {
		if (!text.empty()) {
			pieces_.emplace_back(tokenizer::Text{.utf8 = text, .spelling = tokenizer::Spelling::Characters});
		}
	}

	/** How many blocks have opened: the ordinal of the next `<|im_start|>`. */
	[[nodiscard]] auto blocks() const -> std::size_t {
		return blocks_;
	}

	[[nodiscard]] auto pieces() const -> std::span<tokenizer::Piece const> {
		return pieces_;
	}

private:
	std::vector<tokenizer::Piece> pieces_;
	std::size_t blocks_ = 0;
};

/** The template's tool call: the name, then each argument as a parameter block. */
struct CallText {
	std::string_view name;
	/** Each argument's name and text: a string as itself, any other value as `tojson` writes it. */
	std::vector<std::pair<std::string_view, std::string_view>> arguments;
};

/**
 * Renders chats exactly as the checkpoint's chat template does, into token ids. It owns the reader that reads a
 * tool definition or a JSON argument again to write it as the template's `tojson` does (Python's `json.dumps`, with
 * `", "` and `": "` between parts and no ASCII escaping), which is not how the client wrote it. A renderer stays in
 * place while it renders.
 */
export class Renderer {
public:
	[[nodiscard]] static auto make() -> core::Result<Renderer, json::ReadError> {
		auto reader = json::Reader<json::fragment_bounds>::make();
		if (!reader) {
			return std::unexpected(reader.error());
		}
		return Renderer{std::move(*reader)};
	}

	/** The prompt for the request's chat, ending with the generation prompt. */
	[[nodiscard]] auto render(tokenizer::Tokenizer const& tokenizer, json::ChatRequest const& request) -> core::Result<Prompt, RenderError>;

private:
	explicit Renderer(json::Reader<json::fragment_bounds> reader) : reader_{std::move(reader)} {}

	/** A JSON text as the template's `tojson` writes it, kept in `held` while the pieces borrow it. */
	[[nodiscard]] auto tojson(std::deque<std::string>& held, json::JsonText const& text)
	    -> core::Result<std::string_view, json::ReadError> {
		auto value = load(reader_, text.text());
		if (!value) {
			return std::unexpected(value.error());
		}
		return std::string_view{held.emplace_back(dumps(*value))};
	}

	json::Reader<json::fragment_bounds> reader_;
};

auto Renderer::render(tokenizer::Tokenizer const& tokenizer, json::ChatRequest const& request) -> core::Result<Prompt, RenderError> {
	auto const& messages = request.messages;
	auto const& options = request.template_options;
	auto const refuse = [](RenderFault fault) {
		return std::unexpected(RenderError{.cause = fault});
	};

	auto const last_query = [&] -> std::optional<std::size_t> {
		for (auto index = messages.size(); index > 0; --index) {
			auto const& message = messages[index - 1];
			if (std::holds_alternative<json::UserMessage>(message) && !tool_response_text(std::get<json::UserMessage>(message).content)) {
				return index - 1;
			}
		}
		return std::nullopt;
	}();
	if (!last_query) {
		return refuse(RenderFault::NoUserQuery);
	}
	auto const system = messages.empty() || !std::holds_alternative<json::SystemMessage>(messages.front())
	                        ? std::optional<std::string_view>{}
	                        : std::optional<std::string_view>{std::get<json::SystemMessage>(messages.front()).content};
	if (std::ranges::any_of(messages | std::views::drop(1), [](json::Message const& message) {
		    return std::holds_alternative<json::SystemMessage>(message);
	    })) {
		return refuse(RenderFault::MisplacedSystem);
	}

	// Text the pieces borrow that the request does not hold as written: definitions and arguments as `tojson`.
	auto held = std::deque<std::string>{};
	auto tools = std::vector<std::string_view>{};
	for (auto const& tool : request.tools) {
		auto const text = tojson(held, tool.definition);
		if (!text) {
			return std::unexpected(RenderError{.cause = text.error()});
		}
		tools.push_back(*text);
	}
	auto calls = std::vector<std::vector<CallText>>(messages.size());
	for (auto const [index, message] : std::views::enumerate(messages)) {
		if (!std::holds_alternative<json::AssistantMessage>(message)) {
			continue;
		}
		for (auto const& call : std::get<json::AssistantMessage>(message).tool_calls) {
			auto text = CallText{.name = call.name, .arguments = {}};
			for (auto const& argument : call.arguments) {
				if (std::holds_alternative<std::string>(argument.value)) {
					text.arguments.emplace_back(argument.name, std::get<std::string>(argument.value));
					continue;
				}
				auto const json = tojson(held, std::get<json::JsonText>(argument.value));
				if (!json) {
					return std::unexpected(RenderError{.cause = json.error()});
				}
				text.arguments.emplace_back(argument.name, *json);
			}
			calls[core::narrow<std::size_t>(index)].push_back(std::move(text));
		}
	}

	auto out = Pieces{};
	auto const instructions = reasoning_instructions(options);
	auto const system_text = system ? strip(*system) : std::string_view{};
	if (!tools.empty()) {
		out.token(tokenizer::Added::im_start);
		out.literal("system\n");
		if (!instructions.empty()) {
			out.literal(instructions);
			out.literal("\n\n");
		}
		out.literal(tools_heading);
		for (auto const tool : tools) {
			out.literal("\n");
			out.client(tool);
		}
		out.literal("\n</tools>");
		out.literal(tools_instructions);
		if (!system_text.empty()) {
			out.literal("\n\n");
			out.client(system_text);
		}
		out.token(tokenizer::Added::im_end);
		out.literal("\n");
	} else if (!system_text.empty() || !instructions.empty()) {
		out.token(tokenizer::Added::im_start);
		out.literal("system\n");
		out.literal(instructions);
		if (!instructions.empty() && !system_text.empty()) {
			out.literal("\n\n");
		}
		out.client(system_text);
		out.token(tokenizer::Added::im_end);
		out.literal("\n");
	}

	auto const first_turn = out.blocks();
	auto last_query_block = std::size_t{0};
	auto const keep_reasoning = options.past_thinking != json::PastThinking::Dropped;
	for (auto const [signed_index, message] : std::views::enumerate(messages)) {
		auto const index = core::narrow<std::size_t>(signed_index);
		if (std::holds_alternative<json::UserMessage>(message)) {
			if (index == *last_query) {
				last_query_block = out.blocks();
			}
			out.token(tokenizer::Added::im_start);
			out.literal("user\n");
			out.client(strip(std::get<json::UserMessage>(message).content));
			out.token(tokenizer::Added::im_end);
			out.literal("\n");
		} else if (std::holds_alternative<json::AssistantMessage>(message)) {
			auto const& assistant = std::get<json::AssistantMessage>(message);
			auto const content = strip(assistant.content);
			out.token(tokenizer::Added::im_start);
			out.literal("assistant\n");
			if (keep_reasoning || index > *last_query) {
				out.token(tokenizer::Added::think);
				out.literal("\n");
				out.client(strip(assistant.reasoning_content));
				out.literal("\n");
				out.token(tokenizer::Added::think_end);
				out.literal("\n\n");
			}
			out.client(content);
			for (auto const [call_index, call] : std::views::enumerate(calls[index])) {
				if (call_index > 0) {
					out.literal("\n");
				} else if (!content.empty()) {
					out.literal("\n\n");
				}
				out.token(tokenizer::Added::tool_call);
				out.literal("\n<function=");
				out.client(call.name);
				out.literal(">\n");
				for (auto const& [name, text] : call.arguments) {
					out.literal("<parameter=");
					out.client(name);
					out.literal(">\n");
					out.client(text);
					out.literal("\n</parameter>\n");
				}
				out.literal("</function>\n");
				out.token(tokenizer::Added::tool_call_end);
			}
			out.token(tokenizer::Added::im_end);
			out.literal("\n");
		} else if (std::holds_alternative<json::ToolMessage>(message)) {
			// Consecutive tool responses share one user block. The template opens it after any other message, and
			// opens none for a tool response that comes first of all, which then follows the system block bare.
			if (index > 0 && !std::holds_alternative<json::ToolMessage>(messages[index - 1])) {
				out.token(tokenizer::Added::im_start);
				out.literal("user");
			}
			out.literal("\n");
			out.token(tokenizer::Added::tool_response);
			out.literal("\n");
			out.client(strip(std::get<json::ToolMessage>(message).content));
			out.literal("\n");
			out.token(tokenizer::Added::tool_response_end);
			if (index + 1 == messages.size() || !std::holds_alternative<json::ToolMessage>(messages[index + 1])) {
				out.token(tokenizer::Added::im_end);
				out.literal("\n");
			}
		}
	}

	auto const generation_block = out.blocks();
	out.token(tokenizer::Added::im_start);
	out.literal("assistant\n");
	out.token(tokenizer::Added::think);
	if (thinking_on(options)) {
		out.literal("\n");
	} else {
		out.literal("\n\n");
		out.token(tokenizer::Added::think_end);
		out.literal("\n\n");
	}

	auto ids = tokenizer.encode(out.pieces());
	if (!ids) {
		return refuse(RenderFault::NotUtf8);
	}
	// Client text reads as characters and the template spells no `<|im_start|>`, so the n-th one in the ids is the
	// n-th block opened.
	auto starts = std::vector<std::size_t>{};
	for (auto const [offset, id] : std::views::enumerate(*ids)) {
		if (id == tokenizer::id(tokenizer::Added::im_start)) {
			starts.push_back(core::narrow<std::size_t>(offset));
		}
	}
	core::invariant(starts.size() == generation_block + 1);
	return Prompt{
	    .ids = std::move(*ids),
	    .first_turn = starts.at(first_turn),
	    .last_query = starts.at(last_query_block),
	    .generation = starts.at(generation_block),
	};
}

}

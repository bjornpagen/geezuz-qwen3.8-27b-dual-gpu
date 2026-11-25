// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:render;

import std;
import engine.core;
import engine.json;
import engine.tokenizer;
import :text;
import :dumps;
import :conversation;

namespace engine::chat {

/** Why a chat that the template renders still cannot become ids. */
export enum class RenderFault : std::uint8_t {
	/** Client text is not utf-8. */
	NotUtf8,
};

export struct [[nodiscard]] RenderError {
	/** The fault, or a tool definition or call's arguments that do not read back as JSON within bounds. */
	std::variant<RenderFault, json::ReadError> cause;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		if (std::holds_alternative<json::ReadError>(cause)) {
			return std::get<json::ReadError>(cause).retry();
		}
		switch (std::get<RenderFault>(cause)) {
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

/**
 * What a continuation appends to a stored response's tokens, which end with its assistant turn's `<|im_end|>`: the
 * ids the whole chat's render holds after that point, ending with the generation prompt.
 */
export struct Continuation {
	std::vector<tokenizer::Id> ids;
	/** The offset of the generation prompt's `<|im_start|>` in `ids`. */
	std::size_t generation;

	[[nodiscard]] auto operator==(Continuation const&) const -> bool = default;
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

/** The system block's reasoning instructions; medium effort has none, and a turn that does not think has none. */
[[nodiscard]] constexpr auto reasoning_instructions(Options options) -> std::string_view {
	switch (options.thinking) {
	case Thinking::Disabled:
		return {};
	case Thinking::Enabled:
		break;
	}
	switch (options.effort) {
	case Effort::XHigh:
		return extra_high_instructions;
	case Effort::Medium:
		return {};
	case Effort::Low:
		return low_instructions;
	}
	std::unreachable();
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

/** The turn before the turns being rendered: none (the chat's start), or one that is not a tool turn. */
enum class Before : std::uint8_t {
	Start,
	Other,
};

/** The generation prompt: the assistant block, inside the thinking block or past an empty one. */
auto generation_prompt(Pieces& out, Options options) -> void {
	out.token(tokenizer::Added::im_start);
	out.literal("assistant\n");
	out.token(tokenizer::Added::think);
	switch (options.thinking) {
	case Thinking::Enabled:
		out.literal("\n");
		return;
	case Thinking::Disabled:
		out.literal("\n\n");
		out.token(tokenizer::Added::think_end);
		out.literal("\n\n");
		return;
	}
	std::unreachable();
}

/** The offsets of every `<|im_start|>` in `ids`: client text reads as characters, so each is a block the renderer opened. */
[[nodiscard]] auto block_starts(std::span<tokenizer::Id const> ids) -> std::vector<std::size_t> {
	auto starts = std::vector<std::size_t>{};
	for (auto offset = std::size_t{0}; offset < ids.size(); ++offset) {
		if (ids[offset] == tokenizer::id(tokenizer::Added::im_start)) {
			starts.push_back(offset);
		}
	}
	return starts;
}

/**
 * Renders chats exactly as the checkpoint's chat template does, into token ids. It owns the reader that reads a
 * tool definition or a call's JSON arguments again to write them as the template's `tojson` does (Python's
 * `json.dumps`, with `", "` and `": "` between parts and no ASCII escaping), which is not how the client wrote them.
 * A renderer stays in place while it renders.
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

	/** The prompt for a chat, ending with the generation prompt. */
	[[nodiscard]] auto render(tokenizer::Tokenizer const& tokenizer, Conversation const& conversation) -> core::Result<Prompt, RenderError>;

	/**
	 * The ids a continuation appends after a stored response, whose tokens end with its assistant turn's `<|im_end|>`:
	 * the new turns as the whole chat renders them there, then the generation prompt. The stored tokens already hold
	 * the system block, so the turns hold no system turn.
	 */
	[[nodiscard]] auto continuation(tokenizer::Tokenizer const& tokenizer, std::span<Turn const> turns, Options options)
	    -> core::Result<Continuation, RenderError>;

private:
	explicit Renderer(json::Reader<json::fragment_bounds> reader) : reader_{std::move(reader)} {}

	/** A JSON value as the template's `tojson` writes it, kept in `held` while the pieces borrow it. */
	[[nodiscard]] auto tojson(std::deque<std::string>& held, Json const& value) -> std::string_view {
		return std::string_view{held.emplace_back(dumps(value))};
	}

	[[nodiscard]] auto call_text(std::deque<std::string>& held, Call const& call) -> core::Result<CallText, json::ReadError>;

	/**
	 * Appends `turns`. `last_query` is the index in `turns` of the chat's last user query, or nothing when it stands
	 * before them; an assistant turn after it keeps its reasoning even when past reasoning is dropped. Returns the
	 * block the last query opened, when it is among `turns`.
	 */
	[[nodiscard]] auto append_turns(Pieces& out, std::deque<std::string>& held, std::span<Turn const> turns, Before before, Options options,
	                                std::optional<std::size_t> last_query) -> core::Result<std::optional<std::size_t>, RenderError>;

	json::Reader<json::fragment_bounds> reader_;
};

auto Renderer::call_text(std::deque<std::string>& held, Call const& call) -> core::Result<CallText, json::ReadError> {
	auto text = CallText{.name = call.name, .arguments = {}};
	if (auto const custom = core::alternative<CustomInput>(call.arguments)) {
		text.arguments.emplace_back("input", custom->text);
		return text;
	}
	auto const loaded = load(reader_, std::get<json::JsonText>(call.arguments).text());
	if (!loaded) {
		return std::unexpected(loaded.error());
	}
	// A function call's arguments are one JSON object, which the request's decoding established.
	core::invariant(std::holds_alternative<Object>(loaded->value));
	auto const& members = std::get<Object>(loaded->value).members;
	for (auto const& member : members) {
		if (auto const string = core::alternative<std::string>(member.value.value)) {
			text.arguments.emplace_back(std::string_view{held.emplace_back(member.name)}, std::string_view{held.emplace_back(*string)});
		} else {
			text.arguments.emplace_back(std::string_view{held.emplace_back(member.name)}, tojson(held, member.value));
		}
	}
	return text;
}

auto Renderer::append_turns(Pieces& out, std::deque<std::string>& held, std::span<Turn const> turns, Before before, Options options,
                            std::optional<std::size_t> last_query) -> core::Result<std::optional<std::size_t>, RenderError> {
	auto last_query_block = std::optional<std::size_t>{};
	for (auto index = std::size_t{0}; index < turns.size(); ++index) {
		auto const& turn = turns[index];
		if (auto const user = core::alternative<UserTurn>(turn)) {
			if (index == last_query) {
				last_query_block = out.blocks();
			}
			out.token(tokenizer::Added::im_start);
			out.literal("user\n");
			out.client(strip(user->text));
			out.token(tokenizer::Added::im_end);
			out.literal("\n");
		} else if (auto const assistant = core::alternative<AssistantTurn>(turn)) {
			auto const content = strip(assistant->content);
			out.token(tokenizer::Added::im_start);
			out.literal("assistant\n");
			auto const keep_reasoning = options.past_thinking == PastThinking::Preserved || !last_query || index > *last_query;
			if (keep_reasoning) {
				out.token(tokenizer::Added::think);
				out.literal("\n");
				out.client(strip(assistant->reasoning));
				out.literal("\n");
				out.token(tokenizer::Added::think_end);
				out.literal("\n\n");
			}
			out.client(content);
			for (auto call_index = std::size_t{0}; call_index < assistant->calls.size(); ++call_index) {
				auto const call = call_text(held, assistant->calls[call_index]);
				if (!call) {
					return std::unexpected(RenderError{.cause = call.error()});
				}
				if (call_index > 0) {
					out.literal("\n");
				} else if (!content.empty()) {
					out.literal("\n\n");
				}
				out.token(tokenizer::Added::tool_call);
				out.literal("\n<function=");
				out.client(call->name);
				out.literal(">\n");
				for (auto const& [name, text] : call->arguments) {
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
		} else if (auto const tool = core::alternative<ToolTurn>(turn)) {
			// Consecutive tool responses share one user block. The template opens it after any other message, and
			// opens none for a tool response that comes first of all, which then follows the system block bare.
			auto const after_other = index == 0 ? before == Before::Other : !std::holds_alternative<ToolTurn>(turns[index - 1]);
			if (after_other) {
				out.token(tokenizer::Added::im_start);
				out.literal("user");
			}
			out.literal("\n");
			out.token(tokenizer::Added::tool_response);
			out.literal("\n");
			out.client(strip(tool->content));
			out.literal("\n");
			out.token(tokenizer::Added::tool_response_end);
			if (index + 1 == turns.size() || !std::holds_alternative<ToolTurn>(turns[index + 1])) {
				out.token(tokenizer::Added::im_end);
				out.literal("\n");
			}
		}
		// A system turn renders in the system block, before the turns; the template writes nothing in its place.
	}
	return last_query_block;
}

auto Renderer::render(tokenizer::Tokenizer const& tokenizer, Conversation const& conversation) -> core::Result<Prompt, RenderError> {
	auto const turns = conversation.turns();
	auto const options = conversation.options();
	auto const last_query = [&] {
		auto index = turns.size();
		while (index > 0 && !user_query(turns[index - 1])) {
			--index;
		}
		// A conversation always holds a user query: `Conversation::make` refuses one that does not.
		core::invariant(index > 0);
		return index - 1;
	}();
	auto const system = turns.empty() ? std::optional<SystemTurn const&>{} : core::alternative<SystemTurn>(turns.front());

	// Text the pieces borrow that the conversation does not hold as written: definitions and arguments as `tojson`.
	auto held = std::deque<std::string>{};
	auto tools = std::vector<std::string_view>{};
	for (auto const& definition : conversation.tools()) {
		auto const value = load(reader_, definition);
		if (!value) {
			return std::unexpected(RenderError{.cause = value.error()});
		}
		tools.push_back(tojson(held, *value));
	}

	auto out = Pieces{};
	auto const instructions = reasoning_instructions(options);
	auto const system_text = system ? strip(system->text) : std::string_view{};
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
	auto const last_query_block = append_turns(out, held, turns, Before::Start, options, last_query);
	if (!last_query_block) {
		return std::unexpected(last_query_block.error());
	}
	auto const generation_block = out.blocks();
	generation_prompt(out, options);

	auto ids = tokenizer.encode(out.pieces());
	if (!ids) {
		return std::unexpected(RenderError{.cause = RenderFault::NotUtf8});
	}
	auto const starts = block_starts(*ids);
	core::invariant(starts.size() == generation_block + 1);
	return Prompt{
	    .ids = std::move(*ids),
	    .first_turn = starts.at(first_turn),
	    // The last query is among the turns, so its block was opened; `value()` aborts were it not.
	    .last_query = starts.at(last_query_block->value()),
	    .generation = starts.at(generation_block),
	};
}

auto Renderer::continuation(tokenizer::Tokenizer const& tokenizer, std::span<Turn const> turns, Options options)
    -> core::Result<Continuation, RenderError> {
	core::invariant(std::ranges::none_of(turns, [](Turn const& turn) { return std::holds_alternative<SystemTurn>(turn); }));
	// The chat's last user query is the last among the new turns, or stands in the stored part before all of them.
	auto last_query = std::optional<std::size_t>{};
	for (auto index = std::size_t{0}; index < turns.size(); ++index) {
		if (user_query(turns[index])) {
			last_query = index;
		}
	}
	auto held = std::deque<std::string>{};
	auto out = Pieces{};
	// The template follows every assistant turn's `<|im_end|>` with a line break, which the stored tokens end before.
	out.literal("\n");
	auto const appended = append_turns(out, held, turns, Before::Other, options, last_query);
	if (!appended) {
		return std::unexpected(appended.error());
	}
	auto const generation_block = out.blocks();
	generation_prompt(out, options);
	auto ids = tokenizer.encode(out.pieces());
	if (!ids) {
		return std::unexpected(RenderError{.cause = RenderFault::NotUtf8});
	}
	auto const starts = block_starts(*ids);
	core::invariant(starts.size() == generation_block + 1);
	return Continuation{.ids = std::move(*ids), .generation = starts.back()};
}

}

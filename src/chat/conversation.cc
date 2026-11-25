// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:conversation;

import std;
import engine.core;
import engine.json;
import :text;

/**
 * A chat as the checkpoint's chat template takes it, and how a Responses request's input becomes one. The template
 * renders messages by role; a Responses input is a list of items, so items map onto the template's messages here,
 * once, and the renderer sees only the template's shapes.
 */
namespace engine::chat {

/** `enable_thinking`: the generation prompt opens the thinking block, or closes it empty. */
export enum class Thinking : std::uint8_t {
	Enabled,
	Disabled,
};

/** `reasoning_effort`, as the template defines it; it matters only while thinking. */
export enum class Effort : std::uint8_t {
	Low,
	Medium,
	XHigh,
};

/** `preserve_thinking`: whether assistant turns before the last user query keep their reasoning. */
export enum class PastThinking : std::uint8_t {
	Preserved,
	Dropped,
};

export struct Options {
	Thinking thinking;
	Effort effort;
	PastThinking past_thinking;

	[[nodiscard]] auto operator==(Options const&) const -> bool = default;
};

export struct SystemTurn {
	std::string text;

	[[nodiscard]] auto operator==(SystemTurn const&) const -> bool = default;
};

export struct UserTurn {
	std::string text;

	[[nodiscard]] auto operator==(UserTurn const&) const -> bool = default;
};

/** A custom tool call's one string parameter, `input`. */
export struct CustomInput {
	std::string text;

	[[nodiscard]] auto operator==(CustomInput const&) const -> bool = default;
};

/** A past tool call: a function's arguments as one JSON object, or a custom tool's input. */
export struct Call {
	std::string name;
	std::variant<json::JsonText, CustomInput> arguments;

	[[nodiscard]] auto operator==(Call const&) const -> bool = default;
};

export struct AssistantTurn {
	std::string reasoning;
	std::string content;
	std::vector<Call> calls;

	[[nodiscard]] auto operator==(AssistantTurn const&) const -> bool = default;
};

/** A tool's result, which the template renders as a tool response inside a user block. */
export struct ToolTurn {
	std::string content;

	[[nodiscard]] auto operator==(ToolTurn const&) const -> bool = default;
};

export using Turn = std::variant<SystemTurn, UserTurn, AssistantTurn, ToolTurn>;

/** Why a chat cannot be rendered: each is a case where the chat template raises. */
export enum class ShapeFault : std::uint8_t {
	/** Every user turn is wholly a tool response, or there is none. */
	NoUserQuery,
	/** A system turn stands after the first turn. */
	MisplacedSystem,
};

export struct [[nodiscard]] ShapeError {
	ShapeFault fault;
	/** The input item at fault, when the chat came from Responses input items and one item is. */
	std::optional<std::size_t> item;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (fault) {
		case ShapeFault::NoUserQuery:
		case ShapeFault::MisplacedSystem:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator==(ShapeError const&) const -> bool = default;
};

/** A user turn the template takes for a tool response: one whose stripped text is wrapped in the response tags. */
[[nodiscard]] auto tool_response_text(std::string_view content) -> bool {
	auto const text = strip(content);
	return text.starts_with("<tool_response>") && text.ends_with("</tool_response>");
}

/** Whether a turn is a user query the template can answer: a user turn that is not wholly a tool response. */
[[nodiscard]] auto user_query(Turn const& turn) -> bool {
	auto const user = core::alternative<UserTurn>(turn);
	return user.has_value() && !tool_response_text(user->text);
}

/**
 * A chat the template renders without raising, and the tools its system block lists (each a tool definition as
 * JSON text, the template's `tools`). Only `make` builds one, so the renderer never meets a chat the template
 * refuses.
 */
export class Conversation {
public:
	[[nodiscard]] static auto make(std::vector<Turn> turns, std::vector<std::string> tool_definitions, Options options)
	    -> core::Result<Conversation, ShapeError> {
		auto const late_system = std::ranges::any_of(turns | std::views::drop(1), [](Turn const& turn) {
			return std::holds_alternative<SystemTurn>(turn);
		});
		if (late_system) {
			return std::unexpected(ShapeError{.fault = ShapeFault::MisplacedSystem, .item = std::nullopt});
		}
		if (!std::ranges::any_of(turns, user_query)) {
			return std::unexpected(ShapeError{.fault = ShapeFault::NoUserQuery, .item = std::nullopt});
		}
		return Conversation{std::move(turns), std::move(tool_definitions), options};
	}

	[[nodiscard]] auto turns() const -> std::span<Turn const> {
		return turns_;
	}

	[[nodiscard]] auto tools() const -> std::span<std::string const> {
		return tools_;
	}

	[[nodiscard]] auto options() const -> Options {
		return options_;
	}

	[[nodiscard]] auto operator==(Conversation const&) const -> bool = default;

private:
	Conversation(std::vector<Turn> turns, std::vector<std::string> tools, Options options)
	    : turns_{std::move(turns)}, tools_{std::move(tools)}, options_{options} {}

	std::vector<Turn> turns_;
	std::vector<std::string> tools_;
	Options options_;
};

/** What a Responses request renders: its instructions, its input items, its tools and its reasoning effort. */
export struct Input {
	std::optional<std::string_view> instructions;
	std::span<json::InputItem const> items;
	std::span<json::Tool const> tools;
	json::Effort effort;
};

/** The template's options for a request's reasoning effort: `none` turns thinking off, and past reasoning stays. */
export [[nodiscard]] constexpr auto options_of(json::Effort effort) -> Options {
	switch (effort) {
	case json::Effort::None:
		return {.thinking = Thinking::Disabled, .effort = Effort::XHigh, .past_thinking = PastThinking::Preserved};
	case json::Effort::Low:
		return {.thinking = Thinking::Enabled, .effort = Effort::Low, .past_thinking = PastThinking::Preserved};
	case json::Effort::Medium:
		return {.thinking = Thinking::Enabled, .effort = Effort::Medium, .past_thinking = PastThinking::Preserved};
	case json::Effort::XHigh:
		return {.thinking = Thinking::Enabled, .effort = Effort::XHigh, .past_thinking = PastThinking::Preserved};
	}
	std::unreachable();
}

/**
 * A Responses tool as the template's `tools` takes it: a chat-format function. A custom tool reaches the model as a
 * function with one string parameter, `input`, whose value its format constrains.
 */
export [[nodiscard]] auto definition(json::Tool const& tool) -> std::string {
	auto const custom = std::holds_alternative<json::CustomTool>(tool.input);
	auto const function = core::alternative<json::FunctionTool>(tool.input);
	return json::encode([&](auto& out) {
		json::write_object(out, [&]<class O>(json::Members<O>& element) {
			element.string("type", "function");
			element.object("function", [&](json::Members<O>& written) {
				written.string("name", tool.name);
				if (tool.description) {
					written.string("description", *tool.description);
				}
				if (custom) {
					written.object("parameters", [](json::Members<O>& schema) {
						schema.string("type", "object");
						schema.object("properties", [](json::Members<O>& properties) {
							properties.object("input", [](json::Members<O>& input) { input.string("type", "string"); });
						});
						schema.array("required", [](json::Elements<O>& required) { required.string("input"); });
					});
				} else if (function && function->parameters) {
					written.json("parameters", *function->parameters);
				}
			});
		});
	});
}

/** How far the assistant turn being built has come, in the template's order: reasoning, then content, then calls. */
enum class Stage : std::uint8_t {
	Reasoning,
	Content,
	Calls,
};

/** Where system and developer messages may stand: first of all, or nowhere (a continuation's new items). */
enum class SystemPlace : std::uint8_t {
	First,
	Nowhere,
};

/** Items mapped onto turns, with the assistant turn still open for extension, if the last item opened one. */
class Builder {
public:
	explicit Builder(std::vector<Turn> turns) : turns_{std::move(turns)} {}

	[[nodiscard]] auto add(json::InputItem const& item, std::size_t index, SystemPlace place) -> core::Result<void, ShapeError> {
		if (auto const message = core::alternative<json::MessageItem>(item)) {
			return add_message(*message, index, place);
		}
		if (auto const reasoning = core::alternative<json::ReasoningItem>(item)) {
			// Reasoning leads a turn: whatever turn is open is finished before it.
			stage_.reset();
			open_turn().reasoning = reasoning->text;
			stage_ = Stage::Reasoning;
			return {};
		}
		if (auto const call = core::alternative<json::FunctionCallItem>(item)) {
			add_call(Call{.name = call->name, .arguments = call->arguments});
			return {};
		}
		if (auto const call = core::alternative<json::CustomCallItem>(item)) {
			add_call(Call{.name = call->name, .arguments = CustomInput{.text = call->input}});
			return {};
		}
		if (auto const output = core::alternative<json::FunctionOutputItem>(item)) {
			close(ToolTurn{.content = output->output});
			return {};
		}
		close(ToolTurn{.content = std::get<json::CustomOutputItem>(item).output});
		return {};
	}

	[[nodiscard]] auto turns() && -> std::vector<Turn> {
		return std::move(turns_);
	}

private:
	[[nodiscard]] auto add_message(json::MessageItem const& message, std::size_t index, SystemPlace place) -> core::Result<void, ShapeError> {
		switch (message.role) {
		case json::Role::System:
		case json::Role::Developer:
			if (place == SystemPlace::Nowhere || !turns_.empty()) {
				return std::unexpected(ShapeError{.fault = ShapeFault::MisplacedSystem, .item = index});
			}
			close(SystemTurn{.text = message.text});
			return {};
		case json::Role::User:
			close(UserTurn{.text = message.text});
			return {};
		case json::Role::Assistant:
			// A message extends a turn that holds only reasoning; after content or calls it opens the next turn.
			if (stage_ != Stage::Reasoning) {
				stage_.reset();
			}
			open_turn().content = message.text;
			stage_ = Stage::Content;
			return {};
		}
		std::unreachable();
	}

	auto add_call(Call call) -> void {
		open_turn().calls.push_back(std::move(call));
		stage_ = Stage::Calls;
	}

	/** The assistant turn being built, opened now when none is. A reasoning item always opens a new one. */
	[[nodiscard]] auto open_turn() -> AssistantTurn& {
		if (!stage_) {
			turns_.emplace_back(AssistantTurn{});
		}
		return std::get<AssistantTurn>(turns_.back());
	}

	auto close(Turn turn) -> void {
		turns_.push_back(std::move(turn));
		stage_.reset();
	}

	std::vector<Turn> turns_;
	/** Present while the last turn is an assistant turn items may still extend. */
	std::optional<Stage> stage_;
};

/** The turns of `items` appended to `turns`. */
[[nodiscard]] auto map_items(std::vector<Turn> turns, std::span<json::InputItem const> items, SystemPlace place)
    -> core::Result<std::vector<Turn>, ShapeError> {
	auto builder = Builder{std::move(turns)};
	for (auto index = std::size_t{0}; index < items.size(); ++index) {
		if (auto added = builder.add(items[index], index, place); !added) {
			return std::unexpected(added.error());
		}
	}
	return std::move(builder).turns();
}

/** The chat a Responses request renders: instructions as the leading system turn, then its items, then its tools. */
export [[nodiscard]] auto conversation(Input const& input) -> core::Result<Conversation, ShapeError> {
	auto leading = std::vector<Turn>{};
	if (input.instructions) {
		leading.emplace_back(SystemTurn{.text = std::string{*input.instructions}});
	}
	auto turns = map_items(std::move(leading), input.items, SystemPlace::First);
	if (!turns) {
		return std::unexpected(turns.error());
	}
	auto definitions = input.tools | std::views::transform([](json::Tool const& tool) { return definition(tool); }) |
	                   std::ranges::to<std::vector<std::string>>();
	return Conversation::make(*std::move(turns), std::move(definitions), options_of(input.effort));
}

/**
 * The turns a continuation's new items add after a stored response. Its system block is already in the stored
 * tokens, so a system or developer message anywhere in them is misplaced.
 */
export [[nodiscard]] auto continuation(std::span<json::InputItem const> items) -> core::Result<std::vector<Turn>, ShapeError> {
	return map_items({}, items, SystemPlace::Nowhere);
}

}

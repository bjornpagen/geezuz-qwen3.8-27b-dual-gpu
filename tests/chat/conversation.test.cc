// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.chat;
import engine.testing;

namespace {

namespace chat = engine::chat;
namespace json = engine::json;

[[nodiscard]] auto object(std::string_view text) -> json::JsonText {
	auto reader = json::Reader<json::fragment_bounds>::make();
	engine::core::invariant(reader.has_value());
	auto read = json::JsonText::from(*reader, text);
	engine::core::invariant(read.has_value());
	return *std::move(read);
}

[[nodiscard]] auto message(json::Role role, std::string text) -> json::InputItem {
	return json::MessageItem{.role = role, .text = std::move(text)};
}

[[nodiscard]] auto user(std::string text) -> json::InputItem {
	return message(json::Role::User, std::move(text));
}

[[nodiscard]] auto said(std::string text) -> json::InputItem {
	return message(json::Role::Assistant, std::move(text));
}

[[nodiscard]] auto thought(std::string text) -> json::InputItem {
	return json::ReasoningItem{.text = std::move(text)};
}

[[nodiscard]] auto called(std::string name) -> json::InputItem {
	return json::FunctionCallItem{.call_id = "c", .name = std::move(name), .arguments = object(R"({"a": 1})")};
}

[[nodiscard]] auto custom(std::string name, std::string input) -> json::InputItem {
	return json::CustomCallItem{.call_id = "c", .name = std::move(name), .input = std::move(input)};
}

[[nodiscard]] auto output(std::string text) -> json::InputItem {
	return json::FunctionOutputItem{.call_id = "c", .output = std::move(text)};
}

[[nodiscard]] auto function_call(std::string name) -> chat::Call {
	return chat::Call{.name = std::move(name), .arguments = object(R"({"a": 1})")};
}

[[nodiscard]] auto assistant(std::string reasoning, std::string content, std::vector<chat::Call> calls) -> chat::Turn {
	return chat::AssistantTurn{.reasoning = std::move(reasoning), .content = std::move(content), .calls = std::move(calls)};
}

[[nodiscard]] auto mapped(std::optional<std::string_view> instructions, std::vector<json::InputItem> const& items)
    -> engine::core::Result<chat::Conversation, chat::ShapeError> {
	return chat::conversation(chat::Input{.instructions = instructions, .items = items, .tools = {}, .effort = json::Effort::XHigh});
}

[[nodiscard]] auto maps_to(std::optional<std::string_view> instructions, std::vector<json::InputItem> const& items, std::vector<chat::Turn> const& turns)
    -> bool {
	auto const made = mapped(instructions, items);
	return made && std::ranges::equal(made->turns(), turns);
}

[[nodiscard]] auto refused(std::optional<std::string_view> instructions, std::vector<json::InputItem> const& items, chat::ShapeError error) -> bool {
	auto const made = mapped(instructions, items);
	return !made && made.error() == error;
}

/** A turn's items in template order (reasoning, content, calls) build one assistant turn; outputs are tool turns. */
[[nodiscard]] auto items_fold_into_turns() -> bool {
	auto const items = std::vector{
	    user("q"),  thought("r1"), said("a1"), called("f"), custom("g", "x"), output("o"), thought("r2"), called("h"),
	    user("q2"), said("only"),
	};
	auto const turns = std::vector<chat::Turn>{
	    chat::SystemTurn{.text = "be brief"},
	    chat::UserTurn{.text = "q"},
	    assistant("r1", "a1", {function_call("f"), chat::Call{.name = "g", .arguments = chat::CustomInput{.text = "x"}}}),
	    chat::ToolTurn{.content = "o"},
	    assistant("r2", "", {function_call("h")}),
	    chat::UserTurn{.text = "q2"},
	    assistant("", "only", {}),
	};
	return maps_to("be brief", items, turns);
}

/** An item that cannot extend the open assistant turn in template order opens the next one. */
[[nodiscard]] auto out_of_order_items_open_turns() -> bool {
	return maps_to(std::nullopt, {user("q"), said("a"), said("b")}, {chat::UserTurn{.text = "q"}, assistant("", "a", {}), assistant("", "b", {})}) &&
	       maps_to(std::nullopt, {user("q"), called("f"), said("a")}, {chat::UserTurn{.text = "q"}, assistant("", "", {function_call("f")}), assistant("", "a", {})}) &&
	       maps_to(std::nullopt, {user("q"), said("a"), thought("r")}, {chat::UserTurn{.text = "q"}, assistant("", "a", {}), assistant("r", "", {})}) &&
	       maps_to(std::nullopt, {user("q"), thought("r1"), thought("r2")}, {chat::UserTurn{.text = "q"}, assistant("r1", "", {}), assistant("r2", "", {})}) &&
	       maps_to(std::nullopt, {user("q"), said("a"), called("f"), called("g")},
	               {chat::UserTurn{.text = "q"}, assistant("", "a", {function_call("f"), function_call("g")})});
}

/** A system or developer message stands only first, and instructions take that place; the refusal names the item. */
[[nodiscard]] auto system_stands_only_first() -> bool {
	using chat::ShapeFault;
	return maps_to(std::nullopt, {message(json::Role::Developer, "d"), user("q")}, {chat::SystemTurn{.text = "d"}, chat::UserTurn{.text = "q"}}) &&
	       maps_to(std::nullopt, {message(json::Role::System, "s"), user("q")}, {chat::SystemTurn{.text = "s"}, chat::UserTurn{.text = "q"}}) &&
	       refused("i", {message(json::Role::Developer, "d"), user("q")}, {.fault = ShapeFault::MisplacedSystem, .item = 0}) &&
	       refused(std::nullopt, {user("q"), said("a"), message(json::Role::System, "s")}, {.fault = ShapeFault::MisplacedSystem, .item = 2});
}

/** The template answers a user query: an input whose user messages are all tool responses, or that has none, is refused. */
[[nodiscard]] auto a_query_is_required() -> bool {
	using chat::ShapeFault;
	auto const none = chat::ShapeError{.fault = ShapeFault::NoUserQuery, .item = std::nullopt};
	return refused("i", {}, none) && refused(std::nullopt, {user(" <tool_response>x</tool_response>\n")}, none) &&
	       refused(std::nullopt, {said("a"), output("o")}, none) && mapped(std::nullopt, {user("<tool_response>x</tool_response>"), user("q")}).has_value();
}

/** A continuation's items are mapped alike, with no system message anywhere and no query required. */
[[nodiscard]] auto continuations_map_alike() -> bool {
	auto const items = std::vector{output("o"), user("q")};
	auto const turns = chat::continuation(items);
	auto const system = std::vector{output("o"), message(json::Role::Developer, "d")};
	auto const refusal = chat::continuation(system);
	auto const empty = chat::continuation({});
	return turns && *turns == std::vector<chat::Turn>{chat::ToolTurn{.content = "o"}, chat::UserTurn{.text = "q"}} && !refusal &&
	       refusal.error() == chat::ShapeError{.fault = chat::ShapeFault::MisplacedSystem, .item = 1} && empty && empty->empty();
}

[[nodiscard]] auto efforts_map_to_options() -> bool {
	using chat::Thinking;
	auto const preserved = [](chat::Options options) { return options.past_thinking == chat::PastThinking::Preserved; };
	return chat::options_of(json::Effort::None).thinking == Thinking::Disabled &&
	       chat::options_of(json::Effort::Low) == chat::Options{.thinking = Thinking::Enabled, .effort = chat::Effort::Low, .past_thinking = chat::PastThinking::Preserved} &&
	       chat::options_of(json::Effort::Medium).effort == chat::Effort::Medium && chat::options_of(json::Effort::XHigh).effort == chat::Effort::XHigh &&
	       std::ranges::all_of(std::array{json::Effort::None, json::Effort::Low, json::Effort::Medium, json::Effort::XHigh},
	                           [&](json::Effort effort) { return preserved(chat::options_of(effort)); });
}

/** A function tool keeps its schema; a custom tool becomes a function of one string parameter, `input`. */
[[nodiscard]] auto tools_become_template_functions() -> bool {
	auto const function = json::Tool{.name = "f", .description = "d", .input = json::FunctionTool{.parameters = object(R"({"type":"object"})")}};
	auto const bare = json::Tool{.name = "b", .description = std::nullopt, .input = json::FunctionTool{.parameters = std::nullopt}};
	auto const free = json::Tool{.name = "g", .description = std::nullopt, .input = json::CustomTool{.grammar = std::nullopt}};
	auto const tools = std::array{function, free};
	auto const items = std::vector{user("q")};
	auto const made = chat::conversation(chat::Input{.instructions = std::nullopt, .items = items, .tools = tools, .effort = json::Effort::Low});
	return chat::definition(function) == R"({"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}})" &&
	       chat::definition(bare) == R"({"type":"function","function":{"name":"b"}})" &&
	       chat::definition(free) ==
	           R"({"type":"function","function":{"name":"g","parameters":{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}}})" &&
	       made && std::ranges::equal(made->tools(), std::array{chat::definition(function), chat::definition(free)}) &&
	       made->options() == chat::options_of(json::Effort::Low);
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a turn's items in template order fold into one assistant turn; outputs are tool turns",
		                      .holds = items_fold_into_turns()},
	    engine::testing::Case{.name = "an item out of template order opens the next assistant turn", .holds = out_of_order_items_open_turns()},
	    engine::testing::Case{.name = "a system or developer message stands only first, and the refusal names the item",
		                      .holds = system_stands_only_first()},
	    engine::testing::Case{.name = "an input without a user query is refused", .holds = a_query_is_required()},
	    engine::testing::Case{.name = "a continuation's items map alike, with no system message and no query required",
		                      .holds = continuations_map_alike()},
	    engine::testing::Case{.name = "each reasoning effort is the template's options", .holds = efforts_map_to_options()},
	    engine::testing::Case{.name = "tools become the template's functions; a custom tool takes one string input",
		                      .holds = tools_become_template_functions()},
	};
	return engine::testing::report(cases);
}

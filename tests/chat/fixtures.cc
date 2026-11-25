// SPDX-License-Identifier: AGPL-3.0-only
export module engine_test_chat.fixtures;

import std;
import engine.core;
import engine.json;
import engine.tokenizer;
import engine.chat;
import engine.chat_fixtures;

namespace engine_test_chat {

namespace json = engine::json;
using engine::tokenizer::Id;

struct Node;
struct Field;

struct Null {
	[[nodiscard]] auto operator==(Null const&) const -> bool = default;
};

using Nodes = std::vector<Node>;
using Fields = std::vector<Field>;

/** A fixture file's JSON, read whole: the fixtures hold only nulls, booleans, integers, strings, arrays and objects. */
struct Node {
	std::variant<Null, bool, std::int64_t, std::string, Nodes, Fields> value;

	[[nodiscard]] auto at(std::string_view name) const -> std::optional<Node const&>;

	[[nodiscard]] auto text() const -> std::string {
		return std::holds_alternative<std::string>(value) ? std::get<std::string>(value) : std::string{};
	}

	[[nodiscard]] auto items() const -> std::span<Node const> {
		return std::holds_alternative<Nodes>(value) ? std::span<Node const>{std::get<Nodes>(value)} : std::span<Node const>{};
	}

	[[nodiscard]] auto flag() const -> bool {
		return std::holds_alternative<bool>(value) && std::get<bool>(value);
	}
};

struct Field {
	std::string name;
	Node value;
};

auto Node::at(std::string_view name) const -> std::optional<Node const&> {
	if (!std::holds_alternative<Fields>(value)) {
		return std::nullopt;
	}
	auto const& fields = std::get<Fields>(value);
	auto const found = std::ranges::find(fields, name, &Field::name);
	if (found == fields.end()) {
		return std::nullopt;
	}
	return found->value;
}

inline constexpr json::Bounds fixture_bounds{
    .document_bytes = std::size_t{8} << 20U,
    .depth = 16,
    .string_bytes = std::size_t{1} << 20U,
    .members = std::size_t{1} << 20U,
    .number_chars = 32,
};

using Value = json::Value<fixture_bounds>;

[[nodiscard]] auto read_node(Value value) -> std::optional<Node> {
	auto const kind = value.kind();
	if (!kind) {
		return std::nullopt;
	}
	switch (*kind) {
	case json::Kind::Null:
		return value.as_null() ? std::optional{Node{.value = Null{}}} : std::nullopt;
	case json::Kind::Boolean: {
		auto const read = value.as_boolean();
		return read ? std::optional{Node{.value = *read}} : std::nullopt;
	}
	case json::Kind::Integer: {
		auto const read = value.as_integer();
		return read ? std::optional{Node{.value = *read}} : std::nullopt;
	}
	case json::Kind::Real:
		return std::nullopt;
	case json::Kind::String: {
		auto const read = value.as_string();
		return read ? std::optional{Node{.value = std::string{*read}}} : std::nullopt;
	}
	case json::Kind::Array: {
		auto array = value.as_array();
		auto nodes = Nodes{};
		while (array) {
			auto element = array->next();
			if (!element) {
				return std::nullopt;
			}
			if (!*element) {
				return Node{.value = std::move(nodes)};
			}
			auto node = read_node(std::move(**element));
			if (!node) {
				return std::nullopt;
			}
			nodes.push_back(std::move(*node));
		}
		return std::nullopt;
	}
	case json::Kind::Object: {
		auto object = value.as_object();
		auto fields = Fields{};
		while (object) {
			auto member = object->next();
			if (!member) {
				return std::nullopt;
			}
			if (!*member) {
				return Node{.value = std::move(fields)};
			}
			auto name = std::string{(*member)->name};
			auto node = read_node(std::move((*member)->value));
			if (!node) {
				return std::nullopt;
			}
			fields.push_back(Field{.name = std::move(name), .value = std::move(*node)});
		}
		return std::nullopt;
	}
	}
	std::unreachable();
}

[[nodiscard]] auto read_document(std::string_view text) -> std::optional<Node> {
	auto reader = json::Reader<fixture_bounds>::make();
	if (!reader) {
		return std::nullopt;
	}
	auto root = reader->read_copy(text);
	if (!root) {
		return std::nullopt;
	}
	auto node = read_node(std::move(*root));
	if (!node || !reader->finish()) {
		return std::nullopt;
	}
	return node;
}

[[nodiscard]] auto ids_of(std::optional<Node const&> node) -> std::vector<Id> {
	auto out = std::vector<Id>{};
	if (!node) {
		return out;
	}
	for (auto const& item : node->items()) {
		if (std::holds_alternative<std::int64_t>(item.value)) {
			out.push_back(engine::core::narrow<Id>(std::get<std::int64_t>(item.value)));
		}
	}
	return out;
}

/** What a case's client text does with added-token spellings. */
export enum class Spellings : std::uint8_t {
	/** None is spelled where the reference would read a token the engine does not place. */
	Absent,
	/** Some client text spells one, which the reference reads as the token and the engine as characters. */
	Forged,
};

/** A chat request, the template's text for it, and the ids: the engine's model of its pieces, and the reference's. */
export struct RenderCase {
	std::string name;
	std::string request;
	/** The template's refusal, named as `RenderFault` names it; the rest of the case is empty then. */
	std::optional<std::string> error;
	std::string text;
	std::vector<Id> ids;
	std::vector<Id> reference_ids;
	Spellings spellings;
};

/** A model's reply to a request with `tools`, and the reference's parse of it. */
export struct ParseCase {
	std::string name;
	engine::chat::Thinking thinking;
	std::vector<std::string> tools;
	std::string output;
	engine::chat::Reply expected;
};

/** A JSON number literal and the text `tojson` writes for it, or nothing where Python reads an infinity. */
export struct NumberCase {
	std::string literal;
	std::optional<std::string> dumps;
};

/** A fixture file: the SHA-256s of the checkpoint files it was made from, as lowercase hex, and its cases. */
export template<class Case>
struct Fixture {
	/** `chat_template.jinja`, which renders chats. */
	std::string template_sha256;
	/** `tokenizer_config.json`, whose response template parses replies. */
	std::string tokenizer_config_sha256;
	std::vector<Case> cases;
};

[[nodiscard]] auto text_at(Node const& node, std::string_view name) -> std::string {
	auto const found = node.at(name);
	return found ? found->text() : std::string{};
}

[[nodiscard]] auto items_at(Node const& node, std::string_view name) -> std::span<Node const> {
	auto const found = node.at(name);
	return found ? found->items() : std::span<Node const>{};
}

[[nodiscard]] auto flag_at(Node const& node, std::string_view name) -> bool {
	auto const found = node.at(name);
	return found && found->flag();
}

template<class Case, class F>
[[nodiscard]] auto read_fixture(std::string_view text, F const& read_case) -> std::optional<Fixture<Case>> {
	auto const document = read_document(text);
	if (!document) {
		return std::nullopt;
	}
	auto fixture = Fixture<Case>{
	    .template_sha256 = text_at(*document, "template_sha256"),
	    .tokenizer_config_sha256 = text_at(*document, "tokenizer_config_sha256"),
	    .cases = {},
	};
	auto const cases = document->at("cases");
	if (!cases) {
		return std::nullopt;
	}
	for (auto const& node : cases->items()) {
		fixture.cases.push_back(read_case(node));
	}
	return fixture;
}

export [[nodiscard]] auto render_fixture() -> std::optional<Fixture<RenderCase>> {
	return read_fixture<RenderCase>(engine::chat_fixtures::render(), [](Node const& node) {
		auto const error = node.at("error");
		return RenderCase{
		    .name = text_at(node, "name"),
		    .request = text_at(node, "request"),
		    .error = error ? std::optional{error->text()} : std::nullopt,
		    .text = text_at(node, "text"),
		    .ids = ids_of(node.at("ids")),
		    .reference_ids = ids_of(node.at("reference_ids")),
		    .spellings = flag_at(node, "forged") ? Spellings::Forged : Spellings::Absent,
		};
	});
}

export [[nodiscard]] auto parse_fixture() -> std::optional<Fixture<ParseCase>> {
	return read_fixture<ParseCase>(engine::chat_fixtures::parse(), [](Node const& node) {
		auto tools = std::vector<std::string>{};
		for (auto const& tool : items_at(node, "tools")) {
			tools.push_back(tool.text());
		}
		auto calls = std::vector<engine::chat::ParsedCall>{};
		for (auto const& call : items_at(node, "calls")) {
			calls.push_back(engine::chat::ParsedCall{.name = text_at(call, "name"), .arguments = text_at(call, "arguments")});
		}
		return ParseCase{
		    .name = text_at(node, "name"),
		    .thinking = flag_at(node, "thinking") ? engine::chat::Thinking::Enabled : engine::chat::Thinking::Disabled,
		    .tools = std::move(tools),
		    .output = text_at(node, "output"),
		    .expected = engine::chat::Reply{.reasoning = text_at(node, "reasoning"),
			                                .content = text_at(node, "content"),
			                                .calls = std::move(calls)},
		};
	});
}

export [[nodiscard]] auto number_fixture() -> std::optional<Fixture<NumberCase>> {
	return read_fixture<NumberCase>(engine::chat_fixtures::numbers(), [](Node const& node) {
		auto const dumps = node.at("dumps");
		return NumberCase{.literal = text_at(node, "literal"), .dumps = dumps ? std::optional{dumps->text()} : std::nullopt};
	});
}

namespace chat = engine::chat;

/**
 * A chat-format request body as the chat template takes it, read here from the template's own input format: its
 * turns, its tools as written (the template's `tools`), its options from `chat_template_kwargs`, and the tools a
 * reply parser casts by.
 */
export struct TemplateInput {
	std::vector<chat::Turn> turns;
	std::vector<std::string> tools;
	chat::Options options;
	std::vector<json::Tool> parser_tools;
};

using Fragments = json::Reader<json::fragment_bounds>;

/** Message content: a string, `null`, or text parts read as their concatenation. */
[[nodiscard]] auto content_of(Value value) -> std::optional<std::string> {
	auto const kind = value.kind();
	if (!kind) {
		return std::nullopt;
	}
	if (*kind == json::Kind::Null) {
		return value.as_null() ? std::optional{std::string{}} : std::nullopt;
	}
	if (*kind == json::Kind::String) {
		auto const read = value.as_string();
		return read ? std::optional{std::string{*read}} : std::nullopt;
	}
	auto parts = value.as_array();
	auto joined = std::string{};
	while (parts) {
		auto part = parts->next();
		if (!part) {
			return std::nullopt;
		}
		if (!*part) {
			return joined;
		}
		auto object = (*part)->as_object();
		auto text = object.and_then([](auto& fields) { return fields.member("text"); });
		auto const read = text.and_then([](auto& field) { return field.as_string(); });
		if (!read) {
			return std::nullopt;
		}
		joined += *read;
	}
	return std::nullopt;
}

/** A past call's arguments: an object as written, or a string holding one (empty for none). */
[[nodiscard]] auto arguments_of(Value value, Fragments& fragments) -> std::optional<json::JsonText> {
	auto const kind = value.kind();
	if (!kind) {
		return std::nullopt;
	}
	if (*kind == json::Kind::Object) {
		auto held = value.capture();
		return held ? std::optional{std::move(*held)} : std::nullopt;
	}
	auto const text = value.as_string();
	if (!text) {
		return std::nullopt;
	}
	auto held = json::JsonText::from(fragments, text->empty() ? std::string_view{"{}"} : *text);
	return held ? std::optional{std::move(*held)} : std::nullopt;
}

[[nodiscard]] auto calls_of(Value value, Fragments& fragments) -> std::optional<std::vector<chat::Call>> {
	auto const kind = value.kind();
	if (kind && *kind == json::Kind::Null) {
		return std::vector<chat::Call>{};
	}
	auto list = value.as_array();
	auto calls = std::vector<chat::Call>{};
	while (list) {
		auto element = list->next();
		if (!element) {
			return std::nullopt;
		}
		if (!*element) {
			return calls;
		}
		auto call = (*element)->as_object();
		auto function = call.and_then([](auto& members) { return members.member("function"); });
		auto fields = function.and_then([](auto& inner) { return inner.as_object(); });
		if (!fields) {
			return std::nullopt;
		}
		auto name = std::optional<std::string>{};
		auto arguments = std::optional<json::JsonText>{};
		while (true) {
			auto member = fields->next();
			if (!member) {
				return std::nullopt;
			}
			if (!*member) {
				break;
			}
			if ((*member)->name == "name") {
				auto const read = (*member)->value.as_string();
				if (!read) {
					return std::nullopt;
				}
				name = std::string{*read};
			} else if ((*member)->name == "arguments") {
				arguments = arguments_of(std::move((*member)->value), fragments);
				if (!arguments) {
					return std::nullopt;
				}
			}
		}
		if (!name) {
			return std::nullopt;
		}
		if (!arguments) {
			arguments = json::JsonText::from(fragments, "{}").value();
		}
		calls.push_back(chat::Call{.name = std::move(*name), .arguments = std::move(*arguments)});
	}
	return std::nullopt;
}

[[nodiscard]] auto turn_of(Value value, Fragments& fragments) -> std::optional<chat::Turn> {
	auto message = value.as_object();
	if (!message) {
		return std::nullopt;
	}
	auto role = std::string{};
	auto content = std::string{};
	auto reasoning = std::string{};
	auto calls = std::vector<chat::Call>{};
	while (true) {
		auto member = message->next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			break;
		}
		auto const name = (*member)->name;
		auto& field = (*member)->value;
		if (name == "role") {
			auto const read = field.as_string();
			if (!read) {
				return std::nullopt;
			}
			role = std::string{*read};
		} else if (name == "content") {
			auto read = content_of(std::move(field));
			if (!read) {
				return std::nullopt;
			}
			content = std::move(*read);
		} else if (name == "reasoning_content") {
			auto read = content_of(std::move(field));
			if (!read) {
				return std::nullopt;
			}
			reasoning = std::move(*read);
		} else if (name == "tool_calls") {
			auto read = calls_of(std::move(field), fragments);
			if (!read) {
				return std::nullopt;
			}
			calls = std::move(*read);
		}
	}
	if (role == "system" || role == "developer") {
		return chat::SystemTurn{.text = std::move(content)};
	}
	if (role == "user") {
		return chat::UserTurn{.text = std::move(content)};
	}
	if (role == "assistant") {
		return chat::AssistantTurn{.reasoning = std::move(reasoning), .content = std::move(content), .calls = std::move(calls)};
	}
	if (role == "tool") {
		return chat::ToolTurn{.content = std::move(content)};
	}
	return std::nullopt;
}

/** The tool a reply parser casts by: a chat-format tool's name and parameter schema. */
export [[nodiscard]] auto parser_tool(std::string_view definition, Fragments& fragments) -> std::optional<json::Tool> {
	auto root = fragments.read_copy(definition);
	auto tool = root.and_then([](auto& inner) { return inner.as_object(); });
	auto function = tool.and_then([](auto& members) { return members.member("function"); });
	auto fields = function.and_then([](auto& inner) { return inner.as_object(); });
	if (!fields) {
		return std::nullopt;
	}
	auto name = std::optional<std::string>{};
	auto parameters = std::optional<json::JsonText>{};
	while (true) {
		auto member = fields->next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			break;
		}
		if ((*member)->name == "name") {
			auto const read = (*member)->value.as_string();
			if (!read) {
				return std::nullopt;
			}
			name = std::string{*read};
		} else if ((*member)->name == "parameters") {
			auto const kind = (*member)->value.kind();
			if (kind && *kind == json::Kind::Object) {
				auto held = (*member)->value.capture();
				if (!held) {
					return std::nullopt;
				}
				parameters = std::move(*held);
			}
		}
	}
	if (!name) {
		return std::nullopt;
	}
	return json::Tool{.name = std::move(*name), .description = std::nullopt, .input = json::FunctionTool{.parameters = std::move(parameters)}};
}

[[nodiscard]] auto options_of(Value value) -> std::optional<chat::Options> {
	auto options = chat::Options{.thinking = chat::Thinking::Enabled, .effort = chat::Effort::XHigh, .past_thinking = chat::PastThinking::Preserved};
	auto kwargs = value.as_object();
	while (kwargs) {
		auto member = kwargs->next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			return options;
		}
		auto const name = (*member)->name;
		auto& field = (*member)->value;
		if (name == "enable_thinking" || name == "preserve_thinking") {
			auto const flag = field.as_boolean();
			if (!flag) {
				return std::nullopt;
			}
			if (name == "enable_thinking" && !*flag) {
				options.thinking = chat::Thinking::Disabled;
			}
			if (name == "preserve_thinking" && !*flag) {
				options.past_thinking = chat::PastThinking::Dropped;
			}
		} else if (name == "reasoning_effort") {
			auto const effort = field.as_string();
			if (!effort) {
				return std::nullopt;
			}
			if (*effort == "low") {
				options.effort = chat::Effort::Low;
			} else if (*effort == "medium") {
				options.effort = chat::Effort::Medium;
			} else if (*effort == "xhigh") {
				options.effort = chat::Effort::XHigh;
			} else {
				return std::nullopt;
			}
		}
	}
	return std::nullopt;
}

export [[nodiscard]] auto template_input(std::string_view body) -> std::optional<TemplateInput> {
	auto reader = json::Reader<fixture_bounds>::make();
	auto fragments = Fragments::make();
	if (!reader || !fragments) {
		return std::nullopt;
	}
	auto root = reader->read_copy(body);
	auto request = root.and_then([](auto& inner) { return inner.as_object(); });
	if (!request) {
		return std::nullopt;
	}
	auto input = TemplateInput{
	    .turns = {},
	    .tools = {},
	    .options = {.thinking = chat::Thinking::Enabled, .effort = chat::Effort::XHigh, .past_thinking = chat::PastThinking::Preserved},
	    .parser_tools = {},
	};
	while (true) {
		auto member = request->next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			break;
		}
		auto const name = (*member)->name;
		auto& field = (*member)->value;
		if (name == "messages") {
			auto messages = field.as_array();
			while (messages) {
				auto message = messages->next();
				if (!message) {
					return std::nullopt;
				}
				if (!*message) {
					break;
				}
				auto turn = turn_of(std::move(**message), *fragments);
				if (!turn) {
					return std::nullopt;
				}
				input.turns.push_back(std::move(*turn));
			}
			if (!messages) {
				return std::nullopt;
			}
		} else if (name == "tools") {
			auto tools = field.as_array();
			while (tools) {
				auto tool = tools->next();
				if (!tool) {
					return std::nullopt;
				}
				if (!*tool) {
					break;
				}
				auto held = (*tool)->capture();
				if (!held) {
					return std::nullopt;
				}
				input.tools.emplace_back(held->text());
			}
			if (!tools) {
				return std::nullopt;
			}
		} else if (name == "chat_template_kwargs") {
			auto options = options_of(std::move(field));
			if (!options) {
				return std::nullopt;
			}
			input.options = *options;
		}
	}
	if (!reader->finish()) {
		return std::nullopt;
	}
	for (auto const& definition : input.tools) {
		auto tool = parser_tool(definition, *fragments);
		if (!tool) {
			return std::nullopt;
		}
		input.parser_tools.push_back(std::move(*tool));
	}
	return input;
}

/** The tools a parse case's reply answers, as a reply parser takes them. */
export [[nodiscard]] auto parser_tools(ParseCase const& parse) -> std::optional<std::vector<json::Tool>> {
	auto fragments = Fragments::make();
	if (!fragments) {
		return std::nullopt;
	}
	auto tools = std::vector<json::Tool>{};
	for (auto const& definition : parse.tools) {
		auto tool = parser_tool(definition, *fragments);
		if (!tool) {
			return std::nullopt;
		}
		tools.push_back(std::move(*tool));
	}
	return tools;
}

}

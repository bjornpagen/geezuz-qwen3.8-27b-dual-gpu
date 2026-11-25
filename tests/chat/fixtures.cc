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
	json::Thinking thinking;
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
		auto calls = std::vector<engine::chat::Call>{};
		for (auto const& call : items_at(node, "calls")) {
			calls.push_back(engine::chat::Call{.name = text_at(call, "name"), .arguments = text_at(call, "arguments")});
		}
		return ParseCase{
		    .name = text_at(node, "name"),
		    .thinking = flag_at(node, "thinking") ? json::Thinking::Enabled : json::Thinking::Disabled,
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

/**
 * The request a parse case's reply answers: one user message, the case's tools as written, and its thinking mode.
 * The tools are JSON texts, so the request is their concatenation.
 */
export [[nodiscard]] auto parse_request(ParseCase const& parse) -> std::string {
	auto tools = std::string{};
	for (auto const& tool : parse.tools) {
		tools += tools.empty() ? tool : "," + tool;
	}
	return std::format(R"({{"messages":[{{"role":"user","content":"q"}}],"tools":[{}],"chat_template_kwargs":{{"enable_thinking":{}}}}})",
	                   tools, parse.thinking == json::Thinking::Enabled ? "true" : "false");
}

}

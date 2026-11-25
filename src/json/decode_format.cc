// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_format;

import std;
import engine.core;
import :bounds;
import :error;
import :text;
import :value;
import :containers;
import :reader;
import :decode_error;
import :request;
import :decode_support;
import :decode_neutral;

namespace engine::json {

/** A custom tool's `format`: `{"type": "text"}`, or `{"type": "grammar", "syntax", "definition"}`; `null` is text. */
[[nodiscard]] auto custom_format(Body& value) -> Decoded<std::optional<Grammar>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null()).transform([] { return std::optional<Grammar>{}; });
	}
	auto type = std::optional<std::string>{};
	auto syntax = std::optional<std::string>{};
	auto definition = std::optional<std::string>{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(value, [&](std::string_view name, Body& member) -> Decoded<void> {
		seen.emplace_back(name);
		if (name == "type") {
			return text(member).transform([&type](std::string read) { type = std::move(read); });
		}
		if (name == "syntax") {
			return text(member).transform([&syntax](std::string read) { syntax = std::move(read); });
		}
		if (name == "definition") {
			return text(member).transform([&definition](std::string read) { definition = std::move(read); });
		}
		// Which members belong depends on the type, which may come later: decided after the loop.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!type) {
		return refuse_at("type", Violation::MissingField);
	}
	if (*type != "text" && *type != "grammar") {
		return refuse_at("type", Violation::UnknownValue);
	}
	auto const grammar = *type == "grammar";
	for (auto const& name : seen) {
		if (name != "type" && !(grammar && (name == "syntax" || name == "definition"))) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	if (!grammar) {
		return std::optional<Grammar>{};
	}
	if (!syntax) {
		return refuse_at("syntax", Violation::MissingField);
	}
	if (!definition) {
		return refuse_at("definition", Violation::MissingField);
	}
	if (*syntax == "lark") {
		return Grammar{.syntax = GrammarSyntax::Lark, .definition = std::move(*definition)};
	}
	if (*syntax == "regex") {
		return Grammar{.syntax = GrammarSyntax::Regex, .definition = std::move(*definition)};
	}
	return refuse_at("syntax", Violation::UnknownValue);
}

/** `text.format`: free text, any JSON object, or JSON matching a schema. */
[[nodiscard]] auto text_format(Body& value) -> Decoded<TextFormat> {
	auto type = std::optional<std::string>{};
	auto name = std::optional<std::string>{};
	auto description = std::optional<std::string>{};
	auto schema = std::optional<JsonText>{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(value, [&](std::string_view field, Body& member) -> Decoded<void> {
		seen.emplace_back(field);
		if (field == "type") {
			return text(member).transform([&type](std::string read) { type = std::move(read); });
		}
		if (field == "name") {
			return text(member).transform([&name](std::string read) { name = std::move(read); });
		}
		if (field == "description") {
			return nullable<std::string>(member, [](Body& inner) { return text(inner); }).transform([&description](std::optional<std::string> read) {
				description = std::move(read);
			});
		}
		if (field == "schema") {
			return schema_of(member).transform([&schema](std::optional<JsonText> read) { schema = std::move(read); });
		}
		if (field == "strict") {
			return any_boolean(member);
		}
		// Which members belong depends on the type, which may come later: decided after the loop.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!type) {
		return refuse_at("type", Violation::MissingField);
	}
	if (*type != "text" && *type != "json_object" && *type != "json_schema") {
		return refuse_at("type", Violation::UnknownValue);
	}
	auto const takes_schema = *type == "json_schema";
	for (auto const& field : seen) {
		if (field != "type" && !takes_schema) {
			return refuse_at(field, Violation::UnknownField);
		}
	}
	if (*type == "text") {
		return TextFormatMode::Text;
	}
	if (*type == "json_object") {
		return TextFormatMode::JsonObject;
	}
	if (!name) {
		return refuse_at("name", Violation::MissingField);
	}
	if (!schema) {
		return refuse_at("schema", Violation::MissingField);
	}
	return JsonSchemaFormat{.name = std::move(*name), .description = std::move(description), .schema = std::move(*schema)};
}

/** `text`: its format, and a verbosity the model has no control over. */
[[nodiscard]] auto text_config(Body& value) -> Decoded<TextFormat> {
	auto format = TextFormat{TextFormatMode::Text};
	auto const visited = each_member(value, [&format](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "format") {
			return nullable<TextFormat>(member, [](Body& inner) { return text_format(inner); }).transform([&format](std::optional<TextFormat> read) {
				format = std::move(read).value_or(TextFormat{TextFormatMode::Text});
			});
		}
		if (name == "verbosity") {
			return neutral(member, Neutral::Medium);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return format;
}

}

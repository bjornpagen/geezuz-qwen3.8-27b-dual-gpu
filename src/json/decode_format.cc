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
import :request;
import :decode_support;
import :decode_neutral;

namespace engine::json {

using Fragment = Value<fragment_bounds>;

/** A grammar's members as read, before the format's type says which of them belong. */
struct GrammarDraft {
	std::optional<std::string> syntax;
	std::optional<std::string> definition;
};

[[nodiscard]] auto grammar_fields(Fragment& value) -> Decoded<GrammarDraft> {
	auto draft = GrammarDraft{};
	auto const visited = each_member(value, [&draft](std::string_view name, Fragment& member) -> Decoded<void> {
		if (name == "syntax") {
			return text(member).transform([&draft](std::string read) { draft.syntax = std::move(read); });
		}
		if (name == "definition") {
			return text(member).transform([&draft](std::string read) { draft.definition = std::move(read); });
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return draft;
}

/** A grammar from its members; `within` is where they sit, innermost first. */
[[nodiscard]] auto grammar(GrammarDraft draft, std::vector<PathStep> const& within) -> Decoded<Grammar> {
	auto const at = [&within](std::string_view name, DecodeFault fault) {
		auto path = std::vector<PathStep>{std::string{name}};
		path.insert(path.end(), within.begin(), within.end());
		return std::unexpected(DecodeError{.fault = std::move(fault), .path = std::move(path)});
	};
	if (!draft.syntax) {
		return at("syntax", Violation::MissingField);
	}
	if (!draft.definition) {
		return at("definition", Violation::MissingField);
	}
	if (*draft.syntax == "lark") {
		return Grammar{.syntax = GrammarSyntax::Lark, .definition = std::move(*draft.definition)};
	}
	if (*draft.syntax == "regex") {
		return Grammar{.syntax = GrammarSyntax::Regex, .definition = std::move(*draft.definition)};
	}
	return at("syntax", Violation::UnknownValue);
}

/**
 * A custom tool's `format`: free text, or a grammar. The grammar's syntax and definition are taken nested under
 * `grammar`, as Chat Completions documents it, or beside `type`, as the Responses API spells it.
 */
[[nodiscard]] auto custom_format(Fragment& value) -> Decoded<std::optional<Grammar>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null()).transform([] { return std::optional<Grammar>{}; });
	}
	auto type = std::optional<std::string>{};
	auto nested = std::optional<GrammarDraft>{};
	auto flat = GrammarDraft{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(value, [&](std::string_view name, Fragment& member) -> Decoded<void> {
		seen.emplace_back(name);
		if (name == "type") {
			return text(member).transform([&type](std::string read) { type = std::move(read); });
		}
		if (name == "grammar") {
			return grammar_fields(member).transform([&nested](GrammarDraft read) { nested = std::move(read); });
		}
		if (name == "syntax") {
			return text(member).transform([&flat](std::string read) { flat.syntax = std::move(read); });
		}
		if (name == "definition") {
			return text(member).transform([&flat](std::string read) { flat.definition = std::move(read); });
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
	auto constexpr text_fields = std::to_array<std::string_view>({"type"});
	auto constexpr nested_fields = std::to_array<std::string_view>({"type", "grammar"});
	auto constexpr flat_fields = std::to_array<std::string_view>({"type", "syntax", "definition"});
	auto const allowed = *type == "text" ? std::span<std::string_view const>{text_fields}
	                     : *type != "grammar" ? std::span<std::string_view const>{}
	                     : nested ? std::span<std::string_view const>{nested_fields}
	                              : std::span<std::string_view const>{flat_fields};
	if (*type != "text" && *type != "grammar") {
		return refuse_at("type", Violation::UnknownValue);
	}
	for (auto const& name : seen) {
		if (std::ranges::find(allowed, name) == allowed.end()) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	if (*type == "text") {
		return std::optional<Grammar>{};
	}
	auto const read = nested ? grammar(std::move(*nested), {"grammar"}) : grammar(std::move(flat), {});
	return read.transform([](Grammar decoded) { return std::optional<Grammar>{std::move(decoded)}; });
}

/** A `json_schema` response format's members, read before the format's type says whether they belong. */
[[nodiscard]] auto json_schema(Body& value) -> Decoded<JsonSchemaFormat> {
	auto name = std::optional<std::string>{};
	auto schema = std::optional<JsonText>{};
	auto const visited = each_member(value, [&](std::string_view field, Body& member) -> Decoded<void> {
		if (field == "name") {
			return text(member).transform([&name](std::string read) { name = std::move(read); });
		}
		if (field == "description") {
			return nullable<std::string>(member, [](Body& description) { return text(description); }).transform([](auto const&) {});
		}
		if (field == "schema") {
			return schema_of(member).transform([&schema](std::optional<JsonText> read) { schema = std::move(read); });
		}
		if (field == "strict") {
			return any_boolean(member);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!name) {
		return refuse_at("name", Violation::MissingField);
	}
	return JsonSchemaFormat{.name = std::move(*name), .schema = std::move(schema)};
}

/** `response_format`: what the content region must be. */
[[nodiscard]] auto response_format(Body& value) -> Decoded<ResponseFormat> {
	auto type = std::optional<std::string>{};
	auto schema = std::optional<JsonSchemaFormat>{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(value, [&](std::string_view name, Body& member) -> Decoded<void> {
		seen.emplace_back(name);
		if (name == "type") {
			return text(member).transform([&type](std::string read) { type = std::move(read); });
		}
		if (name == "json_schema") {
			return json_schema(member).transform([&schema](JsonSchemaFormat read) { schema = std::move(read); });
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
	for (auto const& name : seen) {
		if (name != "type" && !(takes_schema && name == "json_schema")) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	if (*type == "text") {
		return ResponseFormatMode::Text;
	}
	if (*type == "json_object") {
		return ResponseFormatMode::JsonObject;
	}
	if (!schema) {
		return refuse_at("json_schema", Violation::MissingField);
	}
	return std::move(*schema);
}

}

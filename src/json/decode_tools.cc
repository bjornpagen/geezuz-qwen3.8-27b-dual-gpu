// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_tools;

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
import :decode_format;

namespace engine::json {

/** Every member a tool has, read once, before its type says which of them belong. */
struct ToolDraft {
	std::optional<std::string> type;
	std::optional<std::string> name;
	std::optional<std::string> description;
	/** Present (even as `null`) or not: a function tool requires the member. */
	std::optional<std::optional<JsonText>> parameters;
	std::optional<std::monostate> strict;
	std::optional<std::optional<Grammar>> format;
	std::vector<std::string> seen;
};

constexpr auto function_tool_members =
    std::to_array<std::string_view>({"type", "name", "description", "parameters", "strict", "defer_loading", "allowed_callers", "output_schema"});
constexpr auto custom_tool_members = std::to_array<std::string_view>({"type", "name", "description", "format", "defer_loading", "allowed_callers"});

[[nodiscard]] auto tool_draft(Body& value) -> Decoded<ToolDraft> {
	auto read = ToolDraft{};
	auto const visited = each_member(value, [&read](std::string_view name, Body& member) -> Decoded<void> {
		read.seen.emplace_back(name);
		if (name == "type") {
			return text(member).transform([&read](std::string decoded) { read.type = std::move(decoded); });
		}
		if (name == "name") {
			return text(member).transform([&read](std::string decoded) { read.name = std::move(decoded); });
		}
		if (name == "description") {
			return nullable<std::string>(member, [](Body& inner) { return text(inner); }).transform([&read](std::optional<std::string> decoded) {
				read.description = std::move(decoded);
			});
		}
		if (name == "parameters") {
			return schema_of(member).transform([&read](std::optional<JsonText> schema) { read.parameters = std::move(schema); });
		}
		if (name == "strict") {
			return any_boolean(member).transform([&read] { read.strict = std::monostate{}; });
		}
		if (name == "format") {
			return custom_format(member).transform([&read](std::optional<Grammar> grammar) { read.format = std::move(grammar); });
		}
		if (name == "output_schema") {
			return neutral(member, Neutral::Null);
		}
		if (auto const wanted = tolerated(tool_tolerated, name)) {
			return neutral(member, *wanted);
		}
		// Which members belong depends on the type, which may come later: decided after the loop.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return read;
}

/** A `tools` element: a function tool or a custom tool. */
[[nodiscard]] auto tool(Body& value) -> Decoded<Tool> {
	auto read = tool_draft(value);
	if (!read) {
		return std::unexpected(read.error());
	}
	if (!read->type) {
		return refuse_at("type", Violation::MissingField);
	}
	auto const function = *read->type == "function";
	if (!function && *read->type != "custom") {
		return refuse_type(unsupported_tool_types, *read->type);
	}
	auto const allowed = function ? std::span<std::string_view const>{function_tool_members} : std::span<std::string_view const>{custom_tool_members};
	for (auto const& name : read->seen) {
		if (std::ranges::find(allowed, name) == allowed.end()) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	if (!read->name) {
		return refuse_at("name", Violation::MissingField);
	}
	if (function) {
		if (!read->parameters) {
			return refuse_at("parameters", Violation::MissingField);
		}
		if (!read->strict) {
			return refuse_at("strict", Violation::MissingField);
		}
		return Tool{.name = std::move(*read->name), .description = std::move(read->description), .input = FunctionTool{.parameters = std::move(*read->parameters)}};
	}
	return Tool{
	    .name = std::move(*read->name),
	    .description = std::move(read->description),
	    .input = CustomTool{.grammar = read->format ? std::move(*read->format) : std::nullopt},
	};
}

/** `tool_choice`: a mode, or an object naming one tool. */
[[nodiscard]] auto tool_choice(Body& value) -> Decoded<ToolChoice> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::String) {
		auto const mode = text(value);
		if (!mode) {
			return std::unexpected(mode.error());
		}
		if (*mode == "none") {
			return ToolChoiceMode::None;
		}
		if (*mode == "auto") {
			return ToolChoiceMode::Auto;
		}
		if (*mode == "required") {
			return ToolChoiceMode::Required;
		}
		return refuse(Violation::UnknownValue);
	}
	if (*kind != Kind::Object) {
		return refuse(WrongShape{.expected = Expected::ToolChoice, .found = *kind});
	}
	auto type = std::optional<std::string>{};
	auto name = std::optional<std::string>{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(value, [&](std::string_view member, Body& inner) -> Decoded<void> {
		seen.emplace_back(member);
		if (member == "type") {
			return text(inner).transform([&type](std::string read) { type = std::move(read); });
		}
		if (member == "name") {
			return text(inner).transform([&name](std::string read) { name = std::move(read); });
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
	if (*type != "function" && *type != "custom") {
		return refuse_type(unsupported_choice_types, *type);
	}
	for (auto const& member : seen) {
		if (member != "type" && member != "name") {
			return refuse_at(member, Violation::UnknownField);
		}
	}
	if (!name) {
		return refuse_at("name", Violation::MissingField);
	}
	if (*type == "function") {
		return ForcedFunction{.name = std::move(*name)};
	}
	return ForcedCustom{.name = std::move(*name)};
}

}

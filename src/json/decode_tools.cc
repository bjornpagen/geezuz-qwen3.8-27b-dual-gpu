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
import :request;
import :decode_support;
import :decode_neutral;
import :decode_format;

namespace engine::json {

/** A function tool's `function` member. */
[[nodiscard]] auto function_tool(Fragment& value, std::optional<std::string>& name) -> Decoded<FunctionTool> {
	auto tool = FunctionTool{.parameters = std::nullopt};
	auto const visited = each_member(value, [&](std::string_view field, Fragment& member) -> Decoded<void> {
		if (field == "name") {
			return text(member).transform([&name](std::string read) { name = std::move(read); });
		}
		if (field == "description") {
			return nullable<std::string>(member, [](Fragment& description) { return text(description); }).transform([](auto const&) {});
		}
		if (field == "parameters") {
			return schema_of(member).transform([&tool](std::optional<JsonText> schema) { tool.parameters = std::move(schema); });
		}
		if (field == "strict") {
			return any_boolean(member);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return tool;
}

/** A custom tool's `custom` member. */
[[nodiscard]] auto custom_tool(Fragment& value, std::optional<std::string>& name) -> Decoded<CustomTool> {
	auto tool = CustomTool{.grammar = std::nullopt};
	auto const visited = each_member(value, [&](std::string_view field, Fragment& member) -> Decoded<void> {
		if (field == "name") {
			return text(member).transform([&name](std::string read) { name = std::move(read); });
		}
		if (field == "description") {
			return nullable<std::string>(member, [](Fragment& description) { return text(description); }).transform([](auto const&) {});
		}
		if (field == "format") {
			return custom_format(member).transform([&tool](std::optional<Grammar> read) { tool.grammar = std::move(read); });
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return tool;
}

/** A tools element: kept whole as written, then read again on its own for what the engine needs from it. */
[[nodiscard]] auto tool(Body& value, Fragments& fragments) -> Decoded<Tool> {
	auto definition = lift(value.capture());
	if (!definition) {
		return std::unexpected(definition.error());
	}
	auto root = lift(fragments.read_copy(definition->text()));
	if (!root) {
		return std::unexpected(root.error());
	}
	auto type = std::optional<std::string>{};
	auto function_name = std::optional<std::string>{};
	auto custom_name = std::optional<std::string>{};
	auto function = std::optional<FunctionTool>{};
	auto custom = std::optional<CustomTool>{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(*root, [&](std::string_view member, Fragment& inner) -> Decoded<void> {
		seen.emplace_back(member);
		if (member == "type") {
			return text(inner).transform([&type](std::string read) { type = std::move(read); });
		}
		if (member == "function") {
			return function_tool(inner, function_name).transform([&function](FunctionTool read) { function = std::move(read); });
		}
		if (member == "custom") {
			return custom_tool(inner, custom_name).transform([&custom](CustomTool read) { custom = std::move(read); });
		}
		// Which members belong depends on the type, which may come later: decided after the loop.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (auto const finished = lift(fragments.finish()); !finished) {
		return std::unexpected(finished.error());
	}
	if (!type) {
		return refuse_at("type", Violation::MissingField);
	}
	if (*type != "function" && *type != "custom") {
		return refuse_type(unsupported_tool_types, *type);
	}
	auto const& payload = *type;
	for (auto const& name : seen) {
		if (name != "type" && name != payload) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	auto const missing_name = [&payload] { return std::unexpected(DecodeError{.fault = Violation::MissingField, .path = {"name", payload}}); };
	if (payload == "function") {
		if (!function) {
			return refuse_at("function", Violation::MissingField);
		}
		if (!function_name) {
			return missing_name();
		}
		return Tool{.name = std::move(*function_name), .input = std::move(*function), .definition = std::move(*definition)};
	}
	if (!custom) {
		return refuse_at("custom", Violation::MissingField);
	}
	if (!custom_name) {
		return missing_name();
	}
	return Tool{.name = std::move(*custom_name), .input = std::move(*custom), .definition = std::move(*definition)};
}

[[nodiscard]] auto tool_choice(Body& value) -> Decoded<ToolChoice> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::String) {
		auto const mode = lift(value.as_string());
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
	auto function = std::optional<Named>{};
	auto other = std::optional<std::string>{};
	auto const visited = each_member(value, [&](std::string_view member, Body& inner) -> Decoded<void> {
		if (member == "type") {
			return text(inner).transform([&type](std::string read) { type = std::move(read); });
		}
		if (member != "function") {
			other = other.value_or(std::string{member});
			return {};
		}
		function.emplace();
		return each_member(inner, [&function](std::string_view field, Body& field_value) -> Decoded<void> {
			if (field == "name") {
				return text(field_value).transform([&named = *function](std::string read) { named.name = std::move(read); });
			}
			return refuse(Violation::UnknownField);
		});
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!type) {
		return refuse_at("type", Violation::MissingField);
	}
	if (*type != "function") {
		return refuse_type(unsupported_tool_types, *type);
	}
	if (other) {
		return refuse_at(*other, Violation::UnknownField);
	}
	if (!function) {
		return refuse_at("function", Violation::MissingField);
	}
	if (!function->name) {
		return std::unexpected(DecodeError{.fault = Violation::MissingField, .path = {"name", "function"}});
	}
	return ForcedFunction{.name = std::move(*function->name)};
}

}

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

namespace engine::json {

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
	auto parameters = std::optional<JsonText>{};
	auto function = std::optional<Named>{};
	auto other = std::optional<std::string>{};
	auto const visited = each_member(*root, [&](std::string_view member, Value<fragment_bounds>& inner) -> Decoded<void> {
		if (member == "type") {
			return text(inner).transform([&type](std::string read) { type = std::move(read); });
		}
		if (member != "function") {
			other = other.value_or(std::string{member});
			return {};
		}
		function.emplace();
		return each_member(inner, [&](std::string_view field, Value<fragment_bounds>& field_value) -> Decoded<void> {
			if (field == "name") {
				return text(field_value).transform([&named = *function](std::string read) { named.name = std::move(read); });
			}
			if (field == "description") {
				return nullable<std::string>(field_value, [](Value<fragment_bounds>& description) { return text(description); }).transform([](auto const&) {});
			}
			if (field == "parameters") {
				auto const kind = kind_of(field_value);
				if (!kind) {
					return std::unexpected(kind.error());
				}
				if (*kind == Kind::Null) {
					return lift(field_value.as_null());
				}
				if (*kind != Kind::Object) {
					return refuse(WrongShape{.expected = Expected::Object, .found = *kind});
				}
				return lift(field_value.capture()).transform([&parameters](JsonText schema) { parameters = std::move(schema); });
			}
			if (field == "strict") {
				return neutral(field_value, Neutral::False);
			}
			return refuse(Violation::UnknownField);
		});
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
	return Tool{.name = std::move(*function->name), .parameters = std::move(parameters), .definition = std::move(*definition)};
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

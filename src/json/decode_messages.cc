// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_messages;

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

/** Message content: a string, text parts, or null. */
[[nodiscard]] auto content(Body& value) -> Decoded<std::string> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	switch (*kind) {
	case Kind::Null:
		return lift(value.as_null()).transform([] { return std::string{}; });
	case Kind::String:
		return text(value);
	case Kind::Array: {
		auto joined = std::string{};
		auto const parts = each_element(value, [&joined](std::size_t, Body& part) -> Decoded<void> {
			auto type = std::optional<std::string>{};
			auto part_text = std::optional<std::string>{};
			auto other = std::optional<std::string>{};
			auto const visited = each_member(part, [&](std::string_view name, Body& member) -> Decoded<void> {
				if (name == "type") {
					return text(member).transform([&type](std::string read) { type = std::move(read); });
				}
				if (name == "text") {
					return text(member).transform([&part_text](std::string read) { part_text = std::move(read); });
				}
				// The type decides whether this member is the part's payload or a stray: decided after the loop.
				other = other.value_or(std::string{name});
				return {};
			});
			if (!visited) {
				return std::unexpected(visited.error());
			}
			if (!type) {
				return refuse_at("type", Violation::MissingField);
			}
			if (*type != "text") {
				return refuse_type(unsupported_part_types, *type);
			}
			if (other) {
				return refuse_at(*other, Violation::UnknownField);
			}
			if (!part_text) {
				return refuse_at("text", Violation::MissingField);
			}
			joined += *part_text;
			return {};
		});
		if (!parts) {
			return std::unexpected(parts.error());
		}
		return joined;
	}
	case Kind::Boolean:
	case Kind::Integer:
	case Kind::Real:
	case Kind::Object:
		return refuse(WrongShape{.expected = Expected::Content, .found = *kind});
	}
	std::unreachable();
}

/** The members of an arguments object, in order: strings as their text, everything else as its JSON. */
template<Bounds B>
[[nodiscard]] auto named_arguments(Value<B>& object) -> Decoded<std::vector<Argument>> {
	auto arguments = std::vector<Argument>{};
	auto const visited = each_member(object, [&arguments](std::string_view name, Value<B>& value) -> Decoded<void> {
		auto const kind = kind_of(value);
		if (!kind) {
			return std::unexpected(kind.error());
		}
		if (*kind == Kind::String) {
			return text(value).transform([&](std::string read) { arguments.push_back(Argument{.name = std::string{name}, .value = std::move(read)}); });
		}
		return lift(value.capture()).transform([&](JsonText json) {
			arguments.push_back(Argument{.name = std::string{name}, .value = std::move(json)});
		});
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return arguments;
}

[[nodiscard]] auto arguments(Body& value, Fragments& fragments) -> Decoded<std::vector<Argument>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null()).transform([] { return std::vector<Argument>{}; });
	}
	if (*kind == Kind::Object) {
		return named_arguments(value);
	}
	if (*kind != Kind::String) {
		return refuse(WrongShape{.expected = Expected::Arguments, .found = *kind});
	}
	auto const held = lift(value.as_string());
	if (!held) {
		return std::unexpected(held.error());
	}
	// The chat template reads an empty arguments string as a call without arguments.
	if (held->empty()) {
		return std::vector<Argument>{};
	}
	auto root = lift(fragments.read_copy(*held));
	if (!root) {
		return std::unexpected(root.error());
	}
	auto const root_kind = kind_of(*root);
	if (!root_kind) {
		return std::unexpected(root_kind.error());
	}
	if (*root_kind != Kind::Object) {
		return refuse(WrongShape{.expected = Expected::Arguments, .found = *root_kind});
	}
	auto named = named_arguments(*root);
	if (!named) {
		return named;
	}
	if (auto const finished = lift(fragments.finish()); !finished) {
		return std::unexpected(finished.error());
	}
	return named;
}

[[nodiscard]] auto tool_call(Body& value, Fragments& fragments) -> Decoded<ToolCall> {
	auto id = std::optional<std::string>{};
	auto type = std::optional<std::string>{};
	auto call_arguments = std::vector<Argument>{};
	auto function = std::optional<Named>{};
	auto other = std::optional<std::string>{};
	auto const visited = each_member(value, [&](std::string_view member, Body& inner) -> Decoded<void> {
		if (member == "id") {
			return text(inner).transform([&id](std::string read) { id = std::move(read); });
		}
		if (member == "type") {
			return text(inner).transform([&type](std::string read) { type = std::move(read); });
		}
		if (member == "index") {
			return neutral(inner, Neutral::AnyInteger);
		}
		if (member != "function") {
			other = other.value_or(std::string{member});
			return {};
		}
		function.emplace();
		return each_member(inner, [&](std::string_view field, Body& field_value) -> Decoded<void> {
			if (field == "name") {
				return text(field_value).transform([&named = *function](std::string read) { named.name = std::move(read); });
			}
			if (field == "arguments") {
				return arguments(field_value, fragments).transform([&call_arguments](std::vector<Argument> read) { call_arguments = std::move(read); });
			}
			return refuse(Violation::UnknownField);
		});
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (type && *type != "function") {
		return refuse_type(unsupported_tool_types, *type);
	}
	if (other) {
		return refuse_at(*other, Violation::UnknownField);
	}
	if (!id) {
		return refuse_at("id", Violation::MissingField);
	}
	if (!function) {
		return refuse_at("function", Violation::MissingField);
	}
	if (!function->name) {
		return std::unexpected(DecodeError{.fault = Violation::MissingField, .path = {"name", "function"}});
	}
	return ToolCall{.id = std::move(*id), .name = std::move(*function->name), .arguments = std::move(call_arguments)};
}

/** The member names each role defines; any other member of a message is refused at its name. */
constexpr auto plain_fields = std::to_array<std::string_view>({"role", "content", "name"});
constexpr auto assistant_fields =
    std::to_array<std::string_view>({"role", "content", "name", "reasoning_content", "tool_calls", "refusal", "audio", "function_call", "annotations"});
constexpr auto tool_fields = std::to_array<std::string_view>({"role", "content", "tool_call_id"});

[[nodiscard]] auto fields_of(std::string_view role) -> std::optional<std::span<std::string_view const>> {
	if (role == "system" || role == "developer" || role == "user") {
		return plain_fields;
	}
	if (role == "assistant") {
		return assistant_fields;
	}
	if (role == "tool") {
		return tool_fields;
	}
	return std::nullopt;
}

[[nodiscard]] auto message(Body& value, Fragments& fragments) -> Decoded<Message> {
	auto role = std::optional<std::string>{};
	auto body = std::string{};
	auto reasoning = std::string{};
	auto calls = std::vector<ToolCall>{};
	auto call_id = std::optional<std::string>{};
	auto seen = std::vector<std::string>{};
	auto const visited = each_member(value, [&](std::string_view name, Body& member) -> Decoded<void> {
		seen.emplace_back(name);
		if (name == "role") {
			return text(member).transform([&role](std::string read) { role = std::move(read); });
		}
		if (name == "content") {
			return content(member).transform([&body](std::string read) { body = std::move(read); });
		}
		if (name == "name") {
			return neutral(member, Neutral::AnyString);
		}
		if (name == "reasoning_content") {
			return nullable<std::string>(member, [](Body& inner) { return text(inner); }).transform([&reasoning](std::optional<std::string> read) {
				reasoning = std::move(read).value_or(std::string{});
			});
		}
		if (name == "tool_calls") {
			return nullable<std::vector<ToolCall>>(member, [&fragments](Body& inner) -> Decoded<std::vector<ToolCall>> {
				       auto read = std::vector<ToolCall>{};
				       auto const elements = each_element(inner, [&](std::size_t, Body& element) -> Decoded<void> {
					       return tool_call(element, fragments).transform([&read](ToolCall call) { read.push_back(std::move(call)); });
				       });
				       if (!elements) {
					       return std::unexpected(elements.error());
				       }
				       return read;
			       })
			    .transform([&calls](std::optional<std::vector<ToolCall>> read) { calls = std::move(read).value_or(std::vector<ToolCall>{}); });
		}
		if (name == "tool_call_id") {
			return text(member).transform([&call_id](std::string read) { call_id = std::move(read); });
		}
		if (auto const echoed = tolerated(assistant_tolerated, name)) {
			return neutral(member, *echoed);
		}
		// Whether this member is defined depends on the role, which may come later: decided after the loop.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!role) {
		return refuse_at("role", Violation::MissingField);
	}
	auto const fields = fields_of(*role);
	if (!fields) {
		return refuse_at("role", *role == "function" ? Violation::Unsupported : Violation::UnknownValue);
	}
	for (auto const& name : seen) {
		if (std::ranges::find(*fields, name) == fields->end()) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	if (*role == "assistant") {
		return AssistantMessage{.content = std::move(body), .reasoning_content = std::move(reasoning), .tool_calls = std::move(calls)};
	}
	if (*role == "tool") {
		if (!call_id) {
			return refuse_at("tool_call_id", Violation::MissingField);
		}
		return ToolMessage{.tool_call_id = std::move(*call_id), .content = std::move(body)};
	}
	if (*role == "user") {
		return UserMessage{.content = std::move(body)};
	}
	return SystemMessage{.content = std::move(body)};
}

}

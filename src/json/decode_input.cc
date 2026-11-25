// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_input;

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

/** The members one kind of item defines beyond those it requires. */
using Allowed = std::span<std::string_view const>;

constexpr auto message_members = std::to_array<std::string_view>({"type", "role", "content", "id", "status", "phase"});
constexpr auto function_call_members = std::to_array<std::string_view>({"type", "call_id", "name", "arguments", "id", "status", "caller", "namespace"});
constexpr auto custom_call_members = std::to_array<std::string_view>({"type", "call_id", "name", "input", "id", "caller", "namespace"});
constexpr auto output_members = std::to_array<std::string_view>({"type", "call_id", "output", "id", "status", "caller", "name", "namespace"});
constexpr auto custom_output_members = std::to_array<std::string_view>({"type", "call_id", "output", "id", "caller"});
constexpr auto reasoning_members = std::to_array<std::string_view>({"type", "id", "summary", "content", "encrypted_content", "status"});

/** The part types a content array may hold, by where it stands. */
enum class Parts : std::uint8_t {
	/** A user, system or developer message, or a call's output: `input_text`. */
	Input,
	/** An assistant message resent: `output_text`, or `input_text` as the easy message form spells it. */
	Assistant,
	/** A reasoning item: `reasoning_text`. */
	Reasoning,
};

/** One content part as read, before the item's kind says which types and members may stand in it. */
struct PartDraft {
	std::optional<std::string> type;
	std::optional<std::string> text;
	std::vector<std::string> seen;
};

/** Content as written: a string, or parts. */
using ContentDraft = std::variant<std::string, std::vector<PartDraft>>;

[[nodiscard]] auto part_draft(Body& value) -> Decoded<PartDraft> {
	auto read = PartDraft{};
	auto const visited = each_member(value, [&read](std::string_view name, Body& member) -> Decoded<void> {
		read.seen.emplace_back(name);
		if (name == "type") {
			return text(member).transform([&read](std::string decoded) { read.type = std::move(decoded); });
		}
		if (name == "text") {
			return text(member).transform([&read](std::string decoded) { read.text = std::move(decoded); });
		}
		if (name == "prompt_cache_breakpoint") {
			return neutral(member, Neutral::Null);
		}
		if (name == "annotations" || name == "logprobs") {
			return neutral(member, Neutral::EmptyArray);
		}
		// Which members belong depends on the type and the item: decided once both are known.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return read;
}

[[nodiscard]] auto content_draft(Body& value) -> Decoded<ContentDraft> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::String) {
		return text(value).transform([](std::string read) { return ContentDraft{std::move(read)}; });
	}
	if (*kind != Kind::Array) {
		return refuse(WrongShape{.expected = Expected::Content, .found = *kind});
	}
	auto parts = std::vector<PartDraft>{};
	auto const visited = each_element(value, [&parts](std::size_t, Body& part) -> Decoded<void> {
		return part_draft(part).transform([&parts](PartDraft read) { parts.push_back(std::move(read)); });
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return ContentDraft{std::move(parts)};
}

/** A refusal inside part `index` of the member `member`, innermost first as the decoder collects paths. */
[[nodiscard]] auto refuse_in_part(std::string_view member, std::size_t index, std::string_view name, DecodeFault fault) -> std::unexpected<DecodeError> {
	return std::unexpected(DecodeError{.fault = std::move(fault), .path = {std::string{name}, index, std::string{member}}});
}

/** Content checked against what may stand where it is, read as the concatenation of its parts. */
[[nodiscard]] auto join(ContentDraft draft, Parts parts, std::string_view member) -> Decoded<std::string> {
	if (auto const whole = core::alternative<std::string>(draft)) {
		// A string is a message's or an output's whole text; a reasoning item has only parts.
		if (parts == Parts::Reasoning) {
			return std::unexpected(DecodeError{.fault = WrongShape{.expected = Expected::Array, .found = Kind::String}, .path = {std::string{member}}});
		}
		return std::move(*whole);
	}
	auto joined = std::string{};
	auto& drafts = std::get<std::vector<PartDraft>>(draft);
	for (auto index = std::size_t{0}; index < drafts.size(); ++index) {
		auto& part = drafts[index];
		if (!part.type) {
			return refuse_in_part(member, index, "type", Violation::MissingField);
		}
		auto const& type = *part.type;
		auto const fits = (type == "input_text" && parts != Parts::Reasoning) || (type == "output_text" && parts == Parts::Assistant) ||
		                  (type == "reasoning_text" && parts == Parts::Reasoning);
		if (!fits) {
			auto const defined = std::ranges::contains(unsupported_part_types, type);
			return refuse_in_part(member, index, "type", defined ? Violation::Unsupported : Violation::UnknownValue);
		}
		for (auto const& name : part.seen) {
			auto const known = name == "type" || name == "text" || (type == "input_text" && name == "prompt_cache_breakpoint") ||
			                   (type == "output_text" && (name == "annotations" || name == "logprobs"));
			if (!known) {
				return refuse_in_part(member, index, name, Violation::UnknownField);
			}
		}
		if (!part.text) {
			return refuse_in_part(member, index, "text", Violation::MissingField);
		}
		joined += *part.text;
	}
	return joined;
}

/** A call's arguments: a string holding one JSON object, kept as written. */
[[nodiscard]] auto arguments(Body& value, Fragments& fragments) -> Decoded<JsonText> {
	auto const written = text(value);
	if (!written) {
		return std::unexpected(written.error());
	}
	auto root = lift(fragments.read_copy(*written));
	if (!root) {
		return std::unexpected(root.error());
	}
	auto const kind = kind_of(*root);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind != Kind::Object) {
		return refuse(WrongShape{.expected = Expected::Object, .found = *kind});
	}
	auto held = lift(root->capture());
	if (!held) {
		return held;
	}
	if (auto const finished = lift(fragments.finish()); !finished) {
		return std::unexpected(finished.error());
	}
	return held;
}

[[nodiscard]] auto role_of(std::string_view role) -> std::optional<Role> {
	if (role == "user") {
		return Role::User;
	}
	if (role == "assistant") {
		return Role::Assistant;
	}
	if (role == "system") {
		return Role::System;
	}
	if (role == "developer") {
		return Role::Developer;
	}
	return std::nullopt;
}

/** Every member an item has, read once, before its type says which of them belong. */
struct Draft {
	std::optional<std::string> type;
	std::optional<std::string> role;
	std::optional<ContentDraft> content;
	std::optional<std::string> call_id;
	std::optional<std::string> name;
	std::optional<JsonText> arguments;
	std::optional<std::string> input;
	std::optional<ContentDraft> output;
	std::vector<std::string> seen;
};

[[nodiscard]] auto draft(Body& value, Fragments& fragments) -> Decoded<Draft> {
	auto read = Draft{};
	auto const visited = each_member(value, [&](std::string_view name, Body& member) -> Decoded<void> {
		read.seen.emplace_back(name);
		auto const assign = [](std::optional<std::string>& field) {
			return [&field](std::string decoded) { field = std::move(decoded); };
		};
		if (name == "type") {
			return text(member).transform(assign(read.type));
		}
		if (name == "role") {
			return text(member).transform(assign(read.role));
		}
		if (name == "call_id") {
			return text(member).transform(assign(read.call_id));
		}
		if (name == "name") {
			return nullable<std::string>(member, [](Body& inner) { return text(inner); }).transform([&read](std::optional<std::string> decoded) {
				read.name = std::move(decoded);
			});
		}
		if (name == "arguments") {
			return arguments(member, fragments).transform([&read](JsonText decoded) { read.arguments = std::move(decoded); });
		}
		if (name == "input") {
			return text(member).transform(assign(read.input));
		}
		if (name == "output") {
			return content_draft(member).transform([&read](ContentDraft decoded) { read.output = std::move(decoded); });
		}
		if (name == "content") {
			return content_draft(member).transform([&read](ContentDraft decoded) { read.content = std::move(decoded); });
		}
		if (name == "summary") {
			return neutral(member, Neutral::EmptyArray);
		}
		if (name == "encrypted_content") {
			return neutral(member, Neutral::Null);
		}
		if (auto const echoed = tolerated(item_tolerated, name)) {
			return neutral(member, *echoed);
		}
		// Which members belong depends on the type, which may come later: decided once it is known.
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return read;
}

[[nodiscard]] auto require(std::optional<std::string>& field, std::string_view name) -> Decoded<std::string> {
	if (!field) {
		return refuse_at(name, Violation::MissingField);
	}
	return std::move(*field);
}

[[nodiscard]] auto only(Draft const& read, Allowed allowed) -> Decoded<void> {
	for (auto const& name : read.seen) {
		if (std::ranges::find(allowed, name) == allowed.end()) {
			return refuse_at(name, Violation::UnknownField);
		}
	}
	return {};
}

[[nodiscard]] auto message(Draft read) -> Decoded<InputItem> {
	if (auto const checked = only(read, message_members); !checked) {
		return std::unexpected(checked.error());
	}
	auto const role_text = require(read.role, "role");
	if (!role_text) {
		return std::unexpected(role_text.error());
	}
	auto const role = role_of(*role_text);
	if (!role) {
		return refuse_at("role", Violation::UnknownValue);
	}
	if (!read.content) {
		return refuse_at("content", Violation::MissingField);
	}
	auto body = join(std::move(*read.content), *role == Role::Assistant ? Parts::Assistant : Parts::Input, "content");
	if (!body) {
		return std::unexpected(body.error());
	}
	return MessageItem{.role = *role, .text = *std::move(body)};
}

[[nodiscard]] auto reasoning(Draft read) -> Decoded<InputItem> {
	if (auto const checked = only(read, reasoning_members); !checked) {
		return std::unexpected(checked.error());
	}
	if (!read.content) {
		return ReasoningItem{.text = {}};
	}
	return join(std::move(*read.content), Parts::Reasoning, "content").transform([](std::string text) { return InputItem{ReasoningItem{.text = std::move(text)}}; });
}

/** One input item, by its type; a message may leave its type out. */
[[nodiscard]] auto input_item(Body& value, Fragments& fragments) -> Decoded<InputItem> {
	auto read = draft(value, fragments);
	if (!read) {
		return std::unexpected(read.error());
	}
	auto const type = read->type.value_or(read->role ? "message" : "");
	if (type.empty()) {
		return refuse_at("type", Violation::MissingField);
	}
	if (type == "message") {
		return message(*std::move(read));
	}
	if (type == "reasoning") {
		return reasoning(*std::move(read));
	}
	if (type == "function_call") {
		if (auto const checked = only(*read, function_call_members); !checked) {
			return std::unexpected(checked.error());
		}
		auto call_id = require(read->call_id, "call_id");
		auto name = require(read->name, "name");
		if (!call_id || !name) {
			return std::unexpected(!call_id ? call_id.error() : name.error());
		}
		if (!read->arguments) {
			return refuse_at("arguments", Violation::MissingField);
		}
		return FunctionCallItem{.call_id = *std::move(call_id), .name = *std::move(name), .arguments = std::move(*read->arguments)};
	}
	if (type == "function_call_output" || type == "custom_tool_call_output") {
		auto const custom = type == "custom_tool_call_output";
		if (auto const checked = only(*read, custom ? Allowed{custom_output_members} : Allowed{output_members}); !checked) {
			return std::unexpected(checked.error());
		}
		auto call_id = require(read->call_id, "call_id");
		if (!call_id) {
			return std::unexpected(call_id.error());
		}
		if (!read->output) {
			return refuse_at("output", Violation::MissingField);
		}
		auto output = join(std::move(*read->output), Parts::Input, "output");
		if (!output) {
			return std::unexpected(output.error());
		}
		if (custom) {
			return CustomOutputItem{.call_id = *std::move(call_id), .output = *std::move(output)};
		}
		return FunctionOutputItem{.call_id = *std::move(call_id), .output = *std::move(output)};
	}
	if (type == "custom_tool_call") {
		if (auto const checked = only(*read, custom_call_members); !checked) {
			return std::unexpected(checked.error());
		}
		auto call_id = require(read->call_id, "call_id");
		auto name = require(read->name, "name");
		auto input = require(read->input, "input");
		if (!call_id || !name || !input) {
			return std::unexpected(!call_id ? call_id.error() : !name ? name.error() : input.error());
		}
		return CustomCallItem{.call_id = *std::move(call_id), .name = *std::move(name), .input = *std::move(input)};
	}
	return refuse_type(unsupported_item_types, type);
}

/** `input`: a string is one user message; an array is items in order. */
[[nodiscard]] auto input(Body& value, Fragments& fragments) -> Decoded<std::vector<InputItem>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::String) {
		return text(value).transform([](std::string read) { return std::vector<InputItem>{MessageItem{.role = Role::User, .text = std::move(read)}}; });
	}
	if (*kind != Kind::Array) {
		return refuse(WrongShape{.expected = Expected::Input, .found = *kind});
	}
	auto items = std::vector<InputItem>{};
	auto const visited = each_element(value, [&](std::size_t, Body& element) -> Decoded<void> {
		return input_item(element, fragments).transform([&items](InputItem item) { items.push_back(std::move(item)); });
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return items;
}

}

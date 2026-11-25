// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode;

import std;
import engine.core;
import :bounds;
import :error;
import :read;
import :request;

namespace engine::json {

/**
 * The bounds of a JSON text a request carries inside itself: a tool call's arguments sent as a string, and a tool
 * definition read again on its own. The first is a string of the request, so no such text is longer than one.
 */
export inline constexpr Bounds fragment_bounds{
    .document_bytes = request_bounds.string_bytes,
    .depth = request_bounds.depth,
    .string_bytes = request_bounds.string_bytes,
    .members = request_bounds.members,
    .number_chars = request_bounds.number_chars,
};

static_assert(Enforceable<fragment_bounds>);

using Body = Value<request_bounds>;
using Fragments = Reader<fragment_bounds>;

template<class T>
using Decoded = core::Result<T, DecodeError>;

/** A `function` member that was present, and the name it gave, if any. */
struct Named {
	std::optional<std::string> name;
};

[[nodiscard]] auto refuse(DecodeFault fault) -> std::unexpected<DecodeError> {
	return std::unexpected(DecodeError{.fault = std::move(fault), .path = {}});
}

/** Refuses the member `name` of the object being decoded. */
[[nodiscard]] auto refuse_at(std::string_view name, DecodeFault fault) -> std::unexpected<DecodeError> {
	return std::unexpected(DecodeError{.fault = std::move(fault), .path = {std::string{name}}});
}

[[nodiscard]] constexpr auto expected_of(Kind kind) -> Expected {
	switch (kind) {
	case Kind::Boolean:
		return Expected::Boolean;
	case Kind::Integer:
		return Expected::Integer;
	case Kind::Real:
		return Expected::Number;
	case Kind::String:
		return Expected::String;
	case Kind::Array:
		return Expected::Array;
	case Kind::Null:
	case Kind::Object:
		return Expected::Object;
	}
	std::unreachable();
}

/** A failed read in the decoder's terms. */
[[nodiscard]] auto decode_error(ReadError const& error) -> DecodeError {
	auto const& cause = error.cause;
	if (std::holds_alternative<KindMismatch>(cause)) {
		auto const& mismatch = std::get<KindMismatch>(cause);
		return DecodeError{.fault = WrongShape{.expected = expected_of(mismatch.expected), .found = mismatch.found}, .path = {}};
	}
	if (std::holds_alternative<MissingMember>(cause)) {
		return DecodeError{.fault = Violation::MissingField, .path = {std::get<MissingMember>(cause).name}};
	}
	return DecodeError{.fault = std::get<ParseError>(cause), .path = {}};
}

/** Lifts a read into the decoder's result. */
template<class T>
[[nodiscard]] auto lift(core::Result<T, ReadError> read) -> Decoded<T> {
	if (!read) {
		return std::unexpected(decode_error(read.error()));
	}
	if constexpr (std::is_void_v<T>) {
		return {};
	} else {
		return *std::move(read);
	}
}

/** Paths are collected innermost first while an error unwinds, and reversed once at the top. */
[[nodiscard]] auto within(PathStep step) {
	return [outer = std::move(step)](DecodeError error) {
		error.path.push_back(outer);
		return error;
	};
}

template<Bounds B>
[[nodiscard]] auto kind_of(Value<B>& value) -> Decoded<Kind> {
	return lift(value.kind());
}

template<Bounds B>
[[nodiscard]] auto text(Value<B>& value) -> Decoded<std::string> {
	return lift(value.as_string()).transform([](std::string_view read) { return std::string{read}; });
}

/** Reads `value` with `read`, or nothing when it is `null`: every optional member may be sent as `null`. */
template<class T, Bounds B, class F>
[[nodiscard]] auto nullable(Value<B>& value, F const& read) -> Decoded<std::optional<T>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null()).transform([] { return std::optional<T>{}; });
	}
	return read(value).transform([](T decoded) { return std::optional<T>{std::move(decoded)}; });
}

/** Visits every member of an object in document order; an error from a member is placed at its name. */
template<Bounds B, class F>
[[nodiscard]] auto each_member(Value<B>& value, F const& visit) -> Decoded<void> {
	auto object = lift(value.as_object());
	if (!object) {
		return std::unexpected(object.error());
	}
	while (true) {
		auto member = lift(object->next());
		if (!member) {
			return std::unexpected(member.error());
		}
		if (!*member) {
			return {};
		}
		auto& [name, inner] = **member;
		if (auto const visited = visit(name, inner); !visited) {
			return std::unexpected(within(std::string{name})(visited.error()));
		}
	}
}

/** Visits every element of an array in order; an error from an element is placed at its index. */
template<Bounds B, class F>
[[nodiscard]] auto each_element(Value<B>& value, F const& visit) -> Decoded<void> {
	auto array = lift(value.as_array());
	if (!array) {
		return std::unexpected(array.error());
	}
	for (auto index = std::size_t{0};; ++index) {
		auto element = lift(array->next());
		if (!element) {
			return std::unexpected(element.error());
		}
		if (!*element) {
			return {};
		}
		if (auto const visited = visit(index, **element); !visited) {
			return std::unexpected(within(index)(visited.error()));
		}
	}
}

/**
 * The value an OpenAI parameter the engine does not implement may carry: the one that leaves the output as the
 * engine produces it. Any other value is refused as unsupported, so no request is served differently from what it
 * asked for.
 */
enum class Neutral : std::uint8_t {
	/** Only `null`: the parameter has no value the engine honours. */
	Null,
	/** Any string: an identifier for the client's own accounting (`user`, `prompt_cache_key`). */
	AnyString,
	/** Any integer: a streaming client's leftover `index` on a resent tool call. */
	AnyInteger,
	One,
	Zero,
	False,
	True,
	EmptyObject,
	EmptyArray,
	/** `{"type": "text"}`: plain text output. */
	TextFormat,
	/** `["text"]`: text output only. */
	TextModalities,
};

struct Tolerated {
	std::string_view name;
	Neutral neutral;
};

template<Bounds B>
[[nodiscard]] auto number(Value<B>& value) -> Decoded<double> {
	return lift(value.as_real());
}

template<Bounds B>
[[nodiscard]] auto is_value(Value<B>& value, double wanted) -> Decoded<bool> {
	return number(value).transform([wanted](double read) { return read == wanted; });
}

template<Bounds B>
[[nodiscard]] auto is_boolean(Value<B>& value, bool wanted) -> Decoded<bool> {
	return lift(value.as_boolean()).transform([wanted](bool read) { return read == wanted; });
}

/** Whether a container holds nothing; reads it whole either way. */
template<Bounds B>
[[nodiscard]] auto is_empty(Value<B>& value, Kind container) -> Decoded<bool> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind != container) {
		return refuse(WrongShape{.expected = expected_of(container), .found = *kind});
	}
	auto empty = true;
	auto const visited = container == Kind::Array ? each_element(value, [&empty](std::size_t, Value<B>&) -> Decoded<void> {
		empty = false;
		return {};
	})
	                                               : each_member(value, [&empty](std::string_view, Value<B>&) -> Decoded<void> {
		                                                 empty = false;
		                                                 return {};
	                                                 });
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return empty;
}

template<Bounds B>
[[nodiscard]] auto is_text_format(Value<B>& value) -> Decoded<bool> {
	auto plain = false;
	auto other = false;
	auto const visited = each_member(value, [&](std::string_view name, Value<B>& member) -> Decoded<void> {
		if (name != "type") {
			other = true;
			return {};
		}
		return text(member).transform([&plain](std::string const& type) { plain = type == "text"; });
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return plain && !other;
}

template<Bounds B>
[[nodiscard]] auto is_text_modalities(Value<B>& value) -> Decoded<bool> {
	auto texts = std::size_t{0};
	auto other = false;
	auto const visited = each_element(value, [&](std::size_t, Value<B>& element) -> Decoded<void> {
		return text(element).transform([&](std::string const& modality) {
			texts += modality == "text" ? 1 : 0;
			other = other || modality != "text";
		});
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return texts > 0 && !other;
}

/** Reads a tolerated parameter whole and refuses any value but its neutral one. */
template<Bounds B>
[[nodiscard]] auto neutral(Value<B>& value, Neutral wanted) -> Decoded<void> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	if (*kind == Kind::Null) {
		return lift(value.as_null());
	}
	auto const holds = [&] -> Decoded<bool> {
		switch (wanted) {
		case Neutral::Null:
			return false;
		case Neutral::AnyString:
			return text(value).transform([](std::string const&) { return true; });
		case Neutral::AnyInteger:
			return lift(value.as_integer()).transform([](std::int64_t) { return true; });
		case Neutral::One:
			return is_value(value, 1.0);
		case Neutral::Zero:
			return is_value(value, 0.0);
		case Neutral::False:
			return is_boolean(value, false);
		case Neutral::True:
			return is_boolean(value, true);
		case Neutral::EmptyObject:
			return is_empty(value, Kind::Object);
		case Neutral::EmptyArray:
			return is_empty(value, Kind::Array);
		case Neutral::TextFormat:
			return *kind == Kind::Object ? is_text_format(value) : Decoded<bool>{false};
		case Neutral::TextModalities:
			return *kind == Kind::Array ? is_text_modalities(value) : Decoded<bool>{false};
		}
		std::unreachable();
	}();
	if (!holds) {
		return std::unexpected(holds.error());
	}
	if (!*holds) {
		return refuse(Violation::Unsupported);
	}
	return {};
}

template<std::size_t N>
[[nodiscard]] auto tolerated(std::array<Tolerated, N> const& table, std::string_view name) -> std::optional<Neutral> {
	auto const found = std::ranges::find(table, name, &Tolerated::name);
	if (found == table.end()) {
		return std::nullopt;
	}
	return found->neutral;
}

/** The OpenAI parameters the engine accepts only with their neutral value, and the identifiers it ignores. */
constexpr auto request_tolerated = std::to_array<Tolerated>({
    {.name = "user", .neutral = Neutral::AnyString},
    {.name = "safety_identifier", .neutral = Neutral::AnyString},
    {.name = "prompt_cache_key", .neutral = Neutral::AnyString},
    {.name = "n", .neutral = Neutral::One},
    {.name = "presence_penalty", .neutral = Neutral::Zero},
    {.name = "frequency_penalty", .neutral = Neutral::Zero},
    {.name = "logprobs", .neutral = Neutral::False},
    {.name = "top_logprobs", .neutral = Neutral::Null},
    {.name = "logit_bias", .neutral = Neutral::EmptyObject},
    {.name = "parallel_tool_calls", .neutral = Neutral::True},
    {.name = "response_format", .neutral = Neutral::TextFormat},
    {.name = "store", .neutral = Neutral::False},
    {.name = "metadata", .neutral = Neutral::Null},
    {.name = "modalities", .neutral = Neutral::TextModalities},
    {.name = "service_tier", .neutral = Neutral::Null},
    {.name = "audio", .neutral = Neutral::Null},
    {.name = "prediction", .neutral = Neutral::Null},
    {.name = "web_search_options", .neutral = Neutral::Null},
    {.name = "functions", .neutral = Neutral::Null},
    {.name = "function_call", .neutral = Neutral::Null},
    {.name = "reasoning_effort", .neutral = Neutral::Null},
    {.name = "verbosity", .neutral = Neutral::Null},
});

/** What a client echoes back from an OpenAI response's assistant message: accepted when empty. */
constexpr auto assistant_tolerated = std::to_array<Tolerated>({
    {.name = "refusal", .neutral = Neutral::Null},
    {.name = "audio", .neutral = Neutral::Null},
    {.name = "function_call", .neutral = Neutral::Null},
    {.name = "annotations", .neutral = Neutral::EmptyArray},
});

/** Content part types OpenAI defines that the engine, text only, does not serve. */
constexpr auto unsupported_part_types = std::to_array<std::string_view>({"image_url", "input_audio", "file", "refusal", "image", "video"});

/** Tool and tool-choice types OpenAI defines beyond functions. */
constexpr auto unsupported_tool_types = std::to_array<std::string_view>({"custom", "allowed_tools", "web_search", "file_search", "code_interpreter"});

/** Refuses a `type` the decoder does not take: unsupported when OpenAI defines it, unknown otherwise. */
template<std::size_t N>
[[nodiscard]] auto refuse_type(std::array<std::string_view, N> const& defined, std::string_view type) -> std::unexpected<DecodeError> {
	return refuse_at("type", std::ranges::find(defined, type) != defined.end() ? Violation::Unsupported : Violation::UnknownValue);
}

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
struct RoleFields {
	std::string_view role;
	std::span<std::string_view const> fields;
};

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

[[nodiscard]] auto stop(Body& value) -> Decoded<std::vector<std::string>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	switch (*kind) {
	case Kind::Null:
		return lift(value.as_null()).transform([] { return std::vector<std::string>{}; });
	case Kind::String:
		return text(value).transform([](std::string read) { return std::vector<std::string>{std::move(read)}; });
	case Kind::Array: {
		auto sequences = std::vector<std::string>{};
		auto const visited = each_element(value, [&sequences](std::size_t index, Body& element) -> Decoded<void> {
			if (index == max_stop_sequences) {
				return refuse(TooMany{.limit = max_stop_sequences});
			}
			return text(element).transform([&sequences](std::string read) { sequences.push_back(std::move(read)); });
		});
		if (!visited) {
			// Too many is a fault of the whole list, not of its last element.
			auto error = visited.error();
			if (std::holds_alternative<TooMany>(error.fault)) {
				error.path.clear();
			}
			return std::unexpected(std::move(error));
		}
		return sequences;
	}
	case Kind::Boolean:
	case Kind::Integer:
	case Kind::Real:
	case Kind::Object:
		return refuse(WrongShape{.expected = Expected::Stop, .found = *kind});
	}
	std::unreachable();
}

/** `stream_options.include_usage`, read from a present `stream_options`. */
[[nodiscard]] auto include_usage(Body& value) -> Decoded<bool> {
	auto include = std::optional<bool>{};
	auto const visited = each_member(value, [&include](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "include_usage") {
			return nullable<bool>(member, [](Body& flag) { return lift(flag.as_boolean()); }).transform([&include](std::optional<bool> read) {
				include.emplace(read.value_or(false));
			});
		}
		if (name == "include_obfuscation") {
			return neutral(member, Neutral::False);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return include.value_or(false);
}

[[nodiscard]] auto template_options(Body& value) -> Decoded<TemplateOptions> {
	auto options = TemplateOptions{};
	auto const visited = each_member(value, [&options](std::string_view name, Body& member) -> Decoded<void> {
		auto const flag = [&member] { return nullable<bool>(member, [](Body& inner) { return lift(inner.as_boolean()); }); };
		if (name == "enable_thinking") {
			return flag().transform([&options](std::optional<bool> read) {
				options.thinking = read.transform([](bool on) { return on ? Thinking::Enabled : Thinking::Disabled; });
			});
		}
		if (name == "preserve_thinking") {
			return flag().transform([&options](std::optional<bool> read) {
				options.past_thinking = read.transform([](bool on) { return on ? PastThinking::Preserved : PastThinking::Dropped; });
			});
		}
		if (name == "reasoning_effort") {
			auto const effort = nullable<std::string>(member, [](Body& inner) { return text(inner); });
			if (!effort) {
				return std::unexpected(effort.error());
			}
			if (!*effort) {
				return {};
			}
			if (**effort == "low") {
				options.reasoning_effort = ReasoningEffort::Low;
			} else if (**effort == "medium") {
				options.reasoning_effort = ReasoningEffort::Medium;
			} else if (**effort == "xhigh") {
				options.reasoning_effort = ReasoningEffort::XHigh;
			} else {
				return refuse(Violation::UnknownValue);
			}
			return {};
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return options;
}

/** An optional member read with `read`; `null` leaves it absent. */
template<class T, class F>
[[nodiscard]] auto assign(std::optional<T>& slot, Body& value, F const& read) -> Decoded<void> {
	return nullable<T>(value, read).transform([&slot](std::optional<T> decoded) { slot = std::move(decoded); });
}

[[nodiscard]] auto request(Body& root, Fragments& fragments) -> Decoded<ChatRequest> {
	auto decoded = ChatRequest{
	    .model = std::nullopt,
	    .messages = {},
	    .tools = {},
	    .tool_choice = std::nullopt,
	    .delivery = Delivery::Whole,
	    .max_tokens = std::nullopt,
	    .temperature = std::nullopt,
	    .top_p = std::nullopt,
	    .top_k = std::nullopt,
	    .min_p = std::nullopt,
	    .seed = std::nullopt,
	    .stop = {},
	    .template_options = {},
	};
	auto messages = std::optional<std::vector<Message>>{};
	auto stream = std::optional<bool>{};
	auto stream_usage = std::optional<bool>{};
	auto max_tokens = std::optional<std::int64_t>{};
	auto max_completion_tokens = std::optional<std::int64_t>{};
	auto const integer = [](Body& value) { return lift(value.as_integer()); };
	auto const real = [](Body& value) { return number(value); };
	auto const visited = each_member(root, [&](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "messages") {
			auto& list = messages.emplace();
			return each_element(member, [&](std::size_t, Body& element) -> Decoded<void> {
				return message(element, fragments).transform([&list](Message read) { list.push_back(std::move(read)); });
			});
		}
		if (name == "tools") {
			return nullable<bool>(member, [&](Body& list) -> Decoded<bool> {
				       return each_element(list, [&](std::size_t, Body& element) -> Decoded<void> {
					              return tool(element, fragments).transform([&decoded](Tool read) { decoded.tools.push_back(std::move(read)); });
				              })
				           .transform([] { return true; });
			       })
			    .transform([](std::optional<bool>) {});
		}
		if (name == "model") {
			return assign(decoded.model, member, [](Body& value) { return text(value); });
		}
		if (name == "tool_choice") {
			return assign(decoded.tool_choice, member, [](Body& value) { return tool_choice(value); });
		}
		if (name == "stream") {
			return assign(stream, member, [](Body& value) { return lift(value.as_boolean()); });
		}
		if (name == "stream_options") {
			return assign(stream_usage, member, [](Body& value) { return include_usage(value); });
		}
		if (name == "max_tokens") {
			return assign(max_tokens, member, integer);
		}
		if (name == "max_completion_tokens") {
			return assign(max_completion_tokens, member, integer);
		}
		if (name == "temperature") {
			return assign(decoded.temperature, member, real);
		}
		if (name == "top_p") {
			return assign(decoded.top_p, member, real);
		}
		if (name == "top_k") {
			return assign(decoded.top_k, member, integer);
		}
		if (name == "min_p") {
			return assign(decoded.min_p, member, real);
		}
		if (name == "seed") {
			return assign(decoded.seed, member, integer);
		}
		if (name == "stop") {
			return stop(member).transform([&decoded](std::vector<std::string> read) { decoded.stop = std::move(read); });
		}
		if (name == "chat_template_kwargs") {
			return nullable<TemplateOptions>(member, [](Body& value) { return template_options(value); }).transform([&decoded](std::optional<TemplateOptions> read) {
				decoded.template_options = read.value_or(TemplateOptions{});
			});
		}
		if (auto const neutral_value = tolerated(request_tolerated, name)) {
			return neutral(member, *neutral_value);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!messages) {
		return refuse_at("messages", Violation::MissingField);
	}
	decoded.messages = std::move(*messages);
	if (max_tokens && max_completion_tokens) {
		return refuse_at("max_completion_tokens", Conflict{.other = "max_tokens"});
	}
	decoded.max_tokens = max_completion_tokens ? max_completion_tokens : max_tokens;
	if (!stream.value_or(false)) {
		if (stream_usage) {
			return refuse_at("stream_options", Conflict{.other = "stream"});
		}
		decoded.delivery = Delivery::Whole;
	} else {
		decoded.delivery = stream_usage.value_or(false) ? Delivery::StreamWithUsage : Delivery::Stream;
	}
	return decoded;
}

/**
 * Decodes OpenAI chat requests into `ChatRequest`, one at a time, refusing every request it cannot serve exactly
 * with the member at fault. It owns its two readers (the body's, and one for JSON a request holds in a string), each
 * allocated once; decoding reads every part of the request, so nothing it accepts goes unchecked.
 *
 * Unknown members are refused by name, at every level. OpenAI's own endpoint refuses unrecognized arguments, so no
 * OpenAI client depends on extras being ignored; and an ignored sampling or output parameter would serve the request
 * differently from what it asked for. Parameters OpenAI defines that the engine does not implement are accepted
 * only with their neutral value (`n: 1`, `presence_penalty: 0`, `logprobs: false`, ...), which clients commonly send
 * as defaults; account identifiers (`user`, `prompt_cache_key`) are accepted and dropped.
 */
export class RequestDecoder {
public:
	[[nodiscard]] static auto make() -> core::Result<RequestDecoder, ReadError> {
		auto body = Reader<request_bounds>::make();
		if (!body) {
			return std::unexpected(body.error());
		}
		auto fragments = Reader<fragment_bounds>::make();
		if (!fragments) {
			return std::unexpected(fragments.error());
		}
		return RequestDecoder{std::make_unique<Reader<request_bounds>>(std::move(*body)), std::make_unique<Fragments>(std::move(*fragments))};
	}

	/** Decodes a body whose storage extends `padding` bytes past its end, in place. */
	[[nodiscard]] auto decode(PaddedView text) -> core::Result<ChatRequest, DecodeError> {
		auto root = lift(body_->read(text));
		if (!root) {
			return std::unexpected(root.error());
		}
		return finish(request(*root, *fragments_));
	}

	/** Copies a body into the decoder's padded buffer, then decodes it. */
	[[nodiscard]] auto decode_copy(std::string_view text) -> core::Result<ChatRequest, DecodeError> {
		auto root = lift(body_->read_copy(text));
		if (!root) {
			return std::unexpected(root.error());
		}
		return finish(request(*root, *fragments_));
	}

private:
	RequestDecoder(std::unique_ptr<Reader<request_bounds>> body, std::unique_ptr<Fragments> fragments)
	    : body_{std::move(body)}, fragments_{std::move(fragments)} {}

	[[nodiscard]] auto finish(Decoded<ChatRequest> decoded) -> core::Result<ChatRequest, DecodeError> {
		if (!decoded) {
			auto error = decoded.error();
			std::ranges::reverse(error.path);
			return std::unexpected(std::move(error));
		}
		if (auto const finished = lift(body_->finish()); !finished) {
			return std::unexpected(finished.error());
		}
		return decoded;
	}

	std::unique_ptr<Reader<request_bounds>> body_;
	std::unique_ptr<Fragments> fragments_;
};

}

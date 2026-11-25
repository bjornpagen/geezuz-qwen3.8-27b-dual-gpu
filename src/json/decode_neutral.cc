// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode_neutral;

import std;
import engine.core;
import :bounds;
import :error;
import :text;
import :value;
import :containers;
import :reader;
import :decode_error;
import :decode_support;

namespace engine::json {

/**
 * The value an sdk member the engine does not implement may carry: the one that leaves the output as the engine
 * produces it. Any other value is refused as unsupported, so no request is served differently from what it asked
 * for. `null` is always accepted, as absence is.
 */
enum class Neutral : std::uint8_t {
	/** Only `null`: the member has no value the engine honours. */
	Null,
	/** Any string: an identifier the engine echoes nowhere and acts on never (an item's own `id`). */
	AnyString,
	False,
	True,
	Zero,
	EmptyArray,
	/** `"completed"`: an item status a resent item may carry. */
	Completed,
	/** `"medium"`: the sdk's default verbosity. */
	Medium,
	/** `"standard"`: the default reasoning mode. */
	Standard,
	/** `"disabled"`: the default truncation. */
	Disabled,
	/** `"auto"` or `"default"`: the service tiers that mean the ordinary one. */
	OrdinaryTier,
	/** `"auto"` or `"all_turns"`: past reasoning stays in the context, as the template keeps it. */
	AllTurns,
	/** `{"type": "direct"}`: a call the model made itself. */
	DirectCaller,
	/** `["direct"]`: only the model calls the tool. */
	DirectCallers,
	/** Deprecated by the sdk: only `null`, and anything else refused as deprecated rather than unsupported. */
	Deprecated,
};

struct Tolerated {
	std::string_view name;
	Neutral neutral;
};

template<Bounds B>
[[nodiscard]] auto is_value(Value<B>& value, double wanted) -> Decoded<bool> {
	return lift(value.as_real()).transform([wanted](double read) { return read == wanted; });
}

template<Bounds B>
[[nodiscard]] auto is_boolean(Value<B>& value, bool wanted) -> Decoded<bool> {
	return lift(value.as_boolean()).transform([wanted](bool read) { return read == wanted; });
}

template<Bounds B>
[[nodiscard]] auto is_word(Value<B>& value, std::initializer_list<std::string_view> words) -> Decoded<bool> {
	return text(value).transform([words](std::string const& read) { return std::ranges::contains(words, read); });
}

/** Whether an array holds nothing; reads it whole either way. */
template<Bounds B>
[[nodiscard]] auto is_empty_array(Value<B>& value) -> Decoded<bool> {
	auto empty = true;
	auto const visited = each_element(value, [&empty](std::size_t, Value<B>&) -> Decoded<void> {
		empty = false;
		return {};
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return empty;
}

template<Bounds B>
[[nodiscard]] auto is_direct_caller(Value<B>& value) -> Decoded<bool> {
	auto direct = false;
	auto other = false;
	auto const visited = each_member(value, [&](std::string_view name, Value<B>& member) -> Decoded<void> {
		if (name != "type") {
			other = true;
			return {};
		}
		return text(member).transform([&direct](std::string const& type) { direct = type == "direct"; });
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return direct && !other;
}

template<Bounds B>
[[nodiscard]] auto is_direct_callers(Value<B>& value) -> Decoded<bool> {
	auto direct = std::size_t{0};
	auto other = false;
	auto const visited = each_element(value, [&](std::size_t, Value<B>& element) -> Decoded<void> {
		return text(element).transform([&](std::string const& caller) {
			direct += caller == "direct" ? 1 : 0;
			other = other || caller != "direct";
		});
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return direct > 0 && !other;
}

/** Reads a tolerated member whole and refuses any value but its neutral one. */
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
		case Neutral::Deprecated:
			return false;
		case Neutral::AnyString:
			return text(value).transform([](std::string const&) { return true; });
		case Neutral::False:
			return is_boolean(value, false);
		case Neutral::True:
			return is_boolean(value, true);
		case Neutral::Zero:
			return is_value(value, 0.0);
		case Neutral::EmptyArray:
			return is_empty_array(value);
		case Neutral::Completed:
			return is_word(value, {"completed"});
		case Neutral::Medium:
			return is_word(value, {"medium"});
		case Neutral::Standard:
			return is_word(value, {"standard"});
		case Neutral::Disabled:
			return is_word(value, {"disabled"});
		case Neutral::OrdinaryTier:
			return is_word(value, {"auto", "default"});
		case Neutral::AllTurns:
			return is_word(value, {"auto", "all_turns"});
		case Neutral::DirectCaller:
			return is_direct_caller(value);
		case Neutral::DirectCallers:
			return is_direct_callers(value);
		}
		std::unreachable();
	}();
	if (!holds) {
		return std::unexpected(holds.error());
	}
	if (!*holds) {
		return refuse(wanted == Neutral::Deprecated ? Violation::Deprecated : Violation::Unsupported);
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

/** The create params the engine accepts only at their neutral value. */
constexpr auto create_tolerated = std::to_array<Tolerated>({
    {.name = "background", .neutral = Neutral::False},
    {.name = "background_options", .neutral = Neutral::Null},
    {.name = "context_management", .neutral = Neutral::Null},
    {.name = "include", .neutral = Neutral::EmptyArray},
    {.name = "max_tool_calls", .neutral = Neutral::Null},
    {.name = "moderation", .neutral = Neutral::Null},
    {.name = "parallel_tool_calls", .neutral = Neutral::True},
    {.name = "prompt_cache_key", .neutral = Neutral::Null},
    {.name = "prompt_cache_options", .neutral = Neutral::Null},
    {.name = "prompt_cache_retention", .neutral = Neutral::Deprecated},
    {.name = "safety_identifier", .neutral = Neutral::Null},
    {.name = "service_tier", .neutral = Neutral::OrdinaryTier},
    {.name = "top_logprobs", .neutral = Neutral::Zero},
    {.name = "truncation", .neutral = Neutral::Disabled},
    {.name = "user", .neutral = Neutral::Deprecated},
    {.name = "image", .neutral = Neutral::Null},
    {.name = "access_programs", .neutral = Neutral::Null},
    {.name = "conversation", .neutral = Neutral::Null},
    {.name = "prompt", .neutral = Neutral::Null},
});

/** Members of `reasoning` the engine accepts only at their neutral value. */
constexpr auto reasoning_tolerated = std::to_array<Tolerated>({
    {.name = "summary", .neutral = Neutral::Null},
    {.name = "generate_summary", .neutral = Neutral::Deprecated},
    {.name = "context", .neutral = Neutral::AllTurns},
    {.name = "mode", .neutral = Neutral::Standard},
});

/** Members of a function or custom tool the engine accepts only at their neutral value. */
constexpr auto tool_tolerated = std::to_array<Tolerated>({
    {.name = "defer_loading", .neutral = Neutral::False},
    {.name = "allowed_callers", .neutral = Neutral::DirectCallers},
});

/** Members a resent input item may carry that change nothing: its id, a finished status, a direct caller. */
constexpr auto item_tolerated = std::to_array<Tolerated>({
    {.name = "id", .neutral = Neutral::AnyString},
    {.name = "status", .neutral = Neutral::Completed},
    {.name = "caller", .neutral = Neutral::DirectCaller},
    {.name = "namespace", .neutral = Neutral::Null},
    {.name = "phase", .neutral = Neutral::Null},
});

/** Content part types the sdk defines that the engine, text only, does not serve. */
constexpr auto unsupported_part_types = std::to_array<std::string_view>({"input_image", "input_file", "refusal"});

/** Tool types the sdk defines beyond functions and custom tools. */
constexpr auto unsupported_tool_types = std::to_array<std::string_view>({
    "file_search", "computer", "computer_use_preview", "web_search", "web_search_2025_08_26", "web_search_preview",
    "web_search_preview_2025_03_11", "mcp", "code_interpreter", "programmatic_tool_calling", "image_generation",
    "local_shell", "shell", "namespace", "tool_search", "apply_patch",
});

/** Input item types the sdk defines that the engine does not serve. */
constexpr auto unsupported_item_types = std::to_array<std::string_view>({
    "file_search_call", "computer_call", "computer_call_output", "web_search_call", "tool_search_call", "tool_search_output",
    "additional_tools", "configuration_update", "compaction", "compaction_trigger", "image_generation_call",
    "code_interpreter_call", "local_shell_call", "local_shell_call_output", "shell_call", "shell_call_output",
    "apply_patch_call", "apply_patch_call_output", "mcp_list_tools", "mcp_approval_request", "mcp_approval_response",
    "mcp_call", "item_reference", "program", "program_output",
});

/** Tool choice types the sdk defines beyond naming a function or a custom tool. */
constexpr auto unsupported_choice_types = std::to_array<std::string_view>({
    "allowed_tools", "file_search", "web_search_preview", "computer", "computer_use_preview", "computer_use",
    "web_search_preview_2025_03_11", "image_generation", "code_interpreter", "mcp", "apply_patch", "shell",
    "programmatic_tool_calling",
});

/** Refuses a `type` the decoder does not take: unsupported when the sdk defines it, unknown otherwise. */
template<std::size_t N>
[[nodiscard]] auto refuse_type(std::array<std::string_view, N> const& defined, std::string_view type) -> std::unexpected<DecodeError> {
	return refuse_at("type", std::ranges::find(defined, type) != defined.end() ? Violation::Unsupported : Violation::UnknownValue);
}

}

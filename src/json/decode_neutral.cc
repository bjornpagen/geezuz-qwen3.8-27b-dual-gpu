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
import :request;
import :decode_support;

namespace engine::json {

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

}

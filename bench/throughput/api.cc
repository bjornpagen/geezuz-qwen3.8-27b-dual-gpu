// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:api;

import std;
import engine.core;
import engine.json;
import :limits;
import :wire;

/**
 * The server's answers as the measurement reads them: an event of a streamed response, the engine's snapshot from
 * /status, a refusal's error object. Each is parsed once, here, into what the client decides on; a text that is not
 * the shape the api gives it is a fault, never a guess.
 */
namespace engine::throughput {

export using WireReader = json::Reader<wire_bounds>;

using Object = json::Object<wire_bounds>;

/** A response's token counts, from its final usage. Cached tokens are part of the input's, reasoning of the output's. */
export struct Usage {
	std::uint64_t input_tokens;
	std::uint64_t cached_tokens;
	std::uint64_t output_tokens;
	std::uint64_t reasoning_tokens;

	[[nodiscard]] constexpr auto operator==(Usage const&) const -> bool = default;
};

/** How a response ended: at its grammar's end, or at its output budget. */
export enum class Ending : std::uint8_t {
	Completed,
	Incomplete,
};

/** The engine accepted the request; nothing is generated yet. */
export struct Opened {
	[[nodiscard]] constexpr auto operator==(Opened const&) const -> bool = default;
};

/** An event that carries generated output: an item, a part, or a piece of text. */
export struct Output {
	[[nodiscard]] constexpr auto operator==(Output const&) const -> bool = default;
};

/** The terminal event of a response that finished, with its final usage. */
export struct Concluded {
	Ending ending;
	Usage usage;

	[[nodiscard]] constexpr auto operator==(Concluded const&) const -> bool = default;
};

/** `response.failed`: the engine failed under the stream. */
export struct Failed {
	std::string message;

	[[nodiscard]] auto operator==(Failed const&) const -> bool = default;
};

/** The `error` event: the stream ends without its response. */
export struct Faulted {
	std::string code;
	std::string message;

	[[nodiscard]] auto operator==(Faulted const&) const -> bool = default;
};

export using ApiEvent = std::variant<Opened, Output, Concluded, Failed, Faulted>;

/** The engine's counts since it started, as far as the measurement reads them. */
export struct Counters {
	/** Decode cycles run. */
	std::uint64_t cycles;
	/** Draft tokens the cycles verified. */
	std::uint64_t drafted_tokens;
	/** Draft tokens accepted: each row's tokens beyond the one a cycle always commits. */
	std::uint64_t accepted_tokens;

	[[nodiscard]] constexpr auto operator==(Counters const&) const -> bool = default;
};

/** The engine has posted no snapshot yet. */
export struct Unreported {
	[[nodiscard]] constexpr auto operator==(Unreported const&) const -> bool = default;
};

/** Requests are running or waiting: the counters are still moving, and not this client's alone. */
export struct Occupied {
	std::uint64_t running;
	std::uint64_t queued;

	[[nodiscard]] constexpr auto operator==(Occupied const&) const -> bool = default;
};

/** Every slot is vacant and nothing waits: the counters are final for everything that ran. */
export struct Idle {
	Counters counters;

	[[nodiscard]] constexpr auto operator==(Idle const&) const -> bool = default;
};

export using EngineState = std::variant<Unreported, Occupied, Idle>;

/** What a refusal says of sending the same request again. */
export enum class Standing : std::uint8_t {
	/** The engine is still loading. */
	Loading,
	/** The engine holds as many requests as it takes. */
	Busy,
	/** The server was short of memory for the request. */
	Short,
	/** Anything else: the request as sent is not served, now or later. */
	Final,
};

/** An answer that is not the response asked for: its status and OpenAI's error object. */
export struct Refusal {
	std::uint16_t status;
	Standing standing;
	/** The api's `code` and `message`, for rendering; decisions read `standing`. */
	std::string code;
	std::string message;

	[[nodiscard]] auto operator==(Refusal const&) const -> bool = default;
};

/** The codes whose refusal passes without the client changing anything. */
[[nodiscard]] constexpr auto standing_of(std::string_view code) -> Standing {
	if (code == "engine_loading") {
		return Standing::Loading;
	}
	if (code == "engine_busy") {
		return Standing::Busy;
	}
	if (code == "out_of_memory") {
		return Standing::Short;
	}
	return Standing::Final;
}

[[nodiscard]] constexpr auto unshaped() -> std::unexpected<WireFault> {
	return broke(Fault::BadJson);
}

/** The object that is all of `text`. */
[[nodiscard]] auto root_of(WireReader& reader, std::string_view text) -> core::Result<Object, WireFault> {
	auto root = reader.read_copy(text);
	if (!root) {
		return unshaped();
	}
	auto object = root->as_object();
	if (!object) {
		return unshaped();
	}
	return std::move(*object);
}

/** The object `object`'s member `name` holds. */
[[nodiscard]] auto object_in(Object& object, std::string_view name) -> core::Result<Object, WireFault> {
	auto value = object.member(name);
	if (!value) {
		return unshaped();
	}
	auto inner = value->as_object();
	if (!inner) {
		return unshaped();
	}
	return std::move(*inner);
}

/** The count `object`'s member `name` holds: an integer that is not negative. */
[[nodiscard]] auto count_in(Object& object, std::string_view name) -> core::Result<std::uint64_t, WireFault> {
	auto value = object.member(name);
	if (!value) {
		return unshaped();
	}
	auto const integer = value->as_integer();
	if (!integer) {
		return unshaped();
	}
	auto const count = core::try_narrow<std::uint64_t>(*integer);
	if (!count) {
		return unshaped();
	}
	return *count;
}

/** The string `object`'s member `name` holds, owned: the reader's own text lasts only until its next document. */
[[nodiscard]] auto text_in(Object& object, std::string_view name) -> core::Result<std::string, WireFault> {
	auto value = object.member(name);
	if (!value) {
		return unshaped();
	}
	auto const text = value->as_string();
	if (!text) {
		return unshaped();
	}
	return std::string{*text};
}

enum class EventKind : std::uint8_t {
	Opened,
	Output,
	Completed,
	Incomplete,
	Failed,
	Error,
};

/** The events that carry generated output: exactly the ones the api documents, so a new one is noticed, not guessed at. */
constexpr auto output_events = std::array<std::string_view, 12>{
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.custom_tool_call_input.delta",
    "response.custom_tool_call_input.done",
};

[[nodiscard]] constexpr auto kind_of(std::string_view type) -> std::optional<EventKind> {
	if (type == "response.created" || type == "response.in_progress") {
		return EventKind::Opened;
	}
	if (std::ranges::contains(output_events, type)) {
		return EventKind::Output;
	}
	if (type == "response.completed") {
		return EventKind::Completed;
	}
	if (type == "response.incomplete") {
		return EventKind::Incomplete;
	}
	if (type == "response.failed") {
		return EventKind::Failed;
	}
	if (type == "error") {
		return EventKind::Error;
	}
	return std::nullopt;
}

/** A finished response's usage; a response that finished without one is not the api's. */
[[nodiscard]] auto usage_of(Object& response) -> core::Result<Usage, WireFault> {
	auto usage = object_in(response, "usage");
	if (!usage) {
		return std::unexpected(usage.error());
	}
	auto const input = count_in(*usage, "input_tokens");
	auto const output = count_in(*usage, "output_tokens");
	auto input_details = object_in(*usage, "input_tokens_details");
	if (!input || !output || !input_details) {
		return unshaped();
	}
	auto const cached = count_in(*input_details, "cached_tokens");
	auto output_details = object_in(*usage, "output_tokens_details");
	if (!cached || !output_details) {
		return unshaped();
	}
	auto const reasoning = count_in(*output_details, "reasoning_tokens");
	if (!reasoning) {
		return unshaped();
	}
	return Usage{.input_tokens = *input, .cached_tokens = *cached, .output_tokens = *output, .reasoning_tokens = *reasoning};
}

[[nodiscard]] auto concluded(Object& event, Ending ending) -> core::Result<ApiEvent, WireFault> {
	auto response = object_in(event, "response");
	if (!response) {
		return std::unexpected(response.error());
	}
	auto const usage = usage_of(*response);
	if (!usage) {
		return std::unexpected(usage.error());
	}
	return Concluded{.ending = ending, .usage = *usage};
}

[[nodiscard]] auto failed(Object& event) -> core::Result<ApiEvent, WireFault> {
	auto response = object_in(event, "response");
	if (!response) {
		return std::unexpected(response.error());
	}
	auto error = object_in(*response, "error");
	if (!error) {
		return std::unexpected(error.error());
	}
	auto message = text_in(*error, "message");
	if (!message) {
		return std::unexpected(message.error());
	}
	return Failed{.message = std::move(*message)};
}

[[nodiscard]] auto faulted(Object& event) -> core::Result<ApiEvent, WireFault> {
	auto code = text_in(event, "code");
	auto message = text_in(event, "message");
	if (!code || !message) {
		return unshaped();
	}
	return Faulted{.code = std::move(*code), .message = std::move(*message)};
}

/** Reads one event's data. */
export [[nodiscard]] auto read_event(WireReader& reader, std::string_view data) -> core::Result<ApiEvent, WireFault> {
	auto event = root_of(reader, data);
	if (!event) {
		return std::unexpected(event.error());
	}
	auto const type = text_in(*event, "type");
	if (!type) {
		return std::unexpected(type.error());
	}
	auto const kind = kind_of(*type);
	if (!kind) {
		return broke(Fault::UnknownEvent);
	}
	switch (*kind) {
	case EventKind::Opened:
		return Opened{};
	case EventKind::Output:
		return Output{};
	case EventKind::Completed:
		return concluded(*event, Ending::Completed);
	case EventKind::Incomplete:
		return concluded(*event, Ending::Incomplete);
	case EventKind::Failed:
		return failed(*event);
	case EventKind::Error:
		return faulted(*event);
	}
	std::unreachable();
}

/** How many of the snapshot's slots hold a request. */
[[nodiscard]] auto running_in(Object& engine) -> core::Result<std::uint64_t, WireFault> {
	auto member = engine.member("slots");
	if (!member) {
		return unshaped();
	}
	auto slots = member->as_array();
	if (!slots) {
		return unshaped();
	}
	auto running = std::uint64_t{0};
	for (;;) {
		auto element = slots->next();
		if (!element) {
			return unshaped();
		}
		if (!*element) {
			return running;
		}
		auto slot = (*element)->as_object();
		if (!slot) {
			return unshaped();
		}
		auto const phase = text_in(*slot, "phase");
		if (!phase) {
			return unshaped();
		}
		// Any phase but vacant holds a request, one this client does not know of included.
		running += *phase == "vacant" ? 0 : 1;
	}
}

/** Reads /status's body. */
export [[nodiscard]] auto read_status(WireReader& reader, std::string_view body) -> core::Result<EngineState, WireFault> {
	auto root = root_of(reader, body);
	if (!root) {
		return std::unexpected(root.error());
	}
	auto member = root->member("engine");
	if (!member) {
		return unshaped();
	}
	auto const kind = member->kind();
	if (!kind) {
		return unshaped();
	}
	if (*kind == json::Kind::Null) {
		return Unreported{};
	}
	auto engine = member->as_object();
	if (!engine) {
		return unshaped();
	}
	auto const running = running_in(*engine);
	auto const queued = count_in(*engine, "queued");
	auto counters = object_in(*engine, "counters");
	if (!running || !queued || !counters) {
		return unshaped();
	}
	auto const cycles = count_in(*counters, "cycles");
	auto const drafted = count_in(*counters, "drafted_tokens");
	auto const accepted = count_in(*counters, "accepted_tokens");
	if (!cycles || !drafted || !accepted) {
		return unshaped();
	}
	if (*running != 0 || *queued != 0) {
		return Occupied{.running = *running, .queued = *queued};
	}
	return Idle{.counters = Counters{.cycles = *cycles, .drafted_tokens = *drafted, .accepted_tokens = *accepted}};
}

/** Reads the body of an answer whose status is not 200: OpenAI's error object. */
export [[nodiscard]] auto read_refusal(WireReader& reader, std::uint16_t status, std::string_view body) -> core::Result<Refusal, WireFault> {
	auto root = root_of(reader, body);
	if (!root) {
		return std::unexpected(root.error());
	}
	auto error = object_in(*root, "error");
	if (!error) {
		return std::unexpected(error.error());
	}
	auto code = text_in(*error, "code");
	auto message = text_in(*error, "message");
	if (!code || !message) {
		return unshaped();
	}
	return Refusal{.status = status, .standing = standing_of(*code), .code = std::move(*code), .message = std::move(*message)};
}

}

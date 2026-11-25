// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:engine;

import std;
import engine.http;
import engine.json;
import engine.model;
import engine.sampling;
import :stop;

/**
 * The boundary between the api and the engine behind it. The service hands the engine validated, typed requests;
 * the engine renders, tokenizes, schedules and decodes them, and posts back typed outputs and status snapshots
 * through the http server's mailbox. Nothing here renders or tokenizes.
 */
namespace engine::openai {

/** The longest function name a tool may have, and the longest an engine may begin a call with. */
export inline constexpr std::size_t max_tool_name_bytes = 64;

/** The completion runs until the end of the turn or until prompt and completion fill the context. */
export struct UntilContextFull {
	[[nodiscard]] constexpr auto operator==(UntilContextFull const&) const -> bool = default;
};

/**
 * At most `tokens` completion tokens, 1 to model::max_positions. A prompt that leaves fewer than `tokens` positions
 * of the context is refused with PromptTooLong, as OpenAI refuses it, never silently shortened.
 */
export struct AtMost {
	std::uint32_t tokens;

	[[nodiscard]] constexpr auto operator==(AtMost const&) const -> bool = default;
};

export using Budget = std::variant<UntilContextFull, AtMost>;

static_assert(model::max_positions <= std::numeric_limits<std::uint32_t>::max());

/**
 * One validated chat request, owned. What validation established, the engine may rely on:
 * - `messages` is nonempty, only its first message may be a system message, and at least one user message is not
 *   wholly a `<tool_response>`: the chat template renders it without raising;
 * - tool names are distinct, each 1 to max_tool_name_bytes; a forced `tool_choice` names one of them, and
 *   `tool_choice` is `none` or `auto` when there are no tools (absent in the request, it is `auto` with tools and
 *   `none` without, as OpenAI defaults it);
 * - `template_options` is as the client sent it, for the template to default;
 * - `sampling` already carries the defaults of the request's thinking mode and its seed.
 */
export struct Submission {
	/** The request's id, for the engine's log lines and its status snapshot. */
	http::RequestId id;
	std::vector<json::Message> messages;
	std::vector<json::Tool> tools;
	json::ToolChoice tool_choice;
	json::TemplateOptions template_options;
	sampling::Params sampling;
	Budget budget;
	/** The engine runs the content through a StopScan built from these, ending the completion at a match. */
	StopSequences stop;

	[[nodiscard]] auto operator==(Submission const&) const -> bool = default;
};

/**
 * The prompt rendered, tokenized and within the context: the request is queued for a slot. Exactly one of Accepted
 * or PromptTooLong is a request's first output.
 */
export struct Accepted {
	[[nodiscard]] constexpr auto operator==(Accepted const&) const -> bool = default;
};

/** The rendered prompt and the budget do not fit the context; the request's only output. */
export struct PromptTooLong {
	std::uint64_t prompt_tokens;

	[[nodiscard]] constexpr auto operator==(PromptTooLong const&) const -> bool = default;
};

/** The next text of the turn's reasoning: whole UTF-8 characters. */
export struct ReasoningText {
	std::string text;

	[[nodiscard]] auto operator==(ReasoningText const&) const -> bool = default;
};

/** The next text of the turn's answer, after the stop scan: whole UTF-8 characters. */
export struct ContentText {
	std::string text;

	[[nodiscard]] auto operator==(ContentText const&) const -> bool = default;
};

/**
 * A tool call begins: its position among the turn's calls, numbered from 0 in order, and its function name (1 to
 * max_tool_name_bytes). The previous call, if any, has ended.
 */
export struct CallBegin {
	std::uint32_t index;
	std::string name;

	[[nodiscard]] auto operator==(CallBegin const&) const -> bool = default;
};

/**
 * The open call ends, with its whole arguments: one JSON object. No part of a call's arguments is final before the
 * call closes, so they arrive whole, never in pieces.
 */
export struct CallEnd {
	std::uint32_t index;
	std::string arguments;

	[[nodiscard]] auto operator==(CallEnd const&) const -> bool = default;
};

/** Why generation ended. Whether the turn called tools is the service's to see, from the calls that began. */
export enum class Ending : std::uint8_t {
	/** The model ended its turn, or the content reached a stop sequence. Every begun call has ended. */
	Stop,
	/** The budget or the context ran out. A call may be open, cut off: it is not reported in a whole response. */
	Length,
};

/** The request's last output. Usage counts the completion up to its ending token. */
export struct Finished {
	Ending ending;
	json::Usage usage;

	[[nodiscard]] auto operator==(Finished const&) const -> bool = default;
};

/** A request's first output: whether the engine takes it. */
export using Verdict = std::variant<Accepted, PromptTooLong>;

/** What an accepted request produces: texts and calls, in order, then Finished. */
export using Produced = std::variant<ReasoningText, ContentText, CallBegin, CallEnd, Finished>;

/**
 * One output of a request, in order: a Verdict, then, if it is Accepted, Produced outputs up to Finished. An output
 * out of this order is a bug in the engine and violates an invariant of the service. After `cancel`, the engine may
 * still post outputs it made before; the service drops them.
 */
export using Output = std::variant<Verdict, Produced>;

export struct Addressed {
	http::Exchange exchange;
	Output output;

	[[nodiscard]] auto operator==(Addressed const&) const -> bool = default;
};

/**
 * Outputs for any number of requests, each request's in order. An engine posts one batch per cycle rather than one
 * message per output, so a cycle of eight slots costs one mailbox entry.
 */
export struct Outputs {
	std::vector<Addressed> items;

	[[nodiscard]] auto operator==(Outputs const&) const -> bool = default;
};

export enum class Readiness : std::uint8_t {
	/** The model is loading: /health answers 503 and chat requests are refused 503, retryable. */
	Loading,
	Serving,
};

export struct Vacant {
	[[nodiscard]] constexpr auto operator==(Vacant const&) const -> bool = default;
};

/** A slot reading its prompt: `prefilled` of `prompt_tokens` positions done, the first `cached_tokens` from cache. */
export struct Prefilling {
	http::RequestId id;
	std::uint64_t prompt_tokens;
	std::uint64_t cached_tokens;
	std::uint64_t prefilled_tokens;

	[[nodiscard]] auto operator==(Prefilling const&) const -> bool = default;
};

export struct Decoding {
	http::RequestId id;
	std::uint64_t prompt_tokens;
	std::uint64_t cached_tokens;
	std::uint64_t generated_tokens;

	[[nodiscard]] auto operator==(Decoding const&) const -> bool = default;
};

export using Slot = std::variant<Vacant, Prefilling, Decoding>;

/** The shared kv pool, in blocks: `used` held by running requests, `cached` held only by the prefix cache. */
export struct Pool {
	std::uint64_t blocks;
	std::uint64_t used_blocks;
	std::uint64_t cached_blocks;

	[[nodiscard]] constexpr auto operator==(Pool const&) const -> bool = default;
};

/** Decode cycles since load, and the draft tokens they proposed and accepted. */
export struct Cycles {
	std::uint64_t cycles;
	std::uint64_t drafted_tokens;
	std::uint64_t accepted_tokens;

	[[nodiscard]] constexpr auto operator==(Cycles const&) const -> bool = default;
};

/** What the engine is doing, as of the snapshot's posting; /status writes the latest one. */
export struct Status {
	Readiness readiness;
	std::array<Slot, model::slots> slots;
	/** Requests accepted and waiting for a slot. */
	std::uint64_t queued;
	/** Requests paused for a full pool since load. */
	std::uint64_t preemptions;
	Pool pool;
	Cycles cycles;

	[[nodiscard]] auto operator==(Status const&) const -> bool = default;
};

/** The status before the engine has posted any. */
export [[nodiscard]] auto loading_status() -> Status {
	return Status{.readiness = Readiness::Loading, .slots = {}, .queued = 0, .preemptions = 0, .pool = {}, .cycles = {}};
}

/** What the engine posts to the service, through the `http::Poster<Update>` of the server it serves behind. */
export using Update = std::variant<Outputs, Status>;

/**
 * The engine as the service needs it, called on the server's owner thread, bounded and nonblocking:
 *
 * - `submit` takes a request; its outputs come back as `Outputs`, posted. At most one request per connection is in
 *   flight, so the engine's intake never holds more than http::max_connections.
 * - `cancel` ends a request whose client is gone or too slow; its slot is released at the next cycle boundary and
 *   it posts nothing more that matters.
 *
 * Status snapshots are posted as `Status` whenever they change; the first `Serving` one opens the server for chat.
 */
export template<class E>
concept Engine = std::movable<E> && requires(E& engine, http::Exchange exchange, Submission submission) {
	{ engine.submit(exchange, std::move(submission)) } -> std::same_as<void>;
	{ engine.cancel(exchange) } -> std::same_as<void>;
};

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:boundary;

import std;
import engine.chat;
import engine.grammar;
import engine.http;
import engine.json;
import engine.pipeline;
import engine.sampling;
import engine.sched;
import engine.tokenizer;

/**
 * The boundary between the api and the engine behind it. The service hands the engine validated, typed submissions;
 * the engine's front renders them, runs them under their grammar, and posts back typed outputs, status snapshots and,
 * if it fails, why. Nothing on the service's side renders or tokenizes.
 */
namespace engine::openai {

/** The one model this server serves, as `/v1/models` lists it and every response names it. */
export inline constexpr std::string_view model_id = "Qwen3.8-27B-FP8";

/**
 * What a request's output must obey, as the request gave it: the grammar's input, and the parser's. Validation has
 * established that tool names are distinct and well formed and that a forced choice names a tool of its kind.
 */
export struct Constraints {
	std::vector<json::Tool> tools;
	json::ToolChoice tool_choice;
	json::TextFormat format;
	/** Whether the reply starts inside the thinking block: the grammar's reasoning region, the parser's first. */
	chat::Thinking thinking;

	[[nodiscard]] auto operator==(Constraints const&) const -> bool = default;
};

/** A request with no stored predecessor: its whole chat, rendered from the start. */
export struct Fresh {
	chat::Conversation conversation;

	[[nodiscard]] auto operator==(Fresh const&) const -> bool = default;
};

/**
 * A continuation of a stored response: its exact token sequence (prompt and reply, through the token that ended its
 * turn), then the new turns rendered after it with the stored response's options, never a re-render of the past.
 */
export struct Continued {
	std::vector<tokenizer::Id> sequence;
	std::vector<chat::Turn> turns;
	chat::Options options;

	[[nodiscard]] auto operator==(Continued const&) const -> bool = default;
};

export using Prompt = std::variant<Fresh, Continued>;

/** One validated request, owned: what the engine's front prepares on the engine's thread. */
export struct Submission {
	/** The request's id, naming it in status snapshots. */
	std::string tag;
	Prompt prompt;
	Constraints constraints;
	/** Already carrying its effort's defaults and a fresh seed. */
	sampling::Params sampling;
	sched::Limit limit;

	[[nodiscard]] auto operator==(Submission const&) const -> bool = default;
};

/** The prompt is rendered and fits: the request is queued for a slot. */
export struct Accepted {
	[[nodiscard]] constexpr auto operator==(Accepted const&) const -> bool = default;
};

/** The rendered prompt and the budget do not fit the context; the request's only output. */
export struct PromptTooLong {
	std::uint64_t prompt_tokens;

	[[nodiscard]] constexpr auto operator==(PromptTooLong const&) const -> bool = default;
};

/** More of the reply's reasoning: whole utf-8 characters, final. */
export struct Reasoning {
	std::string text;

	[[nodiscard]] auto operator==(Reasoning const&) const -> bool = default;
};

/** More of the reply's answer: whole utf-8 characters, final. */
export struct Content {
	std::string text;

	[[nodiscard]] auto operator==(Content const&) const -> bool = default;
};

/** Which kind of tool a call calls: what the response reports it as. */
export enum class Callee : std::uint8_t {
	Function,
	/** A custom tool: its call carries the one `input` string. */
	Custom,
};

/** A tool call opens: the reply's `index`-th, from zero; the call before it has ended. */
export struct CallBegin {
	std::uint32_t index;
	std::string name;
	Callee callee;

	[[nodiscard]] auto operator==(CallBegin const&) const -> bool = default;
};

/** How a call ended: its closing tag, or the reply's end, or the budget cutting it off. */
export enum class Closing : std::uint8_t {
	Closed,
	CutOff,
};

/**
 * The open call ends with its whole text: a function's arguments as one JSON object, a custom tool's input. No part
 * of a call is final before it closes, so it arrives whole.
 */
export struct CallEnd {
	std::uint32_t index;
	std::string text;
	Closing closing;

	[[nodiscard]] auto operator==(CallEnd const&) const -> bool = default;
};

/** The request's end: why, with the exact sequence it leaves, and its usage. */
export struct Finished {
	pipeline::Ending ending;
	json::Usage usage;

	[[nodiscard]] auto operator==(Finished const&) const -> bool = default;
};

/** The request's grammar outgrew its bounds while the reply was decoded: the reply ended there, without its finish. */
export struct Overran {
	grammar::MatchError fault;

	[[nodiscard]] constexpr auto operator==(Overran const&) const -> bool = default;
};

/** The engine let go of a request the service cancelled. */
export struct Cancelled {
	[[nodiscard]] constexpr auto operator==(Cancelled const&) const -> bool = default;
};

/**
 * One output for one request. Its outputs are: `PromptTooLong` alone; or `Accepted`, the texts and calls in order,
 * then `Finished`, or `Overran` in its place; or, once the service cancelled it, `Cancelled` after whatever was said
 * before. The last of these is its terminal output; nothing follows it. An engine that posts anything else violates
 * the service's invariants.
 */
export using Output = std::variant<Accepted, PromptTooLong, Reasoning, Content, CallBegin, CallEnd, Finished, Overran, Cancelled>;

export using Addressed = pipeline::Addressed<Output>;

/** Outputs for any number of requests, each request's in order: one batch per engine round. */
export struct Outputs {
	std::vector<Addressed> items;

	[[nodiscard]] auto operator==(Outputs const&) const -> bool = default;
};

/** What the engine is doing, each running request named by its submission's tag. */
export using Snapshot = pipeline::Snapshot<std::string>;

/** The engine serves: posted once, by whoever started it, after it loaded. */
export struct Ready {
	[[nodiscard]] constexpr auto operator==(Ready const&) const -> bool = default;
};

/** The engine could not start, or stopped serving; nothing it has not answered will be. */
export struct EngineFailed {
	std::string reason;

	[[nodiscard]] auto operator==(EngineFailed const&) const -> bool = default;
};

/** What reaches the service through the `http::Poster<Update>` of the server it serves behind. */
export using Update = std::variant<Outputs, Snapshot, Ready, EngineFailed>;

/**
 * The engine as the service drives it, on the server's owner thread, bounded and nonblocking: `submit` hands it a
 * request, whose outputs come back posted; `cancel` lets go of one whose client left or fell behind, which the engine
 * answers with `Cancelled` unless it already said the request's terminal output.
 */
export template<class E>
concept Engine = requires(E& engine, http::Exchange exchange, Submission submission) {
	{ engine.submit(exchange, std::move(submission)) } -> std::same_as<void>;
	{ engine.cancel(exchange) } -> std::same_as<void>;
};

/**
 * Where the service's time and randomness come from, passed in so that nothing it does draws on hidden state:
 * `unix_seconds` stamps responses, `draw` makes their ids and every request's fresh seed.
 */
export template<class S>
concept Sources = requires(S& sources) {
	{ sources.unix_seconds() } -> std::same_as<std::int64_t>;
	{ sources.draw() } -> std::same_as<std::uint64_t>;
};

/** An id of the server's: `prefix`, an underscore, and 128 bits from the sources as 32 hex digits. */
template<Sources S>
[[nodiscard]] auto fresh_id(S& sources, std::string_view prefix) -> std::string {
	auto const high = sources.draw();
	auto const low = sources.draw();
	return std::format("{}_{:016x}{:016x}", prefix, high, low);
}

}

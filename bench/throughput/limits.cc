// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:limits;

import std;
import engine.json;
import engine.model;

namespace engine::throughput {

/** Streams one wave holds open at once at most: one per slot of the engine. */
export inline constexpr std::size_t max_streams = model::slots;

/** Prompt files one command line names. */
export inline constexpr std::size_t max_prompts = 64;

/** One prompt's text: far past what fits the model's context beside its reply, far below the server's body bound. */
export inline constexpr std::size_t max_prompt_bytes = std::size_t{1} << 20U;

/** A response's status line and fields with their blank line; this server's heads stay under three hundred bytes. */
export inline constexpr std::size_t max_head_bytes = 4096;

/** A body read whole: /status's snapshot of eight slots, or a refusal's error object. */
export inline constexpr std::size_t max_body_bytes = std::size_t{64} << 10U;

/**
 * One server-sent event's data. The terminal event carries the whole response, whose text is the reply's tokens once:
 * a megabyte leaves a kilobyte for every token of the output budget.
 */
export inline constexpr std::size_t max_event_bytes = std::size_t{1} << 20U;

/** Bytes taken from a connection in one read. */
export inline constexpr std::size_t receive_bytes = std::size_t{64} << 10U;

/** What the client's JSON reader accepts: one event or one whole body, shaped as the api's objects are. */
export inline constexpr json::Bounds wire_bounds{
    .document_bytes = max_event_bytes,
    .depth = 32,
    .string_bytes = max_event_bytes,
    .members = 4096,
    .number_chars = 32,
};

static_assert(json::Enforceable<wire_bounds>);
static_assert(max_body_bytes <= wire_bounds.document_bytes, "a whole body is read by the same reader as an event");

/**
 * The numbers of the measurement: fixed for every run that is compared with another, so they are the program's and not
 * the command line's.
 */
export struct Protocol {
	/** How many times the whole workload is run; the best run is reported beside the spread of all. */
	std::size_t runs;
	/** Every request's output budget, so each stream decodes a comparable stretch. */
	std::uint32_t max_output_tokens;
	/** How long a connection may go without a byte moving either way before the server counts as hung. */
	std::chrono::milliseconds silence;
	/** How long the engine may take to come to rest before and after a wave. */
	std::chrono::milliseconds settle_patience;
	/** How long the client waits before it asks an engine that was not at rest again. */
	std::chrono::milliseconds settle_interval;
};

/**
 * The protocol the outcome table is measured under. Speculative decoding's speed varies from run to run with what it
 * accepts, so the workload is run five times and judged by its best run; a thousand tokens is some two hundred cycles
 * at one stream, enough that the first cycle and the last weigh under a percent.
 */
export inline constexpr Protocol protocol{
    .runs = 5,
    .max_output_tokens = 1024,
    .silence = std::chrono::seconds{120},
    .settle_patience = std::chrono::seconds{10},
    .settle_interval = std::chrono::milliseconds{5},
};

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:limits;

import std;

namespace engine::http {

/** The request line and every header field, with their terminating blank line. */
export inline constexpr std::size_t max_head_bytes = 8192;

/** Header fields per request; the openai clients send fewer than twenty. */
export inline constexpr std::size_t max_header_count = 48;

/** Parameters in one request's query: more than an openai route sends, `include` repeated for every includable. */
export inline constexpr std::size_t max_query_parameters = 16;

/** The id a route's path names; the engine's response ids are 37 bytes. */
export inline constexpr std::size_t max_resource_id_bytes = 64;

/** A whole (non-streamed) response's body; a longer one is answered 500 instead. */
export inline constexpr std::size_t max_response_body_bytes = std::size_t{16} << 20U;

/**
 * Framed stream bytes a connection buffers for a client that has not read them, besides the one event it may hold
 * alone: the bound on a stream's ordinary events, so a client that stops reading is refused deltas early.
 */
export inline constexpr std::size_t max_stream_buffer_bytes = std::size_t{64} << 10U;

/** One server-sent event's data: a terminal event carries a whole response, so it is as long as a whole body. */
export inline constexpr std::size_t max_event_bytes = max_response_body_bytes;

/** Connections open at once, each with its buffers allocated when the server opens. */
export inline constexpr std::size_t max_connections = 128;

/** Messages from other threads not yet drained by the owner; reaching it means the owner is stalled. */
export inline constexpr std::size_t max_posted_messages = 4096;

/** A client's X-Request-Id longer than this is not sane, and the server issues its own. */
export inline constexpr std::size_t max_request_id_bytes = 64;

/**
 * After a connection-closing response, how long the server keeps reading and discarding before it closes: closing
 * with unread client bytes resets the connection, and a reset can destroy the response before the client reads it.
 */
export inline constexpr auto linger_duration = std::chrono::seconds{2};

}

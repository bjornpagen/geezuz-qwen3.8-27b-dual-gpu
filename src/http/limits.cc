// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:limits;

import std;

namespace engine::http {

/** The request line and every header field, with their terminating blank line. */
export inline constexpr std::size_t max_head_bytes = 8192;

/** Header fields per request; the openai clients send fewer than twenty. */
export inline constexpr std::size_t max_header_count = 48;

/** A whole (non-streamed) response's body; a longer one is answered 500 instead. */
export inline constexpr std::size_t max_response_body_bytes = std::size_t{16} << 20U;

/**
 * Framed stream bytes a connection holds for a client that has not read them: the per-connection backpressure
 * window. A slow client costs at most this much server memory, plus what the kernel buffers.
 */
export inline constexpr std::size_t max_stream_buffer_bytes = std::size_t{64} << 10U;

/** One server-sent event's data; a longer delta is split by the service before it is sent. */
export inline constexpr std::size_t max_event_bytes = std::size_t{4} << 10U;

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

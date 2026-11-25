// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:exchange;

import std;
import engine.net;
import :request;
import :write;

namespace engine::http {

/** A generational reference to one connection in the server's table. */
export struct ConnectionHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator<=>(ConnectionHandle const&) const = default;
};

/**
 * One request and its answer on one connection: the connection's handle and the request's position on it. A plain
 * value, safe to carry across threads in a pipeline's messages; once the exchange is over, every operation on it
 * reports `Gone`, even after the connection carries later requests.
 */
export struct Exchange {
	ConnectionHandle connection;
	std::uint32_t sequence;

	[[nodiscard]] constexpr auto operator<=>(Exchange const&) const = default;
};

/** The answer is a stream of server-sent events (200, text/event-stream), sent through the port. */
export struct OpenStream {
	[[nodiscard]] constexpr auto operator==(OpenStream const&) const -> bool = default;
};

/** The answer comes later, through the port: a whole response or a stream. */
export struct Deferred {
	[[nodiscard]] constexpr auto operator==(Deferred const&) const -> bool = default;
};

/** What a service returns for a request: the whole response now, a stream from now on, or an answer later. */
export using Answer = std::variant<Response, OpenStream, Deferred>;

export enum class Delivery : std::uint8_t {
	Accepted,
	/** The exchange is over: its client disconnected or stalled, and the service was told it was abandoned. */
	Gone,
};

export enum class Flow : std::uint8_t {
	Accepted,
	/**
	 * The client has not read enough: the event was not taken. The service keeps it (or coalesces it with later
	 * deltas) and sends again after `drained` for this exchange.
	 */
	Full,
	Gone,
};

export struct ServerConfig {
	/** Where to listen; port zero takes an ephemeral port. */
	net::Endpoint endpoint;
	/**
	 * How long a connection may take to deliver one whole request, counted from when it starts waiting for one; an
	 * idle keep-alive connection is closed after it too.
	 */
	std::chrono::milliseconds read_timeout;
	/** How long pending output may go without progress before the client counts as gone. */
	std::chrono::milliseconds write_timeout;
	/** Seeds issued request ids, so ids from two boots differ. */
	std::uint64_t id_seed;
};

}

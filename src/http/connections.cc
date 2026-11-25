// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:connections;

import std;
import engine.core;
import engine.handle;
import engine.net;
import :exchange;
import :inbound;
import :limits;
import :outbound;
import :request;
import :write;

namespace engine::http {

/** Waiting for, or reading, the next request. */
struct Idle {
	net::Deadline read_by;
};

/** The service owes this request an answer. */
struct Awaiting {
	std::uint32_t sequence;
	RequestId id;
	Persistence persistence;
};

/** Whether the service was refused an event and is owed a `drained` notice. */
enum class Backpressure : std::uint8_t {
	Clear,
	Owed,
};

/** The service is sending events. */
struct Streaming {
	std::uint32_t sequence;
	Persistence persistence;
	Backpressure backpressure;
};

/** The answer is complete; its bytes are being sent. */
struct Flushing {
	Persistence persistence;
};

/** The server has sent its last byte and shut its side; it reads and discards until the client closes. */
struct Lingering {
	net::Deadline until;
};

using Phase = std::variant<Idle, Awaiting, Streaming, Flushing, Lingering>;

enum class Queued : std::uint8_t {
	No,
	Yes,
};

/** One connection's slot: everything it owns, allocated with the table when the server opens. */
template<BodyBounds B>
struct Connection {
	net::Socket socket;
	Inbound<B> inbound;
	Outbound outbound;
	Phase phase;
	/** Set while output is pending: when the client must have read some of it. */
	std::optional<net::Deadline> write_by;
	/** The latest exchange's position on the connection. */
	std::uint32_t sequence;
	Queued queued;
};

/** A request for the service. */
template<BodyBounds B>
struct Arrived {
	Exchange exchange;
	Request<B> request;
};

/** The exchange ended before the service finished it: the client disconnected or stalled. */
struct Abandoned {
	Exchange exchange;
};

/** A stream that refused an event has room for any event again. */
struct Drained {
	Exchange exchange;
};

template<BodyBounds B>
using Notice = std::variant<Arrived<B>, Abandoned, Drained>;

enum class Mode : std::uint8_t {
	Serving,
	/** No new connection or request is taken; exchanges in flight finish and their connections close. */
	Stopping,
};

enum class Listening : std::uint8_t {
	Unwatched,
	Watched,
};

enum class Flush : std::uint8_t {
	Open,
	Broken,
};

enum class Intake : std::uint8_t {
	Quiet,
	Progressed,
	/** The client closed or reset the connection. */
	Ended,
};

constexpr std::uint64_t listener_token = 0;
constexpr std::uint8_t generation_bits = 40;
constexpr auto accept_backoff = std::chrono::milliseconds{100};

/** A refused stream is told to resume once the buffer has room for the longest event and the stream's end. */
constexpr std::size_t drain_mark = max_stream_buffer_bytes - max_encoded_event_bytes - stream_end.size();

template<BodyBounds B>
using Table = handle::Table<Connection<B>, ConnectionHandle, max_connections, generation_bits>;

// A connection's token is its handle's bits; generations start at one, so no handle is the listener's zero.
static_assert(std::bit_width(max_connections - 1) + generation_bits < 63, "connection tokens stay below the poller's token limit");

template<BodyBounds B>
[[nodiscard]] auto deadline_of(Connection<B> const& connection) -> std::optional<net::Deadline> {
	auto phase = std::optional<net::Deadline>{};
	if (auto const idle = core::alternative<Idle>(connection.phase)) {
		phase = idle->read_by;
	} else if (auto const lingering = core::alternative<Lingering>(connection.phase)) {
		phase = lingering->until;
	}
	if (phase && connection.write_by) {
		return std::min(*phase, *connection.write_by);
	}
	return phase ? phase : connection.write_by;
}

/** Whether the connection reads now: for the next request, to see its client leave, or to drain it before closing. */
template<BodyBounds B>
[[nodiscard]] auto reads(Connection<B> const& connection) -> bool {
	return !std::holds_alternative<Flushing>(connection.phase) && !connection.inbound.full();
}

[[nodiscard]] auto abandonment(ConnectionHandle handle, Phase const& phase) -> std::optional<Exchange> {
	if (auto const awaiting = core::alternative<Awaiting>(phase)) {
		return Exchange{.connection = handle, .sequence = awaiting->sequence};
	}
	if (auto const streaming = core::alternative<Streaming>(phase)) {
		return Exchange{.connection = handle, .sequence = streaming->sequence};
	}
	return std::nullopt;
}

/**
 * The owner-driven state machine of every connection: readiness facts and service answers queue connections, and
 * advancing a queued connection moves its bytes and phase as far as they go, yielding at most one notice for the
 * service. Single-threaded: only the server's owner thread touches it.
 */
template<BodyBounds B>
class Connections {
public:
	[[nodiscard]] static auto open(ServerConfig config) -> core::Result<Connections, net::NetError> {
		core::invariant(config.read_timeout > std::chrono::milliseconds::zero());
		core::invariant(config.write_timeout > std::chrono::milliseconds::zero());
		auto poller = net::Poller::open();
		if (!poller) {
			return std::unexpected(poller.error());
		}
		auto listener = net::Listener::open(config.endpoint);
		if (!listener) {
			return std::unexpected(listener.error());
		}
		if (auto watched = poller->watch(*listener, listener_token); !watched) {
			return std::unexpected(watched.error());
		}
		return Connections{config, std::move(*poller), std::move(*listener)};
	}

	[[nodiscard]] auto port() const -> std::uint16_t {
		return port_;
	}

	[[nodiscard]] auto waker() const -> net::Waker {
		return poller_.waker();
	}

	[[nodiscard]] auto stop_on_signals() -> core::Result<void, net::NetError> {
		return poller_.stop_on_signals();
	}

	[[nodiscard]] auto stopping() const -> bool {
		return mode_ == Mode::Stopping;
	}

	[[nodiscard]] auto idle() const -> bool {
		return table_->live_count() == 0;
	}

	/** Waits until a fact, `until`, or the nearest connection or accept deadline. */
	[[nodiscard]] auto wait(std::optional<net::Deadline> until) -> core::Result<net::Facts, net::NetError> {
		auto deadline = until;
		auto const consider = [&deadline](std::optional<net::Deadline> candidate) {
			if (candidate && (!deadline || *candidate < *deadline)) {
				deadline = candidate;
			}
		};
		for (auto const handle : table_->handles()) {
			consider(deadline_of(slot(handle)));
		}
		consider(accept_after_);
		return poller_.wait_until(deadline);
	}

	/** Queues every connection whose deadline has passed: advancing it closes it. */
	auto expire(net::Deadline now) -> void {
		for (auto const handle : table_->handles()) {
			if (auto const deadline = deadline_of(slot(handle)); deadline && *deadline <= now) {
				enqueue(handle);
			}
		}
		if (accept_after_ && *accept_after_ <= now) {
			accept_after_.reset();
		}
	}

	[[nodiscard]] auto ready(net::Ready fact, net::Deadline now) -> core::Result<void, net::NetError> {
		if (fact.token == listener_token) {
			listening_ = Listening::Unwatched;
			return accept_all(now);
		}
		auto const handle = ConnectionHandle{.bits = fact.token};
		if (table_->try_get(handle)) {
			enqueue(handle);
		}
		return {};
	}

	/** Advances queued connections until one yields a notice; nullopt when the queue is empty. */
	[[nodiscard]] auto next_notice(net::Deadline now) -> std::optional<Notice<B>> {
		while (pending_count_ > 0) {
			auto const handle = pending_[pending_head_];
			pending_head_ = (pending_head_ + 1) % pending_.size();
			--pending_count_;
			auto found = table_->try_get(handle);
			if (!found) {
				continue;
			}
			found->queued = Queued::No;
			if (auto notice = advance(handle, now)) {
				return notice;
			}
		}
		return std::nullopt;
	}

	/** Watches the listener again when the server can take a connection. */
	[[nodiscard]] auto listen() -> core::Result<void, net::NetError> {
		if (mode_ == Mode::Stopping || !listener_ || listening_ == Listening::Watched || accept_after_ ||
		    table_->live_count() == max_connections) {
			return {};
		}
		if (auto watched = poller_.watch(*listener_, listener_token); !watched) {
			return std::unexpected(watched.error());
		}
		listening_ = Listening::Watched;
		return {};
	}

	/** Stops listening and closes every connection between requests; exchanges in flight finish. */
	auto begin_stop() -> void {
		mode_ = Mode::Stopping;
		listener_.reset();
		// Collected first: closing releases slots, and the table must hold still while its handles are listed.
		auto const idle = table_->handles() | std::views::filter([this](ConnectionHandle handle) {
			                  return std::holds_alternative<Idle>(slot(handle).phase);
		                  }) |
		                  std::ranges::to<std::vector>();
		for (auto const handle : idle) {
			std::ignore = close(handle);
		}
	}

	/** Closes one connection at once; the exchange it abandoned, if any. */
	[[nodiscard]] auto close_any() -> std::optional<Exchange> {
		auto handles = table_->handles();
		core::invariant(!std::ranges::empty(handles));
		return close(*handles.begin());
	}

	auto answer(Exchange exchange, Answer answer) -> void {
		if (auto const response = core::alternative<Response>(answer)) {
			core::invariant(respond(exchange, *response) == Delivery::Accepted);
		} else if (std::holds_alternative<OpenStream>(answer)) {
			core::invariant(open_stream(exchange) == Delivery::Accepted);
		}
	}

	[[nodiscard]] auto respond(Exchange exchange, Response const& response) -> Delivery {
		auto found = find(exchange);
		if (!found) {
			return Delivery::Gone;
		}
		auto& connection = *found;
		auto const awaiting = core::alternative<Awaiting>(connection.phase);
		// A service answers each exchange once: a second answer is a bug in the service, not a client's doing.
		core::invariant(awaiting.has_value());
		auto const persistence = awaiting->persistence;
		connection.outbound.adopt(response.body.size() <= max_response_body_bytes ? encode_response(response, persistence, awaiting->id)
		                                                                          : encode_oversized(persistence, awaiting->id));
		connection.phase = Flushing{.persistence = persistence};
		enqueue(exchange.connection);
		return Delivery::Accepted;
	}

	[[nodiscard]] auto open_stream(Exchange exchange) -> Delivery {
		auto found = find(exchange);
		if (!found) {
			return Delivery::Gone;
		}
		auto& connection = *found;
		auto const awaiting = core::alternative<Awaiting>(connection.phase);
		core::invariant(awaiting.has_value());
		auto const persistence = awaiting->persistence;
		connection.outbound.append(encode_stream_head(persistence, awaiting->id));
		connection.phase = Streaming{.sequence = exchange.sequence, .persistence = persistence, .backpressure = Backpressure::Clear};
		enqueue(exchange.connection);
		return Delivery::Accepted;
	}

	[[nodiscard]] auto send(Exchange exchange, Event const& event) -> Flow {
		auto found = find(exchange);
		if (!found) {
			return Flow::Gone;
		}
		auto& connection = *found;
		auto const streaming = core::alternative<Streaming>(connection.phase);
		// Events go only to an exchange the service opened as a stream and has not finished.
		core::invariant(streaming.has_value());
		core::invariant(event.data.size() <= max_event_bytes);
		auto const size = encoded_size(event);
		if (connection.outbound.queued() + size + stream_end.size() > max_stream_buffer_bytes) {
			streaming->backpressure = Backpressure::Owed;
			return Flow::Full;
		}
		connection.outbound.emplace(size, [&event](std::span<char> into) {
			encode_event(event, into);
		});
		enqueue(exchange.connection);
		return Flow::Accepted;
	}

	[[nodiscard]] auto finish(Exchange exchange) -> Delivery {
		auto found = find(exchange);
		if (!found) {
			return Delivery::Gone;
		}
		auto& connection = *found;
		auto const streaming = core::alternative<Streaming>(connection.phase);
		core::invariant(streaming.has_value());
		connection.outbound.append(stream_end);
		connection.phase = Flushing{.persistence = streaming->persistence};
		enqueue(exchange.connection);
		return Delivery::Accepted;
	}

private:
	Connections(ServerConfig config, net::Poller poller, net::Listener listener)
	    : config_{config},
	      poller_{std::move(poller)},
	      port_{listener.port()},
	      listener_{std::move(listener)},
	      table_{std::make_unique<Table<B>>()},
	      ids_{config.id_seed} {}

	/** The connection of a handle known to be live. */
	[[nodiscard]] auto slot(ConnectionHandle handle) -> Connection<B>& {
		auto& table = *table_;
		return table.get(handle);
	}

	[[nodiscard]] auto find(Exchange exchange) -> std::optional<Connection<B>&> {
		auto found = table_->try_get(exchange.connection);
		if (!found || found->sequence != exchange.sequence) {
			return std::nullopt;
		}
		return found;
	}

	auto enqueue(ConnectionHandle handle) -> void {
		auto& connection = slot(handle);
		if (connection.queued == Queued::Yes) {
			return;
		}
		// A connection is queued at most once and closed only after it leaves the queue, so the queue never holds
		// more than every live connection.
		core::invariant(pending_count_ < pending_.size());
		connection.queued = Queued::Yes;
		pending_[(pending_head_ + pending_count_) % pending_.size()] = handle;
		++pending_count_;
	}

	[[nodiscard]] auto close(ConnectionHandle handle) -> std::optional<Exchange> {
		auto abandoned = abandonment(handle, slot(handle).phase);
		table_->release(handle);
		return abandoned;
	}

	[[nodiscard]] auto closed(ConnectionHandle handle) -> std::optional<Notice<B>> {
		if (auto abandoned = close(handle)) {
			return Abandoned{.exchange = *abandoned};
		}
		return std::nullopt;
	}

	[[nodiscard]] auto accept_all(net::Deadline now) -> core::Result<void, net::NetError> {
		while (mode_ == Mode::Serving && listener_ && table_->live_count() < max_connections) {
			auto accepted = listener_->accept();
			if (!accepted) {
				if (accepted.error().retry() == core::Retry::Transient) {
					accept_after_ = now + accept_backoff;
					return {};
				}
				return std::unexpected(accepted.error());
			}
			if (!*accepted) {
				return {};
			}
			auto inserted = table_->insert(Connection<B>{
			    .socket = std::move(**accepted),
			    .inbound = {},
			    .outbound = {},
			    .phase = Idle{.read_by = now + config_.read_timeout},
			    .write_by = std::nullopt,
			    .sequence = 0,
			    .queued = Queued::No,
			});
			// Every free slot has exhausted its generations: the connection is dropped as it goes out of scope.
			if (!inserted) {
				continue;
			}
			enqueue(*inserted);
		}
		return {};
	}

	[[nodiscard]] auto receive(Connection<B>& connection) -> Intake {
		auto progressed = false;
		while (reads(connection)) {
			auto const received = connection.socket.receive(connection.inbound.space());
			if (auto const moved = core::alternative<net::Transferred>(received)) {
				connection.inbound.commit(moved->count);
				if (std::holds_alternative<Lingering>(connection.phase)) {
					connection.inbound.discard();
				}
				progressed = true;
				continue;
			}
			if (std::holds_alternative<net::WouldBlock>(received)) {
				break;
			}
			return Intake::Ended;
		}
		return progressed ? Intake::Progressed : Intake::Quiet;
	}

	[[nodiscard]] auto flush(Connection<B>& connection, net::Deadline now) const -> Flush {
		auto progressed = false;
		while (!connection.outbound.empty()) {
			auto const sent = connection.socket.send(connection.outbound.front());
			if (auto const moved = core::alternative<net::Transferred>(sent)) {
				connection.outbound.consume(moved->count);
				progressed = true;
				continue;
			}
			if (std::holds_alternative<net::WouldBlock>(sent)) {
				break;
			}
			return Flush::Broken;
		}
		if (connection.outbound.empty()) {
			connection.write_by.reset();
		} else if (progressed || !connection.write_by) {
			connection.write_by = now + config_.write_timeout;
		}
		return Flush::Open;
	}

	/** Moves one connection's bytes and phase as far as they go without waiting. */
	[[nodiscard]] auto advance(ConnectionHandle handle, net::Deadline now) -> std::optional<Notice<B>> {
		for (;;) {
			auto& connection = slot(handle);
			if (auto const deadline = deadline_of(connection); deadline && *deadline <= now) {
				return closed(handle);
			}
			if (!connection.outbound.empty() && flush(connection, now) == Flush::Broken) {
				return closed(handle);
			}
			if (auto const streaming = core::alternative<Streaming>(connection.phase);
			    streaming.has_value() && streaming->backpressure == Backpressure::Owed && connection.outbound.queued() <= drain_mark) {
				streaming->backpressure = Backpressure::Clear;
				enqueue(handle);
				return Drained{.exchange = Exchange{.connection = handle, .sequence = streaming->sequence}};
			}
			if (auto const flushing = core::alternative<Flushing>(connection.phase); flushing.has_value() && connection.outbound.empty()) {
				if (flushing->persistence == Persistence::KeepAlive && mode_ == Mode::Serving) {
					connection.phase = Idle{.read_by = now + config_.read_timeout};
				} else {
					connection.socket.shut_write();
					connection.inbound.discard();
					connection.phase = Lingering{.until = now + linger_duration};
				}
				continue;
			}
			auto const intake = receive(connection);
			if (intake == Intake::Ended) {
				return closed(handle);
			}
			if (std::holds_alternative<Idle>(connection.phase)) {
				auto assembly = connection.inbound.assemble();
				if (auto const assembled = core::alternative<Assembled<B>>(assembly)) {
					++connection.sequence;
					auto id = assembled->client_id ? std::move(*assembled->client_id) : ids_.issue();
					auto const persistence = mode_ == Mode::Serving ? assembled->persistence : Persistence::Close;
					connection.phase = Awaiting{.sequence = connection.sequence, .id = id, .persistence = persistence};
					enqueue(handle);
					return Arrived<B>{
					    .exchange = Exchange{.connection = handle, .sequence = connection.sequence},
					    .request = Request<B>{.id = std::move(id), .route = std::move(assembled->route)},
					};
				}
				if (std::holds_alternative<Continue>(assembly)) {
					connection.outbound.append(interim_continue);
					continue;
				}
				if (auto const refused = core::alternative<Refused>(assembly)) {
					connection.outbound.adopt(encode_refusal(refused->error.kind, ids_.issue()));
					connection.phase = Flushing{.persistence = Persistence::Close};
					continue;
				}
			}
			if (intake == Intake::Quiet) {
				return settle(handle);
			}
		}
	}

	/** Watches the connection for what it waits on; a connection the kernel refuses to watch cannot progress. */
	[[nodiscard]] auto settle(ConnectionHandle handle) -> std::optional<Notice<B>> {
		auto const& connection = slot(handle);
		auto const read = reads(connection);
		auto const write = !connection.outbound.empty();
		if (!read && !write) {
			return std::nullopt;
		}
		auto const interest = read && write ? net::Interest::ReadWrite : (read ? net::Interest::Read : net::Interest::Write);
		if (!poller_.watch(connection.socket, handle.bits, interest)) {
			return closed(handle);
		}
		return std::nullopt;
	}

	ServerConfig config_;
	net::Poller poller_;
	std::uint16_t port_;
	std::optional<net::Listener> listener_;
	Listening listening_ = Listening::Watched;
	std::optional<net::Deadline> accept_after_;
	std::unique_ptr<Table<B>> table_;
	std::array<ConnectionHandle, max_connections> pending_{};
	std::size_t pending_head_ = 0;
	std::size_t pending_count_ = 0;
	IdIssuer ids_;
	Mode mode_ = Mode::Serving;
};

}

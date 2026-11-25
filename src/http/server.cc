// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:server;

import std;
import engine.core;
import engine.net;
import :connections;
import :exchange;
import :limits;
import :request;
import :write;

namespace engine::http {

export template<std::movable Message, BodyBounds B>
class Server;

/**
 * What a service may do to exchanges, on the owner thread, during `deliver` and `drained`. Every operation names
 * an exchange by value; one that is over reports `Gone`. Using an exchange against its protocol (answering it twice,
 * sending events to an exchange that is not an open stream, an event longer than max_event_bytes) is a bug in the
 * service and violates an invariant.
 */
export template<BodyBounds B>
class Port {
public:
	Port(Port const&) = delete;
	auto operator=(Port const&) -> Port& = delete;
	Port(Port&&) = delete;
	auto operator=(Port&&) -> Port& = delete;
	~Port() = default;

	/** Answers a deferred exchange with a whole response. */
	[[nodiscard]] auto respond(Exchange exchange, Response const& response) -> Delivery {
		return connections_.respond(exchange, response);
	}

	/** Answers a deferred exchange with a stream: 200, text/event-stream, chunked. */
	[[nodiscard]] auto open_stream(Exchange exchange) -> Delivery {
		return connections_.open_stream(exchange);
	}

	/** Queues one event on an open stream, or refuses it when the client is behind by a full buffer. */
	[[nodiscard]] auto send(Exchange exchange, Event const& event) -> Flow {
		return connections_.send(exchange, event);
	}

	/** Ends an open stream; the connection carries the next request once the client has read it. */
	[[nodiscard]] auto finish(Exchange exchange) -> Delivery {
		return connections_.finish(exchange);
	}

private:
	explicit Port(Connections<B>& connections) : connections_{connections} {}

	Connections<B>& connections_;

	template<std::movable, BodyBounds>
	friend class Server;
};

/**
 * The application behind the server: everything the server tells it, on the owner thread. Every call is bounded
 * and nonblocking; blocking work is a sender on another scheduler, whose results come back as messages.
 *
 * - `serve`: a request arrived; answer it now or later.
 * - `deliver`: a message another thread posted arrived.
 * - `drained`: a stream that refused an event (Flow::Full) has room for any event again.
 * - `abandoned`: the exchange ended before the service finished it (client disconnected or stalled, or the server
 *   stopped hard); the service cancels its work. After it, the exchange is `Gone`.
 */
export template<class S, class Message, BodyBounds B>
concept Service = requires(S& service, Exchange exchange, Request<B> request, Message message, Port<B>& port) {
	{ service.serve(exchange, std::move(request)) } -> std::same_as<Answer>;
	{ service.deliver(std::move(message), port) } -> std::same_as<void>;
	{ service.drained(exchange, port) } -> std::same_as<void>;
	{ service.abandoned(exchange) } -> std::same_as<void>;
};

export enum class Turn : std::uint8_t {
	Running,
	/** Stopped and drained (or stopped hard): no connection is open. */
	Stopped,
};

/** Posts messages to a server's service from any thread. */
export template<class Message>
using Poster = net::Poster<Message, max_posted_messages>;

template<class S, BodyBounds B>
struct Dispatch {
	S& service;
	Connections<B>& connections;
	Port<B>& port;

	auto operator()(Arrived<B>& arrived) const -> void {
		connections.answer(arrived.exchange, service.serve(arrived.exchange, std::move(arrived.request)));
	}

	auto operator()(Abandoned const& abandoned) const -> void {
		service.abandoned(abandoned.exchange);
	}

	auto operator()(Drained const& drained) const -> void {
		service.drained(drained.exchange, port);
	}
};

/**
 * The http server: one owner thread runs it, and it owns every connection. `B` bounds and pads the body of a
 * response to create. The first SIGINT or SIGTERM (once subscribed) or `stop` stops taking connections and requests
 * and lets exchanges in flight finish; a second stop closes everything at once, abandoning what is in flight.
 */
export template<std::movable Message, BodyBounds B>
class Server {
public:
	using Mailbox = net::Mailbox<Message, max_posted_messages>;

	[[nodiscard]] static auto open(ServerConfig config) -> core::Result<Server, net::NetError> {
		auto connections = Connections<B>::open(config);
		if (!connections) {
			return std::unexpected(connections.error());
		}
		auto mailbox = std::make_unique<Mailbox>(connections->waker());
		return Server{std::move(*connections), std::move(mailbox)};
	}

	/** The port the server listens on (the kernel's choice when the config asked for zero). */
	[[nodiscard]] auto port() const -> std::uint16_t {
		return connections_.port();
	}

	/** A poster for work the service starts on other threads. The server outlives that work. */
	[[nodiscard]] auto poster() -> Poster<Message> {
		return Poster<Message>{*mailbox_};
	}

	/** Makes SIGINT and SIGTERM stop the server, for its lifetime. On linux, call before starting other threads. */
	[[nodiscard]] auto stop_on_signals() -> core::Result<void, net::NetError> {
		return connections_.stop_on_signals();
	}

	/** Stops as the first signal does. */
	auto stop() -> void {
		if (!connections_.stopping()) {
			connections_.begin_stop();
		}
	}

	/** Turns until the server has stopped. */
	template<Service<Message, B> S>
	[[nodiscard]] auto run(S& service) -> core::Result<void, net::NetError> {
		for (;;) {
			auto const turned = turn(service, std::nullopt);
			if (!turned) {
				return std::unexpected(turned.error());
			}
			if (*turned == Turn::Stopped) {
				return {};
			}
		}
	}

	/**
	 * One wait (until a fact or `until`) and everything it makes possible: stop facts first, then expiries,
	 * messages, accepts, and every connection advanced as far as it goes.
	 */
	template<Service<Message, B> S>
	[[nodiscard]] auto turn(S& service, std::optional<net::Deadline> until) -> core::Result<Turn, net::NetError> {
		if (connections_.stopping() && connections_.idle()) {
			return Turn::Stopped;
		}
		auto const facts = connections_.wait(until);
		if (!facts) {
			return std::unexpected(facts.error());
		}
		auto const view = facts->view();
		auto const stops = std::ranges::count_if(view, [](net::Fact const& fact) {
			return std::holds_alternative<net::StopRequested>(fact);
		});
		if (stops > 0) {
			if (connections_.stopping() || stops > 1) {
				while (!connections_.idle()) {
					if (auto const abandoned = connections_.close_any()) {
						service.abandoned(*abandoned);
					}
				}
				return Turn::Stopped;
			}
			connections_.begin_stop();
		}

		auto const now = net::Clock::now();
		connections_.expire(now);
		auto port = Port<B>{connections_};
		for (auto const& fact : view) {
			if (std::holds_alternative<net::Woken>(fact)) {
				mailbox_->drain([&service, &port](Message&& message) {
					service.deliver(std::move(message), port);
				});
			} else if (auto const ready = core::alternative<net::Ready>(fact)) {
				if (auto accepted = connections_.ready(*ready, now); !accepted) {
					return std::unexpected(accepted.error());
				}
			}
		}
		auto const dispatch = Dispatch<S, B>{.service = service, .connections = connections_, .port = port};
		while (auto notice = connections_.next_notice(now)) {
			std::visit(dispatch, *notice);
		}

		if (connections_.stopping() && connections_.idle()) {
			return Turn::Stopped;
		}
		if (auto listening = connections_.listen(); !listening) {
			return std::unexpected(listening.error());
		}
		return Turn::Running;
	}

private:
	Server(Connections<B> connections, std::unique_ptr<Mailbox> mailbox)
	    : connections_{std::move(connections)}, mailbox_{std::move(mailbox)} {}

	Connections<B> connections_;
	std::unique_ptr<Mailbox> mailbox_;
};

}

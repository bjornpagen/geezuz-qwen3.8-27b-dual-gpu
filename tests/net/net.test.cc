// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.net;
import engine.testing;

namespace net = engine::net;
using namespace std::chrono_literals;

namespace {

[[nodiscard]] auto within(std::chrono::milliseconds span) -> net::Deadline {
	return net::Clock::now() + span;
}

[[nodiscard]] auto ready_on(std::uint64_t token) {
	return [token](net::Fact const& fact) {
		return std::holds_alternative<net::Ready>(fact) && std::get<net::Ready>(fact).token == token;
	};
}

/** Waits for a fact satisfying `wanted`, for at most two seconds. */
template<class F>
[[nodiscard]] auto await_fact(net::Poller& poller, F wanted) -> bool {
	auto const deadline = within(2000ms);
	while (net::Clock::now() < deadline) {
		auto const facts = poller.wait_until(deadline);
		if (!facts) {
			return false;
		}
		if (std::ranges::any_of(facts->view(), wanted)) {
			return true;
		}
	}
	return false;
}

struct Pair {
	net::Socket client;
	net::Socket server;
};

constexpr std::uint64_t listener_token = 1;
constexpr std::uint64_t server_token = 2;

[[nodiscard]] auto connected_pair(net::Poller& poller) -> std::optional<Pair> {
	auto listener = net::Listener::open(net::Endpoint{.address = net::loopback, .port = 0});
	if (!listener) {
		return std::nullopt;
	}
	auto client = net::Socket::connect(net::Endpoint{.address = net::loopback, .port = listener->port()});
	if (!client || !poller.watch(*listener, listener_token) || !await_fact(poller, ready_on(listener_token))) {
		return std::nullopt;
	}
	auto accepted = listener->accept();
	if (!accepted || !*accepted) {
		return std::nullopt;
	}
	return Pair{.client = std::move(*client), .server = std::move(**accepted)};
}

/** Receives from `socket` until it reports something other than WouldBlock, waiting on its readiness. */
[[nodiscard]] auto receive_next(net::Poller& poller, net::Socket& socket, std::span<char> into) -> net::Received {
	auto const deadline = within(2000ms);
	for (;;) {
		auto received = socket.receive(into);
		if (!std::holds_alternative<net::WouldBlock>(received) || net::Clock::now() >= deadline) {
			return received;
		}
		if (!poller.watch(socket, server_token, net::Interest::Read) || !poller.wait_until(deadline)) {
			return received;
		}
	}
}

[[nodiscard]] auto classification_is_the_policy() -> bool {
	auto const retry = [](net::Cause cause) {
		return net::NetError{.stage = net::Stage::Accept, .cause = cause, .code = 0}.retry();
	};
	using enum engine::core::Retry;
	return retry(net::Cause::Exhausted) == Transient && retry(net::Cause::AddressInUse) == Transient &&
	       retry(net::Cause::Refused) == Transient && retry(net::Cause::Denied) == Permanent && retry(net::Cause::Invalid) == Permanent;
}

[[nodiscard]] auto connecting_to_a_closed_port_is_refused() -> bool {
	auto port = std::uint16_t{0};
	{
		auto const listener = net::Listener::open(net::Endpoint{.address = net::loopback, .port = 0});
		if (!listener) {
			return false;
		}
		port = listener->port();
	}
	auto const client = net::Socket::connect(net::Endpoint{.address = net::loopback, .port = port});
	return !client && client.error().stage == net::Stage::Connect && client.error().cause == net::Cause::Refused &&
	       client.error().retry() == engine::core::Retry::Transient;
}

/** Bytes sent in random pieces arrive whole and in order, however the kernel splits them. */
[[nodiscard]] auto bytes_arrive_intact(std::uint64_t seed) -> bool {
	auto poller = net::Poller::open();
	if (!poller) {
		return false;
	}
	auto pair = connected_pair(*poller);
	if (!pair) {
		return false;
	}
	auto rng = engine::testing::generator(seed);
	auto sent_bytes = std::string(std::size_t{1} << 18U, '\0');
	std::ranges::generate(sent_bytes, [&rng] {
		return static_cast<char>(rng() & 0xffU);
	});
	auto received_bytes = std::string{};
	auto sent = std::size_t{0};
	auto buffer = std::array<char, 9000>{};
	auto const deadline = within(10000ms);
	while (received_bytes.size() < sent_bytes.size() && net::Clock::now() < deadline) {
		if (sent < sent_bytes.size()) {
			auto const piece = std::min<std::size_t>(1 + (rng() % 20000), sent_bytes.size() - sent);
			auto const outcome = pair->client.send(std::span{sent_bytes}.subspan(sent, piece));
			if (std::holds_alternative<net::Transferred>(outcome)) {
				sent += std::get<net::Transferred>(outcome).count;
			} else if (std::holds_alternative<net::Broken>(outcome)) {
				return false;
			}
		}
		auto const into = std::span{buffer}.first(1 + (rng() % buffer.size()));
		auto const received = pair->server.receive(into);
		if (std::holds_alternative<net::Transferred>(received)) {
			auto const arrived = into.first(std::get<net::Transferred>(received).count);
			received_bytes.append(arrived.begin(), arrived.end());
		} else if (std::holds_alternative<net::WouldBlock>(received)) {
			if (!poller->watch(pair->server, server_token, net::Interest::Read) || !poller->wait_until(within(5ms))) {
				return false;
			}
		} else {
			return false;
		}
	}
	return received_bytes == sent_bytes;
}

[[nodiscard]] auto an_empty_socket_would_block() -> bool {
	auto poller = net::Poller::open();
	auto pair = poller ? connected_pair(*poller) : std::nullopt;
	auto buffer = std::array<char, 16>{};
	return pair && std::holds_alternative<net::WouldBlock>(pair->server.receive(buffer));
}

[[nodiscard]] auto a_half_close_reads_as_end_of_stream_after_the_data() -> bool {
	auto poller = net::Poller::open();
	auto pair = poller ? connected_pair(*poller) : std::nullopt;
	if (!pair) {
		return false;
	}
	constexpr std::string_view farewell = "bye";
	if (pair->client.send(farewell) != net::Sent{net::Transferred{.count = farewell.size()}}) {
		return false;
	}
	pair->client.shut_write();
	auto buffer = std::array<char, 16>{};
	auto const first = receive_next(*poller, pair->server, buffer);
	auto const second = receive_next(*poller, pair->server, buffer);
	return first == net::Received{net::Transferred{.count = farewell.size()}} &&
	       std::string_view{buffer.data(), farewell.size()} == farewell && std::holds_alternative<net::EndOfStream>(second);
}

[[nodiscard]] auto an_aborted_peer_reads_as_broken() -> bool {
	auto poller = net::Poller::open();
	auto pair = poller ? connected_pair(*poller) : std::nullopt;
	if (!pair) {
		return false;
	}
	std::move(pair->client).abort();
	auto buffer = std::array<char, 16>{};
	return std::holds_alternative<net::Broken>(receive_next(*poller, pair->server, buffer));
}

[[nodiscard]] auto a_send_to_a_reset_peer_is_broken_without_a_signal() -> bool {
	auto poller = net::Poller::open();
	auto pair = poller ? connected_pair(*poller) : std::nullopt;
	if (!pair) {
		return false;
	}
	std::move(pair->client).abort();
	auto buffer = std::array<char, 16>{};
	std::ignore = receive_next(*poller, pair->server, buffer);
	// Writing to a reset connection raises SIGPIPE unless the socket suppresses it; the process must survive.
	constexpr std::string_view payload = "after the reset";
	auto broken = false;
	for (auto attempt = 0; attempt < 4 && !broken; ++attempt) {
		broken = std::holds_alternative<net::Broken>(pair->server.send(payload));
	}
	return broken;
}

[[nodiscard]] auto a_wait_ends_at_its_deadline() -> bool {
	auto poller = net::Poller::open();
	if (!poller) {
		return false;
	}
	auto const started = net::Clock::now();
	auto const deadline = started + 50ms;
	auto const facts = poller->wait_until(deadline);
	auto const past = poller->wait_until(started);
	return facts && facts->count == 0 && net::Clock::now() >= deadline && past && past->count == 0;
}

[[nodiscard]] auto wakes_between_waits_coalesce() -> bool {
	auto poller = net::Poller::open();
	if (!poller) {
		return false;
	}
	poller->waker().wake();
	poller->waker().wake();
	auto const facts = poller->wait_until(within(1000ms));
	auto const after = poller->wait_until(within(20ms));
	auto const woken = [](net::Fact const& fact) {
		return std::holds_alternative<net::Woken>(fact);
	};
	return facts && std::ranges::count_if(facts->view(), woken) == 1 && after && after->count == 0;
}

[[nodiscard]] auto a_mailbox_delivers_in_order_within_its_bound() -> bool {
	auto poller = net::Poller::open();
	if (!poller) {
		return false;
	}
	auto mailbox = net::Mailbox<int, 4>{poller->waker()};
	auto const poster = net::Poster<int, 4>{mailbox};
	auto holds = true;
	for (auto value = 0; value < 4; ++value) {
		holds = holds && poster.post(value) == net::Posted::Accepted;
	}
	holds = holds && poster.post(4) == net::Posted::Full;
	holds = holds && await_fact(*poller, [](net::Fact const& fact) {
		        return std::holds_alternative<net::Woken>(fact);
	        });
	auto drained = std::vector<int>{};
	mailbox.drain([&](int value) {
		drained.push_back(value);
		// A handler may post: the message waits for the next drain, and the owner is woken for it.
		if (value == 0) {
			holds = holds && poster.post(10) == net::Posted::Accepted;
		}
	});
	holds = holds && drained == std::vector{0, 1, 2, 3};
	holds = holds && await_fact(*poller, [](net::Fact const& fact) {
		        return std::holds_alternative<net::Woken>(fact);
	        });
	drained.clear();
	mailbox.drain([&](int value) {
		drained.push_back(value);
	});
	return holds && drained == std::vector{10};
}

[[nodiscard]] auto stop_signals_arrive_first_in_the_batch(net::StopSignal signal) -> bool {
	auto poller = net::Poller::open();
	if (!poller || !poller->stop_on_signals()) {
		return false;
	}
	poller->waker().wake();
	net::raise_stop(signal);
	auto const facts = poller->wait_until(within(1000ms));
	return facts && facts->count >= 1 && std::holds_alternative<net::StopRequested>(facts->view().front());
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "retryability: pressure, a held address and a refusal are transient, the rest permanent",
		                      .holds = classification_is_the_policy()},
	    engine::testing::Case{.name = "connecting to a closed port is a transient refusal",
		                      .holds = connecting_to_a_closed_port_is_refused()},
	    engine::testing::Case{.name = "256 KiB sent in random pieces arrive intact (seed 1)", .holds = bytes_arrive_intact(1)},
	    engine::testing::Case{.name = "256 KiB sent in random pieces arrive intact (seed 2)", .holds = bytes_arrive_intact(2)},
	    engine::testing::Case{.name = "receiving from a socket with nothing pending would block", .holds = an_empty_socket_would_block()},
	    engine::testing::Case{.name = "a half-closed peer's data arrives, then end of stream",
		                      .holds = a_half_close_reads_as_end_of_stream_after_the_data()},
	    engine::testing::Case{.name = "an aborted peer reads as a broken connection", .holds = an_aborted_peer_reads_as_broken()},
	    engine::testing::Case{.name = "sending to a reset peer is broken and raises no SIGPIPE",
		                      .holds = a_send_to_a_reset_peer_is_broken_without_a_signal()},
	    engine::testing::Case{.name = "a wait with nothing ready ends at its deadline, a past deadline at once",
		                      .holds = a_wait_ends_at_its_deadline()},
	    engine::testing::Case{.name = "two wakes between waits are one woken fact", .holds = wakes_between_waits_coalesce()},
	    engine::testing::Case{.name = "a mailbox delivers in order, refuses past its bound, and takes posts from a drain",
		                      .holds = a_mailbox_delivers_in_order_within_its_bound()},
	    engine::testing::Case{.name = "SIGTERM is a stop fact, first in its batch",
		                      .holds = stop_signals_arrive_first_in_the_batch(net::StopSignal::Terminate)},
	    engine::testing::Case{.name = "SIGINT is a stop fact, first in its batch",
		                      .holds = stop_signals_arrive_first_in_the_batch(net::StopSignal::Interrupt)},
	};
	return engine::testing::report(cases);
}

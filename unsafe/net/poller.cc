// SPDX-License-Identifier: AGPL-3.0-only
module;

// The stop signals and kill(2), for the test double that delivers one.
#include <csignal>
#include <unistd.h>

export module engine.net:poller;

import std;
import engine.core;
import :error;
import :fact;
import :socket;
import :wait;

namespace engine::net {

/**
 * Wakes one poller's owner from any thread: its next wait reports `Woken`. Wakes between two waits coalesce. A
 * waker borrows its poller: the poller's owner outlives every thread that holds one.
 */
export class Waker {
public:
	auto wake() const noexcept -> void {
		net::wake(target_);
	}

private:
	explicit Waker(int target) : target_{target} {}

	/* SAFETY: a descriptor number borrowed from the poller, which outlives every waker it hands out */
	int target_;

	friend class Poller;
};

/**
 * The readiness half of the reactor: one owner thread prepares one-shot interests under integer tokens, waits until
 * an absolute deadline, and receives facts naming those tokens. It never holds an address and never calls back.
 */
export class Poller {
public:
	[[nodiscard]] static auto open() -> core::Result<Poller, NetError> {
		auto queue = Queue::open();
		if (!queue) {
			return std::unexpected(queue.error());
		}
		return Poller{std::move(*queue)};
	}

	/** Reports `socket` once under `token` when it can make progress in the direction `interest` names. */
	[[nodiscard]] auto watch(Socket const& socket, std::uint64_t token, Interest interest) -> core::Result<void, NetError> {
		core::invariant(token < token_limit);
		return queue_.watch(socket.fd_.get(), token, interest);
	}

	/** Reports `listener` once under `token` when a connection is pending. */
	[[nodiscard]] auto watch(Listener const& listener, std::uint64_t token) -> core::Result<void, NetError> {
		core::invariant(token < token_limit);
		return queue_.watch(listener.fd_.get(), token, Interest::Read);
	}

	/** From now until the poller is destroyed, SIGINT and SIGTERM arrive as `StopRequested` facts. */
	[[nodiscard]] auto stop_on_signals() -> core::Result<void, NetError> {
		return queue_.subscribe_stop();
	}

	/** Blocks until a fact or the deadline; no deadline waits for a fact. Stop facts come first in the batch; the order of the rest carries
	 * no meaning. */
	[[nodiscard]] auto wait_until(std::optional<Deadline> deadline) -> core::Result<Facts, NetError> {
		auto facts = queue_.wait_until(deadline);
		if (facts) {
			auto const items = std::span{facts->items}.first(facts->count);
			std::ignore = std::ranges::partition(items, [](Fact const& fact) {
				return std::holds_alternative<StopRequested>(fact);
			});
		}
		return facts;
	}

	[[nodiscard]] auto waker() const -> Waker {
		return Waker{queue_.wake_target()};
	}

private:
	explicit Poller(Queue queue) : queue_{std::move(queue)} {}

	Queue queue_;
};

export enum class StopSignal : std::uint8_t {
	Interrupt,
	Terminate,
};

/**
 * Sends SIGINT or SIGTERM to this process, as an operator's kill does: a test double for stopping the server. It is
 * process-directed on purpose; darwin's signal filter does not see a thread-directed raise of an ignored signal.
 */
export auto raise_stop(StopSignal signal) -> void {
	switch (signal) {
	case StopSignal::Interrupt:
		std::ignore = ::kill(::getpid(), SIGINT);
		return;
	case StopSignal::Terminate:
		std::ignore = ::kill(::getpid(), SIGTERM);
		return;
	}
}

}

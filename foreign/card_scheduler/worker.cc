// SPDX-License-Identifier: AGPL-3.0-only
export module engine.card_scheduler:worker;

import std;
import engine.core;
import engine.net;

namespace engine::card_scheduler {

export template<std::movable M, std::size_t Capacity, std::movable Outcome>
class Worker;

/**
 * The worker thread's end of its intake: messages posted from other threads, a wait that sleeps until one arrives,
 * and whether the worker is being stopped. Only the worker's thread uses it.
 */
export template<std::movable M, std::size_t Capacity>
class Inbox {
public:
	Inbox(Inbox const&) = delete;
	auto operator=(Inbox const&) -> Inbox& = delete;
	Inbox(Inbox&&) = delete;
	auto operator=(Inbox&&) -> Inbox& = delete;
	~Inbox() = default;

	/** Hands every message posted so far to `each`, oldest first; never blocks. */
	template<std::invocable<M&&> F>
	auto drain(F each) -> void {
		mailbox_.drain(std::move(each));
	}

	/** Sleeps until a message is posted, the worker is asked to stop, or `deadline` passes, whichever is first. */
	[[nodiscard]] auto wait(std::optional<net::Deadline> deadline) -> core::Result<void, net::NetError> {
		if (stopping()) {
			return {};
		}
		return poller_.wait_until(deadline).transform([](net::Facts const& /*facts*/) {});
	}

	/** The worker's owner is stopping it: the thread finishes what it holds and returns. */
	[[nodiscard]] auto stopping() const -> bool {
		return stop_.stop_requested();
	}

private:
	Inbox(net::Poller poller, std::stop_token stop) : poller_{std::move(poller)}, mailbox_{poller_.waker()}, stop_{std::move(stop)} {}

	net::Poller poller_;
	net::Mailbox<M, Capacity> mailbox_;
	std::stop_token stop_;

	template<std::movable, std::size_t, std::movable>
	friend class Worker;
};

/**
 * One dedicated thread and its intake: the execution context of work that may block, such as the engine's cycles,
 * which wait on the cards for milliseconds to seconds and so cannot run on the reactor's owner thread. Other threads
 * reach it only by posting owned messages; it owns everything its body holds, and hands back what the body returns.
 *
 * Stopping or destroying the worker asks the body to stop, wakes it, and joins it, so nothing the thread uses
 * outlives this owner and no message is handled after it is gone.
 */
export template<std::movable M, std::size_t Capacity, std::movable Outcome>
class Worker {
	/** Only `start` constructs a worker, and the worker must be in place before its thread starts. */
	struct Key {};

public:
	/** Starts a thread running `body(inbox)`. The body returns once `inbox.stopping()`, or when it cannot go on. */
	template<class Body>
	    requires std::same_as<std::invoke_result_t<Body&, Inbox<M, Capacity>&>, Outcome>
	[[nodiscard]] static auto start(Body body) -> core::Result<std::unique_ptr<Worker>, net::NetError> {
		auto poller = net::Poller::open();
		if (!poller) {
			return std::unexpected(poller.error());
		}
		auto worker = std::make_unique<Worker>(Key{}, std::move(*poller));
		worker->thread_ = std::jthread{[self = worker.get(), run = std::move(body)]() mutable {
			/* SAFETY: the worker joins this thread before any member it touches is destroyed, and reads the outcome only after the join */
			self->outcome_.emplace(std::invoke(run, self->inbox_));
		}};
		return worker;
	}

	Worker(Key /*key*/, net::Poller poller) : inbox_{std::move(poller), stop_.get_token()} {}

	Worker(Worker const&) = delete;
	auto operator=(Worker const&) -> Worker& = delete;
	Worker(Worker&&) = delete;
	auto operator=(Worker&&) -> Worker& = delete;

	~Worker() {
		halt();
	}

	/** Any thread but the worker's own. */
	[[nodiscard]] auto post(M message) -> net::Posted {
		return inbox_.mailbox_.post(std::move(message));
	}

	/** Stops the body, waits for it, and returns what it returned. */
	[[nodiscard]] auto stop() -> Outcome {
		halt();
		core::invariant(outcome_.has_value());
		return *std::exchange(outcome_, std::nullopt);
	}

private:
	auto halt() -> void {
		stop_.request_stop();
		inbox_.poller_.waker().wake();
		if (thread_.joinable()) {
			thread_.join();
		}
	}

	std::stop_source stop_;
	Inbox<M, Capacity> inbox_;
	std::optional<Outcome> outcome_;
	// Last, so the thread is joined before anything it reads or writes is destroyed.
	std::jthread thread_;
};

}

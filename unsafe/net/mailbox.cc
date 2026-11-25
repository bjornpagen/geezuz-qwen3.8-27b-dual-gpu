// SPDX-License-Identifier: AGPL-3.0-only
export module engine.net:mailbox;

import std;
import engine.core;
import :poller;

namespace engine::net {

export enum class Posted : std::uint8_t {
	Accepted,
	/** The owner has not drained `Capacity` messages: it is stalled, and the message was dropped. */
	Full,
};

/**
 * The one way into a reactor's owner thread from another thread: a bounded queue of owned messages, drained by the
 * owner between waits. The lock guards only the queue swap; no message is handled under it, so a handler may post.
 */
export template<std::movable T, std::size_t Capacity>
class Mailbox {
public:
	static_assert(Capacity > 0);

	explicit Mailbox(Waker waker) : waker_{waker} {
		queue_.reserve(Capacity);
		taken_.reserve(Capacity);
	}

	Mailbox(Mailbox&&) = delete;
	auto operator=(Mailbox&&) -> Mailbox& = delete;
	Mailbox(Mailbox const&) = delete;
	auto operator=(Mailbox const&) -> Mailbox& = delete;
	~Mailbox() = default;

	/** Any thread. Wakes the owner when the queue was empty; a nonempty queue already has a wake pending. */
	[[nodiscard]] auto post(T message) -> Posted {
		auto first = false;
		{
			auto const guard = std::lock_guard{lock_};
			if (queue_.size() == Capacity) {
				return Posted::Full;
			}
			first = queue_.empty();
			queue_.push_back(std::move(message));
		}
		if (first) {
			waker_.wake();
		}
		return Posted::Accepted;
	}

	/** Owner thread only: hands every message posted so far to `each`, oldest first. */
	template<std::invocable<T&&> F>
	auto drain(F each) -> void {
		core::invariant(taken_.empty());
		{
			auto const guard = std::lock_guard{lock_};
			std::swap(queue_, taken_);
		}
		for (auto& message : taken_) {
			each(std::move(message));
		}
		taken_.clear();
	}

private:
	std::mutex lock_;
	std::vector<T> queue_;
	std::vector<T> taken_;
	Waker waker_;
};

/**
 * A copyable capability to post into one mailbox from any thread. It borrows the mailbox: the mailbox's owner
 * outlives every piece of work that holds a poster (it joins that work before it is destroyed).
 */
export template<std::movable T, std::size_t Capacity>
class Poster {
public:
	explicit Poster(Mailbox<T, Capacity>& mailbox) : mailbox_{&mailbox} {}

	[[nodiscard]] auto post(T message) const -> Posted {
		return mailbox_->post(std::move(message));
	}

private:
	/* SAFETY: borrowed from the mailbox's owner, which outlives every poster it hands out */
	Mailbox<T, Capacity>* mailbox_;
};

}

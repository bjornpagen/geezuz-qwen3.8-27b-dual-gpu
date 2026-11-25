// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline:outbox;

import std;
import engine.http;
import engine.net;

namespace engine::pipeline {

/** Whether everything the outbox holds has reached the server's mailbox. */
export enum class Sent : std::uint8_t {
	All,
	/** The server's mailbox is full: its owner is behind. What is held goes first on the next flush. */
	Held,
};

/**
 * The engine's posts to the api's service, from the engine's thread: one batch per round (a cycle of eight slots
 * costs one mailbox entry) and the latest status. A batch the mailbox refuses is kept and posted again before
 * anything newer, so every request's outputs arrive in order; a status the mailbox refuses is replaced by a newer
 * one, since only the latest matters.
 */
export template<std::copyable Update>
class Outbox {
public:
	explicit Outbox(http::Poster<Update> poster) : poster_{poster} {}

	/** A round's outputs, posted after everything held before them. */
	auto add(Update batch) -> void {
		held_.push_back(std::move(batch));
	}

	auto report(Update status) -> void {
		status_ = std::move(status);
	}

	/** Posts everything held, oldest first, until the mailbox refuses one. */
	[[nodiscard]] auto flush() -> Sent {
		while (!held_.empty()) {
			// A refused post drops its message, so the held batch is posted as a copy and released once taken.
			if (poster_.post(held_.front()) == net::Posted::Full) {
				return Sent::Held;
			}
			held_.pop_front();
		}
		if (status_) {
			if (poster_.post(*status_) == net::Posted::Full) {
				return Sent::Held;
			}
			status_.reset();
		}
		return Sent::All;
	}

private:
	http::Poster<Update> poster_;
	std::deque<Update> held_;
	std::optional<Update> status_;
};

}

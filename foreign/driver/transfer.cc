// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:transfer;

import std;
import engine.core;
import :abi;
import :error;
import :stream;

namespace engine::driver {

/** What a finished transfer hands back: both owners, and whether the copy happened. */
export template<class Destination, class Source>
struct Delivered {
	Destination destination;
	Source source;
	core::Result<void, DeviceError> copied;
};

export class Context;

/**
 * An asynchronous copy in flight, holding both of its buffers. The copy reads and writes them after the call that
 * enqueued it has returned, so the buffers are moved in when it is enqueued and handed back only once its completion
 * event has fired: a buffer cannot be freed, reused or written by the host under a running copy, because nothing else
 * holds it. Destroying a transfer whose copy is still running is an invariant violation; once complete, dropping it
 * frees both buffers.
 *
 * A transfer whose copy could not be enqueued is complete at once, with the failure, and still hands both buffers
 * back: enqueueing never loses an owner.
 */
export template<class Destination, class Source>
class Transfer {
public:
	Transfer(Transfer const&) = delete;
	auto operator=(Transfer const&) -> Transfer& = delete;

	Transfer(Transfer&& other) noexcept : owners_{std::exchange(other.owners_, std::nullopt)}, progress_{std::move(other.progress_)} {}

	auto operator=(Transfer&& other) noexcept -> Transfer& {
		std::swap(owners_, other.owners_);
		std::swap(progress_, other.progress_);
		return *this;
	}

	~Transfer() {
		// Freeing a buffer the copy engine is still reading or writing is undefined on the card: the owner must wait.
		if (owners_) {
			core::invariant(poll() == EventState::Complete);
		}
	}

	/** Whether the copy has finished (or failed); a failed query ends the copy with that failure. */
	[[nodiscard]] auto poll() -> EventState {
		auto const* const event = std::get_if<Event>(&progress_);
		if (event == nullptr) {
			return EventState::Complete;
		}
		auto const state = event->query();
		if (!state) {
			progress_ = state.error();
			return EventState::Complete;
		}
		return *state;
	}

	/** Polls until the copy has finished or `deadline` passes; the state when it stopped. */
	[[nodiscard]] auto wait_until(std::chrono::steady_clock::time_point deadline) -> EventState {
		constexpr auto interval = std::chrono::microseconds{50};
		for (;;) {
			auto const state = poll();
			if (state == EventState::Complete || std::chrono::steady_clock::now() >= deadline) {
				return state;
			}
			std::this_thread::sleep_until(std::min(std::chrono::steady_clock::now() + interval, deadline));
		}
	}

	/** Both buffers and the copy's outcome once it has finished; nullopt while it runs or after they were taken. */
	[[nodiscard]] auto try_finish() -> std::optional<Delivered<Destination, Source>> {
		if (!owners_ || poll() == EventState::Pending) {
			return std::nullopt;
		}
		auto owners = std::exchange(owners_, std::nullopt);
		auto copied = std::holds_alternative<DeviceError>(progress_)
		                  ? core::Result<void, DeviceError>{std::unexpected(std::get<DeviceError>(progress_))}
		                  : core::Result<void, DeviceError>{};
		return Delivered<Destination, Source>{
		    .destination = std::move(owners->destination),
		    .source = std::move(owners->source),
		    .copied = std::move(copied),
		};
	}

private:
	friend class Context;

	struct Owners {
		Destination destination;
		Source source;
	};

	Transfer(Destination destination, Source source, std::variant<Event, DeviceError> progress)
	    : owners_{Owners{.destination = std::move(destination), .source = std::move(source)}}, progress_{std::move(progress)} {}

	std::optional<Owners> owners_;
	std::variant<Event, DeviceError> progress_;
};

}

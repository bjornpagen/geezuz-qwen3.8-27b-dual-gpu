// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:stream;

import std;
import engine.core;
import :abi;
import :error;

namespace engine::driver {

export class Context;
export class Event;

/** A non-blocking stream of one card: work on it is ordered, and it never waits on the legacy default stream. */
export class Stream {
public:
	Stream(Stream const&) = delete;
	auto operator=(Stream const&) -> Stream& = delete;

	Stream(Stream&& other) noexcept
	    : destroy_{std::exchange(other.destroy_, {})},
	      wait_event_{other.wait_event_},
	      ordinal_{other.ordinal_},
	      handle_{std::exchange(other.handle_, nullptr)} {}

	auto operator=(Stream&& other) noexcept -> Stream& {
		std::swap(destroy_, other.destroy_);
		std::swap(wait_event_, other.wait_event_);
		std::swap(ordinal_, other.ordinal_);
		std::swap(handle_, other.handle_);
		return *this;
	}

	~Stream() {
		// A moved-from stream holds nothing; a failed destroy at teardown has no remedy.
		if (destroy_.function != nullptr) {
			std::ignore = destroy_(handle_);
		}
	}

	/** Orders this stream's later work after `event`'s completion; the event may belong to the other card. */
	[[nodiscard]] auto wait(Event const& event) const -> core::Result<void, DeviceError>;

	[[nodiscard]] auto ordinal() const -> CUdevice {
		return ordinal_;
	}

private:
	friend class Context;
	friend class Event;

	Stream(Api const& api, CUdevice ordinal, CUstream handle)
	    : destroy_{api.stream_destroy}, wait_event_{api.stream_wait_event}, ordinal_{ordinal}, handle_{handle} {}

	decltype(Api::stream_destroy) destroy_;
	decltype(Api::stream_wait_event) wait_event_;
	CUdevice ordinal_;
	CUstream handle_;
};

/** Whether the work an event was recorded after has finished. */
export enum class EventState : std::uint8_t {
	Pending,
	Complete,
};

/** A completion marker of one card, without timing. */
export class Event {
public:
	Event(Event const&) = delete;
	auto operator=(Event const&) -> Event& = delete;

	Event(Event&& other) noexcept
	    : destroy_{std::exchange(other.destroy_, {})},
	      record_{other.record_},
	      query_{other.query_},
	      ordinal_{other.ordinal_},
	      handle_{std::exchange(other.handle_, nullptr)} {}

	auto operator=(Event&& other) noexcept -> Event& {
		std::swap(destroy_, other.destroy_);
		std::swap(record_, other.record_);
		std::swap(query_, other.query_);
		std::swap(ordinal_, other.ordinal_);
		std::swap(handle_, other.handle_);
		return *this;
	}

	~Event() {
		// A moved-from event holds nothing; a failed destroy at teardown has no remedy.
		if (destroy_.function != nullptr) {
			std::ignore = destroy_(handle_);
		}
	}

	/** Marks the point after everything enqueued on `stream` so far; the stream must be this card's. */
	[[nodiscard]] auto record(Stream const& stream) const -> core::Result<void, DeviceError> {
		core::invariant(stream.ordinal_ == ordinal_);
		return call(record_, handle_, stream.handle_);
	}

	/** Whether the recorded work has finished; an event never recorded counts as complete. */
	[[nodiscard]] auto query() const -> core::Result<EventState, DeviceError> {
		// Not ready is the driver's answer to a question, not a failure.
		if (auto const status = query_(handle_); status == Status::NotReady) {
			return EventState::Pending;
		} else if (status != Status::Success) {
			return std::unexpected(DeviceError{.reason = Failure{.symbol = query_.symbol, .status = status}});
		}
		return EventState::Complete;
	}

	/** Polls until the recorded work has finished or `deadline` passes; the state when it stopped. */
	[[nodiscard]] auto wait_until(std::chrono::steady_clock::time_point deadline) const -> core::Result<EventState, DeviceError> {
		constexpr auto poll = std::chrono::microseconds{50};
		for (;;) {
			auto const state = query();
			if (!state || *state == EventState::Complete || std::chrono::steady_clock::now() >= deadline) {
				return state;
			}
			std::this_thread::sleep_until(std::min(std::chrono::steady_clock::now() + poll, deadline));
		}
	}

	[[nodiscard]] auto ordinal() const -> CUdevice {
		return ordinal_;
	}

private:
	friend class Context;
	friend class Stream;

	Event(Api const& api, CUdevice ordinal, CUevent handle)
	    : destroy_{api.event_destroy}, record_{api.event_record}, query_{api.event_query}, ordinal_{ordinal}, handle_{handle} {}

	decltype(Api::event_destroy) destroy_;
	decltype(Api::event_record) record_;
	decltype(Api::event_query) query_;
	CUdevice ordinal_;
	CUevent handle_;
};

auto Stream::wait(Event const& event) const -> core::Result<void, DeviceError> {
	return call(wait_event_, handle_, event.handle_, 0U);
}

}

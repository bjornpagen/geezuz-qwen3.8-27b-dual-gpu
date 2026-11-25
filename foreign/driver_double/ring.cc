// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver_double:ring;

import std;
import engine.core;
import engine.driver;

namespace engine::driver_double {

/** What one step of the modelled persistent kernel did. */
export enum class Served : std::uint8_t {
	Idle,
	Ran,
	Exited,
};

/**
 * The card's half of the resident ring, written in host C++ against the same slot layout: the acquire and release
 * here are what `ld.acquire.sys` and `st.release.sys` do on the card. Under the fake, unified addressing makes the
 * ring's device address the host's own.
 */
export template<driver::Wire D, driver::Wire R, std::size_t Capacity>
class RingDevice {
public:
	explicit RingDevice(driver::CUdeviceptr base)
	    : /* SAFETY: under the fake the device address of mapped memory is its host address, and the ring outlives this. */
	      slots_{reinterpret_cast<device::ring::Slot<D, R>*>(base)} {}

	/** Serves the next ticket if the host has published it: `body` maps a descriptor to its result. */
	template<class Body>
	[[nodiscard]] auto step(Body& body) -> Served {
		auto& slot = slots_[device::ring::index(next_, Capacity)];
		if (std::atomic_ref{slot.request.ticket}.load(std::memory_order_acquire) != next_) {
			return Served::Idle;
		}
		auto const command = slot.request.command;
		core::invariant(command == device::ring::Command::Run || command == device::ring::Command::Exit);
		if (command == device::ring::Command::Run) {
			slot.response.result = body(slot.request.descriptor);
		}
		std::atomic_ref{slot.response.ticket}.store(next_, std::memory_order_release);
		++next_;
		return command == device::ring::Command::Run ? Served::Ran : Served::Exited;
	}

private:
	device::ring::Slot<D, R>* slots_;
	std::uint64_t next_ = 1;
};

/**
 * The modelled persistent kernel on its own thread, as the card runs it beside the host: it serves tickets until it
 * serves an exit, or until its owner dies. The ring it serves must outlive it.
 */
export template<driver::Wire D, driver::Wire R, std::size_t Capacity, class Body>
class RingKernel {
public:
	RingKernel(driver::CUdeviceptr base, Body serve)
	    : thread_{[device = RingDevice<D, R, Capacity>{base}, body = std::move(serve)](std::stop_token const& stop) mutable {
		      while (!stop.stop_requested()) {
			      if (device.step(body) == Served::Exited) {
				      return;
			      }
			      std::this_thread::yield();
		      }
	      }} {}

private:
	std::jthread thread_;
};

}

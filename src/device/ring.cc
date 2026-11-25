// SPDX-License-Identifier: AGPL-3.0-only
export module engine.device:ring;

import std;
import engine.core;
import engine.driver;

namespace engine::device {

/** One exchange through the ring, numbered from 1 in submission order; tickets never repeat. */
export struct Ticket {
	std::uint64_t value;

	[[nodiscard]] constexpr auto operator<=>(Ticket const&) const = default;
};

/** The kernel's answer to a `Run` ticket. */
export template<class R>
struct Ran {
	Ticket ticket;
	R result;

	[[nodiscard]] constexpr auto operator==(Ran const&) const -> bool = default;
};

/** The kernel's acknowledgement of the exit ticket: it has left its loop and serves nothing more. */
export struct Exited {
	Ticket ticket;

	[[nodiscard]] constexpr auto operator==(Exited const&) const -> bool = default;
};

export template<class R>
using Completion = std::variant<Ran<R>, Exited>;

/**
 * The host's end of the resident ring between the host and one card's persistent kernel: `Capacity` slots of mapped,
 * pinned memory through which the host submits descriptors and the kernel returns results, in submission order. The
 * slot layout and the acquire/release protocol per ticket are `driver::RingSlot`'s; this owner keeps the host's
 * bookkeeping, which makes the protocol's one rule a fact of its state: at most `Capacity` tickets are in flight, so
 * the host never rewrites a slot the kernel has not answered or the host has not collected.
 *
 * The persistent kernel takes `device_address()` and `Capacity` and serves tickets 1, 2, 3, ... until the exit ticket.
 * The ring must outlive the kernel's last access, and die before the context it was allocated from.
 */
export template<driver::Wire D, driver::Wire R, std::size_t Capacity>
class Ring {
public:
	static_assert(Capacity > 0, "a ring holds at least one ticket in flight");

	using Slots = driver::MappedArray<driver::RingSlot<D, R>, Capacity>;

	[[nodiscard]] static auto create(driver::Context const& context) -> core::Result<Ring, driver::DeviceError> {
		return context.allocate_mapped<driver::RingSlot<D, R>, Capacity>().transform([](Slots slots) {
			return Ring{std::move(slots)};
		});
	}

	/** Where the kernel finds slot 0. */
	[[nodiscard]] auto device_address() const -> std::uint64_t {
		return slots_.device_address();
	}

	/** Tickets submitted and not yet collected. */
	[[nodiscard]] auto in_flight() const -> std::size_t {
		return core::narrow<std::size_t>(published_ - collected_);
	}

	/** Submits a descriptor for the kernel to run; nullopt while every slot is in flight. */
	[[nodiscard]] auto try_submit(D const& descriptor) -> std::optional<Ticket> {
		core::invariant(!exit_.has_value());
		return claim().transform([&](Ticket ticket) {
			driver::publish_run(slot(ticket), ticket.value, descriptor);
			return ticket;
		});
	}

	/** Submits the exit ticket, after which nothing more can be submitted; nullopt while every slot is in flight. */
	[[nodiscard]] auto try_exit() -> std::optional<Ticket> {
		core::invariant(!exit_.has_value());
		return claim().transform([&](Ticket ticket) {
			driver::publish_exit(slot(ticket), ticket.value);
			exit_ = ticket;
			return ticket;
		});
	}

	/** The oldest submitted ticket's completion once the kernel has answered it; nullopt until then. */
	[[nodiscard]] auto try_collect() -> std::optional<Completion<R>> {
		if (collected_ == published_) {
			return std::nullopt;
		}
		auto const ticket = Ticket{.value = collected_ + 1};
		auto& answered = slot(ticket);
		if (driver::answered(answered) != ticket.value) {
			return std::nullopt;
		}
		collected_ = ticket.value;
		if (exit_ == ticket) {
			return Completion<R>{Exited{.ticket = ticket}};
		}
		return Completion<R>{Ran<R>{.ticket = ticket, .result = answered.result}};
	}

private:
	explicit Ring(Slots slots) : slots_{std::move(slots)} {}

	[[nodiscard]] auto claim() -> std::optional<Ticket> {
		if (in_flight() == Capacity) {
			return std::nullopt;
		}
		++published_;
		return Ticket{.value = published_};
	}

	[[nodiscard]] auto slot(Ticket ticket) -> driver::RingSlot<D, R>& {
		return slots_.host()[driver::ring_index(ticket.value, Capacity)];
	}

	Slots slots_;
	std::uint64_t published_ = 0;
	std::uint64_t collected_ = 0;
	std::optional<Ticket> exit_;
};

}

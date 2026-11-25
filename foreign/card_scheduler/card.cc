// SPDX-License-Identifier: AGPL-3.0-only
export module engine.card_scheduler:card;

import std;
import engine.core;
import engine.driver;
import engine.exec;

namespace engine::card_scheduler {

namespace ex = engine::exec;

/**
 * Builds an immovable value in place: the conversion's result initializes the storage directly, so an operation
 * state can be emplaced into an optional without ever being moved.
 */
template<class Make>
struct Emplace {
	Make make;

	// Implicit on purpose: `std::optional::emplace` initializes its value from this, and only a conversion reaches the
	// operation state's guaranteed elision.
	operator std::invoke_result_t<Make&>() && {
		return make();
	}
};

/** No queries: the tick that polls a completion is not cancellable, so it asks its scheduler nothing. */
struct Quiet {};

/** The work a card's stream held when the sender started has finished: an event recorded then has fired. */
struct Recorded {
	using Outcome = core::Result<void, driver::DeviceError>;
	using Signatures =
	    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(driver::DeviceError), ex::set_stopped_t()>;

	/* SAFETY: borrowed from the card's owner, which outlives every wait on the card */
	driver::Context const* context;
	/* SAFETY: as `context` */
	driver::Stream const* stream;
	std::optional<driver::Event> event;

	[[nodiscard]] auto begin() -> std::optional<Outcome> {
		auto created = context->create_event();
		if (!created) {
			return Outcome{std::unexpected(created.error())};
		}
		if (auto recorded = created->record(*stream); !recorded) {
			return Outcome{std::unexpected(recorded.error())};
		}
		event.emplace(std::move(*created));
		return std::nullopt;
	}

	[[nodiscard]] auto check() const -> std::optional<Outcome> {
		auto const state = event.value().query();
		if (!state) {
			return Outcome{std::unexpected(state.error())};
		}
		if (*state == driver::EventState::Pending) {
			return std::nullopt;
		}
		return Outcome{};
	}

	template<class R>
	static auto deliver(R&& receiver, Outcome outcome) -> void {
		if (outcome) {
			ex::set_value(std::forward<R>(receiver));
		} else {
			ex::set_error(std::forward<R>(receiver), std::move(outcome).error());
		}
	}
};

/** An asynchronous copy has finished: both of its buffers come back with its outcome. */
template<class Destination, class Source>
struct Copying {
	using Outcome = driver::Delivered<Destination, Source>;
	using Signatures = ex::completion_signatures<ex::set_value_t(Outcome), ex::set_stopped_t()>;

	driver::Transfer<Destination, Source> transfer;

	[[nodiscard]] auto begin() -> std::optional<Outcome> {
		return transfer.try_finish();
	}

	[[nodiscard]] auto check() -> std::optional<Outcome> {
		return transfer.try_finish();
	}

	template<class R>
	static auto deliver(R&& receiver, Outcome outcome) -> void {
		ex::set_value(std::forward<R>(receiver), std::move(outcome));
	}
};

/**
 * The operation of a watched completion. It polls on the scheduler of its receiver's environment, the loop of the
 * thread that waits for it: each tick asks the driver once and, while the work runs, schedules the next tick behind
 * whatever else that loop holds (another card's poll, host work overlapping the cycle). The cards' work cannot be
 * called back, so stop requests are not observed: the wait ends when the card does.
 *
 * Ticks alternate between two storage cells, because a tick's operation is still on the stack when it schedules the
 * next one; the cell it reuses belongs to the tick before, long finished.
 */
template<class W, class R>
class Polling {
public:
	Polling(W watch, R receiver)
	    : watch_{std::move(watch)}, receiver_{std::move(receiver)}, scheduler_{ex::get_scheduler(ex::get_env(receiver_))} {}

	Polling(Polling const&) = delete;
	auto operator=(Polling const&) -> Polling& = delete;
	Polling(Polling&&) = delete;
	auto operator=(Polling&&) -> Polling& = delete;
	~Polling() = default;

	auto start() & noexcept -> void {
		if (auto done = watch_.begin()) {
			W::deliver(std::move(receiver_), std::move(*done));
			return;
		}
		again();
	}

private:
	struct Tick {
		using receiver_concept = ex::receiver_t;

		/* SAFETY: the operation owns the tick's operation, so it outlives every completion of the tick */
		Polling* operation;

		auto set_value() noexcept -> void {
			operation->poll();
		}

		/** The loop's schedule declares an exception channel it never takes: there are no exceptions. */
		template<class E>
		auto set_error(E&& /*error*/) noexcept -> void {
			core::invariant(false);
		}

		/** The loop ran out before the card finished: the wait is abandoned with it. */
		auto set_stopped() noexcept -> void {
			ex::set_stopped(std::move(operation->receiver_));
		}

		[[nodiscard]] auto get_env() const noexcept -> Quiet {
			return {};
		}
	};

	using Scheduler = std::remove_cvref_t<decltype(ex::get_scheduler(ex::get_env(std::declval<R const&>())))>;
	using Inner = decltype(ex::connect(ex::schedule(std::declval<Scheduler&>()), std::declval<Tick>()));

	auto again() -> void {
		auto& cell = cells_[turn_ % cells_.size()];
		turn_ += 1;
		cell.emplace(Emplace{[this] {
			return ex::connect(ex::schedule(scheduler_), Tick{.operation = this});
		}});
		ex::start(*cell);
	}

	auto poll() -> void {
		if (auto done = watch_.check()) {
			W::deliver(std::move(receiver_), std::move(*done));
			return;
		}
		again();
	}

	W watch_;
	R receiver_;
	Scheduler scheduler_;
	std::array<std::optional<Inner>, 2> cells_;
	std::size_t turn_ = 0;
};

/** A sender that completes when a watched card completion has fired. */
template<class W>
struct Watched {
	using sender_concept = ex::sender_t;
	using completion_signatures = typename W::Signatures;

	W watch;

	template<class R>
	[[nodiscard]] auto connect(R receiver) && -> Polling<W, R> {
		return Polling<W, R>{std::move(watch), std::move(receiver)};
	}
};

/**
 * One card's scheduler: work scheduled on it runs once the card has finished everything enqueued on its stream
 * before the scheduled sender started. That is the only execution a card offers the host (nvexec, which runs
 * senders on the card itself, does not target clang), and it is what the engine composes: `when_all` over both
 * cards' schedulers joins the two halves of an upload or a prefill chunk.
 *
 * Its senders poll the driver from the loop of whoever waits for them (`exec::wait` on the engine's own thread),
 * so readiness needs no second thread, no lock and no callback from the driver. A scheduler borrows its card's
 * context and stream: their owner outlives every sender it hands out.
 */
export class Card {
public:
	Card(driver::Context const& context, driver::Stream const& stream) : context_{&context}, stream_{&stream} {
		core::invariant(context.ordinal() == stream.ordinal());
	}

	[[nodiscard]] auto schedule() const noexcept -> Watched<Recorded> {
		return Watched<Recorded>{.watch = Recorded{.context = context_, .stream = stream_, .event = std::nullopt}};
	}

	[[nodiscard]] auto operator==(Card const&) const -> bool = default;

private:
	/* SAFETY: borrowed from the card's owner, which outlives every scheduler and sender made from it */
	driver::Context const* context_;
	/* SAFETY: as `context_` */
	driver::Stream const* stream_;
};

static_assert(ex::scheduler<Card>);

/** A copy in flight, as a sender: it completes with both buffers and the copy's outcome once the copy has finished. */
export template<class Destination, class Source>
[[nodiscard]] auto finished(driver::Transfer<Destination, Source> transfer) -> Watched<Copying<Destination, Source>> {
	return Watched<Copying<Destination, Source>>{.watch = Copying<Destination, Source>{.transfer = std::move(transfer)}};
}

/**
 * A caller's question about work in flight: nothing while it runs, then its outcome, a value or a typed error. It is
 * asked once when its sender starts and then once per tick until it answers; it must not block.
 */
export template<class P>
concept Poll = std::move_constructible<P> && std::invocable<P&> && requires {
	typename std::invoke_result_t<P&>::value_type;
	requires std::same_as<std::invoke_result_t<P&>, std::optional<typename std::invoke_result_t<P&>::value_type>>;
	requires core::Classified<typename std::invoke_result_t<P&>::value_type::error_type>;
	requires std::same_as<typename std::invoke_result_t<P&>::value_type,
	                      core::Result<typename std::invoke_result_t<P&>::value_type::value_type,
	                                   typename std::invoke_result_t<P&>::value_type::error_type>>;
};

/** The value channel of a poll's sender: none for a poll whose outcome carries no value. */
template<class V>
struct ValueSignature {
	using type = ex::set_value_t(V);
};

template<>
struct ValueSignature<void> {
	using type = ex::set_value_t();
};

/** A caller's poll, watched: its outcome is the sender's completion. */
template<Poll P>
struct Polled {
	using Outcome = std::invoke_result_t<P&>::value_type;
	using Value = Outcome::value_type;
	using Error = Outcome::error_type;
	using Signatures =
	    ex::completion_signatures<typename ValueSignature<Value>::type, ex::set_error_t(Error), ex::set_stopped_t()>;

	P poll;

	[[nodiscard]] auto begin() -> std::optional<Outcome> {
		return std::invoke(poll);
	}

	[[nodiscard]] auto check() -> std::optional<Outcome> {
		return std::invoke(poll);
	}

	template<class R>
	static auto deliver(R&& receiver, Outcome outcome) -> void {
		if (!outcome) {
			ex::set_error(std::forward<R>(receiver), std::move(outcome).error());
		} else if constexpr (std::is_void_v<Value>) {
			ex::set_value(std::forward<R>(receiver));
		} else {
			ex::set_value(std::forward<R>(receiver), std::move(*outcome));
		}
	}
};

/**
 * Work the caller polls, as a sender: it completes with the poll's value, or its error, once the poll answers. The
 * poll runs on the loop of whoever waits for the sender, as a card's scheduled senders do, so a host-side completion
 * (a ring's answer, a mapped flag) joins the cards' events in one wait.
 */
export template<Poll P>
[[nodiscard]] auto polled(P poll) -> Watched<Polled<P>> {
	return Watched<Polled<P>>{.watch = Polled<P>{.poll = std::move(poll)}};
}

}

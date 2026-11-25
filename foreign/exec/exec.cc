// SPDX-License-Identifier: AGPL-3.0-only
export module engine.exec;

import std;
import stdexec;
import engine.core;

/**
 * The sender vocabulary, under the name it keeps when libc++ ships std::execution: this unit is the one place that
 * spells stdexec, and the swap rebinds these declarations without touching a caller.
 */
export namespace engine::exec {

using stdexec::operation_state;
using stdexec::receiver;
using stdexec::scheduler;
using stdexec::sender;
using stdexec::sender_in;

using stdexec::completion_signatures;
using stdexec::operation_state_t;
using stdexec::receiver_t;
using stdexec::scheduler_t;
using stdexec::sender_t;
using stdexec::set_error_t;
using stdexec::set_stopped_t;
using stdexec::set_value_t;

using stdexec::connect;
using stdexec::get_env;
using stdexec::get_scheduler;
using stdexec::get_stop_token;
using stdexec::set_error;
using stdexec::set_stopped;
using stdexec::set_value;
using stdexec::start;

using stdexec::continues_on;
using stdexec::into_variant;
using stdexec::just;
using stdexec::just_error;
using stdexec::just_stopped;
using stdexec::let_error;
using stdexec::let_stopped;
using stdexec::let_value;
using stdexec::on;
using stdexec::schedule;
using stdexec::starts_on;
using stdexec::stopped_as_optional;
using stdexec::then;
using stdexec::upon_error;
using stdexec::upon_stopped;
using stdexec::when_all;

using stdexec::inplace_stop_source;
using stdexec::inplace_stop_token;
using stdexec::run_loop;

}

namespace engine::exec {

struct Pending {};

struct Stopped {};

/**
 * The environment a waited sender sees: the run loop driving the wait answers every scheduler query, so a sender
 * that returns to "the caller's context" completes back on that loop. Completion signatures are computed in this
 * same environment.
 */
struct WaitEnv {
	stdexec::run_loop& loop;

	[[nodiscard]] auto query(stdexec::get_scheduler_t) const noexcept -> stdexec::run_loop::scheduler {
		return loop.get_scheduler();
	}

	[[nodiscard]] auto query(stdexec::get_start_scheduler_t) const noexcept -> stdexec::run_loop::scheduler {
		return loop.get_scheduler();
	}

	[[nodiscard]] auto query(stdexec::get_delegation_scheduler_t) const noexcept -> stdexec::run_loop::scheduler {
		return loop.get_scheduler();
	}
};

template<class Tuple, class E>
struct WaitState {
	std::variant<Pending, Stopped, std::expected<Tuple, E>> outcome;
	stdexec::run_loop loop;
};

template<class Tuple, class E>
struct WaitReceiver {
	using receiver_concept = stdexec::receiver_t;

	WaitState<Tuple, E>& state;

	template<class... Values>
	auto set_value(Values&&... values) noexcept -> void {
		state.outcome.template emplace<std::expected<Tuple, E>>(std::in_place, std::forward<Values>(values)...);
		state.loop.finish();
	}

	auto set_error(E error) noexcept -> void {
		state.outcome.template emplace<std::expected<Tuple, E>>(std::unexpect, std::move(error));
		state.loop.finish();
	}

	/** stdexec declares an exception channel on some adaptors even without exceptions; nothing can complete on it. */
	auto set_error(std::exception_ptr const&) noexcept -> void {
		core::invariant(false);
	}

	auto set_stopped() noexcept -> void {
		state.outcome.template emplace<Stopped>();
		state.loop.finish();
	}

	[[nodiscard]] auto get_env() const noexcept -> WaitEnv {
		return WaitEnv{state.loop};
	}
};

template<class... Values>
using ValueTuple = std::tuple<Values...>;

template<class Value>
using Single = Value;

/**
 * Runs `sndr` to completion on the calling thread: its values as an engaged expected of a tuple, its error `E` as
 * unexpected with the payload intact, and stopped as nullopt. It replaces `sync_wait`, which cannot return an error
 * as a value without exceptions.
 */
export template<class E, class Sndr>
[[nodiscard]] auto wait(Sndr&& sndr) -> std::optional<std::expected<stdexec::value_types_of_t<Sndr, WaitEnv, ValueTuple, Single>, E>> {
	using Tuple = stdexec::value_types_of_t<Sndr, WaitEnv, ValueTuple, Single>;
	auto state = WaitState<Tuple, E>{};
	auto operation = stdexec::connect(std::forward<Sndr>(sndr), WaitReceiver<Tuple, E>{state});
	stdexec::start(operation);
	state.loop.run();
	if (std::holds_alternative<Stopped>(state.outcome)) {
		return std::nullopt;
	}
	return std::move(std::get<std::expected<Tuple, E>>(state.outcome));
}

}

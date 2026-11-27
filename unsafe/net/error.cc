// SPDX-License-Identifier: AGPL-3.0-only
module;

// errno values are macros: the boundary maps them to a closed cause once.
#include <cerrno>

export module engine.net:error;

import std;
import engine.core;

namespace engine::net {

/** The system call that failed. */
export enum class Stage : std::uint8_t {
	Socket,
	Option,
	Bind,
	Listen,
	Resolve,
	Accept,
	Connect,
	Queue,
	Watch,
	Wait,
	Signal,
};

/** Why a system call failed, as the decision it drives. */
export enum class Cause : std::uint8_t {
	/** Descriptors, buffers or memory ran out: the same call can succeed once pressure eases. */
	Exhausted,
	/** The address is still held, typically by a socket in TIME_WAIT. */
	AddressInUse,
	/** Nothing listens at the address yet. */
	Refused,
	/** The process lacks the privilege (a low port, a sandbox). */
	Denied,
	/** Any other refusal: misuse or an environment the engine does not support. */
	Invalid,
};

export struct [[nodiscard]] NetError {
	Stage stage;
	Cause cause;
	/** The system's error number, for rendering only; decisions read `cause`. */
	std::int32_t code;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (cause) {
		case Cause::Exhausted:
		case Cause::AddressInUse:
		case Cause::Refused:
			return core::Retry::Transient;
		case Cause::Denied:
		case Cause::Invalid:
			return core::Retry::Permanent;
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(NetError const&) const -> bool = default;
};

/** The call a stage names, as an operator reads it. */
[[nodiscard]] constexpr auto call_of(Stage stage) -> std::string_view {
	switch (stage) {
	case Stage::Socket:
		return "opening a socket";
	case Stage::Option:
		return "setting a socket option";
	case Stage::Bind:
		return "binding the listening address";
	case Stage::Listen:
		return "listening";
	case Stage::Resolve:
		return "resolving an address";
	case Stage::Accept:
		return "accepting a connection";
	case Stage::Connect:
		return "connecting";
	case Stage::Queue:
		return "opening the readiness queue";
	case Stage::Watch:
		return "watching a descriptor";
	case Stage::Wait:
		return "waiting for readiness";
	case Stage::Signal:
		return "subscribing to the stop signals";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto reason_of(Cause cause) -> std::string_view {
	switch (cause) {
	case Cause::Exhausted:
		return "descriptors, buffers or memory ran out";
	case Cause::AddressInUse:
		return "the address is still in use";
	case Cause::Refused:
		return "nothing listens at the address";
	case Cause::Denied:
		return "the process lacks the privilege";
	case Cause::Invalid:
		return "the system refused the call";
	}
	std::unreachable();
}

/** One sentence for an operator: the call that failed, why, and the system's error number. */
export [[nodiscard]] auto describe(NetError const& error) -> std::string {
	return std::format("{} failed: {} (errno {})", call_of(error.stage), reason_of(error.cause), error.code);
}

[[nodiscard]] auto cause_of(int code) -> Cause {
	switch (code) {
	case EMFILE:
	case ENFILE:
	case ENOBUFS:
	case ENOMEM:
		return Cause::Exhausted;
	case EADDRINUSE:
		return Cause::AddressInUse;
	case ECONNREFUSED:
		return Cause::Refused;
	case EACCES:
	case EPERM:
		return Cause::Denied;
	// errno is an open set the kernel may extend: every number not named above is a failure the engine does not
	// expect to recover from.
	default:
		return Cause::Invalid;
	}
}

/** The failure of the call at `stage` that just set errno to `code`. */
[[nodiscard]] auto failure(Stage stage, int code) -> NetError {
	return NetError{.stage = stage, .cause = cause_of(code), .code = code};
}

}

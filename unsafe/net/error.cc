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

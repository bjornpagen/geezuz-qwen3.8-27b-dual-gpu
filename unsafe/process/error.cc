// SPDX-License-Identifier: AGPL-3.0-only
module;

// errno values are macros: the boundary maps them to a closed cause once.
#include <cerrno>

export module engine.process:error;

import std;
import engine.core;

namespace engine::process {

/** The step that failed. */
export enum class Stage : std::uint8_t {
	Pipe,
	Spawn,
	Read,
	Signal,
	Wait,
};

/** Why it failed, as the decision it drives. */
export enum class Cause : std::uint8_t {
	/** No program at the path, or a path component is not a directory. */
	NotFound,
	/** The program may not be executed by this process. */
	Denied,
	/** Descriptors, processes or memory ran out: the same call can succeed once pressure eases. */
	Exhausted,
	/** The child wrote a line longer than `max_line_bytes`: its output is not the line protocol the caller reads. */
	LineTooLong,
	/** Any other refusal: misuse or an environment the boundary does not support. */
	Invalid,
};

export struct [[nodiscard]] ProcessError {
	Stage stage;
	Cause cause;
	/** The system's error number, for rendering only; decisions read `cause`. Zero where no call failed. */
	std::int32_t code;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (cause) {
		case Cause::Exhausted:
			return core::Retry::Transient;
		case Cause::NotFound:
		case Cause::Denied:
		case Cause::LineTooLong:
		case Cause::Invalid:
			return core::Retry::Permanent;
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(ProcessError const&) const -> bool = default;
};

[[nodiscard]] auto cause_of(int error) -> Cause {
	switch (error) {
	case ENOENT:
	case ENOTDIR:
		return Cause::NotFound;
	case EACCES:
	case EPERM:
		return Cause::Denied;
	case EAGAIN:
	case EMFILE:
	case ENFILE:
	case ENOMEM:
		return Cause::Exhausted;
	default:
		return Cause::Invalid;
	}
}

[[nodiscard]] auto failed(Stage stage, int error) -> std::unexpected<ProcessError> {
	return std::unexpected(ProcessError{.stage = stage, .cause = cause_of(error), .code = error});
}

}

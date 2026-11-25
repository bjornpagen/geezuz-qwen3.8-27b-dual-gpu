// SPDX-License-Identifier: AGPL-3.0-only
module;

// The darwin spelling of keeping a child's descriptors from it: no pipe2, so the pipe's ends are marked close-on-exec
// after they exist, and POSIX_SPAWN_CLOEXEC_DEFAULT closes every descriptor the file actions do not name, so a
// descriptor another thread opens without the flag still never reaches the child. The environment of a darwin
// process is reached through _NSGetEnviron.
#include <cerrno>
#include <crt_externs.h>
#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>

export module engine.process:spawn;

import std;
import engine.core;
import :error;

namespace engine::process {

/** A pipe whose two ends are not inherited across exec: [read, write]. */
[[nodiscard]] auto open_pipe() -> core::Result<std::array<int, 2>, ProcessError> {
	auto ends = std::array<int, 2>{-1, -1};
	if (::pipe(ends.data()) != 0) {
		return failed(Stage::Pipe, errno);
	}
	for (auto const end : ends) {
		if (::fcntl(end, F_SETFD, FD_CLOEXEC) != 0) {
			auto const error = errno;
			std::ignore = ::close(ends[0]);
			std::ignore = ::close(ends[1]);
			return failed(Stage::Pipe, error);
		}
	}
	return ends;
}

/**
 * Makes the child start with only its standard input, output and error: `flags` gains what the attributes must set.
 * Zero, or the errno of the refusal.
 */
[[nodiscard]] auto keep_only_standard(posix_spawn_file_actions_t& actions, std::uint32_t& flags) -> int {
	flags |= std::uint32_t{POSIX_SPAWN_CLOEXEC_DEFAULT};
	if (auto const error = ::posix_spawn_file_actions_addinherit_np(&actions, STDIN_FILENO); error != 0) {
		return error;
	}
	return ::posix_spawn_file_actions_addinherit_np(&actions, STDERR_FILENO);
}

/** This process's environment, which the child inherits. */
[[nodiscard]] auto environment() -> char** {
	return *::_NSGetEnviron();
}

}

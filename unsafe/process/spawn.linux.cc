// SPDX-License-Identifier: AGPL-3.0-only
module;

// The linux spelling of keeping a child's descriptors from it: pipe2 creates the pipe close-on-exec at once, and the
// child closes every descriptor from 3 up before it executes, so a descriptor another thread opens without the flag
// still never reaches it.
#include <cerrno>
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
	if (::pipe2(ends.data(), O_CLOEXEC) != 0) {
		return failed(Stage::Pipe, errno);
	}
	return ends;
}

/**
 * Makes the child start with only its standard input, output and error: `flags` gains what the attributes must set.
 * Zero, or the errno of the refusal.
 */
[[nodiscard]] auto keep_only_standard(posix_spawn_file_actions_t& actions, std::uint32_t& /*flags*/) -> int {
	return ::posix_spawn_file_actions_addclosefrom_np(&actions, STDERR_FILENO + 1);
}

/** This process's environment, which the child inherits. */
[[nodiscard]] auto environment() -> char** {
	return environ;
}

}

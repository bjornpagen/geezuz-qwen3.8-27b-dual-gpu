// SPDX-License-Identifier: AGPL-3.0-only
module;

// stdout is a macro of the C runtime's, so this process's own standard output is reached here.
#include <cstdio>

export module engine.process:output;

import std;

namespace engine::process {

/**
 * Writes `text` to this process's standard output and flushes it, so a reader at the other end of a pipe (the
 * journal, a supervising test) has it at once rather than when a buffer fills. A closed or broken output loses the
 * text: the program has nowhere else to say it.
 */
export auto output(std::string_view text) -> void {
	/* SAFETY: stdout is the C runtime's process-lifetime stream, borrowed for this call only. */
	std::print(stdout, "{}", text);
	std::ignore = std::fflush(stdout);
}

/** `output` of one line: `line` and its newline. */
export auto output_line(std::string_view line) -> void {
	/* SAFETY: as in `output`. */
	std::println(stdout, "{}", line);
	std::ignore = std::fflush(stdout);
}

}

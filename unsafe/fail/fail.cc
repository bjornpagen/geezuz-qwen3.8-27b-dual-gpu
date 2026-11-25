// SPDX-License-Identifier: AGPL-3.0-only
module;

#include <cstdio>

export module engine.fail;

import std;

namespace engine::fail {

/**
 * Impossible programmer state: names the violated site on standard error and aborts. Never used for a failure the
 * input or the environment can cause; that is a `Result`. Always on, in every build. In a constant expression a
 * violation does not compile.
 *
 * It lives in quarantine for one reason: capturing the caller's site needs `std::source_location::current()` as a
 * default argument, the one default argument in the engine. Dialect code reaches it as `engine::core::invariant`.
 */
export constexpr auto invariant(bool holds, std::source_location where = std::source_location::current()) -> void {
	if (holds) {
		return;
	}
	auto const line = std::format("invariant violated at {}:{} in {}", where.file_name(), where.line(), where.function_name());
	/* SAFETY: stderr is the C runtime's process-lifetime, unbuffered stream, borrowed for this call only. */
	std::println(stderr, "{}", line);
	std::abort();
}

}

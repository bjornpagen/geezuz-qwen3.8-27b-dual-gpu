// SPDX-License-Identifier: AGPL-3.0-only
//
// The reference's fork-join pool: every index of a call runs exactly once, whatever the worker count, and the call
// returns only after all have run, so a pool's result is the serial one.
import std;
import engine.testing;
import engine.thread_pool;

namespace {

/** Runs `calls` calls of growing size, counting how often each index ran, against the serial count of one each. */
[[nodiscard]] auto every_index_once(std::size_t workers, std::size_t calls) -> bool {
	auto pool = engine::thread_pool::Pool{workers};
	for (auto call = std::size_t{0}; call < calls; ++call) {
		auto const count = call * 37;
		auto runs = std::vector<std::size_t>(count, 0);
		pool.share(count, [&](std::size_t index) {
			// Each index is its own slot, so concurrent runs never touch one place.
			runs[index] += 1 + index;
		});
		for (auto index = std::size_t{0}; index < count; ++index) {
			if (runs[index] != 1 + index) {
				return false;
			}
		}
	}
	return pool.threads() == workers + 1;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "with no workers the caller runs every index once", .holds = every_index_once(0, 20)},
	    engine::testing::Case{.name = "with one worker every index runs once, call after call", .holds = every_index_once(1, 40)},
	    engine::testing::Case{.name = "with seven workers every index runs once, call after call", .holds = every_index_once(7, 200)},
	};
	return engine::testing::report(cases);
}

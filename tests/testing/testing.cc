// SPDX-License-Identifier: AGPL-3.0-only
export module engine.testing;

import std;

namespace engine::testing {

/** One checked property: what it states, and whether it held. */
export struct Case {
	std::string_view name;
	bool holds;
};

/** Prints every case and returns the process exit status: zero exactly when every case held. */
export [[nodiscard]] auto report(std::span<Case const> cases) -> int {
	auto failures = std::size_t{0};
	for (auto const& check : cases) {
		std::println("{}: {}", check.holds ? "pass" : "FAIL", check.name);
		failures += check.holds ? 0 : 1;
	}
	return failures == 0 ? 0 : 1;
}

/**
 * The generator every property test draws from. A fixed seed makes a failure a replayable input; tests that want
 * breadth run several seeds, each named in the case.
 */
export [[nodiscard]] auto generator(std::uint64_t seed) -> std::mt19937_64 {
	return std::mt19937_64{seed};
}

}

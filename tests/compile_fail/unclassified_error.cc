// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;

struct Unclassified {
	int code;

	[[nodiscard]] constexpr auto operator==(Unclassified const&) const -> bool = default;
};

auto main() -> int {
	auto const result = engine::core::Result<int, Unclassified>{1};
	return *result;
}

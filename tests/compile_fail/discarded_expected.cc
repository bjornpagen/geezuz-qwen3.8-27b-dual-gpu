// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;

auto main() -> int {
	engine::core::try_narrow<std::uint8_t>(300);
	return 0;
}

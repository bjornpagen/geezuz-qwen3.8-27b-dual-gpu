// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;

auto main() -> int {
	auto const wide = 256;
	return engine::core::narrow<std::uint8_t>(wide);
}

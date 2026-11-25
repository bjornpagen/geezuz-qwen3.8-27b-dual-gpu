// SPDX-License-Identifier: AGPL-3.0-only
//
// An event made without timing has no timestamp: asking the time between two of them must not compile.
import std;
import engine.device;

[[nodiscard]] auto elapsed(engine::device::Event const& start, engine::device::Event const& end) -> bool {
	return end.since(start).has_value();
}

auto main() -> int {
	return 0;
}

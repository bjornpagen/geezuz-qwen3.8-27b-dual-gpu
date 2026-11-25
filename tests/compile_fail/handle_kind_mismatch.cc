// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.handle;

struct SlotHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(SlotHandle const&) const -> bool = default;
};

struct BlockHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(BlockHandle const&) const -> bool = default;
};

auto main() -> int {
	auto slots = engine::handle::Table<int, SlotHandle, 8, 8>{};
	auto blocks = engine::handle::Table<int, BlockHandle, 8, 8>{};
	auto const slot = slots.insert(1);
	return blocks.try_get(*slot).has_value() ? 1 : 0;
}

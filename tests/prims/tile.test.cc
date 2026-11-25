// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.prims;
import engine.testing;

namespace {

namespace device = engine::device::prims;
namespace tile = device::tile;
namespace fragment = device::fragment;

using engine::prims::bake_bf16_tile;
using engine::prims::bake_e4m3_tile;

[[nodiscard]] constexpr auto index(int value) -> std::size_t {
	return static_cast<std::size_t>(value);
}

[[nodiscard]] auto e4m3_slots_are_a_bijection() -> bool {
	auto seen = std::vector<int>(index(tile::words) * 4, 0);
	auto holds = true;
	for (auto row = 0; row < tile::e4m3_rows; ++row) {
		for (auto column = 0; column < tile::e4m3_columns; ++column) {
			auto const slot = tile::e4m3_slot(row, column);
			holds = holds && slot.word >= 0 && slot.word < tile::words && slot.place >= 0 && slot.place < 4;
			holds = holds && tile::e4m3_element(slot) == fragment::Coordinate{.row = row, .column = column};
			seen.at(index(slot.word) * 4 + index(slot.place)) += 1;
		}
	}
	return holds && std::ranges::all_of(seen, [](int count) { return count == 1; });
}

[[nodiscard]] auto bf16_slots_are_a_bijection() -> bool {
	auto seen = std::vector<int>(index(tile::words) * 2, 0);
	auto holds = true;
	for (auto row = 0; row < tile::bf16_rows; ++row) {
		for (auto column = 0; column < tile::bf16_columns; ++column) {
			auto const slot = tile::bf16_slot(row, column);
			holds = holds && slot.word >= 0 && slot.word < tile::words && slot.place >= 0 && slot.place < 2;
			holds = holds && tile::bf16_element(slot) == fragment::Coordinate{.row = row, .column = column};
			seen.at(index(slot.word) * 2 + index(slot.place)) += 1;
		}
	}
	return holds && std::ranges::all_of(seen, [](int count) { return count == 1; });
}

/** Every finite code in a fixed order, so a tile filled cyclically holds each code at every kind of position. */
[[nodiscard]] auto finite_codes() -> std::vector<std::uint8_t> {
	auto codes = std::vector<std::uint8_t>{};
	for (auto code = 0; code < 256; ++code) {
		if (code % 128 != 127) {
			codes.push_back(static_cast<std::uint8_t>(code));
		}
	}
	return codes;
}

/**
 * Reads a baked e4m3 tile the way a consumer warp does: lane l of unit u loads its four words, decodes each through
 * the bit-exact model of the device sequence into a-fragment registers for two k16 steps, and every register half
 * must hold the exact bf16 of the code the fragment map places there.
 */
[[nodiscard]] auto decodes_to(engine::prims::TileWords const& words, std::span<std::uint8_t const> row_major) -> bool {
	auto holds = true;
	for (auto unit = 0; unit < tile::units; ++unit) {
		for (auto lane = 0; lane < fragment::lanes; ++lane) {
			auto const base = index(unit) * (tile::unit_bytes / 4) + 4 * index(lane);
			for (auto step = 0; step < 2; ++step) {
				auto const low_k = device::decode::model(words.at(base + static_cast<std::size_t>(2 * step)));
				auto const high_k = device::decode::model(words.at(base + static_cast<std::size_t>(2 * step + 1)));
				auto const registers = std::array{low_k.top, low_k.bottom, high_k.top, high_k.bottom};
				for (auto reg = 0; reg < 4; ++reg) {
					for (auto half = 0; half < 2; ++half) {
						auto const at = fragment::a(lane, reg, half);
						auto const row = 16 * (unit / 4) + at.row;
						auto const column = 32 * (unit % 4) + 16 * step + at.column;
						auto const code = device::E4m3{.bits = row_major[index(row) * tile::e4m3_columns + index(column)]};
						auto const pair = registers.at(static_cast<std::size_t>(reg));
						auto const got = half == 0 ? device::first(pair) : device::second(pair);
						holds = holds && got == device::to_bf16(code);
					}
				}
			}
		}
	}
	return holds;
}

[[nodiscard]] auto e4m3_tiles_round_trip(std::uint64_t seed) -> bool {
	auto const codes = finite_codes();
	auto generator = engine::testing::generator(seed);
	auto pick = std::uniform_int_distribution<std::size_t>{0, codes.size() - 1};
	auto row_major = std::vector<std::uint8_t>(tile::bytes);
	for (auto& code : row_major) {
		code = codes[pick(generator)];
	}
	auto const baked = bake_e4m3_tile(std::span<std::uint8_t const, tile::bytes>{row_major});
	return baked.has_value() && decodes_to(*baked, row_major);
}

[[nodiscard]] auto e4m3_tile_of_every_code_round_trips() -> bool {
	auto const codes = finite_codes();
	auto row_major = std::vector<std::uint8_t>(tile::bytes);
	for (auto index = std::size_t{0}; index < row_major.size(); ++index) {
		// 254 is coprime to neither 128 nor 2, so step by 129 to put every code in every row and column parity.
		row_major[index] = codes[(index * 129) % codes.size()];
	}
	auto const baked = bake_e4m3_tile(std::span<std::uint8_t const, tile::bytes>{row_major});
	return baked.has_value() && decodes_to(*baked, row_major);
}

[[nodiscard]] auto a_nan_code_anywhere_is_refused(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto position = std::uniform_int_distribution<std::size_t>{0, tile::bytes - 1};
	auto holds = true;
	for (auto trial = 0; trial < 64; ++trial) {
		auto row_major = std::vector<std::uint8_t>(tile::bytes, std::uint8_t{0x38});
		row_major[position(generator)] = trial % 2 == 0 ? std::uint8_t{0x7f} : std::uint8_t{0xff};
		holds = holds && !bake_e4m3_tile(std::span<std::uint8_t const, tile::bytes>{row_major}).has_value();
	}
	return holds;
}

[[nodiscard]] auto bf16_tiles_round_trip(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto bits = std::uniform_int_distribution<std::uint32_t>{0, 0xffff};
	auto row_major = std::vector<std::uint16_t>(tile::bytes / 2);
	for (auto& value : row_major) {
		value = static_cast<std::uint16_t>(bits(generator));
	}
	auto const words = bake_bf16_tile(std::span<std::uint16_t const, tile::bytes / 2>{row_major});
	auto holds = true;
	for (auto unit = 0; unit < tile::units; ++unit) {
		for (auto lane = 0; lane < fragment::lanes; ++lane) {
			for (auto reg = 0; reg < 4; ++reg) {
				auto const word = words.at(index(unit) * (tile::unit_bytes / 4) + 4 * index(lane) + index(reg));
				for (auto half = 0; half < 2; ++half) {
					auto const at = fragment::a(lane, reg, half);
					auto const row = 16 * (unit / 4) + at.row;
					auto const column = 16 * (unit % 4) + at.column;
					auto const expected = row_major[index(row) * tile::bf16_columns + index(column)];
					holds = holds && ((word >> (16U * static_cast<std::uint32_t>(half))) & 0xffffU) == expected;
				}
			}
		}
	}
	return holds;
}

/** A warp owning m16 block b reads 2 KiB in a row, and they hold exactly rows 16b .. 16b + 15. */
[[nodiscard]] auto each_row_block_is_one_contiguous_run() -> bool {
	auto holds = true;
	for (auto word = 0; word < tile::words; ++word) {
		auto const block = word / (4 * tile::unit_bytes / 4);
		for (auto place = 0; place < 4; ++place) {
			holds = holds && tile::e4m3_element({.word = word, .place = place}).row / 16 == block;
		}
		for (auto place = 0; place < 2; ++place) {
			holds = holds && tile::bf16_element({.word = word, .place = place}).row / 16 == block;
		}
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every e4m3 element has exactly one slot, and the slot names it back", .holds = e4m3_slots_are_a_bijection()},
	    engine::testing::Case{.name = "every bf16 element has exactly one slot, and the slot names it back", .holds = bf16_slots_are_a_bijection()},
	    engine::testing::Case{.name = "a tile cycling through every finite code decodes back exactly through the device sequence's model",
	                          .holds = e4m3_tile_of_every_code_round_trips()},
	    engine::testing::Case{.name = "random e4m3 tiles decode back exactly (seed 1)", .holds = e4m3_tiles_round_trip(1)},
	    engine::testing::Case{.name = "random e4m3 tiles decode back exactly (seed 2)", .holds = e4m3_tiles_round_trip(2)},
	    engine::testing::Case{.name = "random e4m3 tiles decode back exactly (seed 3)", .holds = e4m3_tiles_round_trip(3)},
	    engine::testing::Case{.name = "a nan code anywhere in a tile is refused (seed 4)", .holds = a_nan_code_anywhere_is_refused(4)},
	    engine::testing::Case{.name = "random bf16 tiles land in a-fragment registers exactly (seed 5)", .holds = bf16_tiles_round_trip(5)},
	    engine::testing::Case{.name = "each m16 row block is one contiguous 2 KiB run", .holds = each_row_block_is_one_contiguous_run()},
	};
	return engine::testing::report(cases);
}

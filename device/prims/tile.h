// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight tile encoding: the byte order and bit encoding the baker writes and the consumer warps read, held once
// for both. A tile is one contiguous 16 KiB run of the transaction stream. Inside it, 512-byte units each hold one
// warp's operand for one m16 row block: lane l's 16 bytes are lane l's `mma.m16n8k16` a-fragment registers, so a warp
// fetches a unit with one conflict-free `ld.shared.v4` per lane and needs no `ldmatrix`.
//
// An e4m3 tile is 128 rows x 128 k (one block scale). Its unit u = 4 * b + s covers rows 16b .. 16b + 15 and k
// 32s .. 32s + 31 (two k16 steps). Lane l's four words are, in order, step 0's register pairs {0, 1} and {2, 3} and
// step 1's, each word holding four codes re-encoded by `decode::encode` (positions as `fragment::weight` names them).
// A warp owning m16 block b reads units 4b .. 4b + 3: 2 KiB in a row.
//
// A bf16 tile is 128 rows x 64 k (16 KiB). Its unit u = 4 * b + s covers rows 16b .. 16b + 15 and k 16s .. 16s + 15
// (one k16 step); lane l's four words are its four a-fragment registers, first element in the low half.
//
// Words are little-endian 32-bit values at byte offset 4 * word.
#pragma once

#include "prims/fragment.h"
#include "prims/numeric.h"

namespace engine::device::prims::tile {

inline constexpr int unit_bytes = 512;
inline constexpr int bytes = 16 * 1024;
inline constexpr int words = bytes / 4;
inline constexpr int units = bytes / unit_bytes;

inline constexpr int e4m3_rows = 128;
inline constexpr int e4m3_columns = 128;
inline constexpr int bf16_rows = 128;
inline constexpr int bf16_columns = 64;

static_assert(e4m3_rows * e4m3_columns == bytes && bf16_rows * bf16_columns * 2 == bytes);

/**
 * The encoding's identity: everything the device decode depends on. The image's description digest covers it, so an
 * image baked under another encoding is refused at the boundary rather than decoded wrongly.
 */
struct Encoding {
	unsigned version;
	int bytes;
	int unit_bytes;
	int e4m3_rows;
	int e4m3_columns;
	int bf16_rows;
	int bf16_columns;
	unsigned placed_mask;
	unsigned left_mask;
	unsigned right_mask;
	unsigned select_lut;
	unsigned two_to_120;
	unsigned negative_zero;
};

inline constexpr Encoding encoding = {
    .version = 1,
    .bytes = bytes,
    .unit_bytes = unit_bytes,
    .e4m3_rows = e4m3_rows,
    .e4m3_columns = e4m3_columns,
    .bf16_rows = bf16_rows,
    .bf16_columns = bf16_columns,
    .placed_mask = decode::placed_mask,
    .left_mask = decode::left_mask,
    .right_mask = decode::right_mask,
    .select_lut = decode::select_lut,
    .two_to_120 = decode::two_to_120.bits,
    .negative_zero = decode::negative_zero.bits,
};

/** Where an element lives in a tile: its 32-bit word, and its place in it (an e4m3 position 0-3, a bf16 half 0-1). */
struct Slot {
	int word;
	int place;

	[[nodiscard]] constexpr auto operator==(Slot const&) const -> bool = default;
};

/** The e4m3 element (row, column) of a tile, row < 128, column < 128, in the encoding position `decode::encode` takes. */
[[nodiscard]] constexpr auto e4m3_slot(int row, int column) -> Slot {
	int const block = row / 16;
	int const step_pair = column / 32;
	int const in_row = row % 16;
	int const in_column = column % 32;
	// Within a unit: k16 step, then register pair (k 0-7 or 8-15 of the step), then the lane's quad and thread.
	int const step = in_column / 16;
	int const pair = (in_column % 16) / 8;
	int const thread = (in_column % 8) / 2;
	int const half = in_column % 2;
	int const group = in_row % 8;
	int const bottom = in_row / 8;
	int const lane = 4 * group + thread;
	int const word_in_lane = 2 * step + pair;
	return {.word = (4 * block + step_pair) * (unit_bytes / 4) + 4 * lane + word_in_lane, .place = 2 * bottom + half};
}

/** The e4m3 element a word and position hold: the inverse of `e4m3_slot`. */
[[nodiscard]] constexpr auto e4m3_element(Slot slot) -> fragment::Coordinate {
	int const unit = slot.word / (unit_bytes / 4);
	int const lane = (slot.word % (unit_bytes / 4)) / 4;
	int const word_in_lane = slot.word % 4;
	fragment::Coordinate const within = fragment::weight(lane, word_in_lane, slot.place);
	return {.row = 16 * (unit / 4) + within.row, .column = 32 * (unit % 4) + within.column};
}

/** The bf16 element (row, column) of a tile, row < 128, column < 64; the place is the register half. */
[[nodiscard]] constexpr auto bf16_slot(int row, int column) -> Slot {
	int const block = row / 16;
	int const step = column / 16;
	int const in_row = row % 16;
	int const in_column = column % 16;
	int const lane = 4 * (in_row % 8) + (in_column % 8) / 2;
	int const reg = (in_row / 8) + 2 * (in_column / 8);
	return {.word = (4 * block + step) * (unit_bytes / 4) + 4 * lane + reg, .place = in_column % 2};
}

/** The bf16 element a word and half hold: the inverse of `bf16_slot`. */
[[nodiscard]] constexpr auto bf16_element(Slot slot) -> fragment::Coordinate {
	int const unit = slot.word / (unit_bytes / 4);
	int const lane = (slot.word % (unit_bytes / 4)) / 4;
	fragment::Coordinate const within = fragment::a(lane, slot.word % 4, slot.place);
	return {.row = 16 * (unit / 4) + within.row, .column = 16 * (unit % 4) + within.column};
}

}

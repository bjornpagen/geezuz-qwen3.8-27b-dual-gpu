// SPDX-License-Identifier: AGPL-3.0-only
//
// Where each element of an `mma.sync.m16n8k16` bf16 fragment lives, which lane supplies which `ldmatrix` row, the
// order the baker writes a lane's weight bytes in, and the xor swizzle that keeps activation tiles free of bank
// conflicts. Pure constexpr: the device indexes with it, the baker lays the image out with it, and the host tests
// prove each map a bijection and the swizzle conflict-free.
#pragma once

namespace engine::device::prims {

/** An m16n8k16 A operand: 4 bf16x2 registers, laid out as `fragment::a`. */
struct AFragment {
	unsigned reg[4];
};

/** An m16n8k16 B operand: 2 bf16x2 registers, laid out as `fragment::b`. */
struct BFragment {
	unsigned reg[2];
};

/** An m16n8 fp32 accumulator: 4 registers, laid out as `fragment::c`. */
struct Accumulator {
	float reg[4];
};

}

namespace engine::device::prims::fragment {

inline constexpr int lanes = 32;

struct Coordinate {
	int row;
	int column;

	[[nodiscard]] constexpr auto operator==(Coordinate const&) const -> bool = default;
};

// In every m16n8k16 fragment, lane l is in quad group g = l / 4 and thread t = l % 4 of it; a bf16x2 register's
// first element (its low half) has the lower column.

/** The A operand (16 x 16, row major, 4 bf16x2 registers): rows g and g + 8, columns 2t + {0, 1} and 2t + 8 + {0, 1}. */
[[nodiscard]] constexpr auto a(int lane, int reg, int half) -> Coordinate {
	return {.row = lane / 4 + 8 * (reg % 2), .column = 2 * (lane % 4) + half + 8 * (reg / 2)};
}

/** The B operand (16 x 8, column major, 2 bf16x2 registers): k rows 2t + {0, 1} and 2t + 8 + {0, 1}, column g. */
[[nodiscard]] constexpr auto b(int lane, int reg, int half) -> Coordinate {
	return {.row = 2 * (lane % 4) + half + 8 * reg, .column = lane / 4};
}

/** The C and D accumulators (16 x 8 fp32, 4 registers): rows g and g + 8, columns 2t and 2t + 1. */
[[nodiscard]] constexpr auto c(int lane, int reg) -> Coordinate {
	return {.row = lane / 4 + 8 * (reg / 2), .column = 2 * (lane % 4) + reg % 2};
}

/** Whether `ldmatrix` hands a lane its matrix's rows (`Plain`) or its columns (`.trans`). */
enum class Transpose : unsigned char { Plain, Transposed };

/**
 * The element of 8 x 8 matrix j that `ldmatrix` puts in half `half` of `lane`'s register j (the same for every j).
 * Lane 8m + r supplies the address of matrix m's row r, sixteen contiguous bytes.
 */
[[nodiscard]] constexpr auto ldmatrix(int lane, int half, Transpose transpose) -> Coordinate {
	Coordinate const plain = {.row = lane / 4, .column = 2 * (lane % 4) + half};
	return transpose == Transpose::Plain ? plain : Coordinate{.row = plain.column, .column = plain.row};
}

/**
 * The weight element (row of the 16-row tile, k within two k16 steps) the baker stores as `position` of `word` in a
 * lane's 16 bytes. Word w holds k step w / 2; even words feed registers 0 and 1, odd words registers 2 and 3.
 * Positions 0 and 1 are the top register's two elements (row g), 2 and 3 the bottom's (row g + 8), in the
 * argument order of `decode::encode`.
 */
[[nodiscard]] constexpr auto weight(int lane, int word, int position) -> Coordinate {
	Coordinate const element = a(lane, 2 * (word % 2) + position / 2, position % 2);
	return {.row = element.row, .column = 16 * (word / 2) + element.column};
}

/**
 * A shared-memory tile of 16-byte chunks: `rows` rows of `row_bytes` each, chunk k of row r stored at chunk
 * k ^ (r mod 8) within its 128-byte line. Eight consecutive rows at one logical chunk, which is what an `ldmatrix`
 * phase reads, and eight consecutive chunks of one row, which is what a row-wise writer stores, then fall in eight
 * different 16-byte bank groups; the type does not exist unless that holds.
 */
template<int row_bytes_, int rows_>
struct XorSwizzle {
	static constexpr int row_bytes = row_bytes_;
	static constexpr int rows = rows_;
	static constexpr int chunk_bytes = 16;
	static constexpr int chunks_per_row = row_bytes / chunk_bytes;
	static constexpr int bytes = row_bytes * rows;

	/** The byte offset of logical chunk `chunk` of row `row`. */
	[[nodiscard]] static constexpr auto offset(int row, int chunk) -> int {
		return row * row_bytes + static_cast<int>(static_cast<unsigned>(chunk) ^ (static_cast<unsigned>(row) % 8U)) * chunk_bytes;
	}

	static_assert(row_bytes % 128 == 0 && rows % 8 == 0 && rows > 0, "the swizzle permutes whole 128-byte lines of 8-row groups");
};

/** The 16-byte bank group (of eight, each four 4-byte banks) a byte offset falls in. */
[[nodiscard]] constexpr auto bank_group(int offset) -> int {
	return (offset / 16) % 8;
}

/**
 * Whether every eight-access phase a layout serves is conflict-free: for every run of eight consecutive rows at one
 * chunk, and every aligned run of eight chunks in one row, the eight 16-byte accesses hit eight distinct bank groups.
 */
template<class Layout>
[[nodiscard]] consteval auto conflict_free() -> bool {
	for (int first = 0; first + 8 <= Layout::rows; ++first) {
		for (int chunk = 0; chunk < Layout::chunks_per_row; ++chunk) {
			unsigned seen = 0;
			for (int row = first; row < first + 8; ++row) {
				seen |= 1U << static_cast<unsigned>(bank_group(Layout::offset(row, chunk)));
			}
			if (seen != 0xffU) {
				return false;
			}
		}
	}
	for (int row = 0; row < Layout::rows; ++row) {
		for (int first = 0; first + 8 <= Layout::chunks_per_row; first += 8) {
			unsigned seen = 0;
			for (int chunk = first; chunk < first + 8; ++chunk) {
				seen |= 1U << static_cast<unsigned>(bank_group(Layout::offset(row, chunk)));
			}
			if (seen != 0xffU) {
				return false;
			}
		}
	}
	return true;
}

}

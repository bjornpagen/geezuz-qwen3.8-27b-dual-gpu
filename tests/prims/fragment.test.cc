// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.prims;
import engine.testing;

namespace {

namespace fragment = engine::device::prims::fragment;

using fragment::Coordinate;

/** Whether `coordinates` visits every cell of a rows x columns grid exactly once. */
[[nodiscard]] auto covers_once(std::vector<Coordinate> const& coordinates, int rows, int columns) -> bool {
	auto seen = std::vector<int>(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns), 0);
	for (auto const& at : coordinates) {
		if (at.row < 0 || at.row >= rows || at.column < 0 || at.column >= columns) {
			return false;
		}
		seen.at(static_cast<std::size_t>(at.row) * static_cast<std::size_t>(columns) + static_cast<std::size_t>(at.column)) += 1;
	}
	return coordinates.size() == seen.size() && std::ranges::all_of(seen, [](int count) { return count == 1; });
}

[[nodiscard]] auto operand_maps_cover_their_matrices() -> bool {
	auto a = std::vector<Coordinate>{};
	auto b = std::vector<Coordinate>{};
	auto c = std::vector<Coordinate>{};
	auto weight = std::vector<Coordinate>{};
	for (auto lane = 0; lane < fragment::lanes; ++lane) {
		for (auto reg = 0; reg < 4; ++reg) {
			c.push_back(fragment::c(lane, reg));
			for (auto half = 0; half < 2; ++half) {
				a.push_back(fragment::a(lane, reg, half));
				if (reg < 2) {
					b.push_back(fragment::b(lane, reg, half));
				}
			}
			for (auto position = 0; position < 4; ++position) {
				weight.push_back(fragment::weight(lane, reg, position));
			}
		}
	}
	return covers_once(a, 16, 16) && covers_once(b, 16, 8) && covers_once(c, 16, 8) && covers_once(weight, 16, 32);
}

/** Each register's two halves are adjacent columns, the lower in the low half, as `mma` pairs them. */
[[nodiscard]] auto register_halves_are_adjacent_columns() -> bool {
	auto holds = true;
	for (auto lane = 0; lane < fragment::lanes; ++lane) {
		for (auto reg = 0; reg < 4; ++reg) {
			auto const low = fragment::a(lane, reg, 0);
			auto const high = fragment::a(lane, reg, 1);
			holds = holds && high.row == low.row && high.column == low.column + 1 && low.column % 2 == 0;
		}
	}
	return holds;
}

[[nodiscard]] auto ldmatrix_covers_each_matrix_and_transposes() -> bool {
	auto plain = std::vector<Coordinate>{};
	auto holds = true;
	for (auto lane = 0; lane < fragment::lanes; ++lane) {
		for (auto half = 0; half < 2; ++half) {
			auto const at = fragment::ldmatrix(lane, half, fragment::Transpose::Plain);
			auto const flipped = fragment::ldmatrix(lane, half, fragment::Transpose::Transposed);
			plain.push_back(at);
			holds = holds && flipped == Coordinate{.row = at.column, .column = at.row};
		}
	}
	return holds && covers_once(plain, 8, 8);
}

/**
 * `ldmatrix.x2` from a tile stored token by token (k contiguous), matrix j holding k 8j .. 8j + 7, delivers exactly
 * the B operand: register j, half h of every lane is the element `fragment::b` places there.
 */
[[nodiscard]] auto ldmatrix_of_token_rows_is_the_b_operand() -> bool {
	auto holds = true;
	for (auto lane = 0; lane < fragment::lanes; ++lane) {
		for (auto reg = 0; reg < 2; ++reg) {
			for (auto half = 0; half < 2; ++half) {
				auto const in_matrix = fragment::ldmatrix(lane, half, fragment::Transpose::Plain);
				auto const loaded = Coordinate{.row = 8 * reg + in_matrix.column, .column = in_matrix.row};
				holds = holds && loaded == fragment::b(lane, reg, half);
			}
		}
	}
	return holds;
}

/** The unswizzled layout a tile would have without the xor: rows of 16-byte chunks in order. */
template<int row_bytes_, int rows_>
struct Unswizzled {
	static constexpr int row_bytes = row_bytes_;
	static constexpr int rows = rows_;
	static constexpr int chunk_bytes = 16;
	static constexpr int chunks_per_row = row_bytes / chunk_bytes;
	static constexpr int bytes = row_bytes * rows;

	[[nodiscard]] static constexpr auto offset(int row, int chunk) -> int {
		return row * row_bytes + chunk * chunk_bytes;
	}
};

/**
 * The oracle for one phase: eight 16-byte accesses are conflict-free when their 32 four-byte words fall in 32
 * distinct banks of the 32.
 */
[[nodiscard]] auto distinct_banks(std::array<int, 8> const& offsets) -> bool {
	auto banks = std::bitset<32>{};
	for (auto const offset : offsets) {
		for (auto word = 0; word < 4; ++word) {
			banks.set(static_cast<std::size_t>((offset / 4 + word) % 32));
		}
	}
	return banks.all();
}

/** Every ldmatrix phase (eight consecutive rows at one chunk) and every row-store phase, by the bank oracle. */
template<class Layout>
[[nodiscard]] auto phases_hit_distinct_banks() -> bool {
	auto holds = true;
	for (auto first = 0; first + 8 <= Layout::rows; ++first) {
		for (auto chunk = 0; chunk < Layout::chunks_per_row; ++chunk) {
			auto offsets = std::array<int, 8>{};
			for (auto row = 0; row < 8; ++row) {
				offsets.at(static_cast<std::size_t>(row)) = Layout::offset(first + row, chunk);
			}
			holds = holds && distinct_banks(offsets);
		}
	}
	for (auto row = 0; row < Layout::rows; ++row) {
		for (auto first = 0; first + 8 <= Layout::chunks_per_row; first += 8) {
			auto offsets = std::array<int, 8>{};
			for (auto chunk = 0; chunk < 8; ++chunk) {
				offsets.at(static_cast<std::size_t>(chunk)) = Layout::offset(row, first + chunk);
			}
			holds = holds && distinct_banks(offsets);
		}
	}
	return holds;
}

/** The swizzle stores every chunk of the tile exactly once, inside the tile, 16-byte aligned. */
template<class Layout>
[[nodiscard]] auto swizzle_is_a_permutation_of_chunks() -> bool {
	auto seen = std::vector<int>(static_cast<std::size_t>(Layout::bytes / 16), 0);
	for (auto row = 0; row < Layout::rows; ++row) {
		for (auto chunk = 0; chunk < Layout::chunks_per_row; ++chunk) {
			auto const offset = Layout::offset(row, chunk);
			if (offset < 0 || offset >= Layout::bytes || offset % 16 != 0) {
				return false;
			}
			seen.at(static_cast<std::size_t>(offset / 16)) += 1;
		}
	}
	return std::ranges::all_of(seen, [](int count) { return count == 1; });
}

template<class Layout>
[[nodiscard]] auto swizzle_holds() -> bool {
	static_assert(fragment::conflict_free<Layout>());
	return phases_hit_distinct_banks<Layout>() && swizzle_is_a_permutation_of_chunks<Layout>();
}

// The check the swizzle type rests on rejects the layout it replaces, so it can fail.
static_assert(!fragment::conflict_free<Unswizzled<256, 16>>());

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the a, b, c and baked-weight maps each cover their matrix once", .holds = operand_maps_cover_their_matrices()},
	    engine::testing::Case{.name = "a register's halves are adjacent columns, lower first", .holds = register_halves_are_adjacent_columns()},
	    engine::testing::Case{.name = "ldmatrix covers each 8x8 matrix once and .trans transposes it", .holds = ldmatrix_covers_each_matrix_and_transposes()},
	    engine::testing::Case{.name = "ldmatrix of token-major rows delivers exactly the b operand", .holds = ldmatrix_of_token_rows_is_the_b_operand()},
	    engine::testing::Case{.name = "the swizzle is conflict-free and a permutation (128-byte rows)", .holds = swizzle_holds<fragment::XorSwizzle<128, 16>>()},
	    engine::testing::Case{.name = "the swizzle is conflict-free and a permutation (256-byte rows)", .holds = swizzle_holds<fragment::XorSwizzle<256, 32>>()},
	    engine::testing::Case{.name = "the swizzle is conflict-free and a permutation (512-byte rows)", .holds = swizzle_holds<fragment::XorSwizzle<512, 64>>()},
	    engine::testing::Case{.name = "the bank oracle finds the conflicts of an unswizzled tile", .holds = !phases_hit_distinct_banks<Unswizzled<256, 16>>()},
	};
	return engine::testing::report(cases);
}

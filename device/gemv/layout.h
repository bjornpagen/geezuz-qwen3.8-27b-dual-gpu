// SPDX-License-Identifier: AGPL-3.0-only
//
// Where each warp of the weight matmuls reads its operands and where its accumulators belong: the byte offsets of a
// lane's weight chunk in a ring stage, the activation row whose address a lane hands `ldmatrix`, and the output and
// verify row of every accumulator register. Pure constexpr, so the kernels index with it and the host emulation of
// every warp reads through the very same offsets.
#pragma once

#include "gemv/order.h"
#include "prims/fragment.h"
#include "prims/tile.h"

namespace engine::device::gemv::layout {

/** One unit's k span of activations for every row: `8 row_tiles` rows of bf16, xor-swizzled for conflict-free `ldmatrix`. */
template<model::Format format, int row_tiles>
using ActivationTile = prims::fragment::XorSwizzle<static_cast<int>(schedule::tile_columns(format) * 2), 8 * row_tiles>;

/** A lane's 16 bytes of unit `slab * 4 + chunk` of a weight tile: slab is the m16 block, chunk the step pair (or step). */
[[nodiscard]] constexpr auto weight_offset(unsigned slab, unsigned chunk, unsigned lane) -> unsigned {
	return (slab * 4U + chunk) * static_cast<unsigned>(prims::tile::unit_bytes) + lane * 16U;
}

/**
 * The activation tile offset lane `lane` hands `ldmatrix.x4` for the step pair `pair` of the n8 tile whose first row is
 * `first_row`: lane 8m + r supplies row r of matrix m, and matrix m is k chunk 4 pair + m (two k16 steps).
 */
template<class Activations>
[[nodiscard]] constexpr auto activation_offset(int first_row, int pair, unsigned lane) -> int {
	return Activations::offset(first_row + static_cast<int>(lane % 8U), 4 * pair + static_cast<int>(lane / 8U));
}

/** The output (within a 128-row tile) and the verify row an accumulator register holds. */
struct Element {
	int output;
	int row;

	[[nodiscard]] constexpr auto operator==(Element const&) const -> bool = default;
};

/** The decode: register `reg` of n8 tile `n` of the warp owning m16 slab `slab`. */
[[nodiscard]] constexpr auto element(unsigned slab, unsigned lane, int n, int reg) -> Element {
	prims::fragment::Coordinate const at = prims::fragment::c(static_cast<int>(lane), reg);
	return {.output = static_cast<int>(slab) * 16 + at.row, .row = 8 * n + at.column};
}

/** Where an element sits in a slot (and an output tile) of [tensor][row][128 outputs] fp32 values. */
[[nodiscard]] constexpr auto slot_index(int tensor, int rows, Element at) -> int {
	return (tensor * rows + at.row) * static_cast<int>(tile_rows) + at.output;
}

namespace prefill {

/** Rows of one work item: the n side of the block's tile. */
inline constexpr int block_rows = 32;
/** A consumer warp's share of a work item: two m16 slabs by two n8 tiles. */
inline constexpr int warp_slabs = 2;
inline constexpr int warp_row_tiles = 2;
inline constexpr int output_groups = static_cast<int>(tile_rows) / (16 * warp_slabs);
inline constexpr int row_groups = block_rows / (8 * warp_row_tiles);

/** A work item's activation tile: its 32 rows of one unit's k span. */
template<model::Format format>
using ActivationTile = prims::fragment::XorSwizzle<static_cast<int>(schedule::tile_columns(format) * 2), block_rows>;

/** The m16 slab of the tile that slab `slab` of consumer warp `warp` is. */
[[nodiscard]] constexpr auto tile_slab(unsigned warp, int slab) -> unsigned {
	return (warp % static_cast<unsigned>(output_groups)) * static_cast<unsigned>(warp_slabs) + static_cast<unsigned>(slab);
}

/** The first row (within the 32-row block) of n8 tile `n` of consumer warp `warp`. */
[[nodiscard]] constexpr auto first_row(unsigned warp, int n) -> int {
	return 16 * static_cast<int>(warp / static_cast<unsigned>(output_groups)) + 8 * n;
}

/** Register `reg` of n8 tile `n` of slab `slab` of consumer warp `warp`: its output and its row within the block. */
[[nodiscard]] constexpr auto element(unsigned warp, unsigned lane, int slab, int n, int reg) -> Element {
	prims::fragment::Coordinate const at = prims::fragment::c(static_cast<int>(lane), reg);
	return {.output = static_cast<int>(tile_slab(warp, slab)) * 16 + at.row, .row = first_row(warp, n) + at.column};
}

}

}

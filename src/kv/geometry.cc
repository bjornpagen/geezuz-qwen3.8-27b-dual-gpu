// SPDX-License-Identifier: AGPL-3.0-only
export module engine.kv:geometry;

import std;
import engine.core;
import engine.model;

namespace engine::kv {

export using model::Count;

/**
 * The shape of a block pool. The engine has one, `model_geometry`; the pool is a template over it so the tests can
 * also run it at sizes where exhaustion, sharing and the snapshot cap happen within a few dozen operations.
 */
export struct Geometry {
	/** Positions per block. */
	Count block_tokens;
	/** Blocks of the one pool that kv and snapshots share. */
	Count pool_blocks;
	/** Entries of one slot's block table. */
	Count table_blocks;
	/** Blocks one deltanet snapshot occupies. */
	Count snapshot_blocks;
	/** Snapshots the pool holds at most: the cap on the blocks snapshots may take, by construction. */
	Count snapshot_capacity;
};

/** Whether a geometry describes a pool the bookkeeping can run: every quantity present, and the cap within the pool. */
export [[nodiscard]] constexpr auto sound(Geometry geometry) -> bool {
	return geometry.block_tokens > 0 && geometry.pool_blocks > 0 && geometry.table_blocks > 0 && geometry.snapshot_blocks > 0 &&
	       geometry.snapshot_capacity > 0 && geometry.snapshot_capacity * geometry.snapshot_blocks <= geometry.pool_blocks &&
	       geometry.pool_blocks <= std::numeric_limits<std::uint32_t>::max();
}

/** The most positions one slot's table addresses. */
export [[nodiscard]] constexpr auto table_positions(Geometry geometry) -> Count {
	return geometry.table_blocks * geometry.block_tokens;
}

export inline constexpr Geometry model_geometry = {
    .block_tokens = model::kv_block_tokens,
    .pool_blocks = model::pool_blocks,
    .table_blocks = model::kv_table_blocks,
    .snapshot_blocks = model::snapshot_blocks,
    .snapshot_capacity = model::snapshot_capacity,
};

static_assert(sound(model_geometry));
static_assert(table_positions(model_geometry) >= model::max_positions, "a slot's table addresses its longest sequence");

/** A block's index in the pool. The same index names the same block on both cards, each holding its own kv heads. */
export using BlockIndex = std::uint32_t;

/** A generational reference to a live block of the pool. */
export struct BlockHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(BlockHandle const&) const -> bool = default;
};

/** A generational reference to a live snapshot in the pool. */
export struct SnapshotHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(SnapshotHandle const&) const -> bool = default;
};

/** One of the engine's fixed slots: an index below `model::slots`, by construction. */
export class Slot {
public:
	/** The slot at a program-guaranteed index. */
	[[nodiscard]] static constexpr auto at(Count index) -> Slot {
		core::invariant(index < model::slots);
		return Slot{core::narrow<std::uint8_t>(index)};
	}

	[[nodiscard]] constexpr auto index() const -> Count {
		return index_;
	}

	[[nodiscard]] constexpr auto operator==(Slot const&) const -> bool = default;

private:
	constexpr explicit Slot(std::uint8_t index) : index_{index} {}

	std::uint8_t index_;
};

}

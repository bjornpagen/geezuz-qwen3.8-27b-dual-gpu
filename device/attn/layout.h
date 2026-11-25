// SPDX-License-Identifier: AGPL-3.0-only
//
// The shapes of attention's kernels as data: the tile of query rows a core job computes, the core's shared memory and
// budget, the merge's units, and every kernel's grid. Pure constexpr with no device dependency, so the host's kernel
// declarations launch with exactly the budgets the kernels `static_assert`.
#pragma once

#include "attn/geometry.h"
#include "prims/budget.h"

namespace engine::device::attn {

/** The query rows one job computes: `rows` positions, each with the kv head's six query heads, row-major. */
template<unsigned rows_>
struct RowTile {
	static constexpr unsigned rows = rows_;
	static constexpr unsigned query_rows = rows * static_cast<unsigned>(group_heads);
	static constexpr unsigned n_tiles = (query_rows + 7U) / 8U;
	static constexpr unsigned tile_rows = n_tiles * 8U;
	static_assert(rows >= 1 && rows <= max_rows, "a slot brings 1 to max_rows rows");
	static_assert(n_tiles <= 4, "each consumer warp owns one n8 tile of query rows per m16 half of a chunk");
};

/** Prefill runs the decode core's deepest tile: a job is up to five consecutive prompt rows. */
using PromptTile = RowTile<static_cast<unsigned>(prompt_tile_rows)>;

inline constexpr int ring_stages = 4;

/** Score rows are padded to 36 floats so the accumulator stores of a warp hit 32 different banks. */
inline constexpr unsigned score_stride = 36;

/** A weight part tile: 32 positions of bf16 per query row, two rows per 128-byte line, swizzled by line. */
[[nodiscard]] constexpr auto weight_offset(unsigned row, unsigned chunk) -> unsigned {
	return (row / 2U) * 128U + ((((row % 2U) * 4U) + chunk) ^ ((row / 2U) % 8U)) * 16U;
}

/** One job: the slot's rows, the kv head, and the run of groups. */
struct Job {
	SlotWork work;
	unsigned head;
	unsigned first_group;
	unsigned groups;
	unsigned record_base;
};

/**
 * The prefill attention's jobs on this block: of the prompt chunk's tiles of `prompt_tile_rows` rows, by kv head, every
 * `blocks`-th from the block's own index, the tiles taken last-first so the longest contexts start first.
 */
struct PromptJobs {
	unsigned slot;
	unsigned start;
	unsigned rows;
	unsigned first;
	unsigned blocks;

	[[nodiscard]] constexpr auto tiles() const -> unsigned {
		return static_cast<unsigned>(model::divide_up(rows, prompt_tile_rows));
	}

	[[nodiscard]] constexpr auto count() const -> unsigned {
		unsigned const total = tiles() * static_cast<unsigned>(kv_heads);
		return total > first ? (total - first + blocks - 1) / blocks : 0;
	}

	[[nodiscard]] constexpr auto at(unsigned index) const -> Job {
		unsigned const job = first + index * blocks;
		unsigned const tile = tiles() - 1 - job / static_cast<unsigned>(kv_heads);
		unsigned const begin = tile * static_cast<unsigned>(prompt_tile_rows);
		unsigned const rest = rows - begin;
		SlotWork const work = {.slot = slot,
		                       .position = start + begin,
		                       .rows = rest < prompt_tile_rows ? rest : static_cast<unsigned>(prompt_tile_rows),
		                       .first_row = begin};
		return {.work = work, .head = job % static_cast<unsigned>(kv_heads), .first_group = 0, .groups = static_cast<unsigned>(groups(work)), .record_base = 0};
	}
};

/** How a group's last chunk leaves each row's result in prefill. */
enum class Fold : unsigned char { Keep, Copy, Combine };

/** Sixteen bytes of shared memory: the unit of every copy, `ldmatrix` row and swizzle. */
struct alignas(16) Piece {
	unsigned word[4];
};

/** The kv ring's full and empty mbarriers, one each per stage. */
struct RingStorage {
	alignas(8) unsigned long long full[ring_stages];
	alignas(8) unsigned long long empty[ring_stages];
};

template<class Tile>
struct CoreShared {
	alignas(128) Piece stage[ring_stages][tile_bytes / chunk_bytes];
	alignas(128) Piece query[Tile::tile_rows * row_chunks];
	alignas(128) Piece fresh[kinds][max_rows * row_chunks];
	alignas(128) Piece weights[3][Tile::tile_rows * 4];
	alignas(16) float scores[Tile::tile_rows][score_stride];
	RingStorage ring;
	float maximum[Tile::tile_rows];
	float total[Tile::tile_rows];
	float rescale[Tile::tile_rows];
	unsigned active[Tile::tile_rows];
	float final_maximum[Tile::tile_rows];
	float final_total[Tile::tile_rows];
	float merge_left[Tile::tile_rows];
	float merge_right[Tile::tile_rows];
	Fold fold[Tile::tile_rows];
	Cycle cycle;
	Assignment assignment;
};

/** The budget of a kernel running the core: twelve warps, one block per SM, every register the occupancy leaves. */
template<class Tile>
[[nodiscard]] consteval auto core_budget() -> prims::Budget {
	auto budget = prims::Budget{.threads = 12 * prims::sm86.warp_threads, .registers_per_thread = 0, .shared_bytes = 0, .blocks_per_sm = 1};
	budget.registers_per_thread = prims::register_cap(budget);
	budget.shared_bytes = static_cast<int>(sizeof(CoreShared<Tile>));
	return budget;
}

/** The decode core's and the prefill attention's grid: one block per SM. */
inline constexpr unsigned core_blocks = static_cast<unsigned>(prims::sm86.sms);

// ---------------------------------------------------------------------------------------------------------------------
// The merge.

/** One output of the merge: a row of the cycle and one of the card's query heads. */
struct MergeUnit {
	unsigned work;
	unsigned token;
	unsigned query_head;
};

[[nodiscard]] constexpr auto merge_units(Cycle const& cycle) -> Count {
	auto total = Count{0};
	for (auto index = Count{0}; index < cycle.live; ++index) {
		total += Count{cycle.work[index].rows} * query_heads;
	}
	return total;
}

/** Unit `unit` of the cycle: slots in order, each its rows in order, each row its query heads in order. */
[[nodiscard]] constexpr auto merge_unit(Cycle const& cycle, Count unit) -> MergeUnit {
	auto rest = unit;
	auto work = Count{0};
	while (rest >= Count{cycle.work[work].rows} * query_heads) {
		rest -= Count{cycle.work[work].rows} * query_heads;
		++work;
	}
	return {.work = static_cast<unsigned>(work), .token = static_cast<unsigned>(rest / query_heads), .query_head = static_cast<unsigned>(rest % query_heads)};
}

/** The groups a row of the slot folds: those holding a position it sees. */
[[nodiscard]] constexpr auto row_groups(SlotWork const& work, Count token) -> Count {
	return group_of((Count{work.position} + token) / chunk_positions) + 1;
}

struct MergeShared {
	Cycle cycle;
};

/** Eight warps, one unit each at a time; a cycle has at most 32 rows x 12 heads = 384 units, under one wave. */
inline constexpr prims::Budget merge_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = sizeof(MergeShared), .blocks_per_sm = 2};
inline constexpr unsigned merge_blocks = static_cast<unsigned>(prims::sm86.sms);

/** The prompt append: one warp per (row, kv head, kind). */
inline constexpr prims::Budget append_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 2};
inline constexpr unsigned append_blocks = static_cast<unsigned>(prims::sm86.sms);

static_assert(prims::fits(merge_budget) && prims::fits(append_budget));
static_assert(Count{merge_blocks} * static_cast<unsigned>(merge_budget.threads / 32) >= query_heads * model::rows(model::RowBudget::FourTiles),
              "every merge unit of a cycle has its own warp");

}

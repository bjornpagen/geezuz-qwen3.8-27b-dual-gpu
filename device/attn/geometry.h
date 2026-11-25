// SPDX-License-Identifier: AGPL-3.0-only
//
// Where attention's data lives and how its work divides: the byte layout of a kv block, the fixed groups of chunks
// attention folds a context in, the per-cycle description of each live slot's rows, the balanced assignment of groups
// to SMs, the partial records the decode core leaves for the merge, and the row buffers the phases exchange. Pure
// constexpr with no device dependency, so the kernels, the host reference and the tests read one definition.
#pragma once

#include "model.h"
#include "prims/budget.h"
#include "prims/fragment.h"
#include "schedule/program.h"

namespace engine::device::attn {

using model::Count;

inline constexpr Count head_dim = model::head_dim;
inline constexpr Count kv_heads = model::kv_heads_per_card;
inline constexpr Count query_heads = model::query_heads_per_card;
/** Query heads reading one kv head: one kv tile serves all of them as one block of rows. */
inline constexpr Count group_heads = model::query_heads / model::kv_heads;
inline constexpr Count kv_layers = model::kv_layers;
inline constexpr Count block_positions = model::kv_block_tokens;
inline constexpr Count chunk_positions = model::attention::chunk_positions;
inline constexpr Count max_chunks = model::attention::max_chunks;

static_assert(schedule::attention_chunk_positions == chunk_positions, "the schedule's chunk is the model's");

/** Rows one slot brings into a cycle: its committed token and its deepest drafts. */
inline constexpr Count max_rows = 1 + model::drafts(model::max_draft_depth);

static_assert(group_heads * kv_heads == query_heads, "a card's query heads read its own kv heads");
static_assert(model::max_positions % chunk_positions == 0);

// ---------------------------------------------------------------------------------------------------------------------
// The kv block.
//
// A block holds 64 positions of every kv layer of the card: for each layer, each kv head, the keys and then the values,
// each a plane of 64 rows of 256 bf16 features. A row's sixteen 16-byte chunks of 8 features are stored xor-swizzled
// by the position's low three bits, which is exactly the shared-memory tile `KvTile` that `ldmatrix` reads without bank
// conflicts: a 32-position chunk of one plane is 16 KiB of contiguous bytes that a producer copies verbatim. The
// positions [0, n) of a plane are its first n rows, so a block's prefix is one contiguous run per plane.

enum class Kind : unsigned char { Key, Value };

inline constexpr Count kinds = 2;
inline constexpr Count row_bytes = head_dim * model::element_bytes(model::Format::Bf16);
inline constexpr Count chunk_bytes = 16;
inline constexpr Count row_chunks = row_bytes / chunk_bytes;
inline constexpr Count features_per_chunk = chunk_bytes / 2;
inline constexpr Count tile_bytes = chunk_positions * row_bytes;
inline constexpr Count plane_bytes = block_positions * row_bytes;

using KvTile = prims::fragment::XorSwizzle<static_cast<int>(row_bytes), static_cast<int>(chunk_positions)>;

static_assert(prims::fragment::conflict_free<KvTile>(), "a kv tile is read by ldmatrix without bank conflicts");
static_assert(KvTile::bytes == static_cast<int>(tile_bytes));
static_assert(kv_layers * kv_heads * kinds * plane_bytes == model::kv_block_bytes, "the planes tile the block exactly");
static_assert(block_positions % chunk_positions == 0 && block_positions / chunk_positions == 2, "a block is two chunks");

[[nodiscard]] constexpr auto kind_index(Kind kind) -> Count {
	return kind == Kind::Key ? 0 : 1;
}

/** The byte offset in a block of the plane of (`layer`, `head`, `kind`). */
[[nodiscard]] constexpr auto plane_offset(Count layer, Count head, Kind kind) -> Count {
	return ((layer * kv_heads + head) * kinds + kind_index(kind)) * plane_bytes;
}

/** The stored place of a row's logical 16-byte chunk: the swizzle of the row's position within its tile. */
[[nodiscard]] constexpr auto stored_chunk(Count position, Count chunk) -> Count {
	return chunk ^ (position % 8);
}

/** The byte offset in its block of the row of absolute `position`. */
[[nodiscard]] constexpr auto row_offset(Count layer, Count head, Kind kind, Count position) -> Count {
	return plane_offset(layer, head, kind) + (position % block_positions) * row_bytes;
}

/** The byte offset in its block of feature `feature` of the row of absolute `position`. */
[[nodiscard]] constexpr auto element_offset(Count layer, Count head, Kind kind, Count position, Count feature) -> Count {
	return row_offset(layer, head, kind, position) + stored_chunk(position, feature / features_per_chunk) * chunk_bytes +
	       (feature % features_per_chunk) * 2;
}

/** The byte offset in its block of chunk `chunk`'s tile of one plane: 16 KiB a producer streams as is. */
[[nodiscard]] constexpr auto tile_offset(Count layer, Count head, Kind kind, Count chunk) -> Count {
	return plane_offset(layer, head, kind) + (chunk % 2) * tile_bytes;
}

/** The block-table entry holding chunk `chunk`. */
[[nodiscard]] constexpr auto table_entry(Count chunk) -> Count {
	return chunk * chunk_positions / block_positions;
}

// ---------------------------------------------------------------------------------------------------------------------
// Groups: the unit attention folds and assigns. The model fixes them (`model::attention`): chunks are folded in order
// within a group by an online softmax, and a row's groups are then merged in order. A group is never split across
// SMs, so every row's arithmetic is a function of its own query, its kv and its position alone: not of the SM count,
// the other slots or the rows beside it.

using model::attention::far_group_chunks;
using model::attention::group_begin;
using model::attention::group_of;
using model::attention::near_chunks;
using model::attention::near_group_chunks;

/** The groups holding chunks [0, chunks). */
[[nodiscard]] constexpr auto groups_covering(Count chunks) -> Count {
	return chunks == 0 ? 0 : group_of(chunks - 1) + 1;
}

/** The first group starting at or after chunk `chunk`. */
[[nodiscard]] constexpr auto first_group_from(Count chunk) -> Count {
	return groups_covering(chunk);
}

inline constexpr Count max_groups = groups_covering(max_chunks);

static_assert(max_groups == 240);

// ---------------------------------------------------------------------------------------------------------------------
// One cycle's attention work, as the decode loop describes it.

/**
 * A live slot's rows this cycle: `rows` consecutive positions from `position`, whose fp32 projections sit at rows
 * [`first_row`, `first_row` + `rows`) of the row buffers. Row t attends to positions [0, `position` + t]; its own and
 * the later rows' kv are appended this cycle.
 */
struct SlotWork {
	unsigned slot;
	unsigned position;
	unsigned rows;
	unsigned first_row;
};

/** The live slots of one cycle, in slot order. */
struct Cycle {
	unsigned live;
	SlotWork work[model::slots];
};

/** Positions a slot's rows attend to in all: every earlier position and the rows themselves. */
[[nodiscard]] constexpr auto context(SlotWork const& work) -> Count {
	return Count{work.position} + work.rows;
}

[[nodiscard]] constexpr auto chunks(SlotWork const& work) -> Count {
	return model::divide_up(context(work), chunk_positions);
}

[[nodiscard]] constexpr auto groups(SlotWork const& work) -> Count {
	return groups_covering(chunks(work));
}

/** The query rows one kv head of the slot serves: each row's six query heads, row-major (row t, head h at 6t + h). */
[[nodiscard]] constexpr auto query_rows(SlotWork const& work) -> Count {
	return Count{work.rows} * group_heads;
}

/** Whether a cycle is one the decode loop can describe: every slot distinct, in order, with 1 to `max_rows` rows. */
[[nodiscard]] constexpr auto valid(Cycle const& cycle, Count row_capacity) -> bool {
	if (cycle.live > model::slots) {
		return false;
	}
	auto rows = Count{0};
	for (auto index = Count{0}; index < cycle.live; ++index) {
		auto const& work = cycle.work[index];
		auto const ordered = index == 0 || cycle.work[index - 1].slot < work.slot;
		if (!ordered || work.slot >= model::slots || work.rows < 1 || work.rows > max_rows || context(work) > model::max_positions ||
		    work.first_row != rows) {
			return false;
		}
		rows += work.rows;
	}
	return rows <= row_capacity;
}

// ---------------------------------------------------------------------------------------------------------------------
// The assignment of groups to SMs.
//
// The work is the sequence of (slot, kv head) segments in order, each its groups in order, and a group costs its
// chunks. SM s of `sms` takes exactly the groups whose first chunk's offset in that sequence falls in
// [ceil(s T / sms), ceil((s + 1) T / sms)), T the total: contiguous, every group once, each SM within one group of
// T / sms. The assignment decides only where a group is computed, never how.

/** A run of one segment's groups on one SM. */
struct Item {
	unsigned work;
	unsigned head;
	unsigned first_group;
	unsigned groups;
};

/** A contiguous range meets each segment at most once. */
inline constexpr Count max_items = model::slots * kv_heads;

struct Assignment {
	unsigned count;
	Item item[max_items];
};

[[nodiscard]] constexpr auto total_chunks(Cycle const& cycle) -> Count {
	auto total = Count{0};
	for (auto index = Count{0}; index < cycle.live; ++index) {
		total += kv_heads * chunks(cycle.work[index]);
	}
	return total;
}

/** Hands `emit` each item of the groups SM `sm` of `sms` computes, in work order. */
template<class Emit>
constexpr void for_each_item(Cycle const& cycle, Count sm, Count sms, Emit emit) {
	auto const total = total_chunks(cycle);
	auto const low = model::divide_up(sm * total, sms);
	auto const high = model::divide_up((sm + 1) * total, sms);
	auto offset = Count{0};
	for (auto index = Count{0}; index < cycle.live; ++index) {
		auto const length = chunks(cycle.work[index]);
		for (auto head = Count{0}; head < kv_heads; ++head) {
			// The groups whose first chunk lies in [low, high) once shifted by the segment's offset.
			auto const from = low > offset ? low - offset : 0;
			auto const to = high > offset ? (high - offset < length ? high - offset : length) : 0;
			auto const first = first_group_from(from);
			auto const last = first_group_from(to);
			if (from < length && last > first) {
				emit(Item{.work = static_cast<unsigned>(index),
				          .head = static_cast<unsigned>(head),
				          .first_group = static_cast<unsigned>(first),
				          .groups = static_cast<unsigned>(last - first)});
			}
			offset += length;
		}
	}
}

/** The groups SM `sm` of `sms` computes. */
[[nodiscard]] constexpr auto assignment(Cycle const& cycle, Count sm, Count sms) -> Assignment {
	auto result = Assignment{.count = 0, .item = {}};
	for_each_item(cycle, sm, sms, [&result](Item item) {
		result.item[result.count] = item;
		++result.count;
	});
	return result;
}

/** The chunks [begin, end) an item's groups hold of its slot's context. */
struct ChunkRange {
	Count begin;
	Count end;
};

[[nodiscard]] constexpr auto chunk_range(SlotWork const& work, Count first_group, Count group_count) -> ChunkRange {
	auto const end = group_begin(first_group + group_count);
	auto const length = chunks(work);
	return {.begin = group_begin(first_group), .end = end < length ? end : length};
}

/** The positions of the slot's own rows that fall in a chunk range: the kv this cycle appends there. */
struct PositionRange {
	Count begin;
	Count end;
};

[[nodiscard]] constexpr auto fresh_positions(SlotWork const& work, ChunkRange range) -> PositionRange {
	auto const low = range.begin * chunk_positions;
	auto const high = range.end * chunk_positions;
	auto const begin = Count{work.position} > low ? Count{work.position} : low;
	auto const end = context(work) < high ? context(work) : high;
	return {.begin = begin, .end = end > begin ? end : begin};
}

// ---------------------------------------------------------------------------------------------------------------------
// Partial records: what the decode core leaves for the merge.
//
// One record row per (slot, kv head, group, query row): the group's unnormalized output (256 fp32, scaled by the
// softmax weight scale) and its running maximum and total. They are laid out in segment order, each segment its groups
// in order, each group its slot's query rows, so the offsets follow from the cycle alone.

/** A record row's running maximum and total, padded to one 16-byte chunk. */
struct Running {
	float maximum;
	float total;
	float unused[2];
};

static_assert(sizeof(Running) == chunk_bytes);

/** The first record row of segment (`work`, `head`). */
[[nodiscard]] constexpr auto record_base(Cycle const& cycle, Count work, Count head) -> Count {
	auto base = Count{0};
	for (auto index = Count{0}; index < work; ++index) {
		base += kv_heads * groups(cycle.work[index]) * query_rows(cycle.work[index]);
	}
	return base + head * groups(cycle.work[work]) * query_rows(cycle.work[work]);
}

/** The record row of (`work`, `head`, `group`, query row `row`). */
[[nodiscard]] constexpr auto record_row(Cycle const& cycle, Count work, Count head, Count group, Count row) -> Count {
	return record_base(cycle, work, head) + group * query_rows(cycle.work[work]) + row;
}

[[nodiscard]] constexpr auto record_rows(Cycle const& cycle) -> Count {
	auto rows = Count{0};
	for (auto index = Count{0}; index < cycle.live; ++index) {
		rows += kv_heads * groups(cycle.work[index]) * query_rows(cycle.work[index]);
	}
	return rows;
}

/** The most record rows any valid cycle at `budget` leaves: every row at the longest context. */
[[nodiscard]] constexpr auto max_record_rows(model::RowBudget budget) -> Count {
	return kv_heads * max_groups * group_heads * model::rows(budget);
}

/** The workspace bytes the records take at `budget`: outputs and running pairs. */
[[nodiscard]] constexpr auto record_bytes(model::RowBudget budget) -> Count {
	return max_record_rows(budget) * (head_dim * sizeof(float) + sizeof(Running));
}

// ---------------------------------------------------------------------------------------------------------------------
// The row buffers.
//
// AttentionIn leaves each row's fp32 projections: the query halves of q_proj (12 heads x 256, the card's heads in
// order), k_proj and v_proj (2 heads x 256); AttentionGate leaves the gate halves of q_proj (12 x 256). AttentionMerge
// writes each row's bf16 attention output (12 x 256), o_proj's input.

inline constexpr Count query_row_floats = query_heads * head_dim;
inline constexpr Count kv_row_floats = kv_heads * head_dim;
inline constexpr Count gate_row_floats = query_heads * head_dim;
inline constexpr Count output_row_values = query_heads * head_dim;

/** A device address as kernel parameters carry it. */
using Address = unsigned long long;

/** The fp32 projection rows of the cycle's (or the prompt chunk's) tokens. */
struct Projections {
	Address query;
	Address key;
	Address value;
	Address gate;
};

/** A kv layer's view of the pool: block 0's first byte, the slots' block tables (u32 [slots][table]) and the layer. */
struct KvPool {
	Address blocks;
	Address tables;
	unsigned layer;
	unsigned unused;
};

/** What preparing a row reads: the layer's q_norm and k_norm weights (bf16 [256] each) and the rope table. */
struct Preparation {
	Address query;
	Address key;
	/** `rope_table_bytes`: every (position, pair)'s rotation, filled once by `attn_rope_table`. */
	Address rotations;
};

/** The records' storage: fp32 [rows][256] outputs and `Running` [rows]. */
struct Records {
	Address output;
	Address running;
};

/** The decode core's parameters: one cycle, one layer. */
struct CoreParams {
	Cycle cycle;
	KvPool kv;
	Projections rows;
	Preparation preparation;
	Records records;
};

/** The decode merge's parameters: the core's records, the gate rows and the bf16 output rows. */
struct MergeParams {
	Cycle cycle;
	Records records;
	Address gate;
	Address output;
};

/**
 * A prompt chunk of one slot: `count` positions from `start`, whose rows are [0, count) of the row buffers. The append
 * writes their kv; the prefill attention then reads every position up to each row's own.
 */
struct PromptParams {
	unsigned slot;
	unsigned start;
	unsigned count;
	unsigned unused;
	KvPool kv;
	Projections rows;
	Preparation preparation;
	Address output;
};

/** Prompt rows one prefill item takes: as many as a decode slot's deepest cycle, so both run one core. */
inline constexpr Count prompt_tile_rows = max_rows;

}

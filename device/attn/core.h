// SPDX-License-Identifier: AGPL-3.0-only
//
// The attention core as device building blocks: the decode core and the prefill attention both run a list of jobs, a
// job being one slot's query rows of one kv head against a run of that slot's groups. Three roles share a block:
//
// - one producer warp streams each chunk's key tile and value tile (16 KiB each, contiguous in the block) through a
//   four-stage `cp.async` ring, evict-first: kv tiles are read once;
// - three preparing warps normalize and rotate the job's queries into a bf16 tile, and in decode compute the kv the
//   job's own rows append (written to the pool, and kept to be spliced into the copied tiles, so no copy has to wait
//   for the append);
// - eight consumer warps run each chunk: S^T = K Q^T on the tensor cores (positions are the m16 side, query rows the n8
//   side, so a slot's few rows waste no m16 tile), the online softmax by query row, and O^T += V^T P^T with P carried
//   exactly as three bf16 parts. At a group's end decode writes the group's record; prefill folds it into the row's
//   result and, after the last group, writes the gated bf16 output.
//
// The roles are warps [0, 1), [1, 9) and [9, 12): the decode kernel's producer, consumer warpgroups, and its exchange
// and control warps, which are idle during attention.
#pragma once

#include "attn/address.h"
#include "attn/arithmetic.h"
#include "attn/geometry.h"
#include "attn/layout.h"
#include "prims/budget.h"
#include "prims/fragment.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/reduce.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"

namespace engine::device::attn {

using Producer = prims::Warps<0, 1>;
using Consumers = prims::Warps<1, 8>;
using Preparers = prims::Warps<9, 3>;

inline constexpr unsigned consumer_warps = static_cast<unsigned>(Consumers::count);
inline constexpr unsigned preparer_warps = static_cast<unsigned>(Preparers::count);

/** The consumers among themselves, between a chunk's steps. */
using ConsumerStep = prims::NamedBarrier<1, Consumers::count>;
/** The preparers hand a job's queries (and fresh kv) to the consumers. */
using JobReady = prims::NamedBarrier<2, Consumers::count + Preparers::count>;
/** The consumers release the query tile and the fresh rows to the preparers. */
using JobDone = prims::NamedBarrier<3, Consumers::count + Preparers::count>;

/** A chunk streams its key tile, then its value tile. */
inline constexpr Kind kinds_in_order[kinds] = {Kind::Key, Kind::Value};

using KvRing = prims::StageRing<ring_stages, 32, consumer_warps>;
using Cursor = KvRing::Cursor;

static_assert(sizeof(KvRing::Storage) == sizeof(RingStorage) && alignof(KvRing::Storage) == alignof(RingStorage));

/** The ring over the barriers a core layout holds. */
template<class Tile>
[[nodiscard]] __attribute__((device, always_inline)) inline auto kv_ring(CoreShared<Tile>& shared) -> KvRing {
	return {.full_base = prims::shared_address(shared.ring.full), .empty_base = prims::shared_address(shared.ring.empty)};
}

/** The query tile: query row j at row j of a swizzled tile, like a kv tile's positions. */
template<class Tile>
using QueryTile = prims::fragment::XorSwizzle<static_cast<int>(row_bytes), static_cast<int>(Tile::tile_rows)>;

/** What the consumers do with a finished group. */
enum class Sink : unsigned char {
	/** Write it as a record for the merge (decode). */
	Records,
	/** Fold it into the row's result, and write the gated output after the last group (prefill). */
	Output,
};

/** Whether a job's own rows append kv (decode) or find it already in the pool (prefill). */
enum class Fresh : unsigned char { Appended, Cached };

// ---------------------------------------------------------------------------------------------------------------------
// Device addresses.

/** The 8 fp32 values of a row's logical chunk `chunk`, from a row of fp32 starting at `row`. */
template<prims::Eviction eviction>
__attribute__((device, always_inline)) inline void load_floats(float const* row, unsigned chunk, prims::Policy<eviction> policy, float (&values)[lane_features]) {
	auto const* chunks = reinterpret_cast<prims::Chunk const*>(row);
	prims::Chunk const low = prims::load(chunks + Count{2} * chunk, policy);
	prims::Chunk const high = prims::load(chunks + Count{2} * chunk + 1U, policy);
#pragma unroll
	for (Count word = 0; word < 4; ++word) {
		values[word] = __builtin_bit_cast(float, low.word[word]);
		values[4 + word] = __builtin_bit_cast(float, high.word[word]);
	}
}

/** Eight bf16, the first in the low half of the first word: one stored chunk of a row. */
[[nodiscard]] __attribute__((device, always_inline)) inline auto pack_chunk(float const (&values)[lane_features]) -> prims::Chunk {
	prims::Chunk packed{};
#pragma unroll
	for (Count word = 0; word < 4; ++word) {
		packed.word[word] = prims::pack(prims::round_to_bf16(values[2 * word]), prims::round_to_bf16(values[2 * word + 1])).bits;
	}
	return packed;
}

// ---------------------------------------------------------------------------------------------------------------------
// The preparation of a row, one warp per row, lane L holding features [8L, 8L + 8).

/** q_norm or k_norm, then the rope at `position`: the lane's chunk of the rotated row, still fp32. */
__attribute__((device, always_inline)) inline void normalize_and_rotate(float (&values)[lane_features], prims::Chunk weight, unsigned position) {
	float const inverse = inverse_rms(prims::warp_sum(lane_squares(values)));
#pragma unroll
	for (unsigned index = 0; index < lane_features; ++index) {
		unsigned const pair = weight.word[index / 2];
		Bf16 const w = {.bits = static_cast<unsigned short>(index % 2 == 0 ? pair & 0xffffU : pair >> 16U)};
		values[index] = normalized(values[index], inverse, w);
	}
	unsigned const lane = prims::lane();
	float partner[lane_features] = {};
#pragma unroll
	for (unsigned index = 0; index < lane_features; ++index) {
		partner[index] = __builtin_bit_cast(float, prims::exchange(__builtin_bit_cast(unsigned, values[index]), 4));
	}
	// Lanes 0-3 hold features 0-31, the first of each pair; lanes 4-7 hold 32-63, the second.
	if (lane < 8) {
#pragma unroll
		for (unsigned index = 0; index < lane_features; ++index) {
			int const pair = static_cast<int>((lane % 4U) * lane_features + index);
			SinCos const turn = sincos(angle(position, pair));
			values[index] = lane < 4 ? rotate_first(values[index], partner[index], turn) : rotate_second(partner[index], values[index], turn);
		}
	}
}

template<class Tile>
struct Core {
	using Shared = CoreShared<Tile>;
	using Queries = QueryTile<Tile>;

	// -----------------------------------------------------------------------------------------------------------------
	// The producer.

	/** Streams every chunk of every job, key tile then value tile, and returns the ring's cursor after them. */
	template<class Jobs>
	__attribute__((device, always_inline)) static auto produce(Shared& shared, Jobs const& jobs, KvRing ring, Cursor cursor, KvPool kv) -> Cursor {
		unsigned const lane = prims::lane();
		auto const once = prims::policy<prims::Eviction::First>();
		auto const reused = prims::policy<prims::Eviction::Last>();
		auto const* tables = at<unsigned const>(kv.tables);
		for (unsigned index = 0; index < jobs.count(); ++index) {
			Job const job = jobs.at(index);
			ChunkRange const range = chunk_range(job.work, job.first_group, job.groups);
			unsigned const* table = tables + Count{job.work.slot} * model::kv_table_blocks;
			for (Count chunk = range.begin; chunk < range.end; ++chunk) {
				Count const block = prims::load_word(table + table_entry(chunk), reused);
				for (Kind const kind : kinds_in_order) {
					ring.await_free(cursor, prims::block_backoff);
					Address const tile = kv.blocks + block * model::kv_block_bytes + tile_offset(kv.layer, job.head, kind, chunk);
					auto const* source = at<prims::Chunk const>(tile);
					for (unsigned piece = lane; piece < tile_bytes / chunk_bytes; piece += 32) {
						prims::copy_async<prims::Prefetch::Chunk256>(prims::shared_address(&shared.stage[cursor.stage][piece]), source + piece, once);
					}
					ring.filled_on_copies(cursor);
					cursor = cursor.next();
				}
			}
		}
		return cursor;
	}

	// -----------------------------------------------------------------------------------------------------------------
	// The preparers.

	/** The queries of every job and, in decode, the kv its rows append. */
	template<Fresh fresh, class Jobs>
	__attribute__((device, always_inline)) static void prepare(Shared& shared, Jobs const& jobs, KvPool kv, Projections rows, NormWeights norms) {
		unsigned const warp = Preparers::rank(prims::warp_in_block());
		unsigned const lane = prims::lane();
		auto const reused = prims::policy<prims::Eviction::Last>();
		auto const appended = prims::policy<prims::Eviction::Normal>();
		prims::Chunk const query_weight = prims::load_constant(at<prims::Chunk const>(norms.query) + lane, reused);
		prims::Chunk const key_weight = prims::load_constant(at<prims::Chunk const>(norms.key) + lane, reused);
		prims::SharedAddress const query_tile = prims::shared_address(shared.query);
		for (unsigned index = 0; index < jobs.count(); ++index) {
			if (index > 0) {
				JobDone::sync();
			}
			Job const job = jobs.at(index);
			for (unsigned row = warp; row < Tile::tile_rows; row += preparer_warps) {
				prims::Chunk chunk{};
				// Rows past the slot's own are zero: they are masked, and zeros keep every score they make finite.
				if (row < query_rows(job.work)) {
					unsigned const token = row / static_cast<unsigned>(group_heads);
					unsigned const head = job.head * static_cast<unsigned>(group_heads) + row % static_cast<unsigned>(group_heads);
					float values[lane_features] = {};
					float const* source = at<float const>(rows.query) + (Count{job.work.first_row} + token) * query_row_floats + head * head_dim;
					load_floats(source, lane, reused, values);
					normalize_and_rotate(values, query_weight, job.work.position + token);
					chunk = pack_chunk(values);
				}
				prims::store_shared(query_tile.plus(static_cast<unsigned>(Queries::offset(static_cast<int>(row), static_cast<int>(lane)))), chunk);
			}
			if constexpr (fresh == Fresh::Appended) {
				PositionRange const own = fresh_positions(job.work, chunk_range(job.work, job.first_group, job.groups));
				unsigned const count = static_cast<unsigned>(own.end - own.begin);
				for (unsigned task = warp; task < kinds * count; task += preparer_warps) {
					Kind const kind = task % 2 == 0 ? Kind::Key : Kind::Value;
					Count const position = own.begin + task / 2;
					unsigned const token = static_cast<unsigned>(position - job.work.position);
					Count const row = Count{job.work.first_row} + token;
					float values[lane_features] = {};
					if (kind == Kind::Key) {
						load_floats(at<float const>(rows.key) + row * kv_row_floats + job.head * head_dim, lane, reused, values);
						normalize_and_rotate(values, key_weight, static_cast<unsigned>(position));
					} else {
						load_floats(at<float const>(rows.value) + row * kv_row_floats + job.head * head_dim, lane, reused, values);
					}
					prims::Chunk const packed = pack_chunk(values);
					Count const slot_chunk = stored_chunk(position, lane);
					Count const block = prims::load_word(at<unsigned const>(kv.tables) + Count{job.work.slot} * model::kv_table_blocks + position / block_positions, reused);
					Address const target = kv.blocks + block * model::kv_block_bytes + row_offset(kv.layer, job.head, kind, position) + slot_chunk * chunk_bytes;
					prims::store(at<prims::Chunk>(target), packed, appended);
					unsigned const kept = static_cast<unsigned>((position - own.begin) * row_chunks + slot_chunk);
					prims::store_shared(prims::shared_address(&shared.fresh[kind_index(kind)][kept]), packed);
				}
			}
			JobReady::arrive();
		}
	}

	// -----------------------------------------------------------------------------------------------------------------
	// The consumers.

	/** Replaces the rows of a just-landed tile that this cycle appends with the kept fresh rows. */
	__attribute__((device, always_inline)) static void splice(Shared& shared, Cursor cursor, Kind kind, PositionRange own, Count chunk) {
		Count const low = chunk * chunk_positions;
		Count const begin = own.begin > low ? own.begin : low;
		Count const end = own.end < low + chunk_positions ? own.end : low + chunk_positions;
		unsigned const thread = Consumers::rank(prims::warp_in_block()) * 32U + prims::lane();
		for (Count piece = thread; piece < (end > begin ? end - begin : 0) * row_chunks; piece += Consumers::threads) {
			Count const position = begin + piece / row_chunks;
			Count const stored = piece % row_chunks;
			prims::Chunk const kept = prims::load_shared(prims::shared_address(&shared.fresh[kind_index(kind)][(position - own.begin) * row_chunks + stored]));
			prims::store_shared(prims::shared_address(&shared.stage[cursor.stage][(position - low) * row_chunks + stored]), kept);
		}
	}

	/** S^T for the warp's (m16 half of the chunk, n8 tile of query rows), into the score rows. */
	__attribute__((device, always_inline)) static void scores(Shared& shared, Cursor cursor, unsigned warp) {
		unsigned const half = warp / 4U;
		unsigned const n_tile = warp % 4U;
		if (n_tile >= Tile::n_tiles) {
			return;
		}
		unsigned const lane = prims::lane();
		unsigned const matrix = lane / 8U;
		unsigned const line = lane % 8U;
		prims::SharedAddress const queries = prims::shared_address(shared.query);
		prims::SharedAddress const keys = prims::shared_address(shared.stage[cursor.stage]);
		prims::Accumulator sum = {.reg = {0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
		for (unsigned pair = 0; pair < 8; ++pair) {
			prims::Matrices4 const q = prims::ldmatrix_x4<prims::fragment::Transpose::Plain>(
			    queries.plus(static_cast<unsigned>(Queries::offset(static_cast<int>(8U * n_tile + line), static_cast<int>(4U * pair + matrix)))));
#pragma unroll
			for (unsigned step = 0; step < 2; ++step) {
				unsigned const k_step = 2U * pair + step;
				prims::Matrices4 const k = prims::ldmatrix_x4<prims::fragment::Transpose::Plain>(keys.plus(static_cast<unsigned>(
				    KvTile::offset(static_cast<int>(16U * half + line + 8U * (matrix % 2U)), static_cast<int>(2U * k_step + matrix / 2U)))));
				prims::mma(sum, prims::AFragment{.reg = {k.reg[0], k.reg[1], k.reg[2], k.reg[3]}},
				           prims::BFragment{.reg = {q.reg[step + step], q.reg[step + step + 1U]}});
			}
		}
		for (unsigned reg = 0; reg < 4; ++reg) {
			prims::fragment::Coordinate const at_element = prims::fragment::c(static_cast<int>(lane), static_cast<int>(reg));
			unsigned const position = 16U * half + static_cast<unsigned>(at_element.row);
			unsigned const row = 8U * n_tile + static_cast<unsigned>(at_element.column);
			shared.scores[row][position] = score(sum.reg[reg]);
		}
	}

	/** Row j of the job sees positions [0, limit]; rows past the slot's own see nothing. */
	[[nodiscard]] __attribute__((device, always_inline)) static auto limit(Job const& job, unsigned row) -> Count {
		return Count{job.work.position} + row / static_cast<unsigned>(group_heads);
	}

	/**
	 * The online softmax of one chunk for the warp's query rows (lane = position): the running maximum and total, the
	 * rescale of what ran before, and the weights as bf16 parts. A row that sees none of the chunk keeps its state.
	 */
	template<PvParts parts>
	__attribute__((device, always_inline)) static void softmax(Shared& shared, Job const& job, Count chunk, bool first_of_group, unsigned warp) {
		unsigned const lane = prims::lane();
		Count const start = chunk * chunk_positions;
		auto* weights = reinterpret_cast<unsigned short*>(shared.weights);
		for (unsigned row = warp; row < Tile::tile_rows; row += consumer_warps) {
			bool const sees = row < query_rows(job.work) && start <= limit(job, row);
			float weight_value = 0.0F;
			if (sees) {
				bool const visible = start + lane <= limit(job, row);
				float const value = shared.scores[row][lane];
				float const chunk_maximum = prims::warp_max(visible ? value : -math::infinity);
				float const running = first_of_group ? -math::infinity : shared.maximum[row];
				Advance const step = advance(running, chunk_maximum);
				weight_value = visible ? weight(value, step.maximum) : 0.0F;
				float const sum = prims::warp_sum(weight_value);
				float const kept_total = first_of_group ? 0.0F : shared.total[row];
				if (lane == 0) {
					shared.maximum[row] = step.maximum;
					shared.total[row] = running_total(kept_total, step.rescale, sum);
					shared.rescale[row] = step.rescale;
					shared.active[row] = 1;
				}
			} else if (lane == 0) {
				if (first_of_group) {
					shared.maximum[row] = -math::infinity;
					shared.total[row] = 0.0F;
				}
				shared.rescale[row] = 1.0F;
				shared.active[row] = 0;
			}
			Parts const split_weight = split(weight_value, parts);
			unsigned const element = weight_offset(row, lane / 8U) / 2U + lane % 8U;
			for (int part = 0; part < part_count(parts); ++part) {
				weights[static_cast<unsigned>(part) * Tile::tile_rows * 32U + element] = split_weight.part[part].bits;
			}
		}
	}

	/** The warp's output accumulators: 32 features (two m16 tiles) by every n8 tile of query rows. */
	struct Outputs {
		prims::Accumulator tile[2][Tile::n_tiles];
	};

	/** O^T = O^T rescale + V^T P^T for the warp's 32 features; rows that saw nothing of the chunk keep their values. */
	template<PvParts parts>
	__attribute__((device, always_inline)) static void accumulate(Shared& shared, Cursor cursor, unsigned warp, Outputs& outputs) {
		unsigned const lane = prims::lane();
		unsigned const matrix = lane / 8U;
		unsigned const line = lane % 8U;
		unsigned const first_chunk = 4U * warp;
		prims::SharedAddress const values = prims::shared_address(shared.stage[cursor.stage]);
		prims::SharedAddress const weights = prims::shared_address(shared.weights);
		prims::AFragment value_tiles[2][2] = {};
#pragma unroll
		for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
			for (unsigned step = 0; step < 2; ++step) {
				prims::Matrices4 const v = prims::ldmatrix_x4<prims::fragment::Transpose::Transposed>(values.plus(static_cast<unsigned>(
				    KvTile::offset(static_cast<int>(16U * step + line + 8U * (matrix / 2U)), static_cast<int>(first_chunk + 2U * half + matrix % 2U)))));
				value_tiles[half][step] = {.reg = {v.reg[0], v.reg[1], v.reg[2], v.reg[3]}};
			}
		}
		unsigned const column = 2U * (lane % 4U);
#pragma unroll
		for (unsigned n_tile = 0; n_tile < Tile::n_tiles; ++n_tile) {
			unsigned const rows[2] = {8U * n_tile + column, 8U * n_tile + column + 1U};
			float const rescale[2] = {shared.rescale[rows[0]], shared.rescale[rows[1]]};
			bool const active[2] = {shared.active[rows[0]] != 0, shared.active[rows[1]] != 0};
			prims::Accumulator kept[2] = {outputs.tile[0][n_tile], outputs.tile[1][n_tile]};
#pragma unroll
			for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
				for (unsigned reg = 0; reg < 4; ++reg) {
					outputs.tile[half][n_tile].reg[reg] = outputs.tile[half][n_tile].reg[reg] * rescale[reg % 2];
				}
			}
#pragma unroll
			for (int part = 0; part < part_count(parts); ++part) {
				unsigned const tile = static_cast<unsigned>(part) * Tile::tile_rows * 64U;
				prims::Matrices4 const p = prims::ldmatrix_x4<prims::fragment::Transpose::Plain>(weights.plus(tile + weight_offset(8U * n_tile + line, matrix)));
#pragma unroll
				for (unsigned step = 0; step < 2; ++step) {
#pragma unroll
					for (unsigned half = 0; half < 2; ++half) {
						prims::mma(outputs.tile[half][n_tile], value_tiles[half][step], prims::BFragment{.reg = {p.reg[step + step], p.reg[step + step + 1U]}});
					}
				}
			}
#pragma unroll
			for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
				for (unsigned reg = 0; reg < 4; ++reg) {
					float const now = outputs.tile[half][n_tile].reg[reg];
					outputs.tile[half][n_tile].reg[reg] = active[reg % 2] ? now : kept[half].reg[reg];
				}
			}
		}
	}

	/** The feature and query row of accumulator register `reg` of tile (half, n_tile) in consumer warp `warp`. */
	struct Element {
		unsigned feature;
		unsigned row;
	};

	[[nodiscard]] __attribute__((device, always_inline)) static auto element(unsigned warp, unsigned half, unsigned n_tile, unsigned reg) -> Element {
		prims::fragment::Coordinate const at_element = prims::fragment::c(static_cast<int>(prims::lane()), static_cast<int>(reg));
		return {.feature = 32U * warp + 16U * half + static_cast<unsigned>(at_element.row), .row = 8U * n_tile + static_cast<unsigned>(at_element.column)};
	}

	/** Decode: the group's record rows, the outputs by every consumer warp and the running pairs by warp 0. */
	__attribute__((device, always_inline)) static void write_record(Shared& shared, Job const& job, Records records, unsigned group_index, unsigned warp,
	                                                 Outputs const& outputs) {
		auto const reused = prims::policy<prims::Eviction::Last>();
		unsigned const rows = static_cast<unsigned>(query_rows(job.work));
		Count const first = Count{job.record_base} + Count{group_index} * rows;
		auto* output = at<unsigned>(records.output);
#pragma unroll
		for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
			for (unsigned n_tile = 0; n_tile < Tile::n_tiles; ++n_tile) {
#pragma unroll
				for (unsigned reg = 0; reg < 4; ++reg) {
					Element const place = element(warp, half, n_tile, reg);
					if (place.row < rows) {
						prims::store_word(output + (first + place.row) * head_dim + place.feature, __builtin_bit_cast(unsigned, outputs.tile[half][n_tile].reg[reg]), reused);
					}
				}
			}
		}
		unsigned const lane = prims::lane();
		if (warp == 0 && lane < rows) {
			prims::Chunk const running = {.word = {__builtin_bit_cast(unsigned, shared.maximum[lane]), __builtin_bit_cast(unsigned, shared.total[lane]), 0U, 0U}};
			prims::store(at<prims::Chunk>(records.running) + first + lane, running, reused);
		}
	}

	/**
	 * Prefill, by the softmax warps at a group's last chunk: how each row's result takes the group (the first group is
	 * copied; a later one is merged; a group past the row's position leaves it), and the merged maximum and total.
	 */
	__attribute__((device, always_inline)) static void plan_fold(Shared& shared, Job const& job, Count group, unsigned warp) {
		if (prims::lane() != 0) {
			return;
		}
		for (unsigned row = warp; row < Tile::tile_rows; row += consumer_warps) {
			bool const sees = row < query_rows(job.work) && group_begin(group) * chunk_positions <= limit(job, row);
			if (!sees) {
				shared.fold[row] = Fold::Keep;
			} else if (group == 0) {
				shared.fold[row] = Fold::Copy;
				shared.final_maximum[row] = shared.maximum[row];
				shared.final_total[row] = shared.total[row];
			} else {
				Factors const weights = factors(shared.final_maximum[row], shared.maximum[row]);
				shared.fold[row] = Fold::Combine;
				shared.merge_left[row] = weights.left;
				shared.merge_right[row] = weights.right;
				shared.final_total[row] = combine(shared.final_total[row], shared.total[row], weights);
				shared.final_maximum[row] = weights.maximum;
			}
		}
	}

	/** Prefill, by every consumer: the group's outputs into the rows' results as `plan_fold` decided. */
	__attribute__((device, always_inline)) static void fold(Shared const& shared, unsigned warp, Outputs const& group, Outputs& result) {
#pragma unroll
		for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
			for (unsigned n_tile = 0; n_tile < Tile::n_tiles; ++n_tile) {
#pragma unroll
				for (unsigned reg = 0; reg < 4; ++reg) {
					unsigned const row = element(warp, half, n_tile, reg).row;
					float const now = result.tile[half][n_tile].reg[reg];
					float const added = group.tile[half][n_tile].reg[reg];
					Factors const weights = {.maximum = 0.0F, .left = shared.merge_left[row], .right = shared.merge_right[row]};
					Fold const how = shared.fold[row];
					result.tile[half][n_tile].reg[reg] = how == Fold::Copy ? added : how == Fold::Combine ? combine(now, added, weights) : now;
				}
			}
		}
	}

	/** Prefill: the gated bf16 outputs, staged in the query tile and stored a chunk at a time. */
	__attribute__((device, always_inline)) static void write_output(Shared& shared, Job const& job, Projections rows, Address output, unsigned warp,
	                                                 Outputs const& result) {
		auto const once = prims::policy<prims::Eviction::First>();
		auto const reused = prims::policy<prims::Eviction::Last>();
		unsigned const count = static_cast<unsigned>(query_rows(job.work));
		auto* staged = reinterpret_cast<unsigned short*>(shared.query);
		auto const* gates = at<unsigned const>(rows.gate);
#pragma unroll
		for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
			for (unsigned n_tile = 0; n_tile < Tile::n_tiles; ++n_tile) {
#pragma unroll
				for (unsigned reg = 0; reg < 4; ++reg) {
					Element const place = element(warp, half, n_tile, reg);
					if (place.row < count) {
						Count const token = place.row / group_heads;
						Count const head = Count{job.head} * group_heads + place.row % group_heads;
						Count const index = (Count{job.work.first_row} + token) * gate_row_floats + head * head_dim + place.feature;
						float const gate = __builtin_bit_cast(float, prims::load_word(gates + index, once));
						staged[place.row * head_dim + place.feature] = finish(result.tile[half][n_tile].reg[reg], shared.final_total[place.row], gate).bits;
					}
				}
			}
		}
		ConsumerStep::sync();
		unsigned const thread = warp * 32U + prims::lane();
		auto* target = at<prims::Chunk>(output);
		for (unsigned piece = thread; piece < count * row_chunks; piece += Consumers::threads) {
			unsigned const row = piece / static_cast<unsigned>(row_chunks);
			Count const token = row / group_heads;
			Count const head = Count{job.head} * group_heads + row % group_heads;
			prims::Chunk const value = prims::load_shared(prims::shared_address(&shared.query[piece]));
			prims::store(target + ((Count{job.work.first_row} + token) * output_row_values + head * head_dim) / features_per_chunk + piece % row_chunks, value, reused);
		}
	}

	/** Runs every job's chunks. */
	template<Sink sink, Fresh fresh, PvParts parts, class Jobs>
	__attribute__((device, always_inline)) static auto consume(Shared& shared, Jobs const& jobs, KvRing ring, Cursor cursor, Records records, Projections rows, Address output)
	    -> Cursor {
		unsigned const warp = Consumers::rank(prims::warp_in_block());
		unsigned const lane = prims::lane();
		for (unsigned index = 0; index < jobs.count(); ++index) {
			JobReady::sync();
			Job const job = jobs.at(index);
			ChunkRange const range = chunk_range(job.work, job.first_group, job.groups);
			PositionRange const own = fresh == Fresh::Appended ? fresh_positions(job.work, range) : PositionRange{.begin = 0, .end = 0};
			Outputs group_outputs{};
			[[maybe_unused]] Outputs result{};
			for (unsigned group_index = 0; group_index < job.groups; ++group_index) {
				Count const group = Count{job.first_group} + group_index;
				Count const group_end = group_begin(group + 1) < range.end ? group_begin(group + 1) : range.end;
				group_outputs = {};
				for (Count chunk = group_begin(group); chunk < group_end; ++chunk) {
					bool const fresh_here = own.end > chunk * chunk_positions && own.begin < (chunk + 1) * chunk_positions;
					ring.await_filled(cursor, prims::block_backoff);
					if (fresh_here) {
						splice(shared, cursor, Kind::Key, own, chunk);
						ConsumerStep::sync();
					}
					scores(shared, cursor, warp);
					prims::sync_warp();
					if (lane == 0) {
						ring.drained(cursor);
					}
					cursor = cursor.next();
					ConsumerStep::sync();
					softmax<parts>(shared, job, chunk, chunk == group_begin(group), warp);
					if constexpr (sink == Sink::Output) {
						if (chunk + 1 == group_end) {
							plan_fold(shared, job, group, warp);
						}
					}
					ring.await_filled(cursor, prims::block_backoff);
					if (fresh_here) {
						splice(shared, cursor, Kind::Value, own, chunk);
					}
					ConsumerStep::sync();
					accumulate<parts>(shared, cursor, warp, group_outputs);
					prims::sync_warp();
					if (lane == 0) {
						ring.drained(cursor);
					}
					cursor = cursor.next();
				}
				if constexpr (sink == Sink::Records) {
					write_record(shared, job, records, group_index, warp, group_outputs);
				} else {
					fold(shared, warp, group_outputs, result);
				}
			}
			if constexpr (sink == Sink::Output) {
				write_output(shared, job, rows, output, warp, result);
			}
			if (index + 1 < jobs.count()) {
				JobDone::arrive();
			}
		}
		return cursor;
	}
};

}

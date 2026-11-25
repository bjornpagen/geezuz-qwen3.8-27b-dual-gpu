// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's kernels: the prompt append and the prefill attention, the decode core for each row count a slot can
// bring and the decode merge, the rope table's fill at start, the measurement variants U21 compares, and the self-checks
// that hold the device to the host's arithmetic. The decode loop runs the same building blocks inside its persistent
// kernel; these entries run them alone, for prefill, for the bring-up checks and for the microbenchmarks.
#include "attn/arithmetic.h"
#include "attn/core.h"
#include "attn/geometry.h"
#include "attn/layout.h"
#include "attn/merge.h"
#include "attn/selfcheck.h"
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/order.h"
#include "prims/reduce.h"
#include "prims/selfcheck.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "target.h"

namespace attn = engine::device::attn;
namespace prims = engine::device::prims;

namespace engine::device::attn {

/** A cycle the decode loop could not have built is a bug upstream of attention: trap where device checks are on. */
__attribute__((device)) inline void require(bool holds) {
	if constexpr (checks == Checks::On) {
		if (!holds) {
			__builtin_trap();
		}
	}
}

/** The decode core's jobs on this block: its assignment, held in shared memory. */
template<class Tile>
struct AssignedJobs {
	CoreShared<Tile> const* shared;

	[[nodiscard]] __attribute__((device)) auto count() const -> unsigned {
		return shared->assignment.count;
	}

	[[nodiscard]] __attribute__((device)) auto at(unsigned index) const -> Job {
		Item const item = shared->assignment.item[index];
		Cycle const& cycle = shared->cycle;
		return {.work = cycle.work[item.work],
		        .head = item.head,
		        .first_group = item.first_group,
		        .groups = item.groups,
		        .record_base = static_cast<unsigned>(record_base(cycle, item.work, item.head))};
	}
};

/** The decode core over `sms` blocks: each takes its assignment of the cycle's groups and leaves their records. */
template<unsigned rows, PvParts parts, unsigned sms>
__attribute__((device, always_inline)) inline void decode_core(CoreParams params) {
	using Tile = RowTile<rows>;
	constexpr prims::Budget budget = core_budget<Tile>();
	auto& shared = prims::shared_layout<budget, CoreShared<Tile>>();
	KvRing const ring = kv_ring(shared);
	if (prims::thread_in_block() == 0) {
		shared.cycle = params.cycle;
		Cycle const& cycle = shared.cycle;
		require(valid(cycle, model::rows(model::RowBudget::FourTiles)));
		for (unsigned index = 0; index < cycle.live; ++index) {
			require(cycle.work[index].rows <= rows);
		}
		ring.init();
		Assignment& mine = shared.assignment;
		mine.count = 0;
		for_each_item(cycle, prims::block(), sms, [&mine](Item item) {
			mine.item[mine.count] = item;
			++mine.count;
		});
	}
	prims::sync_block();
	AssignedJobs<Tile> const jobs = {.shared = &shared};
	unsigned const warp = prims::warp_in_block();
	if (Producer::holds(warp)) {
		[[maybe_unused]] Cursor const streamed = Core<Tile>::produce(shared, jobs, ring, Cursor::start(), params.kv);
		prims::wait_all_copies();
	} else if (Consumers::holds(warp)) {
		[[maybe_unused]] Cursor const consumed = Core<Tile>::template consume<Sink::Records, Fresh::Appended, parts>(shared, jobs, ring, Cursor::start(), params.records, params.rows, 0);
	} else {
		Core<Tile>::template prepare<Fresh::Appended>(shared, jobs, params.kv, params.rows, params.preparation);
	}
}


}

using attn::CoreParams;
using attn::MergeParams;
using attn::PromptParams;
using attn::PvParts;

// ---------------------------------------------------------------------------------------------------------------------
// Prefill.

inline constexpr prims::Budget prompt_budget = attn::core_budget<attn::PromptTile>();
static_assert(prims::check(prompt_budget) == prims::Fit::Fits);
static_assert(prims::check(attn::append_budget) == prims::Fit::Fits);
static_assert(prims::tiles<attn::Producer, attn::Consumers, attn::Preparers>(prompt_budget), "the roles tile the block");

/**
 * The prompt chunk's kv: each row's keys (k_norm, then the rope at its position) and values, rounded to bf16 into the
 * slot's blocks, one warp per (row, kv head, kind). Runs before the prefill attention, which reads every position.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_append = attn::append_budget;

extern "C" __attribute__((global, launch_bounds(attn::append_budget.threads, attn::append_budget.blocks_per_sm))) void attn_append(PromptParams params) {
	using namespace engine::device::attn;
	auto const reused = prims::policy<prims::Eviction::Last>();
	auto const appended = prims::policy<prims::Eviction::Normal>();
	unsigned const lane = prims::lane();
	unsigned const warps = static_cast<unsigned>(append_budget.threads / 32);
	prims::Chunk const key_weight = prims::load_constant(at<prims::Chunk const>(params.preparation.key) + lane, reused);
	unsigned const tasks = params.count * static_cast<unsigned>(kv_heads * kinds);
	for (unsigned task = prims::block() * warps + prims::warp_in_block(); task < tasks; task += append_blocks * warps) {
		unsigned const row = task / static_cast<unsigned>(kv_heads * kinds);
		unsigned const head = (task / static_cast<unsigned>(kinds)) % static_cast<unsigned>(kv_heads);
		Kind const kind = task % 2 == 0 ? Kind::Key : Kind::Value;
		Count const position = Count{params.start} + row;
		float values[lane_features] = {};
		if (kind == Kind::Key) {
			load_floats(at<float const>(params.rows.key) + Count{row} * kv_row_floats + Count{head} * head_dim, lane, reused, values);
			normalize_and_rotate(values, key_weight, static_cast<unsigned>(position), params.preparation.rotations);
		} else {
			load_floats(at<float const>(params.rows.value) + Count{row} * kv_row_floats + Count{head} * head_dim, lane, reused, values);
		}
		Count const block = prims::load_word(at<unsigned const>(params.kv.tables) + Count{params.slot} * engine::model::kv_table_blocks + position / block_positions, reused);
		Address const target = params.kv.blocks + block * engine::model::kv_block_bytes + row_offset(params.kv.layer, head, kind, position) +
		                       stored_chunk(position, lane) * chunk_bytes;
		prims::store(at<prims::Chunk>(target), pack_chunk(values), appended);
	}
}

/**
 * Causal attention of a prompt chunk over the slot's kv (the cached positions and the chunk itself, appended first):
 * each job is up to five consecutive rows of one kv head against every group up to the last row, folded as decode
 * folds them, and leaves the gated bf16 outputs.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_prefill = prompt_budget;

extern "C" __attribute__((global, launch_bounds(prompt_budget.threads, prompt_budget.blocks_per_sm))) void attn_prefill(PromptParams params) {
	using namespace engine::device::attn;
	using Tile = PromptTile;
	auto& shared = prims::shared_layout<prompt_budget, CoreShared<Tile>>();
	KvRing const ring = kv_ring(shared);
	if (prims::thread_in_block() == 0) {
		require(Count{params.start} + params.count <= engine::model::max_positions);
		ring.init();
	}
	prims::sync_block();
	PromptJobs const jobs = {.slot = params.slot, .start = params.start, .rows = params.count, .first = prims::block(), .blocks = static_cast<unsigned>(prims::sm86.sms)};
	unsigned const warp = prims::warp_in_block();
	if (Producer::holds(warp)) {
		[[maybe_unused]] Cursor const streamed = Core<Tile>::produce(shared, jobs, ring, Cursor::start(), params.kv);
		prims::wait_all_copies();
	} else if (Consumers::holds(warp)) {
		Records const none = {.output = 0, .running = 0};
		[[maybe_unused]] Cursor const consumed = Core<Tile>::template consume<Sink::Output, Fresh::Cached, PvParts::Three>(shared, jobs, ring, Cursor::start(), none, params.rows, params.output);
	} else {
		Core<Tile>::template prepare<Fresh::Cached>(shared, jobs, params.kv, params.rows, params.preparation);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode: the core for each row count (the mtp steps' one row, then one committed row and one to four drafts), and
// the merge.

inline constexpr unsigned all_sms = static_cast<unsigned>(prims::sm86.sms);

inline constexpr prims::Budget core_rows1_budget = attn::core_budget<attn::RowTile<1>>();
inline constexpr prims::Budget core_rows2_budget = attn::core_budget<attn::RowTile<2>>();
inline constexpr prims::Budget core_rows3_budget = attn::core_budget<attn::RowTile<3>>();
inline constexpr prims::Budget core_rows4_budget = attn::core_budget<attn::RowTile<4>>();
inline constexpr prims::Budget core_rows5_budget = attn::core_budget<attn::RowTile<5>>();
static_assert(prims::check(core_rows1_budget) == prims::Fit::Fits && prims::check(core_rows2_budget) == prims::Fit::Fits);
static_assert(prims::check(core_rows3_budget) == prims::Fit::Fits && prims::check(core_rows4_budget) == prims::Fit::Fits);
static_assert(prims::check(core_rows5_budget) == prims::Fit::Fits);
static_assert(prims::check(attn::merge_budget) == prims::Fit::Fits);

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_decode_core_rows1 = core_rows1_budget;
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_decode_core_rows2 = core_rows2_budget;
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_decode_core_rows3 = core_rows3_budget;
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_decode_core_rows4 = core_rows4_budget;
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_decode_core_rows5 = core_rows5_budget;

/** The decode core for slots of at most one row each (the mtp draft steps). */
extern "C" __attribute__((global, launch_bounds(core_rows1_budget.threads, core_rows1_budget.blocks_per_sm))) void attn_decode_core_rows1(CoreParams params) {
	attn::decode_core<1, PvParts::Three, all_sms>(params);
}

extern "C" __attribute__((global, launch_bounds(core_rows2_budget.threads, core_rows2_budget.blocks_per_sm))) void attn_decode_core_rows2(CoreParams params) {
	attn::decode_core<2, PvParts::Three, all_sms>(params);
}

/** The shipping shape at eight slots: one committed row and two drafts. U21 measures it at three PV parts. */
extern "C" __attribute__((global, launch_bounds(core_rows3_budget.threads, core_rows3_budget.blocks_per_sm))) void attn_decode_core_rows3(CoreParams params) {
	attn::decode_core<3, PvParts::Three, all_sms>(params);
}

extern "C" __attribute__((global, launch_bounds(core_rows4_budget.threads, core_rows4_budget.blocks_per_sm))) void attn_decode_core_rows4(CoreParams params) {
	attn::decode_core<4, PvParts::Three, all_sms>(params);
}

/** The deepest shape: one committed row and four drafts. U21 measures it at three PV parts. */
extern "C" __attribute__((global, launch_bounds(core_rows5_budget.threads, core_rows5_budget.blocks_per_sm))) void attn_decode_core_rows5(CoreParams params) {
	attn::decode_core<5, PvParts::Three, all_sms>(params);
}

/** The decode merge: one warp per (row, query head) of the cycle, over `merge_blocks` blocks of eight warps. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_decode_merge = attn::merge_budget;

extern "C" __attribute__((global, launch_bounds(attn::merge_budget.threads, attn::merge_budget.blocks_per_sm))) void attn_decode_merge(MergeParams params) {
	unsigned const warps = static_cast<unsigned>(attn::merge_budget.threads / 32);
	auto& shared = prims::shared_layout<attn::merge_budget, attn::MergeShared>();
	if (prims::thread_in_block() == 0) {
		shared.cycle = params.cycle;
	}
	prims::sync_block();
	attn::Cycle const& cycle = shared.cycle;
	engine::model::Count const units = attn::merge_units(cycle);
	for (engine::model::Count unit = engine::model::Count{prims::block()} * warps + prims::warp_in_block(); unit < units; unit += engine::model::Count{attn::merge_blocks} * warps) {
		attn::merge(params.records, params.gate, params.output, cycle, attn::merge_unit(cycle, unit));
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// U21: what carrying P exactly costs decode. These entries round P to one bf16 part, which the lossless contract
// forbids in the engine; they exist only so the microbenchmark can time the same core with one PV mma instead of three.

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_u21_rows3_one_part = core_rows3_budget;
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_u21_rows5_one_part = core_rows5_budget;

extern "C" __attribute__((global, launch_bounds(core_rows3_budget.threads, core_rows3_budget.blocks_per_sm))) void attn_u21_rows3_one_part(CoreParams params) {
	attn::decode_core<3, PvParts::One, all_sms>(params);
}

extern "C" __attribute__((global, launch_bounds(core_rows5_budget.threads, core_rows5_budget.blocks_per_sm))) void attn_u21_rows5_one_part(CoreParams params) {
	attn::decode_core<5, PvParts::One, all_sms>(params);
}

// ---------------------------------------------------------------------------------------------------------------------
// Self-checks.

/**
 * The decode core over seven blocks instead of 82: the bring-up check launches it and `attn_decode_core_rows5` on the
 * same cycle, merges both, and requires the two outputs to agree bit for bit, since the assignment may move a group
 * but never change how it is computed.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_check_core_rows5_sms7 = core_rows5_budget;

extern "C" __attribute__((global, launch_bounds(core_rows5_budget.threads, core_rows5_budget.blocks_per_sm))) void attn_check_core_rows5_sms7(CoreParams params) {
	attn::decode_core<5, PvParts::Three, attn::selfcheck::few_sms>(params);
}

static_assert(prims::check(attn::selfcheck::rope_budget) == prims::Fit::Fits);

/**
 * The rope table, filled once at start: `model::rope::rotation` of every (position, pair), evaluated here in ieee
 * double, so every rotation the kernels read is the model's bits. `table` holds `attn::rope_table_bytes`. The bring-up
 * check reads it back and compares it with the host's evaluation of the same function.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_rope_table = attn::selfcheck::rope_budget;

extern "C" __attribute__((global, launch_bounds(attn::selfcheck::rope_budget.threads, attn::selfcheck::rope_budget.blocks_per_sm))) void attn_rope_table(attn::Address table) {
	unsigned const position = prims::block() * static_cast<unsigned>(attn::selfcheck::rope_budget.threads) + prims::thread_in_block();
	auto const written = prims::policy<prims::Eviction::Normal>();
	auto* chunks = attn::at<prims::Chunk>(table + attn::rope_offset(position, 0));
	for (unsigned pair = 0; pair < static_cast<unsigned>(attn::rotary_pairs); pair += 2) {
		engine::model::rope::Rotation const first = engine::model::rope::rotation(position, pair);
		engine::model::rope::Rotation const second = engine::model::rope::rotation(position, pair + 1);
		prims::Chunk const both = {.word = {__builtin_bit_cast(unsigned, first.cosine), __builtin_bit_cast(unsigned, first.sine),
		                                    __builtin_bit_cast(unsigned, second.cosine), __builtin_bit_cast(unsigned, second.sine)}};
		prims::store(chunks + pair / 2, both, written);
	}
}

static_assert(prims::check(attn::selfcheck::split_budget) == prims::Fit::Fits);

/**
 * The weight split on the device over every fp32 value in [0, 1], scaled: the three parts must sum to the scaled value
 * exactly and the last part must hold exactly what the first two leave. `report` starts zeroed; each block reports one
 * comparison per thread and keeps the first mismatches.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_attn_check_split = attn::selfcheck::split_budget;

extern "C" __attribute__((global, launch_bounds(attn::selfcheck::split_budget.threads, attn::selfcheck::split_budget.blocks_per_sm))) void attn_check_split(prims::selfcheck::Report* report) {
	unsigned const thread = prims::block() * static_cast<unsigned>(attn::selfcheck::split_budget.threads) + prims::thread_in_block();
	attn::selfcheck::SplitRun const run = attn::selfcheck::split_run(thread);
	unsigned failures = 0;
	unsigned first_failure = 0;
	for (unsigned long long bits = run.begin; bits < run.end; ++bits) {
		if (!attn::selfcheck::split_exact(static_cast<unsigned>(bits))) {
			first_failure = failures == 0 ? static_cast<unsigned>(bits) : first_failure;
			++failures;
		}
	}
	prims::count(&report->checked, 1U);
	if (failures != 0) {
		unsigned const slot = prims::fetch_add(&report->mismatched, 1U);
		if (slot < prims::selfcheck::kept_mismatches) {
			report->first[slot] = {.input = first_failure, .detail = failures, .got = 0, .expected = 0};
		}
	}
}

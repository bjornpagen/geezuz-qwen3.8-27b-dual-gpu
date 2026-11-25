// SPDX-License-Identifier: AGPL-3.0-only
//
// How the deltanet kernels land on an SM: their warp roles, their shared memory, their budgets and their grids. Pure
// constexpr with no device dependency, so the kernels `static_assert` these budgets and the host's kernel declarations
// launch with the very same ones. It uses no standard headers so that it compiles unchanged as device code and inside
// a module purview.
#pragma once

#include "deltanet/layout.h"
#include "deltanet/math.h"
#include "prims/budget.h"
#include "prims/stage_ring.h"

namespace engine::device::deltanet {

/** A budget at `blocks_per_sm` with every register that occupancy leaves. */
[[nodiscard]] consteval auto budget(int threads, int shared_bytes, int blocks_per_sm) -> prims::Budget {
	auto shape = prims::Budget{.threads = threads, .registers_per_thread = 0, .shared_bytes = shared_bytes, .blocks_per_sm = blocks_per_sm};
	shape.registers_per_thread = prims::register_cap(shape);
	return shape;
}

inline constexpr int warp_threads = 32;

// ---------------------------------------------------------------------------------------------------------------------
// Decode: one item is one live slot's key head group. A producer warp streams the group's state (96 quads, twelve
// 16 KiB stages) through a ring; preparer warps convolve, normalize and gate every event and keep the window and the
// replay ring; consumer warps, one per quad of a stage, run the quads' columns through every event, commit the state
// after the first row, and normalize the rows' outputs.

inline constexpr int state_stages = 4;
inline constexpr int stage_bytes = 16 * 1024;
inline constexpr int stage_chunks = stage_bytes / 16;
inline constexpr int quads_per_stage = stage_bytes / static_cast<int>(State::unit_bytes);
inline constexpr int item_quads = group * State::units;
inline constexpr int item_stages = item_quads / quads_per_stage;
inline constexpr int consumer_warps = quads_per_stage;
inline constexpr int producer_warps = 1;
/** Preparer threads: the convolution's eight channels per thread, then the gates, one event per thread. */
inline constexpr int preparer_threads = 96;
inline constexpr int preparer_warps = preparer_threads / warp_threads;
inline constexpr int gate_threads = max_events;

static_assert(item_quads % quads_per_stage == 0 && State::units % quads_per_stage == 0, "a stage never straddles two heads");
static_assert(group_chunks + gate_threads <= preparer_threads && preparer_threads % warp_threads == 0);
static_assert(2 * max_events <= 2 * (preparer_threads / Column::lanes), "two passes normalize every event's query and key");
static_assert(max_rows * group <= consumer_warps * warp_threads / Column::lanes, "one pass normalizes every row's heads");

/** One stage of the state ring: eight quads, one contiguous run of the item's state. */
struct Stage {
	alignas(128) unsigned char bytes[stage_bytes];
};

/** Every event's prepared inputs: the normalized query and key, the convolved values, each head's gates. */
struct Events {
	/** Each event's query and key: convolved in natural order, then normalized in place into `vector_element<Column>` order. */
	alignas(16) float queries[max_events][dim];
	alignas(16) float keys[max_events][dim];
	/** Each event's convolved values of the group's heads. */
	alignas(16) float values[max_events][group * dim];
	Gates gates[max_events][group];
};

/** The rows' core outputs, fp32, head by head. */
struct RowOutputs {
	alignas(16) float values[max_rows][group * dim];
};

struct DecodeShared {
	Stage stages[state_stages];
	/** The ring's full and empty mbarriers, one each per stage. */
	prims::StageBarriers<state_stages> ring;
	/** Completes when every preparer warp has filled `events`. */
	alignas(8) unsigned long long prepared;
	Events events;
	RowOutputs outputs;
};

inline constexpr prims::Budget decode_budget =
    budget((producer_warps + consumer_warps + preparer_warps) * warp_threads, static_cast<int>(sizeof(DecodeShared)), 1);

static_assert(prims::check(decode_budget) == prims::Fit::Fits);

// ---------------------------------------------------------------------------------------------------------------------
// Prefill: one block per slot and value head. Preparer warps convolve, gate and normalize a block of tokens into one of
// two stages while the chain warps run the previous block through the head's 32 quads, four per warp, held in registers
// for the whole chunk; the preparers then normalize the chain's outputs of that stage.

inline constexpr int chain_warps = 8;
inline constexpr int quads_per_chain_warp = State::units / chain_warps;
inline constexpr int prefill_preparer_threads = 128;
inline constexpr int prefill_preparer_warps = prefill_preparer_threads / warp_threads;
/** The convolution's chunks of one value head: its group's query and key, and its own values. */
inline constexpr int head_chunks = 3 * dim / bf16_per_chunk;
inline constexpr int prefill_stages = 2;

static_assert(head_chunks + prefill_block_tokens <= prefill_preparer_threads);
static_assert(2 * prefill_block_tokens <= 2 * (prefill_preparer_threads / Column::lanes), "two passes normalize a block's vectors");
static_assert(prefill_block_tokens <= prefill_preparer_threads / Column::lanes, "one pass normalizes a block's outputs");

/** The group channel chunk a prefill block's convolution chunk `u` reads: query, key, then its own head's values. */
[[nodiscard]] constexpr auto head_chunk(int value_head, int u) -> int {
	return u < 2 * dim / bf16_per_chunk ? u : u + (value_head % group) * (dim / bf16_per_chunk);
}

/** One block of prompt tokens for one value head: prepared inputs, then the chain's outputs. */
struct TokenBlock {
	/** Each token's query and key: convolved in natural order, then normalized in place into `vector_element<Column>` order. */
	alignas(16) float queries[prefill_block_tokens][dim];
	alignas(16) float keys[prefill_block_tokens][dim];
	/** Each token's convolved values of the block's head. */
	alignas(16) float values[prefill_block_tokens][dim];
	Gates gates[prefill_block_tokens];
	alignas(16) float outputs[prefill_block_tokens][dim];
};

struct PrefillShared {
	TokenBlock blocks[prefill_stages];
	/** Per stage: the preparers filled it; the chain is done with it. */
	alignas(8) unsigned long long prepared[prefill_stages];
	alignas(8) unsigned long long chained[prefill_stages];
};

inline constexpr prims::Budget prefill_budget =
    budget((prefill_preparer_warps + chain_warps) * warp_threads, static_cast<int>(sizeof(PrefillShared)), 1);
inline constexpr prims::Budget window_budget = budget(128, 0, 4);

static_assert(prims::check(prefill_budget) == prims::Fit::Fits && prims::check(window_budget) == prims::Fit::Fits);
static_assert(group_chunks <= window_budget.threads);

// ---------------------------------------------------------------------------------------------------------------------
// The chain probe and the tree self-check.

inline constexpr prims::Budget probe_budget = budget(256, 0, 2);
inline constexpr prims::Budget check_budget = budget(256, 0, 2);

static_assert(prims::check(probe_budget) == prims::Fit::Fits && prims::check(check_budget) == prims::Fit::Fits);

/** Blocks of the probe under tree `T`: each of its warps runs one unit of columns of one head. */
template<class T>
inline constexpr int probe_blocks = probe_heads * StateLayout<T>::units / (probe_budget.threads / warp_threads);

static_assert(StateLayout<Column>::units % (probe_budget.threads / warp_threads) == 0);
static_assert(StateLayout<Tree<32>>::units % (probe_budget.threads / warp_threads) == 0);

}

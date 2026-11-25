// SPDX-License-Identifier: AGPL-3.0-only
//
// Everything about the primitives' self-check kernels that the host drivers also need, held once: each kernel's one
// parameter struct, budget, grid and buffer layouts, the report it writes, the fixed partition of the fp32 inputs and
// the digest both sides fold the transcendentals into. Pure constexpr, shared with the host drivers (the
// microbenchmarks, through `engine.prims`), so a launch and a report are built and read by the definitions the
// kernels compile.
#pragma once

#include "prims/budget.h"
#include "prims/chunk.h"
#include "prims/fragment.h"
#include "prims/stage_ring.h"

namespace engine::device::prims::selfcheck {

/** How many mismatches a report keeps verbatim; the count goes on past it. */
inline constexpr unsigned kept_mismatches = 64;

/** One mismatch: what was fed in, what came out, and what the oracle says should have. */
struct Mismatch {
	unsigned input;
	unsigned detail;
	unsigned got;
	unsigned expected;
};

/** A self-check's verdict: zero mismatches out of `checked` comparisons means the primitive held on the device. */
struct Report {
	unsigned checked;
	unsigned mismatched;
	Mismatch first[kept_mismatches];
};

/** The transcendentals in digest order. */
enum class Function : unsigned char { Exp, Log, Rsqrt, Sigmoid, Silu, Softplus };
inline constexpr unsigned functions = 6;

// Every kernel takes exactly one parameter struct, device addresses as 64-bit integers; the kernel turns each back into
// a typed pointer. Each report, digest array and scratch block starts zeroed.
using Address = unsigned long long;
static_assert(sizeof(Address) == 8, "a device address is 64 bits on both sides");

/** `prims_check_decode`, `prims_check_fragments`, `prims_check_reduce`: where the one `Report` goes. */
struct ReportParams {
	Address report;
};

// prims_check_decode (U7): every pair of codes through `decode_word`, then all 256 through the mma.
inline constexpr Budget decode_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};
inline constexpr unsigned decode_blocks = 1;

// prims_probe_mma (U6): one probe per warp; results[probe * 32 + lane] is that lane's D.
inline constexpr Budget mma_budget = {.threads = 128, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 4};
inline constexpr unsigned mma_blocks = 64;
inline constexpr unsigned mma_probes = mma_blocks * static_cast<unsigned>(mma_budget.threads / 32);

/** One warp's mma probe: every lane's operands as the fragment maps lay them out. */
struct MmaProbe {
	AFragment a[32];
	BFragment b[32];
	Accumulator c[32];
};

/** `probes`: `mma_probes` MmaProbe; `results`: `mma_probes * 32` Accumulator. */
struct MmaParams {
	Address probes;
	Address results;
};

// prims_check_fragments: one warp, one swizzled tile.
using ActivationTile = fragment::XorSwizzle<256, 16>;
static_assert(fragment::conflict_free<ActivationTile>());

struct FragmentLayout {
	alignas(128) unsigned char tile[ActivationTile::bytes];
};

inline constexpr Budget fragment_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = static_cast<int>(sizeof(FragmentLayout)), .blocks_per_sm = 1};
inline constexpr unsigned fragment_blocks = 1;

// prims_check_stage_ring (U8): one producer warp and four consumer warps per block, one block per SM, each block
// streaming its own run of tiles through a 4-stage ring.
inline constexpr int stage_ring_stages = 4;
inline constexpr unsigned stage_ring_tile_chunks = 256;
inline constexpr unsigned stage_ring_tiles_per_block = 64;
inline constexpr int stage_ring_producer_warps = 1;
inline constexpr int stage_ring_consumer_warps = 4;
inline constexpr unsigned stage_ring_blocks = static_cast<unsigned>(sm86.sms);
inline constexpr unsigned long long stage_ring_source_chunks =
    static_cast<unsigned long long>(stage_ring_blocks) * stage_ring_tiles_per_block * stage_ring_tile_chunks;

struct StageRingLayout {
	StageBarriers<stage_ring_stages> barriers;
	Chunk stage[stage_ring_stages][stage_ring_tile_chunks];
};

inline constexpr Budget stage_ring_budget = {
    .threads = 32 * (stage_ring_producer_warps + stage_ring_consumer_warps),
    .registers_per_thread = 64,
    .shared_bytes = static_cast<int>(sizeof(StageRingLayout)),
    .blocks_per_sm = 1,
};

/** The stage-ring check's source pattern: word i of the tile stream. */
[[nodiscard]] constexpr auto ring_word(unsigned index) -> unsigned {
	return index * 0x9E3779B9U + 0x7F4A7C15U;
}

/**
 * `source`: `stage_ring_source_chunks` Chunk, word i holding `ring_word(i)`; `report`: one Report; `elapsed`: one
 * 64-bit clock count per block, the consumers' time.
 */
struct StageRingParams {
	Address source;
	Address report;
	Address elapsed;
};

// prims_check_reduce: one block, `reduce_trials` trials per warp.
inline constexpr Budget reduce_budget = {.threads = 256, .registers_per_thread = 128, .shared_bytes = 0, .blocks_per_sm = 1};
inline constexpr unsigned reduce_blocks = 1;
inline constexpr unsigned reduce_trials = 256;

// prims_digest_transcendentals: the fp32 inputs are split into blocks x threads contiguous runs; thread t of block b
// evaluates inputs [(b * threads + t) * run, ... + run).
inline constexpr Budget digest_budget = {.threads = 256, .registers_per_thread = 128, .shared_bytes = 0, .blocks_per_sm = 2};
inline constexpr unsigned digest_blocks = 1024;
inline constexpr unsigned long long digest_run = (1ULL << 32U) / (static_cast<unsigned long long>(digest_blocks) * digest_budget.threads);
static_assert(digest_run * digest_blocks * digest_budget.threads == 1ULL << 32U, "the runs tile every fp32 input");

/** `digests`: `functions * digest_blocks` 64-bit sums, function-major. */
struct DigestParams {
	Address digests;
};

// prims_check_order: one block per SM, all co-resident (the step-1 grid barrier needs it); two producer warps hand
// shared data to two consumer warps.
inline constexpr unsigned order_blocks = static_cast<unsigned>(sm86.sms);
inline constexpr unsigned order_steps = 8;
inline constexpr int order_producer_warps = 2;
inline constexpr int order_consumer_warps = 2;
static_assert(order_producer_warps == order_consumer_warps, "each consumer thread reads one producer thread's chunk");

struct OrderLayout {
	alignas(8) unsigned long long barrier;
	Chunk copied[32];
	Chunk handed[32 * order_producer_warps];
	Chunk steps[order_steps][32];
};

inline constexpr Budget order_budget = {
    .threads = 32 * (order_producer_warps + order_consumer_warps),
    .registers_per_thread = 128,
    .shared_bytes = static_cast<int>(sizeof(OrderLayout)),
    .blocks_per_sm = 1,
};

/** The global memory `prims_check_order` works in. */
struct OrderScratch {
	unsigned arrivals;
	unsigned counter;
	unsigned long long generation;
	unsigned long long gpu_flags[order_blocks];
	unsigned long long system_flags[order_blocks];
	unsigned long long relaxed_flags[order_blocks];
	Chunk slots[order_blocks];
	Chunk inbox[order_blocks];
	Chunk fenced[order_blocks];
	Line scratch[order_blocks];
	alignas(16) unsigned words[order_blocks][32];
};

/** `scratch`: one OrderScratch; `report`: one Report. */
struct OrderParams {
	Address scratch;
	Address report;
};

/**
 * One input's contribution to a block's digest: a mix of the input and output bits, summed mod 2^64 so the order
 * the device adds them in cannot matter.
 */
[[nodiscard]] constexpr auto digest_term(unsigned input, unsigned output) -> unsigned long long {
	unsigned long long mixed = (static_cast<unsigned long long>(input) << 32U) | output;
	mixed ^= mixed >> 33U;
	mixed *= 0xff51afd7ed558ccdULL;
	mixed ^= mixed >> 33U;
	mixed *= 0xc4ceb9fe1a85ec53ULL;
	mixed ^= mixed >> 33U;
	return mixed;
}

static_assert(check(decode_budget) == Fit::Fits && check(mma_budget) == Fit::Fits && check(fragment_budget) == Fit::Fits);
static_assert(check(stage_ring_budget) == Fit::Fits && check(reduce_budget) == Fit::Fits && check(digest_budget) == Fit::Fits);
static_assert(check(order_budget) == Fit::Fits);

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The microbenchmarks' shapes, parameters and result layouts, held once for both ends: the kernels compile them and
// `engine.ubench_device` re-exports them, so the host launches exactly the geometry a kernel was built for and reads
// its results by the layout that wrote them. Pure constexpr with no device dependency. Every sweep point is its own
// shape and its own kernel: a benchmark's dimensions are compile-time facts like every other kernel's, never launch
// arguments. Parameters carry device addresses as 64-bit integers; the kernels turn them back into typed pointers.
#pragma once

#include "prims/budget.h"
#include "prims/selfcheck.h"

namespace engine::device::ubench {

using Address = unsigned long long;
using prims::Budget;

/** Sixteen bytes as the host types them: the layout of the device's `prims::Chunk`. */
struct alignas(16) Word4 {
	unsigned word[4];
};

inline constexpr int sms = prims::sm86.sms;
inline constexpr int warp_threads = prims::sm86.warp_threads;

/** The budget of a kernel holding one block per SM and at most 128 registers per thread (512 threads leave 128). */
[[nodiscard]] constexpr auto one_block_per_sm(int warps, int shared_bytes) -> Budget {
	return {.threads = warps * warp_threads, .registers_per_thread = 128, .shared_bytes = shared_bytes, .blocks_per_sm = 1};
}

/** A per-block reading of the SM's cycle counter and the card's nanosecond timer over the same interval. */
struct Clocks {
	unsigned long long cycles;
	unsigned long long nanoseconds;
};

/** A per-block throughput reading: the cycles a fixed amount of work took, and a value folded from its results. */
struct Rate {
	unsigned long long cycles;
	unsigned long long sink;
};

/** How a benchmark's waits back off: the engine's production backoff, or a spin that only measures the hardware. */
enum class Wait : unsigned char { Backoff, Spin };

// The weight stream: every SM reads its own contiguous run of whole 16 KiB tiles, as a decode producer does.
namespace stream {

inline constexpr int tile_bytes = 16 * 1024;
inline constexpr int tile_chunks = tile_bytes / 16;
inline constexpr int tiles_per_block = 1536;
inline constexpr unsigned long long block_chunks = static_cast<unsigned long long>(tiles_per_block) * tile_chunks;
inline constexpr unsigned long long chunks = block_chunks * sms;
inline constexpr unsigned long long bytes = chunks * 16;

/** How the stream's tiles lie in memory: whole contiguous 16 KiB runs, or rows of a row-major matrix. */
enum class TileOrder : unsigned char { Contiguous, RowMajor };

// The row-major comparison: 128 tiles side by side form a slab whose 128-byte tile rows sit 16 KiB apart, as a
// 128-row block of a matrix 16 Ki columns wide stores them.
inline constexpr int slab_tiles = 128;
inline constexpr int row_pitch = slab_tiles * 128;
inline constexpr int slab_chunks = slab_tiles * tile_chunks;
static_assert(tiles_per_block % slab_tiles == 0, "a block's run is whole slabs");

/**
 * A pure read stream: `producer_warps` warps issue `cp.async.cg` 16-byte copies with the 256-byte l2 prefetch hint and
 * evict-first, into a 4-stage ring holding `depth_kib` KiB per SM; the consumer warps only wait for each stage and free
 * it. The consumers round the block up to a whole number of scheduler quarters.
 */
template<int producer_warps_, int depth_kib_, TileOrder order_>
struct Read {
	static constexpr int producer_warps = producer_warps_;
	static constexpr int depth_kib = depth_kib_;
	static constexpr TileOrder order = order_;
	static constexpr int consumer_warps = prims::round_up(producer_warps + 1, 4) - producer_warps;
	static constexpr int stages = 4;
	static constexpr int stage_bytes = depth_kib * 1024 / stages;
	static constexpr int stage_chunks = stage_bytes / 16;
	static constexpr int producer_threads = producer_warps * warp_threads;
	static constexpr int stages_per_block = static_cast<int>(block_chunks / stage_chunks);
	static constexpr Budget budget = one_block_per_sm(producer_warps + consumer_warps, depth_kib * 1024 + 128);

	static_assert(stage_chunks % producer_threads == 0, "every producer thread issues whole chunks per stage");
	static_assert(block_chunks % stage_chunks == 0, "a block's run is whole stages");
	static_assert(prims::fits(budget));
};

/** The decode's own shape: one producer warp keeping four whole tiles in flight. */
using Canonical = Read<1, 64, TileOrder::Contiguous>;

struct ReadParams {
	Address source;
};

/**
 * One producer warp and a consumer warpgroup over a ring of `ring_kib` KiB in whole tiles; every `stall_period` tiles
 * the consumers hold the stage for `stall_us` microseconds before freeing it, as a dependency hop would.
 */
template<int ring_kib_, int stall_us_>
struct Stall {
	static constexpr int ring_kib = ring_kib_;
	static constexpr int stall_us = stall_us_;
	static constexpr int stages = ring_kib * 1024 / tile_bytes;
	static constexpr int consumer_warps = 4;
	static constexpr int stall_period = 16;
	static constexpr int stalls = tiles_per_block / stall_period;
	static constexpr Budget budget = one_block_per_sm(1 + consumer_warps, ring_kib * 1024 + 128);

	static_assert(ring_kib * 1024 % tile_bytes == 0 && stages >= 2);
	static_assert(tiles_per_block % stall_period == 0);
	static_assert(prims::fits(budget));
};

/**
 * A decode-like load for the clock under power: one producer warp streams tiles through a 4-tile ring, and four
 * consumer warps decode every chunk and run the eight `mma`s 32 rows make of it. Each block reports its clocks.
 */
struct Sustain {
	static constexpr int stages = 4;
	static constexpr int consumer_warps = 4;
	static constexpr int launches = 4000;
	static constexpr Budget budget = one_block_per_sm(1 + consumer_warps, stages * tile_bytes + 128);
	static_assert(prims::fits(budget));
};

struct SustainParams {
	Address source;
	Address clocks;
};

/** How the activation reuse benchmark assigns l2 priorities: the engine's split, or everything at normal. */
enum class Priorities : unsigned char { Split, Uniform };

/**
 * Activations re-read while the weight stream runs: one producer warp streams tiles; per tile each of four consumer
 * warps reads 512 bytes of a 2 MiB activation buffer every block shares, and counts the cycles those reads take.
 */
template<Priorities priorities_>
struct Reuse {
	static constexpr Priorities priorities = priorities_;
	static constexpr int stages = 4;
	static constexpr int consumer_warps = 4;
	static constexpr int activation_chunks = 2 * 1024 * 1024 / 16;
	static constexpr Budget budget = one_block_per_sm(1 + consumer_warps, stages * tile_bytes + 128);
	static_assert(prims::fits(budget));
};

struct ReuseParams {
	Address source;
	Address activations;
	Address rates;
};

}

// What run-ahead l2 can carry: a region fetched at evict-last, read back after the weight stream has passed.
namespace l2 {

inline constexpr int segments = 48;
inline constexpr int segment_bytes = 128 * 1024;
inline constexpr int segment_chunks = segment_bytes / 16;
inline constexpr int region_chunks = segments * segment_chunks;
/** The stream between fetch and read-back: seven passes over the canonical stream, 13.9 GB. */
inline constexpr int stream_passes = 7;
inline constexpr Budget prefetch_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};
inline constexpr Budget probe_budget = prefetch_budget;
static_assert(segment_bytes % (128 * prefetch_budget.threads) == 0);

struct PrefetchParams {
	Address region;
};

struct ProbeParams {
	Address region;
	Address cycles;
};

}

// Synchronisation inside a block and across SMs.
namespace sync {

inline constexpr unsigned long long calibration_cycles = 200'000'000;
inline constexpr Budget clock_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

struct ClockParams {
	Address clocks;
};

inline constexpr int pingpong_iterations = 10'000;
inline constexpr Budget pingpong_budget = {.threads = 64, .registers_per_thread = 64, .shared_bytes = 128, .blocks_per_sm = 1};

struct CyclesParams {
	Address cycles;
};

inline constexpr int copy_iterations = 10'000;
inline constexpr Budget copy_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = 128, .blocks_per_sm = 1};

struct CopyParams {
	Address source;
	Address cycles;
};

/** The `nanosleep` arguments measured, in nanoseconds. */
inline constexpr unsigned sleeps[] = {0, 32, 64, 128, 256, 512, 1024, 4096};
inline constexpr int sleep_kinds = sizeof(sleeps) / sizeof(sleeps[0]);
inline constexpr int sleep_samples = 1000;
inline constexpr Budget sleep_budget = clock_budget;

/** One `nanosleep` argument's spread over its samples, in cycles. */
struct SleepReading {
	unsigned long long total;
	unsigned long long least;
	unsigned long long most;
};

struct SleepParams {
	Address readings;
};

/**
 * The cross-SM hop: two blocks, each with more shared memory than half an SM holds, so no SM can hold both; they pass
 * a value back and forth through weak stores and release counters.
 */
inline constexpr int hop_iterations = 10'000;
inline constexpr Budget hop_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = 60 * 1024, .blocks_per_sm = 1};
static_assert(2 * (hop_budget.shared_bytes + prims::sm86.shared_reserved_per_block) > prims::sm86.shared_per_sm);

/** The hop's global memory, zeroed by the host: a 128-byte line per direction's counter and value. */
struct HopScratch {
	alignas(128) unsigned forward_count;
	alignas(128) unsigned backward_count;
	alignas(128) unsigned forward_value[4];
	alignas(128) unsigned backward_value[4];
};

/** The first block's round trips in its SM's cycles, where each block ran, and the values each saw out of order. */
struct HopReading {
	unsigned long long cycles;
	unsigned first_sm;
	unsigned second_sm;
	unsigned first_mismatches;
	unsigned second_mismatches;
};

struct HopParams {
	Address scratch;
	Address reading;
};

/** The warp-specialised handoff: one producer warp and two consumer warpgroups trading a 4-stage ring's barriers. */
inline constexpr int handoff_iterations = 100'000;
inline constexpr int handoff_consumer_warps = 8;
inline constexpr Budget handoff_budget = {.threads = 9 * 32, .registers_per_thread = 64, .shared_bytes = 128, .blocks_per_sm = 1};

}

// Arithmetic throughput.
namespace compute {

inline constexpr int warps = 8;
inline constexpr int iterations = 4096;
inline constexpr Budget rate_budget = {.threads = warps * 32, .registers_per_thread = 128, .shared_bytes = 0, .blocks_per_sm = 1};

/**
 * What a tensor-rate kernel runs per iteration: four independent `mma`s; eight `mma`s and the decode of one 16-byte
 * chunk (the 32-row ratio); the decode alone; or the byte-permute widening alone.
 */
enum class Mix : unsigned char { Mma, MmaDecode, Decode, Widen };

struct RateParams {
	Address rates;
};

/** The folded-scale decode: one warp per activation exponent, as the exact self-check's `mma` pass. */
inline constexpr int activation_exponents[] = {-100, -60, -20, -9, 0, 9, 40, 100};
inline constexpr int exponent_count = sizeof(activation_exponents) / sizeof(activation_exponents[0]);
inline constexpr Budget folded_budget = {.threads = exponent_count * 32, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

struct FoldedReading {
	unsigned checked[exponent_count];
	unsigned mismatched[exponent_count];
};

struct FoldedParams {
	Address reading;
};

/** How activation tiles lie in shared memory: xor-swizzled, or plain row-major with its bank conflicts. */
enum class Swizzle : unsigned char { Xor, None };

inline constexpr int fragment_rows = 16;
inline constexpr int fragment_row_bytes = 256;
inline constexpr Budget ldmatrix_budget = {.threads = warps * 32, .registers_per_thread = 128, .shared_bytes = 4096, .blocks_per_sm = 1};

/** Whether the `ldmatrix` + `mma` loop loads step s + 1's operands before step s computes. */
enum class Pipelining : unsigned char { Double, Sequential };

inline constexpr int pipeline_steps = 8;

/** Which warp reduction a rate kernel runs: the fixed-order float butterfly or the integer `redux.sync`. */
enum class Reduction : unsigned char { FloatButterfly, IntegerRedux };

}

// Across the two cards and to the host.
namespace link {

inline constexpr int flag_iterations = 100'000;
inline constexpr Budget flag_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

/** One end of the flag ping-pong: its own flag (written by the peer) and the peer's (written by this end). */
struct FlagParams {
	Address own_flag;
	Address peer_flag;
	Address nanoseconds;
};

/** One exchange item: `rows` rows of 20 KiB of fp32 partials. */
inline constexpr int row_chunks = 20 * 1024 / 16;
inline constexpr int items = 64;
inline constexpr int push_threads = 128;
inline constexpr int max_rows = 32;
inline constexpr int max_senders = 64;
inline constexpr int receive_chunks = max_rows * row_chunks;

template<int rows_, int senders_>
struct Push {
	static constexpr int rows = rows_;
	static constexpr int senders = senders_;
	static constexpr int item_chunks = rows * row_chunks;
	static constexpr int sender_chunks = item_chunks / senders;
	static constexpr Budget budget = {.threads = push_threads, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

	static_assert(rows <= max_rows && senders <= max_senders && senders <= sms);
	static_assert(item_chunks % senders == 0, "every sender pushes an equal share");
};

/**
 * The senders push into `receive` and release on `flags`, both in the receiver's memory, once the receiver has
 * published `go` in the senders' memory: so the receiver's clock is running before the first item lands.
 */
struct PushParams {
	Address receive;
	Address flags;
	Address go;
};

/** The receiving end: one thread per sender watches its flag; the block times the first and the last item. */
template<int senders_>
struct Receive {
	static constexpr int senders = senders_;
	static constexpr Budget budget = {
	    .threads = prims::round_up(senders, 32), .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};
};

struct ReceiveReading {
	unsigned long long first_ns;
	unsigned long long last_ns;
};

struct ReceiveParams {
	Address flags;
	Address go;
	Address reading;
};

struct PullParams {
	Address source;
	Address sink;
};

/** What runs beside the timed `.sys` releases: nothing, stores to the peer, or stores to mapped host memory. */
enum class Background : unsigned char { Idle, PeerStores, HostWrites };

inline constexpr int release_iterations = 10'000;
inline constexpr int background_chunks = 64;
inline constexpr Budget release_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

struct ReleaseParams {
	Address flag;
	Address scratch;
	Address peer;
	Address host;
	Address stop;
	Address cycles;
};

/** The mapped embedding-like table: rows of 10 KiB in pinned host memory. */
inline constexpr int table_rows = 2048;
inline constexpr int table_row_chunks = 10 * 1024 / 16;
inline constexpr int table_chunks = table_rows * table_row_chunks;
inline constexpr int fetch_repeats = 16;
inline constexpr Budget fetch_budget = {.threads = 128, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

template<int rows_>
struct Fetch {
	static constexpr int rows = rows_;
	static_assert(rows * fetch_repeats <= table_rows, "every repeat reads rows no earlier repeat touched");
};

struct FetchReading {
	unsigned long long best_ns;
	unsigned long long total_ns;
	unsigned long long sink;
};

struct FetchParams {
	Address table;
	Address reading;
};

/** The resident mailbox: a request carries its sequence number, the answer the card's time on seeing it. */
struct MailboxRequest {
	unsigned sequence;
};

struct MailboxAnswer {
	unsigned long long seen_ns;
};

inline constexpr unsigned long long mailbox_capacity = 2;
inline constexpr Budget mailbox_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};

struct MailboxParams {
	Address slots;
};

}

}

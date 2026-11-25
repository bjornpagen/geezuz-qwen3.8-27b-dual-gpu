// SPDX-License-Identifier: AGPL-3.0-only
//
// The soak kernels' shapes, parameters and result layouts, held once for both ends: the kernels compile them and
// `engine.soak` re-exports them, so the host launches exactly the geometry a kernel was built for and reads its results
// by the layout that wrote them. Pure constexpr with no device dependency. Parameters carry device addresses as 64-bit
// integers; the kernels turn them back into typed pointers.
#pragma once

#include "model.h"
#include "prims/budget.h"
#include "prims/chunk.h"
#include "soak/sha256.h"

namespace engine::device::soak {

using Address = unsigned long long;
using prims::Budget;
using prims::Chunk;

/**
 * The image digest, carried between launches in memory both the host and the card reach: the eight working words
 * after every block absorbed so far, and how many blocks that is. A pass starts from `sha256::initial` and zero blocks;
 * after the finishing launch the words are the digest, most significant word first.
 */
struct alignas(16) HashState {
	sha256::State state;
	unsigned long long blocks;
	unsigned long long unused;
};

static_assert(sizeof(HashState) == 48, "the state is three whole chunks");

/**
 * Most blocks one absorbing launch takes: 64 MiB, about a second and a half of a single SHA-256 chain on one SM. A pass
 * over a card image is a few hundred such launches, so the host sees progress, holds a deadline per launch, and the
 * pattern rounds on the other stream keep the memory hot meanwhile.
 */
inline constexpr unsigned slice_blocks = 1U << 20U;

inline constexpr unsigned long long block_bytes = 64;

/**
 * How a message splits for the kernels: its whole 64-byte blocks, absorbed in slices, then the 32-bit words after them,
 * which the finishing launch reads. A message is a whole number of words, so no launch reads past its end.
 */
struct Pass {
	unsigned long long blocks;
	unsigned tail_words;
	unsigned long long bytes;
};

[[nodiscard]] constexpr auto pass_of(unsigned long long bytes) -> Pass {
	return {.blocks = bytes / block_bytes, .tail_words = static_cast<unsigned>(bytes % block_bytes / 4), .bytes = bytes};
}

/** One absorbing launch's blocks: `count` blocks from block `first`. */
struct Slice {
	unsigned long long first;
	unsigned count;
};

/** How many absorbing launches a pass takes at `per_slice` blocks each. */
[[nodiscard]] constexpr auto slices(Pass const& pass, unsigned per_slice) -> unsigned long long {
	return (pass.blocks + per_slice - 1) / per_slice;
}

/** Absorbing launch `index` of a pass, the last one taking what remains. */
[[nodiscard]] constexpr auto slice(Pass const& pass, unsigned per_slice, unsigned long long index) -> Slice {
	unsigned long long const first = index * per_slice;
	unsigned long long const left = pass.blocks - first;
	return {.first = first, .count = static_cast<unsigned>(left < per_slice ? left : per_slice)};
}

/** Absorbs `blocks` whole 64-byte blocks starting at `source` into the state at `state`. */
struct AbsorbParams {
	Address state;
	Address source;
	unsigned blocks;
	unsigned unused;
};

/**
 * Pads the message and finishes the digest at `state`: `tail_words` 32-bit words at `tail` are the message's last
 * bytes after its whole blocks, and `total_bytes` its whole length.
 */
struct FinishParams {
	Address state;
	Address tail;
	unsigned tail_words;
	unsigned unused;
	unsigned long long total_bytes;
};

// SHA-256 is one chain from the first block to the last, so the digest is one thread's work; the rest of its warp only
// fills the launch's minimum of one warp. A lone warp may hold the most registers a thread can: the unrolled rounds
// keep the state, the schedule and the next block in flight, and the pattern block beside it on the SM still fits.
inline constexpr Budget hash_budget = {.threads = 32, .registers_per_thread = 255, .shared_bytes = 0, .blocks_per_sm = 1};

static_assert(prims::fits(hash_budget));

/**
 * The memory the pattern test covers on each card: everything the engine writes while it serves (the workspace, the
 * slots' state and the kv pool), so the soak exercises every cell a silent write error could land in. The weights are
 * the image hash's.
 */
inline constexpr unsigned long long region_bytes =
    (model::workspace_bytes + model::slots * model::slot_state_bytes + model::pool_bytes) / sizeof(Chunk) * sizeof(Chunk);
inline constexpr unsigned long long region_chunks = region_bytes / sizeof(Chunk);

// A word's place in the region, chunk * 4 + word, must fit 32 bits for the pattern's bijection to give every word of a
// round its own value.
static_assert(region_chunks * 4 <= 0x1'0000'0000ULL, "every word of the region has a distinct 32-bit index");

/** Every SM runs one block of the pattern kernels; each of its warps reports one tally. */
inline constexpr Budget pattern_budget = {.threads = 512, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};
inline constexpr int pattern_blocks = prims::sm86.sms;
inline constexpr int pattern_warps = pattern_budget.threads / prims::sm86.warp_threads;
inline constexpr int tallies = pattern_blocks * pattern_warps;

/** Each thread keeps this many independent 16-byte loads in flight per step, so the check streams at dram rate. */
inline constexpr int pattern_unroll = 4;

static_assert(prims::fits(pattern_budget));

/** Writes round `round`'s pattern over the region at `region`. */
struct WriteParams {
	Address region;
	unsigned round;
	unsigned unused;
};

/** Checks the region at `region` against round `round`'s pattern, one tally per warp into `tallies`. */
struct CheckParams {
	Address region;
	Address tallies;
	unsigned round;
	unsigned unused;
};

/**
 * One warp's check: how many 32-bit words differed from the pattern, and the lowest chunk index among those that did
 * (meaningful only when any did).
 */
struct alignas(8) Tally {
	unsigned mismatched_words;
	unsigned first_chunk;
};

[[nodiscard]] constexpr auto mix(unsigned value) -> unsigned {
	value ^= value >> 16U;
	value *= 0x85eb'ca6bU;
	value ^= value >> 13U;
	value *= 0xc2b2'ae35U;
	value ^= value >> 16U;
	return value;
}

/**
 * The value word `word` of chunk `chunk` holds in round `round`. Within a round every word gets its own value (the
 * finalizer is a bijection of the word's index, keyed by the round pair), so a read that lands on the wrong address
 * returns a wrong value; an odd round stores the bitwise inverse of the round before it, so over two rounds every bit
 * of every cell holds both a one and a zero.
 */
[[nodiscard]] constexpr auto pattern_word(unsigned round, unsigned chunk, unsigned word) -> unsigned {
	unsigned const key = (round / 2U) * 0x9e37'79b9U;
	unsigned const value = mix((chunk * 4U + word) ^ key);
	return round % 2U == 0U ? value : ~value;
}

}

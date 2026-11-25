// SPDX-License-Identifier: AGPL-3.0-only
//
// What the primitives' self-check kernels report and how the host reads it: the report layout, the fixed partition
// of the fp32 inputs, and the digest both sides fold the transcendentals into. Pure constexpr, shared with the host
// drivers (the microbenchmarks), so a report is decoded by the very definitions that wrote it.
#pragma once

#include "prims/budget.h"

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

// The fp32 inputs are split into blocks x threads contiguous runs; thread t of block b evaluates inputs
// [(b * threads + t) * run, ... + run).
inline constexpr Budget digest_budget = {.threads = 256, .registers_per_thread = 128, .shared_bytes = 0, .blocks_per_sm = 2};
inline constexpr unsigned digest_blocks = 1024;
inline constexpr unsigned long long digest_run = (1ULL << 32U) / (static_cast<unsigned long long>(digest_blocks) * digest_budget.threads);
static_assert(digest_run * digest_blocks * digest_budget.threads == 1ULL << 32U, "the runs tile every fp32 input");

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

/** The ring self-check's source pattern: word i of the tile stream. */
[[nodiscard]] constexpr auto ring_word(unsigned index) -> unsigned {
	return index * 0x9E3779B9U + 0x7F4A7C15U;
}

}

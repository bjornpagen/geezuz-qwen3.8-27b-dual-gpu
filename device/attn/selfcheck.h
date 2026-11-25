// SPDX-License-Identifier: AGPL-3.0-only
//
// What attention's self-check kernels compute and how the host reproduces it: the rope digest's partition and terms,
// the weight split's sweep, and the reduced SM count of the assignment check. Pure constexpr, shared with the host,
// so a report is read by the very definitions that wrote it.
#pragma once

#include "attn/arithmetic.h"
#include "prims/budget.h"
#include "prims/numeric.h"
#include "prims/selfcheck.h"

namespace engine::device::attn::selfcheck {

/** The assignment check's block count: far from 82, so nearly every group lands on a different block than at 82. */
inline constexpr unsigned few_sms = 7;

// The rope digest: one thread per position, its 32 pairs.
inline constexpr prims::Budget digest_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 4};
inline constexpr unsigned digest_blocks = static_cast<unsigned>(model::max_positions) / 256U;

static_assert(Count{digest_blocks} * 256U == model::max_positions, "the digest's threads are exactly the positions");

/** One (position, pair)'s contribution to its block's digest: the sine's and the cosine's bits, each mixed with its input. */
[[nodiscard]] constexpr auto rope_term(unsigned position, int pair) -> unsigned long long {
	unsigned const input = (position << 5U) | static_cast<unsigned>(pair);
	SinCos const turn = sincos(angle(position, pair));
	return prims::selfcheck::digest_term(input, math::bits(turn.sine)) + prims::selfcheck::digest_term(input | 0x80000000U, math::bits(turn.cosine));
}

// The split sweep: every fp32 bit pattern from +0 to 1.0, in contiguous runs, one per thread.
inline constexpr prims::Budget split_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 4};
inline constexpr unsigned split_blocks = 1024;
inline constexpr unsigned long long split_values = 0x3f800001ULL;
inline constexpr unsigned long long split_threads = static_cast<unsigned long long>(split_blocks) * 256U;
inline constexpr unsigned long long split_length = (split_values + split_threads - 1) / split_threads;

struct SplitRun {
	unsigned long long begin;
	unsigned long long end;
};

[[nodiscard]] constexpr auto split_run(unsigned thread) -> SplitRun {
	unsigned long long const begin = thread * split_length;
	unsigned long long const end = begin + split_length;
	return {.begin = begin < split_values ? begin : split_values, .end = end < split_values ? end : split_values};
}

/** Whether the weight with fp32 bits `bits`, scaled, splits exactly: the last part is what the others leave, all three sum to it. */
[[nodiscard]] constexpr auto split_exact(unsigned bits) -> bool {
	float const scaled = math::from_bits(bits) * weight_scale;
	Parts const parts = split(scaled);
	float const after_high = scaled - prims::widen(parts.part[0]);
	float const rest = after_high - prims::widen(parts.part[1]);
	float const low_two = prims::widen(parts.part[1]) + prims::widen(parts.part[2]);
	float const whole = prims::widen(parts.part[0]) + low_two;
	return math::bits(prims::widen(parts.part[2])) == math::bits(rest) && math::bits(whole) == math::bits(scaled);
}

}

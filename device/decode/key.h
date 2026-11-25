// SPDX-License-Identifier: AGPL-3.0-only
//
// A candidate as one unsigned 64-bit key whose numeric order is `sampling::ranks_before`: the logit's bits mapped to an
// unsigned order in the high word, the token id's complement in the low word. Ranking is then a sort of integers, the
// same on both cards and in the host's checks. Key zero is no candidate (a masked row, a missing list entry); no
// candidate maps to it, since a token id's complement is at least 2^31. It uses no standard headers so that it
// compiles unchanged as device code and inside a module purview.
#pragma once

#include "sampling/draw.h"

namespace engine::device::decode {

using Key = unsigned long long;

inline constexpr Key no_candidate = 0;

/**
 * The logit's bits in an unsigned order: negatives reversed below the positives. -0 is taken as +0 first, since the
 * two are equal logits, which `ranks_before` orders by id alone.
 */
[[nodiscard]] constexpr auto ordered(float logit) -> unsigned {
	unsigned const bits = logit == 0.0F ? 0U : __builtin_bit_cast(unsigned, logit);
	return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

[[nodiscard]] constexpr auto logit_of(unsigned order) -> float {
	unsigned const bits = (order & 0x80000000U) != 0 ? order & 0x7fffffffU : ~order;
	return __builtin_bit_cast(float, bits);
}

[[nodiscard]] constexpr auto key_of(float logit, unsigned token) -> Key {
	return (static_cast<Key>(ordered(logit)) << 32U) | static_cast<Key>(0xffffffffU - token);
}

/** The candidate a key stands for; no candidate becomes id -1, which ranks after every real one by its key. */
[[nodiscard]] constexpr auto candidate_of(Key key) -> sampling::Candidate {
	if (key == no_candidate) {
		return {.logit = 0.0F, .id = -1};
	}
	return {.logit = logit_of(static_cast<unsigned>(key >> 32U)), .id = static_cast<int>(0xffffffffU - static_cast<unsigned>(key))};
}

[[nodiscard]] constexpr auto key_of(sampling::Candidate candidate) -> Key {
	return candidate.id < 0 ? no_candidate : key_of(candidate.logit, static_cast<unsigned>(candidate.id));
}

}

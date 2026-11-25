// SPDX-License-Identifier: AGPL-3.0-only
//
// The engine's one random stream, shared bit for bit by the device and the host reference: Philox4x32-10 (Salmon et
// al., "Parallel random numbers: as easy as 1, 2, 3"), a counter-based generator, so a draw is a pure function of its
// key and no generator state exists to carry between cycles, cards or a preemption and resume. Everything here is
// `constexpr`, which clang compiles for both the host and the device from this one definition.
#pragma once

namespace engine::device::sampling {

static_assert(sizeof(unsigned int) == 4 && sizeof(unsigned long long) == 8, "philox works on 32-bit words");

/** One 128-bit philox block: the counter going in, the random bits coming out. */
struct Block {
	unsigned int w0;
	unsigned int w1;
	unsigned int w2;
	unsigned int w3;
};

/** The 64-bit philox key. */
struct Philox_key {
	unsigned int k0;
	unsigned int k1;
};

[[nodiscard]] constexpr auto philox_round(Block counter, Philox_key key) -> Block {
	constexpr unsigned long long multiplier0 = 0xD2511F53ULL;
	constexpr unsigned long long multiplier1 = 0xCD9E8D57ULL;
	unsigned long long const product0 = multiplier0 * counter.w0;
	unsigned long long const product1 = multiplier1 * counter.w2;
	return Block{
	    .w0 = static_cast<unsigned int>(product1 >> 32U) ^ counter.w1 ^ key.k0,
	    .w1 = static_cast<unsigned int>(product1),
	    .w2 = static_cast<unsigned int>(product0 >> 32U) ^ counter.w3 ^ key.k1,
	    .w3 = static_cast<unsigned int>(product0),
	};
}

/** Philox4x32 with its standard ten rounds: the key is bumped by the Weyl constants between rounds. */
[[nodiscard]] constexpr auto philox(Block counter, Philox_key key) -> Block {
	constexpr unsigned int weyl0 = 0x9E3779B9U;
	constexpr unsigned int weyl1 = 0xBB67AE85U;
	Block block = philox_round(counter, key);
	for (int round = 1; round < 10; ++round) {
		key.k0 += weyl0;
		key.k1 += weyl1;
		block = philox_round(block, key);
	}
	return block;
}

// Known answers from the Random123 distribution (kat_vectors, philox4x32 10 rounds), checked by every compiler that
// includes this header: the host compiler and the device compiler each evaluate them.
static_assert(philox({0, 0, 0, 0}, {0, 0}).w0 == 0x6627e8d5U && philox({0, 0, 0, 0}, {0, 0}).w1 == 0xe169c58dU &&
              philox({0, 0, 0, 0}, {0, 0}).w2 == 0xbc57ac4cU && philox({0, 0, 0, 0}, {0, 0}).w3 == 0x9b00dbd8U);
static_assert(philox({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU}, {0xffffffffU, 0xffffffffU}).w0 == 0x408f276dU &&
              philox({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU}, {0xffffffffU, 0xffffffffU}).w1 == 0x41c83b0eU &&
              philox({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU}, {0xffffffffU, 0xffffffffU}).w2 == 0xa20bc7c6U &&
              philox({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU}, {0xffffffffU, 0xffffffffU}).w3 == 0x6d5451fdU);
static_assert(philox({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}, {0xa4093822U, 0x299f31d0U}).w0 == 0xd16cfe09U &&
              philox({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}, {0xa4093822U, 0x299f31d0U}).w1 == 0x94fdccebU &&
              philox({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}, {0xa4093822U, 0x299f31d0U}).w2 == 0x5001e420U &&
              philox({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}, {0xa4093822U, 0x299f31d0U}).w3 == 0x24126ea1U);

/**
 * Why a uniform is drawn. Each decided token position draws at most one uniform per purpose: a plain or bonus
 * sample from the target, the drafter's sample of its proposal, the acceptance test of that proposal, and the
 * residual sample after a rejection.
 */
enum class Purpose : unsigned char { Sample, Draft, Accept, Residual };

/**
 * Everything one uniform is a function of. `seed` is the request's seed; `request` separates independent streams
 * under one seed (the choice index: a request that names its seed passes 0, so the same seed reproduces the same
 * text); `position` is the absolute sequence position of the token being decided, so a preempted request resumes
 * on the same draws; `step` is that token's row within the cycle's draft chain (0 for the first draft).
 */
struct Key {
	unsigned long long seed;
	unsigned int request;
	unsigned int position;
	unsigned int step;
	Purpose purpose;
};

/**
 * The uniform for a key, on the 2^-53 grid of [0, 1): the counter is (position, request, step, purpose), the philox
 * key is the seed's low and high words, and the top 53 bits of the first two output words become the fraction.
 * Every value is exact in binary64, and the largest is 1 - 2^-53.
 */
[[nodiscard]] constexpr auto uniform(Key const& key) -> double {
	Block const bits = philox(
	    Block{
	        .w0 = key.position,
	        .w1 = key.request,
	        .w2 = key.step,
	        .w3 = static_cast<unsigned int>(key.purpose),
	    },
	    Philox_key{.k0 = static_cast<unsigned int>(key.seed), .k1 = static_cast<unsigned int>(key.seed >> 32U)});
	unsigned long long const fraction = (static_cast<unsigned long long>(bits.w0) << 21U) | (bits.w1 >> 11U);
	return static_cast<double>(fraction) * 0x1p-53;
}

}

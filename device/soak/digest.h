// SPDX-License-Identifier: AGPL-3.0-only
//
// How the soak's kernels walk a message: whole blocks in slices, each block's four chunks loaded while the block before
// it compresses, then the tail words, the padding and the length. Generic over how a chunk or a word is fetched, so the
// kernels fetch from card memory and the host's emulation from host memory through the very same steps.
#pragma once

#include "prims/chunk.h"
#include "soak/layout.h"
#include "soak/sha256.h"

namespace engine::device::soak {

/** One block's bytes as the card loads them: four 16-byte chunks of little-endian words. */
struct Quad {
	prims::Chunk chunk[4];
};

[[nodiscard, gnu::always_inline]] constexpr auto message(Quad const& quad) -> sha256::Block {
	sha256::Block block{};
	for (int chunk = 0; chunk < 4; ++chunk) {
		for (int word = 0; word < 4; ++word) {
			block.word[chunk * 4 + word] = sha256::big_endian(quad.chunk[chunk].word[word]);
		}
	}
	return block;
}

/** Block `index` of a message whose chunks `chunks` fetches by index. */
template<class Chunks>
[[nodiscard, gnu::always_inline]] constexpr auto fetch(Chunks const& chunks, unsigned long long index) -> Quad {
	Quad quad{};
	for (int chunk = 0; chunk < 4; ++chunk) {
		quad.chunk[chunk] = chunks(index * 4 + static_cast<unsigned long long>(chunk));
	}
	return quad;
}

/**
 * Absorbs `blocks` whole blocks, block `i` being chunks 4i to 4i + 3 of `chunks`. The next block's loads are issued
 * before the current block compresses, so one load latency hides under each compression.
 */
template<class Chunks>
[[gnu::always_inline]] constexpr void absorb(HashState& hash, Chunks const& chunks, unsigned blocks) {
	if (blocks == 0U) {
		return;
	}
	Quad next = fetch(chunks, 0);
	for (unsigned block = 0; block < blocks; ++block) {
		Quad const current = next;
		if (block + 1U < blocks) {
			next = fetch(chunks, block + 1ULL);
		}
		hash.state = sha256::compress(hash.state, message(current));
	}
	hash.blocks += blocks;
}

/**
 * Finishes the digest: `tail_words` (at most 15) little-endian words fetched by `words` follow the absorbed blocks, then
 * the standard's marker bit, zeros, and the message's length in bits in the last block's final two words. The marker
 * and the length share one block when the tail leaves the two words free, else the length takes a block of its own.
 */
template<class Words>
[[gnu::always_inline]] constexpr void finish(HashState& hash, Words const& words, unsigned tail_words, unsigned long long total_bytes) {
	sha256::Block last{};
	for (unsigned index = 0; index < 16U; ++index) {
		if (index < tail_words) {
			last.word[index] = sha256::big_endian(words(index));
		} else if (index == tail_words) {
			last.word[index] = 0x8000'0000U;
		}
	}
	if (tail_words >= 14U) {
		hash.state = sha256::compress(hash.state, last);
		last = sha256::Block{};
	}
	unsigned long long const bits = total_bytes * 8ULL;
	last.word[14] = static_cast<unsigned>(bits >> 32U);
	last.word[15] = static_cast<unsigned>(bits);
	hash.state = sha256::compress(hash.state, last);
}

}

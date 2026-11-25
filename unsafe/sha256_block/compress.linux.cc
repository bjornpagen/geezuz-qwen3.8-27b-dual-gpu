// SPDX-License-Identifier: AGPL-3.0-only
module;

// The x86 SHA extensions (SHA-NI), which the production box's Zen 3 cores have, plus the SSE4.1 shuffles and blends
// that rearrange the state for them.
#include <immintrin.h>

export module engine.sha256_block:compress;

import std;
import engine.core;
import :constants;

namespace engine::sha256_block {

// The instructions keep the state as two halves, ABEF and CDGH, in the lanes from high to low; the schedule is a ring
// of four vectors of four message words each.
struct Lanes {
	__m128i abef;
	__m128i cdgh;
	std::array<__m128i, 4> words;
};

// The baseline x86-64 target lacks these instructions, and a translation-unit flag would make the module's interface
// disagree with its importers about target features, so every function that uses them names them itself.
template<std::size_t group>
[[gnu::target("sha,sse4.1"), gnu::always_inline]] inline auto four_rounds(Lanes& lanes) -> void {
	/* SAFETY: unaligned 16-byte load of four round constants inside the 64-entry table. */
	auto const constants = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::span{round_constants}.subspan<4 * group, 4>().data()));
	auto scheduled = _mm_add_epi32(lanes.words[group % 4], constants);
	lanes.cdgh = _mm_sha256rnds2_epu32(lanes.cdgh, lanes.abef, scheduled);
	scheduled = _mm_shuffle_epi32(scheduled, 0x0e);
	lanes.abef = _mm_sha256rnds2_epu32(lanes.abef, lanes.cdgh, scheduled);
	// Group g consumes schedule words g..g+3; before the last four groups it expands words g+16..g+19 in place.
	if constexpr (group < 12) {
		auto const tail = _mm_alignr_epi8(lanes.words[(group + 3) % 4], lanes.words[(group + 2) % 4], 4);
		auto const partial = _mm_add_epi32(_mm_sha256msg1_epu32(lanes.words[group % 4], lanes.words[(group + 1) % 4]), tail);
		lanes.words[group % 4] = _mm_sha256msg2_epu32(partial, lanes.words[(group + 3) % 4]);
	}
}

template<std::size_t... group>
[[gnu::target("sha,sse4.1"), gnu::always_inline]] inline auto sixty_four_rounds(Lanes& lanes, std::index_sequence<group...> /*groups*/)
    -> void {
	(four_rounds<group>(lanes), ...);
}

/**
 * Compresses whole 64-byte blocks into `state`, in order. The working state stays in vector registers across the
 * blocks, which is why the call takes many blocks at once.
 */
export [[nodiscard, gnu::target("sha,sse4.1")]] auto compress(State state, std::span<std::byte const> blocks) -> State {
	core::invariant(blocks.size() % block_bytes == 0);
	// The message words are big-endian; the lanes are little-endian.
	auto const big_endian = _mm_set_epi64x(0x0c0d0e0f'08090a0bLL, 0x04050607'00010203LL);
	/* SAFETY: unaligned 16-byte loads of the two halves of the eight-word state. */
	auto const dcba = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<__m128i const*>(state.data())), 0xb1);
	auto const hgfe = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<__m128i const*>(std::span{state}.subspan<4>().data())), 0x1b);
	auto lanes = Lanes{
	    .abef = _mm_alignr_epi8(dcba, hgfe, 8),
	    .cdgh = _mm_blend_epi16(hgfe, dcba, 0xf0),
	    .words = {},
	};
	for (auto offset = std::size_t{0}; offset < blocks.size(); offset += block_bytes) {
		auto const block = blocks.subspan(offset, block_bytes);
		for (auto index = std::size_t{0}; index < lanes.words.size(); ++index) {
			/* SAFETY: unaligned 16-byte load inside the 64-byte block. */
			auto const raw = _mm_loadu_si128(reinterpret_cast<__m128i const*>(block.subspan(16 * index, 16).data()));
			lanes.words.at(index) = _mm_shuffle_epi8(raw, big_endian);
		}
		auto const start_abef = lanes.abef;
		auto const start_cdgh = lanes.cdgh;
		sixty_four_rounds(lanes, std::make_index_sequence<16>{});
		lanes.abef = _mm_add_epi32(lanes.abef, start_abef);
		lanes.cdgh = _mm_add_epi32(lanes.cdgh, start_cdgh);
	}
	auto const feba = _mm_shuffle_epi32(lanes.abef, 0x1b);
	auto const dchg = _mm_shuffle_epi32(lanes.cdgh, 0xb1);
	/* SAFETY: unaligned 16-byte stores of the two halves of the eight-word state. */
	_mm_storeu_si128(reinterpret_cast<__m128i*>(state.data()), _mm_blend_epi16(feba, dchg, 0xf0));
	_mm_storeu_si128(reinterpret_cast<__m128i*>(std::span{state}.subspan<4>().data()), _mm_alignr_epi8(dchg, feba, 8));
	return state;
}

}

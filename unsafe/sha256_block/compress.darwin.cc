// SPDX-License-Identifier: AGPL-3.0-only
module;

// The ARMv8 SHA-256 instructions (FEAT_SHA256), which every Apple arm64 core has.
#include <arm_neon.h>

export module engine.sha256_block:compress;

import std;
import engine.core;
import :constants;

namespace engine::sha256_block {

// The message words are big-endian; the vector lanes are native little-endian. The intrinsics are macros, so their
// arguments are plain names.
[[nodiscard]] auto load_words(std::span<std::uint8_t const, 16> bytes) -> uint32x4_t {
	auto const* const first = bytes.data();
	return vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(first)));
}

// The working state as the instructions keep it, ABCD and EFGH, and the schedule as a ring of four vectors of four
// message words each.
struct Lanes {
	uint32x4_t abcd;
	uint32x4_t efgh;
	std::array<uint32x4_t, 4> words;
};

// Group g of four rounds consumes schedule words g..g+3 and, before the last four groups, expands words g+16..g+19 in
// place. Values in and out, and forced inline, so the whole block's rounds stay in registers.
template<std::size_t group>
[[nodiscard, gnu::always_inline]] inline auto four_rounds(Lanes lanes) -> Lanes {
	auto const* const constants = std::span{round_constants}.subspan<4 * group, 4>().data();
	auto const scheduled = vaddq_u32(lanes.words[group % 4], vld1q_u32(constants));
	auto const before = lanes.abcd;
	lanes.abcd = vsha256hq_u32(lanes.abcd, lanes.efgh, scheduled);
	lanes.efgh = vsha256h2q_u32(lanes.efgh, before, scheduled);
	if constexpr (group < 12) {
		auto const partial = vsha256su0q_u32(lanes.words[group % 4], lanes.words[(group + 1) % 4]);
		lanes.words[group % 4] = vsha256su1q_u32(partial, lanes.words[(group + 2) % 4], lanes.words[(group + 3) % 4]);
	}
	return lanes;
}

template<std::size_t... group>
[[nodiscard, gnu::always_inline]] inline auto sixty_four_rounds(Lanes lanes, std::index_sequence<group...> /*groups*/) -> Lanes {
	((lanes = four_rounds<group>(lanes)), ...);
	return lanes;
}

/**
 * Compresses whole 64-byte blocks into `state`, in order. The working state stays in vector registers across the
 * blocks, which is why the call takes many blocks at once.
 */
export [[nodiscard]] auto compress(State state, std::span<std::byte const> blocks) -> State {
	core::invariant(blocks.size() % block_bytes == 0);
	auto const* const low = state.data();
	auto const* const high = std::span{state}.subspan<4>().data();
	auto lanes = Lanes{.abcd = vld1q_u32(low), .efgh = vld1q_u32(high), .words = {}};
	for (auto offset = std::size_t{0}; offset < blocks.size(); offset += block_bytes) {
		/* SAFETY: a std::byte and a uint8_t share size and alignment; the 64 bytes lie inside `blocks`. */
		auto const block = std::span{reinterpret_cast<std::uint8_t const*>(blocks.subspan(offset, block_bytes).data()), block_bytes};
		lanes.words = {load_words(block.first<16>()), load_words(block.subspan<16, 16>()), load_words(block.subspan<32, 16>()),
		               load_words(block.subspan<48, 16>())};
		auto const start = lanes;
		lanes = sixty_four_rounds(lanes, std::make_index_sequence<16>{});
		lanes.abcd = vaddq_u32(lanes.abcd, start.abcd);
		lanes.efgh = vaddq_u32(lanes.efgh, start.efgh);
	}
	auto* const low_out = state.data();
	auto* const high_out = std::span{state}.subspan<4>().data();
	vst1q_u32(low_out, lanes.abcd);
	vst1q_u32(high_out, lanes.efgh);
	return state;
}

}

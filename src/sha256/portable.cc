// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sha256:portable;

import std;
import engine.core;
import engine.sha256_block;

namespace engine::sha256 {

using sha256_block::block_bytes;
using sha256_block::State;

[[nodiscard]] constexpr auto big_endian_word(std::span<std::byte const, 4> bytes) -> std::uint32_t {
	return (std::to_integer<std::uint32_t>(bytes[0]) << 24U) | (std::to_integer<std::uint32_t>(bytes[1]) << 16U) |
	       (std::to_integer<std::uint32_t>(bytes[2]) << 8U) | std::to_integer<std::uint32_t>(bytes[3]);
}

/**
 * The compression function in plain integer arithmetic, exactly as the standard states it. It exists because the
 * machine's SHA instructions have no constant evaluation: digests the build computes (the model description's) take
 * this path, digests computed at run time take the instructions.
 */
[[nodiscard]] constexpr auto portable_compress(State state, std::span<std::byte const> blocks) -> State {
	core::invariant(blocks.size() % block_bytes == 0);
	for (auto offset = std::size_t{0}; offset < blocks.size(); offset += block_bytes) {
		auto const block = blocks.subspan(offset, block_bytes);
		auto schedule = std::array<std::uint32_t, 64>{};
		for (auto index = std::size_t{0}; index < 16; ++index) {
			schedule.at(index) = big_endian_word(block.subspan(4 * index).first<4>());
		}
		for (auto index = std::size_t{16}; index < schedule.size(); ++index) {
			auto const early = schedule.at(index - 15);
			auto const late = schedule.at(index - 2);
			auto const sigma0 = std::rotr(early, 7) ^ std::rotr(early, 18) ^ (early >> 3U);
			auto const sigma1 = std::rotr(late, 17) ^ std::rotr(late, 19) ^ (late >> 10U);
			schedule.at(index) = schedule.at(index - 16) + sigma0 + schedule.at(index - 7) + sigma1;
		}
		auto work = state;
		for (auto round = std::size_t{0}; round < schedule.size(); ++round) {
			auto const [a, b, c, d, e, f, g, h] = work;
			auto const choose = (e & f) ^ (~e & g);
			auto const majority = (a & b) ^ (a & c) ^ (b & c);
			auto const big_sigma0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
			auto const big_sigma1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
			auto const first = h + big_sigma1 + choose + sha256_block::round_constants.at(round) + schedule.at(round);
			auto const second = big_sigma0 + majority;
			work = State{first + second, a, b, c, d + first, e, f, g};
		}
		for (auto index = std::size_t{0}; index < state.size(); ++index) {
			state.at(index) += work.at(index);
		}
	}
	return state;
}

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The card's image digest on the host: the host's end of the digest cell (starting a pass, reading the digest), and the
// kernels' own steps, from the device headers, fed from host memory in the slices the soak launches. The tests hold the
// emulation against the engine's SHA-256; the build holds the card's constants against the engine's tables.
module;

#include "soak/digest.h"
#include "soak/layout.h"
#include "soak/sha256.h"

export module engine.soak:emulation;

import std;
import engine.core;
import engine.sha256;
import engine.sha256_block;
import :layout;
import :kernels;

namespace engine::soak {

namespace device = engine::device::soak;

static_assert(std::endian::native == std::endian::little, "the host reads the card's little-endian words as the card does");
static_assert(std::ranges::equal(device::sha256::round_constants, sha256_block::round_constants),
              "the card's round constants are the standard's, as the engine's SHA-256 holds them");
static_assert(std::ranges::equal(device::sha256::initial.word, sha256_block::initial_state),
              "the card's initial state is the standard's, as the engine's SHA-256 holds it");

/** The digest a finished state holds: its words in order, each most significant byte first. */
[[nodiscard]] constexpr auto digest_of(State const& state) -> sha256::Digest {
	auto digest = sha256::Digest{};
	for (auto index = std::size_t{0}; index < digest.bytes.size(); ++index) {
		auto const word = std::span{state.word}[index / 4];
		digest.bytes.at(index) = std::byte{core::narrow<std::uint8_t>((word >> (24U - (8U * (index % 4)))) & 0xffU)};
	}
	return digest;
}

/** Starts a pass in `cell`: the standard's initial state, no blocks absorbed. Only while no launch is using it. */
export auto begin_pass(StateCell& cell) -> void {
	cell.host()[0] = HashState{.state = initial_state, .blocks = 0, .unused = 0};
}

/** The digest in `cell` once its pass's finishing launch has completed. */
export [[nodiscard]] auto digest_of(StateCell const& cell) -> sha256::Digest {
	return digest_of(cell.host()[0].state);
}

/** A message's chunks as the card's loads return them. */
struct HostChunks {
	std::span<std::byte const> bytes;

	[[nodiscard]] auto operator()(std::uint64_t index) const -> Chunk {
		auto part = std::array<std::byte, sizeof(Chunk)>{};
		std::ranges::copy(bytes.subspan(core::narrow<std::size_t>(index) * sizeof(Chunk), sizeof(Chunk)), part.begin());
		return std::bit_cast<Chunk>(part);
	}
};

/** A message's words as the card's word loads return them. */
struct HostWords {
	std::span<std::byte const> bytes;

	[[nodiscard]] auto operator()(unsigned index) const -> unsigned {
		auto part = std::array<std::byte, sizeof(unsigned)>{};
		std::ranges::copy(bytes.subspan(std::size_t{index} * sizeof(unsigned), sizeof(unsigned)), part.begin());
		return std::bit_cast<unsigned>(part);
	}
};

/**
 * The digest the kernels compute for `message` (a whole number of 32-bit words), absorbed in slices of `per_slice`
 * blocks exactly as the soak launches them, then finished as the finishing launch does.
 */
export [[nodiscard]] auto emulate(std::span<std::byte const> message, unsigned per_slice) -> sha256::Digest {
	core::invariant(message.size() % sizeof(unsigned) == 0 && per_slice > 0);
	auto const pass = pass_of(message.size());
	auto hash = HashState{.state = initial_state, .blocks = 0, .unused = 0};
	for (auto index = std::uint64_t{0}; index < slices(pass, per_slice); ++index) {
		auto const part = slice(pass, per_slice, index);
		device::absorb(hash, HostChunks{.bytes = message.subspan(core::narrow<std::size_t>(part.first * block_bytes))}, part.count);
	}
	device::finish(hash, HostWords{.bytes = message.subspan(core::narrow<std::size_t>(pass.blocks * block_bytes))}, pass.tail_words, pass.bytes);
	return digest_of(hash.state);
}

}

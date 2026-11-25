// SPDX-License-Identifier: AGPL-3.0-only
//
// SHA-256's compression function in plain 32-bit integer arithmetic, as the standard states it, for the card: the soak
// hashes each card's weight image where it lies in memory and compares the digest with the one the baker stamped in
// the image. Pure constexpr, so the host's emulation runs these very rounds; the host checks them against the engine's
// own SHA-256 and checks the constants against its tables when it compiles them.
#pragma once

namespace engine::device::soak::sha256 {

/** The eight working words between blocks, in the standard's order. */
struct State {
	unsigned word[8];
};

/** One 64-byte block as sixteen message words, each already in the standard's big-endian reading. */
struct Block {
	unsigned word[16];
};

inline constexpr State initial = {{0x6a09'e667U, 0xbb67'ae85U, 0x3c6e'f372U, 0xa54f'f53aU, 0x510e'527fU, 0x9b05'688cU, 0x1f83'd9abU, 0x5be0'cd19U}};

inline constexpr unsigned round_constants[64] = {
    0x428a'2f98U, 0x7137'4491U, 0xb5c0'fbcfU, 0xe9b5'dba5U, 0x3956'c25bU, 0x59f1'11f1U, 0x923f'82a4U, 0xab1c'5ed5U,
    0xd807'aa98U, 0x1283'5b01U, 0x2431'85beU, 0x550c'7dc3U, 0x72be'5d74U, 0x80de'b1feU, 0x9bdc'06a7U, 0xc19b'f174U,
    0xe49b'69c1U, 0xefbe'4786U, 0x0fc1'9dc6U, 0x240c'a1ccU, 0x2de9'2c6fU, 0x4a74'84aaU, 0x5cb0'a9dcU, 0x76f9'88daU,
    0x983e'5152U, 0xa831'c66dU, 0xb003'27c8U, 0xbf59'7fc7U, 0xc6e0'0bf3U, 0xd5a7'9147U, 0x06ca'6351U, 0x1429'2967U,
    0x27b7'0a85U, 0x2e1b'2138U, 0x4d2c'6dfcU, 0x5338'0d13U, 0x650a'7354U, 0x766a'0abbU, 0x81c2'c92eU, 0x9272'2c85U,
    0xa2bf'e8a1U, 0xa81a'664bU, 0xc24b'8b70U, 0xc76c'51a3U, 0xd192'e819U, 0xd699'0624U, 0xf40e'3585U, 0x106a'a070U,
    0x19a4'c116U, 0x1e37'6c08U, 0x2748'774cU, 0x34b0'bcb5U, 0x391c'0cb3U, 0x4ed8'aa4aU, 0x5b9c'ca4fU, 0x682e'6ff3U,
    0x748f'82eeU, 0x78a5'636fU, 0x84c8'7814U, 0x8cc7'0208U, 0x90be'fffaU, 0xa450'6cebU, 0xbef9'a3f7U, 0xc671'78f2U,
};

// The rotation is spelled as two shifts; llvm folds the pair into one funnel shift on the card.
[[nodiscard]] constexpr auto rotate_right(unsigned value, unsigned count) -> unsigned {
	return (value >> count) | (value << (32U - count));
}

/** A 32-bit word read from little-endian memory, as the standard's big-endian message word. */
[[nodiscard]] constexpr auto big_endian(unsigned little) -> unsigned {
	return __builtin_bswap32(little);
}

// One round per instantiation, so every schedule index and round constant is a compile-time index; with every step
// inlined, the whole block unrolls into registers on the card, with no indexed array to spill to local memory.
template<int round>
[[gnu::always_inline]] constexpr void rounds(State& work, Block& schedule) {
	if constexpr (round < 64) {
		if constexpr (round >= 16) {
			unsigned const early = schedule.word[(round - 15) % 16];
			unsigned const late = schedule.word[(round - 2) % 16];
			unsigned const sigma0 = rotate_right(early, 7) ^ rotate_right(early, 18) ^ (early >> 3U);
			unsigned const sigma1 = rotate_right(late, 17) ^ rotate_right(late, 19) ^ (late >> 10U);
			schedule.word[round % 16] += sigma0 + schedule.word[(round - 7) % 16] + sigma1;
		}
		unsigned const a = work.word[0];
		unsigned const e = work.word[4];
		unsigned const choose = (e & work.word[5]) ^ (~e & work.word[6]);
		unsigned const majority = (a & work.word[1]) ^ (a & work.word[2]) ^ (work.word[1] & work.word[2]);
		unsigned const big_sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
		unsigned const big_sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
		unsigned const first = work.word[7] + big_sigma1 + choose + round_constants[round] + schedule.word[round % 16];
		unsigned const second = big_sigma0 + majority;
		work = State{{first + second, a, work.word[1], work.word[2], work.word[3] + first, e, work.word[5], work.word[6]}};
		rounds<round + 1>(work, schedule);
	}
}

/** The state after one more block. */
[[nodiscard, gnu::always_inline]] constexpr auto compress(State state, Block block) -> State {
	State work = state;
	rounds<0>(work, block);
	for (int index = 0; index < 8; ++index) {
		state.word[index] += work.word[index];
	}
	return state;
}

}

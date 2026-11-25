// SPDX-License-Identifier: AGPL-3.0-only
//
// Tensor-core fragments: the bf16 activation b-fragments by `ldmatrix` from a swizzled tile, the weight a-fragments by
// one 16-byte shared load decoded in registers from the baked e4m3, `mma.sync.m16n8k16` bf16 x bf16 -> fp32, and
// register double buffering across k steps.
#pragma once

#include "prims/fragment.h"
#include "prims/isa.h"
#include "prims/memory.h"
#include "prims/numeric.h"
#include "prims/shared.h"

namespace engine::device::prims {

static_assert(available(isa::ldmatrix) && available(isa::mma_bf16));
static_assert(available(isa::fma_bf16x2) && available(isa::integer_logic) && available(isa::byte_permute));

/** An m16n8k16 A operand: 4 bf16x2 registers, laid out as `fragment::a`. */
struct AFragment {
	unsigned reg[4];
};

/** An m16n8k16 B operand: 2 bf16x2 registers, laid out as `fragment::b`. */
struct BFragment {
	unsigned reg[2];
};

/** An m16n8 fp32 accumulator: 4 registers, laid out as `fragment::c`. */
struct Accumulator {
	float reg[4];
};

/** Four 8x8 bf16 matrices as `ldmatrix.x4` delivers them, register j holding matrix j. */
struct Matrices4 {
	unsigned reg[4];
};

/**
 * Loads four 8x8 bf16 matrices; lane 8m + r passes the address of matrix m's row r (16 contiguous bytes in a
 * swizzled tile).
 */
template<fragment::Transpose transpose>
[[nodiscard]] __attribute__((device)) inline auto ldmatrix_x4(SharedAddress row) -> Matrices4 {
	Matrices4 loaded{};
	// Why ptx: `ldmatrix` is gated at ptx 9.3.
	if constexpr (transpose == fragment::Transpose::Plain) {
		asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
		             : "=r"(loaded.reg[0]), "=r"(loaded.reg[1]), "=r"(loaded.reg[2]), "=r"(loaded.reg[3])
		             : "r"(row.value)
		             : "memory");
	} else {
		asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
		             : "=r"(loaded.reg[0]), "=r"(loaded.reg[1]), "=r"(loaded.reg[2]), "=r"(loaded.reg[3])
		             : "r"(row.value)
		             : "memory");
	}
	return loaded;
}

/** One lane's 16 bytes of a shared tile: a warp's 512 contiguous bytes are four conflict-free wavefronts. */
[[nodiscard]] __attribute__((device)) inline auto load_shared(SharedAddress chunk) -> Chunk {
	Chunk loaded{};
	asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
	             : "=r"(loaded.word[0]), "=r"(loaded.word[1]), "=r"(loaded.word[2]), "=r"(loaded.word[3])
	             : "r"(chunk.value)
	             : "memory");
	return loaded;
}

/** Stores one lane's 16 bytes into a shared tile. */
__attribute__((device)) inline void store_shared(SharedAddress chunk, Chunk value) {
	asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
	             :
	             : "r"(chunk.value), "r"(value.word[0]), "r"(value.word[1]), "r"(value.word[2]), "r"(value.word[3])
	             : "memory");
}

/** D = A B + D on the tensor cores: bf16 products, fp32 accumulation. */
__attribute__((device)) inline void mma(Accumulator& accumulator, AFragment const& a, BFragment const& b) {
	// Why ptx: `mma` is gated at ptx 9.3.
	asm volatile(
	    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
	    : "+f"(accumulator.reg[0]), "+f"(accumulator.reg[1]), "+f"(accumulator.reg[2]), "+f"(accumulator.reg[3])
	    : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]), "r"(b.reg[0]), "r"(b.reg[1]));
}

/**
 * Decodes one baked word into its two bf16x2 registers, exactly as `decode::model` specifies: five integer
 * instructions place both pairs' payloads at their bf16 positions, and one `fma.rn.bf16x2` by 2^120 with a -0 addend
 * per pair restores the exponent. Exact for every finite code; the baker admits no other.
 */
[[nodiscard]] __attribute__((device)) inline auto decode_word(unsigned word) -> decode::Pairs {
	unsigned top = 0;
	unsigned bottom = 0;
	// Why ptx: the sequence is the specification (instruction for instruction what the host model proves exact), and
	// `fma.rn.bf16x2` is gated at ptx 9.3.
	asm("{\n\t.reg .b32 left, right;\n\t"
	    "and.b32 %0, %2, %3;\n\t"
	    "and.b32 left, %2, %4;\n\t"
	    "shl.b32 left, left, 4;\n\t"
	    "shr.u32 right, %2, 4;\n\t"
	    "lop3.b32 %1, right, left, %5, 0xE4;\n\t"
	    "fma.rn.bf16x2 %0, %0, %6, %7;\n\t"
	    "fma.rn.bf16x2 %1, %1, %6, %7;\n\t}"
	    : "=&r"(top), "=&r"(bottom)
	    : "r"(word), "r"(decode::placed_mask), "r"(decode::left_mask), "r"(decode::right_mask), "r"(decode::two_to_120.bits),
	      "r"(decode::negative_zero.bits));
	return {.top = {.bits = top}, .bottom = {.bits = bottom}};
}

static_assert(decode::select_lut == 0xE4U, "decode_word spells the select table inline");

/** Two k16 steps of A fragments from one lane's baked 16 bytes (`fragment::weight` order). */
struct AFragments2 {
	AFragment step[2];
};

[[nodiscard]] __attribute__((device)) inline auto decode_chunk(Chunk chunk) -> AFragments2 {
	AFragments2 decoded{};
	for (decltype(sizeof(0)) step = 0; step < 2; ++step) {
		decode::Pairs const low_k = decode_word(chunk.word[2 * step]);
		decode::Pairs const high_k = decode_word(chunk.word[2 * step + 1]);
		decoded.step[step] = {.reg = {low_k.top.bits, low_k.bottom.bits, high_k.top.bits, high_k.bottom.bits}};
	}
	return decoded;
}

/** A bf16x2 register's first element widened to fp32, by a byte permute on the integer pipe. */
[[nodiscard]] __attribute__((device)) inline auto widen_first(Bf16x2 pair) -> float {
	unsigned widened = 0;
	asm("prmt.b32 %0, %1, 0, %2;" : "=r"(widened) : "r"(pair.bits), "n"(widening::first_selector));
	return __builtin_bit_cast(float, widened);
}

/** A bf16x2 register's second element widened to fp32. */
[[nodiscard]] __attribute__((device)) inline auto widen_second(Bf16x2 pair) -> float {
	unsigned widened = 0;
	asm("prmt.b32 %0, %1, 0, %2;" : "=r"(widened) : "r"(pair.bits), "n"(widening::second_selector));
	return __builtin_bit_cast(float, widened);
}

/** Two fp32 rounded to nearest-even bf16 and packed, `first` in the low half (`round_to_bf16` on each). */
[[nodiscard]] __attribute__((device)) inline auto pack_bf16x2(float first, float second) -> Bf16x2 {
	unsigned packed = 0;
	// Why ptx: `cvt.rn.bf16x2.f32` is gated at ptx 9.3; it puts its first source in the upper half.
	asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(second), "f"(first));
	return {.bits = packed};
}

/**
 * Runs `steps` steps with register double buffering: step s + 1's operands are loaded before step s computes, so a
 * load's latency hides under the previous step's math.
 */
template<int steps, class Load, class Compute>
__attribute__((device)) inline void double_buffered(Load load, Compute compute) {
	static_assert(steps >= 1);
	auto current = load(0);
#pragma unroll
	for (int step = 0; step < steps; ++step) {
		decltype(current) following{};
		if (step + 1 < steps) {
			following = load(step + 1);
		}
		compute(step, current);
		current = following;
	}
}

}

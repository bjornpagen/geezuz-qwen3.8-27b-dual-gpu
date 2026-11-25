// SPDX-License-Identifier: AGPL-3.0-only
//
// A bit-exact model of sm_86's `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`: what one tensor core returns for
// D = A B + C, element by element. The kernels' fp32 accumulation runs through it, so the cpu reference, the device
// self-checks and the host emulation of every warp predict the hardware's bits rather than an ieee sum's.
//
// What was measured (an RTX 3060, sm_86, reported identical to the A100; the V100 study first established the
// mechanism with blocks of four and no extra bit):
// - products of bf16 are exact;
// - a block FMA sums 8 products and the addend at once: every term is aligned to the largest exponent among them and
//   truncated (bits shifted out are dropped, no guard or sticky bit) to one bit below fp32's 24-bit significand at that
//   exponent, the aligned terms are summed with carry bits to spare, and only the sum is normalised and truncated to
//   fp32 (round toward zero); subnormal inputs and outputs are kept;
// - a k16 instruction is two block FMAs, (c + T1) + T2, the first block's fp32 result the second's addend.
//
// Assumptions the measurements leave open, each settled by the U6 microbenchmark (crafted operands through the
// hardware mma against this model):
// - the first block holds k 0-7 and the second k 8-15 (the order was measured, not which half is first);
// - truncation is of the magnitude (sign-magnitude, as the V100 study showed), not a two's-complement floor;
// - the alignment exponent is each term's normalised exponent, floor(log2 |term|), not the unnormalised sum of the
//   factors' exponents (they differ when a product's significand reaches 2);
// - the carry bits never overflow: the measured "at least 3" covers 8 products, and the model sums all 9 terms
//   exactly;
// - an exact zero sum is +0 unless every term is a -0;
// - a result beyond fp32's range truncates to the largest finite value of its sign (round toward zero); no kernel
//   reaches it, and infinite or nan operands are outside the model.
export module engine.mma_model;

import std;
import engine.core;
import engine.prims;

namespace engine::mma_model {

namespace prims = engine::device::prims;

export using prims::Bf16;

/** A tensor core's block FMA: products summed per fused block, and alignment bits kept below the significand. */
export struct Datapath {
	int products;
	int extra_alignment_bits;
};

/** sm_86 with bf16 inputs and fp32 output: blocks of 8, one extra alignment bit (a 25-bit window). */
export inline constexpr Datapath sm86 = {.products = 8, .extra_alignment_bits = 1};

/** One term of a block: (-1)^negative * magnitude * 2^exponent, exactly. */
struct Term {
	bool negative;
	std::uint64_t magnitude;
	int exponent;
};

inline constexpr int fp32_significand_bits = 24;
inline constexpr int fp32_least_exponent = -149;
inline constexpr int fp32_greatest_exponent = 127;

[[nodiscard]] constexpr auto finite(Bf16 value) -> bool {
	return (static_cast<std::uint32_t>(value.bits) & 0x7f80U) != 0x7f80U;
}

[[nodiscard]] constexpr auto term(float value) -> Term {
	auto const bits = std::bit_cast<std::uint32_t>(value);
	auto const field = (bits >> 23U) & 0xffU;
	core::invariant(field != 0xffU);
	auto const mantissa = bits & 0x7fffffU;
	auto const negative = (bits & 0x80000000U) != 0;
	if (field == 0) {
		return {.negative = negative, .magnitude = mantissa, .exponent = fp32_least_exponent};
	}
	return {.negative = negative, .magnitude = mantissa | 0x800000U, .exponent = static_cast<int>(field) - 150};
}

[[nodiscard]] constexpr auto product(Bf16 a, Bf16 b) -> Term {
	core::invariant(finite(a) && finite(b));
	auto const x = prims::ptx::exact(a);
	auto const y = prims::ptx::exact(b);
	return {.negative = x.negative != y.negative,
	        .magnitude = std::uint64_t{x.significand} * std::uint64_t{y.significand},
	        .exponent = x.exponent + y.exponent};
}

/** floor(log2 |term|) of a nonzero term. */
[[nodiscard]] constexpr auto top(Term const& term) -> int {
	return std::bit_width(term.magnitude) - 1 + term.exponent;
}

/** The fp32 nearest toward zero to (-1)^negative * magnitude * 2^exponent, magnitude nonzero and below 2^62. */
[[nodiscard]] constexpr auto truncate_to_fp32(bool negative, std::uint64_t magnitude, int exponent) -> float {
	auto const sign = negative ? 0x80000000U : 0U;
	auto const leading = std::bit_width(magnitude) - 1 + exponent;
	if (leading > fp32_greatest_exponent) {
		return std::bit_cast<float>(sign | 0x7f7fffffU);
	}
	auto const quantum = std::max(leading - (fp32_significand_bits - 1), fp32_least_exponent);
	auto kept = magnitude;
	if (quantum > exponent) {
		auto const shift = quantum - exponent;
		kept = shift >= 64 ? 0 : magnitude >> static_cast<unsigned>(shift);
	} else {
		kept = magnitude << static_cast<unsigned>(exponent - quantum);
	}
	if (kept == 0) {
		return std::bit_cast<float>(sign);
	}
	// kept is below 2^24; with 2^23 set the value is normal and its field is quantum + 150, else it is subnormal.
	if (kept >= 0x800000U) {
		auto const field = static_cast<std::uint32_t>(quantum + 150);
		return std::bit_cast<float>(sign | (field << 23U) | (static_cast<std::uint32_t>(kept) & 0x7fffffU));
	}
	return std::bit_cast<float>(sign | static_cast<std::uint32_t>(kept));
}

/**
 * One block FMA: `addend` + sum of `a[i] * b[i]`, every term aligned to the largest and truncated to its window, the
 * aligned sum normalised once and truncated to fp32.
 */
export template<Datapath datapath>
[[nodiscard]] constexpr auto block(float addend, std::span<Bf16 const, static_cast<std::size_t>(datapath.products)> a,
                                   std::span<Bf16 const, static_cast<std::size_t>(datapath.products)> b) -> float {
	static_assert(datapath.products >= 1 && datapath.extra_alignment_bits >= 0);
	auto terms = std::array<Term, static_cast<std::size_t>(datapath.products) + 1>{};
	terms.front() = term(addend);
	for (auto index = std::size_t{0}; index < a.size(); ++index) {
		terms.at(index + 1) = product(a[index], b[index]);
	}
	auto const nonzero = [](Term const& candidate) {
		return candidate.magnitude != 0;
	};
	if (std::ranges::none_of(terms, nonzero)) {
		auto const all_negative = std::ranges::all_of(terms, [](Term const& zero) {
			return zero.negative;
		});
		return all_negative ? -0.0F : 0.0F;
	}
	auto largest = std::numeric_limits<int>::min();
	for (auto const& candidate : terms) {
		if (nonzero(candidate)) {
			largest = std::max(largest, top(candidate));
		}
	}
	// The window's least bit: the fp32 significand at the largest exponent, plus the extra alignment bits.
	auto const least = largest - (fp32_significand_bits - 1) - datapath.extra_alignment_bits;
	auto sum = std::int64_t{0};
	for (auto const& candidate : terms) {
		if (!nonzero(candidate)) {
			continue;
		}
		auto aligned = std::uint64_t{0};
		if (candidate.exponent >= least) {
			aligned = candidate.magnitude << static_cast<unsigned>(candidate.exponent - least);
		} else if (least - candidate.exponent < 64) {
			aligned = candidate.magnitude >> static_cast<unsigned>(least - candidate.exponent);
		}
		auto const signed_aligned = static_cast<std::int64_t>(aligned);
		sum += candidate.negative ? -signed_aligned : signed_aligned;
	}
	if (sum == 0) {
		return 0.0F;
	}
	auto const magnitude = static_cast<std::uint64_t>(sum < 0 ? -sum : sum);
	return truncate_to_fp32(sum < 0, magnitude, least);
}

/** The products of one sm_86 block. */
export inline constexpr std::size_t block_products = static_cast<std::size_t>(sm86.products);

/** The k of one `mma.sync.m16n8k16`. */
export inline constexpr std::size_t instruction_k = 16;

/**
 * A run of mma instructions accumulating in k order from `c`: D for one output element after k / 16 instructions
 * (k / 8 blocks, each block's fp32 result the next one's addend). One instruction is `chain<16>`.
 */
export template<std::size_t k>
[[nodiscard]] constexpr auto chain(float c, std::span<Bf16 const, k> a, std::span<Bf16 const, k> b) -> float {
	static_assert(k % block_products == 0 && k > 0, "the tensor core sums whole blocks");
	auto d = c;
	for (auto first = std::size_t{0}; first < k; first += block_products) {
		d = block<sm86>(d, a.subspan(first).template first<block_products>(), b.subspan(first).template first<block_products>());
	}
	return d;
}

/** One `mma.sync.m16n8k16` element: D = A row . B column + C. */
export [[nodiscard]] constexpr auto step(float c, std::span<Bf16 const, instruction_k> a, std::span<Bf16 const, instruction_k> b) -> float {
	return chain<instruction_k>(c, a, b);
}

/** The A operand as a matrix: 16 rows m, 16 columns k. */
export using MatrixA = std::array<std::array<Bf16, 16>, 16>;
/** The B operand as a matrix: 16 rows k, 8 columns n. */
export using MatrixB = std::array<std::array<Bf16, 8>, 16>;
/** C and D: 16 rows m, 8 columns n. */
export using MatrixC = std::array<std::array<float, 8>, 16>;

/** D = A B + C for one m16n8k16 instruction. */
export [[nodiscard]] constexpr auto mma(MatrixC const& c, MatrixA const& a, MatrixB const& b) -> MatrixC {
	auto d = MatrixC{};
	for (auto row = std::size_t{0}; row < 16; ++row) {
		for (auto column = std::size_t{0}; column < 8; ++column) {
			auto b_column = std::array<Bf16, instruction_k>{};
			for (auto k = std::size_t{0}; k < instruction_k; ++k) {
				b_column.at(k) = b.at(k).at(column);
			}
			d.at(row).at(column) = step(c.at(row).at(column), a.at(row), b_column);
		}
	}
	return d;
}

/**
 * One warp's operands as its 32 lanes hold them in registers (the fragment maps of `prims::fragment`): per lane four
 * bf16x2 words of A, two of B and four fp32 of C, the first element of a pair in the low half.
 */
export struct WarpOperands {
	std::array<std::array<std::uint32_t, 4>, 32> a;
	std::array<std::array<std::uint32_t, 2>, 32> b;
	std::array<std::array<float, 4>, 32> c;
};

/** D per lane, in the accumulator fragment layout. */
export using WarpResult = std::array<std::array<float, 4>, 32>;

[[nodiscard]] constexpr auto half(std::uint32_t word, int which) -> Bf16 {
	return {.bits = static_cast<unsigned short>((word >> (16U * static_cast<std::uint32_t>(which))) & 0xffffU)};
}

/** What one warp's `mma.sync.m16n8k16` returns in each lane: the U6 probe's prediction. */
export [[nodiscard]] constexpr auto warp_mma(WarpOperands const& operands) -> WarpResult {
	auto a = MatrixA{};
	auto b = MatrixB{};
	auto c = MatrixC{};
	for (auto lane = 0; lane < 32; ++lane) {
		auto const at = static_cast<std::size_t>(lane);
		for (auto reg = 0; reg < 4; ++reg) {
			for (auto which = 0; which < 2; ++which) {
				auto const element = prims::fragment::a(lane, reg, which);
				a.at(static_cast<std::size_t>(element.row)).at(static_cast<std::size_t>(element.column)) =
				    half(operands.a.at(at).at(static_cast<std::size_t>(reg)), which);
			}
			auto const accumulator = prims::fragment::c(lane, reg);
			c.at(static_cast<std::size_t>(accumulator.row)).at(static_cast<std::size_t>(accumulator.column)) =
			    operands.c.at(at).at(static_cast<std::size_t>(reg));
		}
		for (auto reg = 0; reg < 2; ++reg) {
			for (auto which = 0; which < 2; ++which) {
				auto const element = prims::fragment::b(lane, reg, which);
				b.at(static_cast<std::size_t>(element.row)).at(static_cast<std::size_t>(element.column)) =
				    half(operands.b.at(at).at(static_cast<std::size_t>(reg)), which);
			}
		}
	}
	auto const d = mma(c, a, b);
	auto result = WarpResult{};
	for (auto lane = 0; lane < 32; ++lane) {
		for (auto reg = 0; reg < 4; ++reg) {
			auto const accumulator = prims::fragment::c(lane, reg);
			result.at(static_cast<std::size_t>(lane)).at(static_cast<std::size_t>(reg)) =
			    d.at(static_cast<std::size_t>(accumulator.row)).at(static_cast<std::size_t>(accumulator.column));
		}
	}
	return result;
}

}

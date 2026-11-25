// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.mma_model;
import engine.prims;
import engine.testing;

namespace {

namespace mma = engine::mma_model;
namespace prims = engine::device::prims;

using mma::Bf16;
using Block = std::array<Bf16, mma::block_products>;

[[nodiscard]] auto bf16(float value) -> Bf16 {
	return prims::round_to_bf16(value);
}

[[nodiscard]] auto wide(Bf16 value) -> double {
	return static_cast<double>(prims::widen(value));
}

/**
 * The block FMA computed a second way, in double precision, from the measured description alone: every term (the
 * products are exact in a double) truncated to the 25-bit window below the largest term's leading bit, the
 * truncated terms summed (exact: they share a quantum and stay below 2^30 of it), the sum truncated toward zero to
 * fp32 by stepping back the nearest fp32 when it lies beyond the sum.
 */
[[nodiscard]] auto oracle(float addend, Block const& a, Block const& b) -> float {
	auto terms = std::vector<double>{static_cast<double>(addend)};
	for (auto index = std::size_t{0}; index < a.size(); ++index) {
		terms.push_back(wide(a.at(index)) * wide(b.at(index)));
	}
	auto largest = std::numeric_limits<int>::min();
	for (auto const term : terms) {
		if (term != 0.0) {
			largest = std::max(largest, std::ilogb(term));
		}
	}
	if (largest == std::numeric_limits<int>::min()) {
		auto const all_negative = std::ranges::all_of(terms, [](double zero) {
			return std::signbit(zero);
		});
		return all_negative ? -0.0F : 0.0F;
	}
	auto const least = largest - 24;
	auto sum = 0.0;
	for (auto const term : terms) {
		sum += std::ldexp(std::trunc(std::ldexp(term, -least)), least);
	}
	if (sum == 0.0) {
		return 0.0F;
	}
	if (std::abs(sum) > static_cast<double>(std::numeric_limits<float>::max())) {
		return std::copysign(std::numeric_limits<float>::max(), static_cast<float>(sum));
	}
	auto nearest = static_cast<float>(sum);
	if (std::abs(static_cast<double>(nearest)) > std::abs(sum)) {
		nearest = std::nextafter(nearest, 0.0F);
	}
	return nearest;
}

[[nodiscard]] auto same_bits(float left, float right) -> bool {
	return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
}

/** A bf16 with a random sign and mantissa, its exponent field drawn from [low, high]; field 0 is a subnormal. */
[[nodiscard]] auto draw_bf16(std::mt19937_64& random, unsigned low, unsigned high) -> Bf16 {
	auto const field = low + static_cast<unsigned>(random() % (high - low + 1));
	auto const bits = static_cast<unsigned>(random() & 0x807fU) | (field << 7U);
	return {.bits = static_cast<unsigned short>(bits)};
}

[[nodiscard]] auto draw_fp32(std::mt19937_64& random, unsigned low, unsigned high) -> float {
	auto const field = low + static_cast<unsigned>(random() % (high - low + 1));
	auto const bits = static_cast<std::uint32_t>(random() & 0x807fffffU) | (field << 23U);
	return std::bit_cast<float>(bits);
}

struct Spread {
	unsigned low;
	unsigned high;
};

/**
 * Random blocks against the double oracle. Narrow exponent spreads force cancellation and carries; wide ones reach
 * subnormal products and terms far below the window; zeros of both signs appear throughout.
 */
[[nodiscard]] auto agrees_with_oracle(std::uint64_t seed, Spread operands, Spread addends) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 200'000; ++trial) {
		auto a = Block{};
		auto b = Block{};
		for (auto index = std::size_t{0}; index < a.size(); ++index) {
			a.at(index) = random() % 16 == 0 ? Bf16{.bits = static_cast<unsigned short>(random() & 0x8000U)}
			                                 : draw_bf16(random, operands.low, operands.high);
			b.at(index) = draw_bf16(random, operands.low, operands.high);
		}
		auto const c = random() % 8 == 0 ? std::bit_cast<float>(static_cast<std::uint32_t>(random() & 0x80000000U))
		                                 : draw_fp32(random, addends.low, addends.high);
		if (!same_bits(mma::block<mma::sm86>(c, a, b), oracle(c, a, b))) {
			return false;
		}
	}
	return true;
}

/** Small integers times one power of two: every term and every sum is exact in the window, so the model is exact. */
[[nodiscard]] auto exact_when_representable(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 100'000; ++trial) {
		auto const scale = static_cast<int>(random() % 60) - 30;
		auto a = std::array<Bf16, 16>{};
		auto b = std::array<Bf16, 16>{};
		auto exact = std::int64_t{0};
		for (auto index = std::size_t{0}; index < 16; ++index) {
			auto const x = static_cast<int>(random() % 255) - 127;
			auto const y = static_cast<int>(random() % 255) - 127;
			a.at(index) = bf16(std::ldexp(static_cast<float>(x), scale));
			b.at(index) = bf16(static_cast<float>(y));
			exact += std::int64_t{x} * y;
		}
		auto const c_integer = static_cast<int>(random() % 4001) - 2000;
		exact += c_integer;
		auto const c = std::ldexp(static_cast<float>(c_integer), scale);
		if (!same_bits(mma::step(c, a, b), std::ldexp(static_cast<float>(exact), scale))) {
			return false;
		}
	}
	return true;
}

/**
 * Terms of one sign: truncation only ever drops magnitude, never adds it, and loses less than one window quantum per
 * term plus one fp32 quantum of the result.
 */
[[nodiscard]] auto truncates_toward_zero(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 100'000; ++trial) {
		auto const negative = random() % 2 == 0;
		auto a = Block{};
		auto b = Block{};
		auto exact = 0.0;
		auto largest = 0.0;
		for (auto index = std::size_t{0}; index < a.size(); ++index) {
			auto const x = draw_bf16(random, 120, 135);
			auto const y = draw_bf16(random, 120, 135);
			a.at(index) = {.bits = static_cast<unsigned short>((x.bits & 0x7fffU) | (negative ? 0x8000U : 0U))};
			b.at(index) = {.bits = static_cast<unsigned short>(y.bits & 0x7fffU)};
			auto const term = wide(a.at(index)) * wide(b.at(index));
			exact += term;
			largest = std::max(largest, std::abs(term));
		}
		auto const result = static_cast<double>(mma::block<mma::sm86>(negative ? -0.0F : 0.0F, a, b));
		auto const window = std::ldexp(1.0, std::ilogb(largest) - 24);
		auto const result_quantum = std::ldexp(1.0, std::ilogb(exact) - 23);
		auto const lost = std::abs(exact) - std::abs(result);
		if (std::signbit(result) != negative || lost < 0.0 || lost >= 8.0 * window + result_quantum) {
			return false;
		}
	}
	return true;
}

/** Within one block the sum is aligned to the largest term and exact before one truncation: order cannot show. */
[[nodiscard]] auto block_ignores_order(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 50'000; ++trial) {
		auto a = Block{};
		auto b = Block{};
		for (auto index = std::size_t{0}; index < a.size(); ++index) {
			a.at(index) = draw_bf16(random, 110, 135);
			b.at(index) = draw_bf16(random, 110, 135);
		}
		auto const c = draw_fp32(random, 110, 140);
		auto const before = mma::block<mma::sm86>(c, a, b);
		auto order = std::array<std::size_t, mma::block_products>{};
		std::ranges::iota(order, std::size_t{0});
		std::ranges::shuffle(order, random);
		auto shuffled_a = Block{};
		auto shuffled_b = Block{};
		for (auto index = std::size_t{0}; index < order.size(); ++index) {
			shuffled_a.at(index) = a.at(order.at(index));
			shuffled_b.at(index) = b.at(order.at(index));
		}
		if (!same_bits(before, mma::block<mma::sm86>(c, shuffled_a, shuffled_b))) {
			return false;
		}
	}
	return true;
}

/**
 * Across blocks the order shows, as measured: (c + T1) + T2 with c = 1, T1 = -1 and T2 = 2^-27 keeps T2, while the
 * other order truncates it away beside c before T1 cancels c.
 */
[[nodiscard]] auto blocks_chain_in_k_order() -> bool {
	auto a = std::array<Bf16, 16>{};
	auto b = std::array<Bf16, 16>{};
	a.fill(bf16(0.0F));
	b.fill(bf16(1.0F));
	a.at(0) = bf16(-1.0F);
	a.at(8) = bf16(0x1p-27F);
	auto swapped = a;
	std::swap(swapped.at(0), swapped.at(8));
	return same_bits(mma::step(1.0F, a, b), 0x1p-27F) && same_bits(mma::step(1.0F, swapped, b), 0.0F);
}

[[nodiscard]] auto one_product(float a, float b) -> std::pair<Block, Block> {
	auto left = Block{};
	auto right = Block{};
	left.fill(bf16(0.0F));
	right.fill(bf16(0.0F));
	left.front() = bf16(a);
	right.front() = bf16(b);
	return {left, right};
}

inline constexpr mma::Datapath v100 = {.products = 4, .extra_alignment_bits = 0};
using V100Block = std::array<Bf16, 4>;

/**
 * The published experiments, replayed: on the V100 datapath (blocks of four, no extra bit) the model returns what
 * that study measured; on sm_86 the extra alignment bit keeps the bit the V100 lost.
 */
[[nodiscard]] auto reproduces_published_experiments() -> bool {
	auto holds = true;
	// The extra alignment bit: 1 + (-1 + 2^-24) is 2^-23 without it and exact with it.
	auto const [one_a, one_b] = one_product(1.0F, 1.0F);
	holds = holds && same_bits(mma::block<mma::sm86>(-1.0F + 0x1p-24F, one_a, one_b), 0x1p-24F);
	auto const unit = V100Block{bf16(1.0F), bf16(0.0F), bf16(0.0F), bf16(0.0F)};
	auto const ones = V100Block{bf16(1.0F), bf16(1.0F), bf16(1.0F), bf16(1.0F)};
	holds = holds && same_bits(mma::block<v100>(-1.0F + 0x1p-24F, unit, ones), 0x1p-23F);
	// Round toward zero: 2 + 3/4 * 2^-22 is 2, and its negation -2.
	auto const [two_a, two_b] = one_product(2.0F, 1.0F);
	auto two_plus = two_a;
	auto two_plus_b = two_b;
	two_plus.at(1) = bf16(0x1.8p-23F);
	two_plus_b.at(1) = bf16(1.0F);
	holds = holds && same_bits(mma::block<mma::sm86>(0.0F, two_plus, two_plus_b), 2.0F);
	auto negated = two_plus_b;
	negated.at(0) = bf16(-1.0F);
	negated.at(1) = bf16(-1.0F);
	holds = holds && same_bits(mma::block<mma::sm86>(0.0F, two_plus, negated), -2.0F);
	// Normalisation once, at the end, and its non-monotonic consequence: four 2^-24 survive beside 1 - 2^-24 but
	// vanish beside 1.
	auto const tiny = V100Block{bf16(0x1p-24F), bf16(0x1p-24F), bf16(0x1p-24F), bf16(0x1p-24F)};
	holds = holds && same_bits(mma::block<v100>(1.0F - 0x1p-24F, tiny, ones), 1.0F + 0x1p-23F);
	holds = holds && same_bits(mma::block<v100>(1.0F, tiny, ones), 1.0F);
	// Carry bits: 1 + 2^-22 + 2^-23 plus 1, 1, 1 and 2^-23 is 4 + 2^-21 in every order.
	auto const carries = V100Block{bf16(1.0F), bf16(1.0F), bf16(1.0F), bf16(0x1p-23F)};
	holds = holds && same_bits(mma::block<v100>(1.0F + 0x1p-22F + 0x1p-23F, carries, ones), 4.0F + 0x1p-21F);
	// Subnormals in and out: a bf16 subnormal times a small normal is an fp32 subnormal, kept exactly.
	auto const [sub_a, sub_b] = one_product(0x1p-133F, 0x1p-10F);
	holds = holds && same_bits(mma::block<mma::sm86>(0.0F, sub_a, sub_b), 0x1p-143F);
	holds = holds && same_bits(mma::block<mma::sm86>(0x1p-149F, one_a, Block{}), 0x1p-149F);
	return holds;
}

[[nodiscard]] auto zero_signs() -> bool {
	auto zeros = Block{};
	auto negative_zeros = Block{};
	zeros.fill(bf16(0.0F));
	negative_zeros.fill(Bf16{.bits = 0x8000U});
	auto const all_negative = mma::block<mma::sm86>(-0.0F, negative_zeros, zeros);
	auto const mixed = mma::block<mma::sm86>(-0.0F, zeros, zeros);
	auto const [one_a, one_b] = one_product(1.0F, 1.0F);
	auto const cancelled = mma::block<mma::sm86>(-1.0F, one_a, one_b);
	return same_bits(all_negative, -0.0F) && same_bits(mixed, 0.0F) && same_bits(cancelled, 0.0F);
}

[[nodiscard]] auto overflow_truncates_to_largest() -> bool {
	auto const [big_a, big_b] = one_product(0x1p100F, 0x1p100F);
	auto negative_b = big_b;
	negative_b.front() = bf16(-0x1p100F);
	return same_bits(mma::block<mma::sm86>(0.0F, big_a, big_b), std::numeric_limits<float>::max()) &&
	       same_bits(mma::block<mma::sm86>(0.0F, big_a, negative_b), -std::numeric_limits<float>::max());
}

/**
 * The warp form places every lane's registers by the fragment maps: small integer operands make D the exact integer
 * product plus C, element by element, wherever each lane's registers put them.
 */
[[nodiscard]] auto warp_form_is_the_matrix_product(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto a = std::array<std::array<int, 16>, 16>{};
	auto b = std::array<std::array<int, 8>, 16>{};
	auto c = std::array<std::array<int, 8>, 16>{};
	for (auto& row : a) {
		std::ranges::generate(row, [&] {
			return static_cast<int>(random() % 31) - 15;
		});
	}
	for (auto& row : b) {
		std::ranges::generate(row, [&] {
			return static_cast<int>(random() % 31) - 15;
		});
	}
	for (auto& row : c) {
		std::ranges::generate(row, [&] {
			return static_cast<int>(random() % 1001) - 500;
		});
	}
	auto const pack = [](int first, int second) {
		return static_cast<std::uint32_t>(bf16(static_cast<float>(first)).bits) |
		       (static_cast<std::uint32_t>(bf16(static_cast<float>(second)).bits) << 16U);
	};
	auto operands = mma::WarpOperands{};
	for (auto lane = 0; lane < 32; ++lane) {
		auto const at = static_cast<std::size_t>(lane);
		for (auto reg = 0; reg < 4; ++reg) {
			auto const low = prims::fragment::a(lane, reg, 0);
			auto const high = prims::fragment::a(lane, reg, 1);
			operands.a.at(at).at(static_cast<std::size_t>(reg)) =
			    pack(a.at(static_cast<std::size_t>(low.row)).at(static_cast<std::size_t>(low.column)),
				     a.at(static_cast<std::size_t>(high.row)).at(static_cast<std::size_t>(high.column)));
			auto const accumulator = prims::fragment::c(lane, reg);
			operands.c.at(at).at(static_cast<std::size_t>(reg)) =
			    static_cast<float>(c.at(static_cast<std::size_t>(accumulator.row)).at(static_cast<std::size_t>(accumulator.column)));
		}
		for (auto reg = 0; reg < 2; ++reg) {
			auto const low = prims::fragment::b(lane, reg, 0);
			auto const high = prims::fragment::b(lane, reg, 1);
			operands.b.at(at).at(static_cast<std::size_t>(reg)) =
			    pack(b.at(static_cast<std::size_t>(low.row)).at(static_cast<std::size_t>(low.column)),
				     b.at(static_cast<std::size_t>(high.row)).at(static_cast<std::size_t>(high.column)));
		}
	}
	auto const result = mma::warp_mma(operands);
	for (auto lane = 0; lane < 32; ++lane) {
		for (auto reg = 0; reg < 4; ++reg) {
			auto const at = prims::fragment::c(lane, reg);
			auto expected = c.at(static_cast<std::size_t>(at.row)).at(static_cast<std::size_t>(at.column));
			for (auto k = std::size_t{0}; k < 16; ++k) {
				expected += a.at(static_cast<std::size_t>(at.row)).at(k) * b.at(k).at(static_cast<std::size_t>(at.column));
			}
			if (!same_bits(result.at(static_cast<std::size_t>(lane)).at(static_cast<std::size_t>(reg)), static_cast<float>(expected))) {
				return false;
			}
		}
	}
	return true;
}

/** The model is usable in constant expressions: the reference and the tests may fold it at compile time. */
constexpr auto folded = [] {
	auto a = std::array<Bf16, 16>{};
	auto b = std::array<Bf16, 16>{};
	a.fill(Bf16{.bits = 0x3f80U});
	b.fill(Bf16{.bits = 0x3f80U});
	return mma::step(0.5F, a, b);
}();
static_assert(folded == 16.5F);

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "block matches the double oracle, cancelling exponents (seed 1)",
		                      .holds = agrees_with_oracle(1, {.low = 120, .high = 135}, {.low = 118, .high = 140})},
	    engine::testing::Case{.name = "block matches the double oracle, wide exponents (seed 2)",
		                      .holds = agrees_with_oracle(2, {.low = 60, .high = 190}, {.low = 1, .high = 254})},
	    engine::testing::Case{.name = "block matches the double oracle, subnormal operands (seed 3)",
		                      .holds = agrees_with_oracle(3, {.low = 0, .high = 40}, {.low = 0, .high = 20})},
	    engine::testing::Case{.name = "step is exact when every term and sum is representable (seed 4)",
		                      .holds = exact_when_representable(4)},
	    engine::testing::Case{.name = "one-signed blocks truncate toward zero within the window (seed 5)",
		                      .holds = truncates_toward_zero(5)},
	    engine::testing::Case{.name = "a block's sum does not depend on product order (seed 6)", .holds = block_ignores_order(6)},
	    engine::testing::Case{.name = "blocks chain as (c + T1) + T2 in k order", .holds = blocks_chain_in_k_order()},
	    engine::testing::Case{.name = "the published tensor-core experiments reproduce", .holds = reproduces_published_experiments()},
	    engine::testing::Case{.name = "zero sums are +0 unless every term is -0", .holds = zero_signs()},
	    engine::testing::Case{.name = "results past fp32 truncate to the largest finite", .holds = overflow_truncates_to_largest()},
	    engine::testing::Case{.name = "the warp form is the matrix product through the fragment maps (seed 7)",
		                      .holds = warp_form_is_the_matrix_product(7)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.sampling;
import engine.testing;

namespace {

using engine::sampling::Block;
using engine::sampling::Key;
using engine::sampling::Philox_key;
using engine::sampling::Purpose;

struct KnownAnswer {
	Block counter;
	Philox_key key;
	Block output;
};

[[nodiscard]] constexpr auto block(std::uint32_t w0, std::uint32_t w1, std::uint32_t w2, std::uint32_t w3) -> Block {
	return Block{.w0 = w0, .w1 = w1, .w2 = w2, .w3 = w3};
}

// Random123's kat_vectors for philox4x32 with 10 rounds.
constexpr auto known_answers = std::array{
    KnownAnswer{.counter = block(0, 0, 0, 0), .key = {.k0 = 0, .k1 = 0}, .output = block(0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)},
    KnownAnswer{
        .counter = block(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
        .key = {.k0 = 0xffffffff, .k1 = 0xffffffff},
        .output = block(0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd),
    },
    KnownAnswer{
        .counter = block(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
        .key = {.k0 = 0xa4093822, .k1 = 0x299f31d0},
        .output = block(0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1),
    },
};

[[nodiscard]] auto same(Block a, Block b) -> bool {
	return a.w0 == b.w0 && a.w1 == b.w1 && a.w2 == b.w2 && a.w3 == b.w3;
}

// Evaluated at run time from values the optimizer cannot see, so the compiled code is checked, not the constant
// evaluator (which the header's own static_asserts check).
[[nodiscard]] auto known_answers_hold_at_run_time() -> bool {
	auto const answers = std::vector(known_answers.begin(), known_answers.end());
	return std::ranges::all_of(answers, [](KnownAnswer const& answer) {
		return same(engine::sampling::philox(answer.counter, answer.key), answer.output);
	});
}

[[nodiscard]] auto random_key(std::mt19937_64& generator) -> Key {
	auto words = std::uniform_int_distribution<std::uint32_t>{};
	auto purposes = std::uniform_int_distribution<int>{0, 3};
	return Key{
	    .seed = generator(),
	    .request = words(generator),
	    .position = words(generator),
	    .step = words(generator),
	    .purpose = static_cast<Purpose>(purposes(generator)),
	};
}

// The uniform is the documented function of the philox block: (position, request, step, purpose) counter, the seed's
// words as key, the top 53 bits of the first two output words as the fraction.
[[nodiscard]] auto uniform_is_the_documented_function(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto holds = true;
	for (auto round = 0; round < 100'000; ++round) {
		auto const key = random_key(generator);
		auto const block = engine::sampling::philox(
		    Block{.w0 = key.position, .w1 = key.request, .w2 = key.step, .w3 = static_cast<std::uint32_t>(key.purpose)},
		    Philox_key{.k0 = static_cast<std::uint32_t>(key.seed), .k1 = static_cast<std::uint32_t>(key.seed >> 32U)});
		auto const fraction = (std::uint64_t{block.w0} << 21U) | (block.w1 >> 11U);
		auto const u = engine::sampling::uniform(key);
		holds = holds && u == std::ldexp(static_cast<double>(fraction), -53) && u >= 0.0 && u < 1.0 &&
		        std::ldexp(u, 53) == std::floor(std::ldexp(u, 53)) && engine::sampling::uniform(key) == u;
	}
	return holds;
}

// Every field of the key feeds the stream: changing any one of them changes the uniform.
[[nodiscard]] auto every_key_field_matters(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto holds = true;
	for (auto round = 0; round < 20'000; ++round) {
		auto const key = random_key(generator);
		auto const u = engine::sampling::uniform(key);
		auto variants = std::array{key, key, key, key, key, key};
		variants[0].seed ^= 1ULL << 40U;
		variants[1].seed ^= 1ULL;
		variants[2].request ^= 1U;
		variants[3].position += 1U;
		variants[4].step += 1U;
		variants[5].purpose = static_cast<Purpose>((static_cast<int>(key.purpose) + 1) % 4);
		holds = holds && std::ranges::none_of(variants, [&](Key const& changed) {
			        return engine::sampling::uniform(changed) == u;
		        });
	}
	return holds;
}

// Consecutive positions under one seed are uniform: a chi-square over 1024 equal buckets with 2^20 draws stays
// under the 1e-6 tail bound (about 1255 for 1023 degrees of freedom).
[[nodiscard]] auto positions_are_uniform(std::uint64_t seed) -> bool {
	auto constexpr buckets = std::size_t{1024};
	auto constexpr draws = std::uint32_t{1U << 20U};
	auto counts = std::vector<double>(buckets, 0.0);
	for (auto position = std::uint32_t{0}; position < draws; ++position) {
		auto const u =
		    engine::sampling::uniform(Key{.seed = seed, .request = 0, .position = position, .step = 0, .purpose = Purpose::Sample});
		counts[static_cast<std::size_t>(u * buckets)] += 1.0;
	}
	auto constexpr expected = double{draws} / buckets;
	auto const statistic = std::ranges::fold_left(counts | std::views::transform([&](double count) {
		                                              return (count - expected) * (count - expected) / expected;
	                                              }),
	                                              0.0, std::plus{});
	return statistic < 1255.0;
}

// The two purposes a rejection uses at one position are independent: their joint 32 x 32 histogram passes chi-square.
[[nodiscard]] auto accept_and_residual_are_independent(std::uint64_t seed) -> bool {
	auto constexpr side = std::size_t{32};
	auto constexpr draws = std::uint32_t{1U << 18U};
	auto counts = std::vector<double>(side * side, 0.0);
	for (auto position = std::uint32_t{0}; position < draws; ++position) {
		auto const base = Key{.seed = seed, .request = 7, .position = position, .step = 2, .purpose = Purpose::Accept};
		auto residual = base;
		residual.purpose = Purpose::Residual;
		auto const row = static_cast<std::size_t>(engine::sampling::uniform(base) * side);
		auto const column = static_cast<std::size_t>(engine::sampling::uniform(residual) * side);
		counts[row * side + column] += 1.0;
	}
	auto constexpr expected = double{draws} / (side * side);
	auto const statistic = std::ranges::fold_left(counts | std::views::transform([&](double count) {
		                                              return (count - expected) * (count - expected) / expected;
	                                              }),
	                                              0.0, std::plus{});
	return statistic < 1255.0;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "philox4x32-10 reproduces Random123's known answers at run time",
		                      .holds = known_answers_hold_at_run_time()},
	    engine::testing::Case{
	        .name = "a uniform is the documented 53-bit function of its key's philox block, in [0, 1), deterministic (seed 3)",
	        .holds = uniform_is_the_documented_function(3),
	    },
	    engine::testing::Case{.name = "changing any key field changes the uniform (seed 4)", .holds = every_key_field_matters(4)},
	    engine::testing::Case{.name = "uniforms over consecutive positions pass chi-square (seed 5)", .holds = positions_are_uniform(5)},
	    engine::testing::Case{.name = "uniforms over consecutive positions pass chi-square (seed 2^63)",
		                      .holds = positions_are_uniform(1ULL << 63U)},
	    engine::testing::Case{
	        .name = "accept and residual uniforms at one position are jointly uniform (seed 6)",
	        .holds = accept_and_residual_are_independent(6),
	    },
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The card's image digest, emulated on the host through the kernels' own steps, against the engine's SHA-256; the
// slicing the soak launches by; and the pattern's values.
import std;
import engine.core;
import engine.sha256;
import engine.soak;
import engine.testing;

namespace {

namespace soak = engine::soak;
namespace sha256 = engine::sha256;

[[nodiscard]] auto random_message(std::mt19937_64& random, std::size_t words) -> std::vector<std::byte> {
	auto bytes = std::vector<std::byte>(words * 4);
	std::ranges::generate(bytes, [&] { return std::byte{static_cast<std::uint8_t>(random() & 0xffU)}; });
	return bytes;
}

/** Every message length of up to four blocks, word by word, so every tail length meets every block count. */
[[nodiscard]] auto every_short_length_matches(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto words = std::size_t{0}; words <= 64; ++words) {
		auto const message = random_message(random, words);
		for (auto const per_slice : {1U, 2U, 3U, soak::slice_blocks}) {
			holds = holds && soak::emulate(message, per_slice) == sha256::digest(message);
		}
	}
	return holds;
}

/** Random messages up to a few hundred blocks, absorbed in random slice sizes. */
[[nodiscard]] auto random_messages_match(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 300; ++trial) {
		auto const words = std::uniform_int_distribution<std::size_t>{0, 6000}(random);
		auto const per_slice = std::uniform_int_distribution<unsigned>{1, 40}(random);
		auto const message = random_message(random, words);
		holds = holds && soak::emulate(message, per_slice) == sha256::digest(message);
	}
	return holds;
}

/** A message one bit away from another hashes differently, wherever the bit is: the digest sees every byte it is given. */
[[nodiscard]] auto every_flipped_bit_changes_the_digest(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const message = random_message(random, 77);
	auto const clean = soak::emulate(message, 1);
	auto holds = true;
	for (auto bit = std::size_t{0}; bit < message.size() * 8; ++bit) {
		auto flipped = message;
		flipped.at(bit / 8) ^= std::byte{static_cast<std::uint8_t>(1U << (bit % 8))};
		holds = holds && soak::emulate(flipped, 1) != clean;
	}
	return holds;
}

/** The slices of a pass cover its blocks in order, each block once, none longer than asked, and the tail is the rest. */
[[nodiscard]] auto slices_tile_the_pass(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 2000; ++trial) {
		auto const words = std::uniform_int_distribution<std::uint64_t>{0, std::uint64_t{1} << 34U}(random);
		auto const per_slice = std::uniform_int_distribution<unsigned>{1, soak::slice_blocks}(random);
		auto const pass = soak::pass_of(words * 4);
		auto next = std::uint64_t{0};
		auto const count = soak::slices(pass, per_slice);
		for (auto index = std::uint64_t{0}; index < count && holds; ++index) {
			auto const part = soak::slice(pass, per_slice, index);
			holds = part.first == next && part.count >= 1 && part.count <= per_slice;
			next += part.count;
		}
		holds = holds && next == pass.blocks && pass.blocks * soak::block_bytes + std::uint64_t{pass.tail_words} * 4 == words * 4 &&
		        pass.tail_words < 16;
	}
	return holds;
}

/** Within a round every word of a 2^16-chunk span holds its own value, for rounds from several pairs. */
[[nodiscard]] auto pattern_words_are_distinct_within_a_round() -> bool {
	auto holds = true;
	for (auto const round : {0U, 1U, 6U, 7U, 1000U}) {
		auto seen = std::vector<std::uint32_t>{};
		seen.reserve(std::size_t{1} << 18U);
		for (auto chunk = 0U; chunk < (1U << 16U); ++chunk) {
			for (auto word = 0U; word < 4U; ++word) {
				seen.push_back(soak::pattern_word(round, chunk, word));
			}
		}
		std::ranges::sort(seen);
		holds = holds && std::ranges::adjacent_find(seen) == seen.end();
	}
	return holds;
}

/**
 * An odd round stores the inverse of the round before it, so over the pair every bit of every cell holds both values;
 * the next pair stores other values.
 */
[[nodiscard]] auto rounds_pair_as_inverses(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	auto unchanged = 0;
	for (auto trial = 0; trial < 100000; ++trial) {
		auto const pair = std::uniform_int_distribution<std::uint32_t>{0, 1U << 20U}(random);
		auto const chunk = static_cast<std::uint32_t>(std::uniform_int_distribution<std::uint64_t>{0, soak::region_chunks - 1}(random));
		auto const word = std::uniform_int_distribution<std::uint32_t>{0, 3}(random);
		auto const even = soak::pattern_word(2 * pair, chunk, word);
		holds = holds && soak::pattern_word(2 * pair + 1, chunk, word) == ~even;
		unchanged += soak::pattern_word(2 * pair + 2, chunk, word) == even ? 1 : 0;
	}
	return holds && unchanged < 10;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every message of 0 to 64 words, in slices of 1, 2, 3 and the soak's own blocks, emulates to its SHA-256 (seed 3)",
	        .holds = every_short_length_matches(3),
	    },
	    engine::testing::Case{
	        .name = "random messages up to 6000 words in random slices emulate to their SHA-256 (seed 41)",
	        .holds = random_messages_match(41),
	    },
	    engine::testing::Case{
	        .name = "flipping any one bit of a 77-word message changes its emulated digest (seed 9)",
	        .holds = every_flipped_bit_changes_the_digest(9),
	    },
	    engine::testing::Case{
	        .name = "a pass's slices cover its blocks once each in order, and the tail is the rest (seed 17)",
	        .holds = slices_tile_the_pass(17),
	    },
	    engine::testing::Case{
	        .name = "within a round, every word of 65536 chunks holds its own value (rounds 0, 1, 6, 7, 1000)",
	        .holds = pattern_words_are_distinct_within_a_round(),
	    },
	    engine::testing::Case{
	        .name = "odd rounds store the inverse of the round before; the next pair stores other values (seed 23)",
	        .holds = rounds_pair_as_inverses(23),
	    },
	};
	return engine::testing::report(cases);
}

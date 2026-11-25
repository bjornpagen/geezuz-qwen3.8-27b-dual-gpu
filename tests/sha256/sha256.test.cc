// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.sha256;
import engine.testing;

namespace {

namespace sha256 = engine::sha256;

using sha256::Digest;

// The standard's own examples and NIST's long-message vectors.
constexpr auto empty_digest = sha256::pinned("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
constexpr auto abc_digest = sha256::pinned("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
constexpr auto two_block_text = std::string_view{"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
constexpr auto two_block_digest = sha256::pinned("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
constexpr auto long_text =
    std::string_view{"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"};
constexpr auto long_digest = sha256::pinned("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
constexpr auto million_a_digest = sha256::pinned("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
// 2^24 repetitions of this 64-byte pattern: one GiB.
constexpr auto gigabyte_pattern = std::string_view{"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"};
constexpr auto gigabyte_repetitions = std::size_t{1} << 24U;
constexpr auto gigabyte_digest = sha256::pinned("50e72a0e26442fe2552dc3938ac58658228c0cbfb1d2ca872ae435266fcd055e");

// The portable rounds the build runs agree with the standard.
static_assert(sha256::digest(std::string_view{}) == empty_digest);
static_assert(sha256::digest("abc") == abc_digest);
static_assert(sha256::digest(two_block_text) == two_block_digest);
static_assert(sha256::digest(long_text) == long_digest);

[[nodiscard]] auto bytes_of(std::string_view text) -> std::vector<std::byte> {
	auto bytes = std::vector<std::byte>(text.size());
	std::ranges::transform(text, bytes.begin(), [](char character) { return std::bit_cast<std::byte>(character); });
	return bytes;
}

[[nodiscard]] auto text_vectors_hold() -> bool {
	return sha256::digest(std::string_view{}) == empty_digest && sha256::digest("abc") == abc_digest &&
	       sha256::digest(two_block_text) == two_block_digest && sha256::digest(long_text) == long_digest;
}

[[nodiscard]] auto million_a_holds() -> bool {
	auto const message = std::vector<std::byte>(1'000'000, std::bit_cast<std::byte>('a'));
	return sha256::digest(message) == million_a_digest;
}

[[nodiscard]] auto gigabyte_holds() -> bool {
	// Fed in 4 MiB pieces of whole patterns: the pattern is one block, so every piece is whole blocks.
	auto const pattern = bytes_of(gigabyte_pattern);
	auto piece = std::vector<std::byte>{};
	constexpr auto per_piece = std::size_t{1} << 16U;
	piece.reserve(per_piece * pattern.size());
	for (auto index = std::size_t{0}; index < per_piece; ++index) {
		piece.insert(piece.end(), pattern.begin(), pattern.end());
	}
	auto hasher = sha256::Hasher{};
	for (auto index = std::size_t{0}; index < gigabyte_repetitions / per_piece; ++index) {
		hasher.update(piece);
	}
	return std::move(hasher).finish() == gigabyte_digest;
}

// A message the build and the run both generate, from its length alone.
[[nodiscard]] constexpr auto mix(std::uint64_t value) -> std::uint64_t {
	value += 0x9e3779b97f4a7c15U;
	value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9U;
	value = (value ^ (value >> 27U)) * 0x94d049bb133111ebU;
	return value ^ (value >> 31U);
}

template<std::size_t length>
[[nodiscard]] consteval auto message() -> std::array<std::byte, length> {
	auto bytes = std::array<std::byte, length>{};
	for (auto index = std::size_t{0}; index < length; ++index) {
		bytes.at(index) = std::byte{static_cast<std::uint8_t>(mix((length << 32U) + index) & 0xffU)};
	}
	return bytes;
}

// The digest the build computes with the portable rounds: the oracle for the machine's instructions.
template<std::size_t length>
constexpr Digest built_digest = sha256::digest(std::span<std::byte const>{message<length>()});

/** Every length up to three blocks and a half, so every padding case and block boundary occurs. */
constexpr auto oracle_lengths = std::size_t{224};

[[nodiscard]] auto run_digest_matches_build_digest() -> bool {
	return []<std::size_t... length>(std::index_sequence<length...>) {
		return ((sha256::digest(std::span<std::byte const>{message<length>()}) == built_digest<length>) && ...);
	}(std::make_index_sequence<oracle_lengths>{});
}

[[nodiscard]] auto random_message(std::mt19937_64& random, std::size_t length) -> std::vector<std::byte> {
	auto bytes = std::vector<std::byte>(length);
	std::ranges::generate(bytes, [&random] { return std::byte{static_cast<std::uint8_t>(random() & 0xffU)}; });
	return bytes;
}

/** Hashing a message cut at any points gives the digest of the whole: cuts are drawn uniformly, so empty pieces and pieces inside one block occur. */
[[nodiscard]] auto split_points_do_not_matter(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 300; ++trial) {
		auto const length = std::uniform_int_distribution<std::size_t>{0, 3000}(random);
		auto const message = random_message(random, length);
		auto cuts = std::vector<std::size_t>(std::uniform_int_distribution<std::size_t>{0, 12}(random));
		std::ranges::generate(cuts, [&] { return std::uniform_int_distribution<std::size_t>{0, length}(random); });
		cuts.push_back(0);
		cuts.push_back(length);
		std::ranges::sort(cuts);
		auto hasher = sha256::Hasher{};
		for (auto const [begin, end] : cuts | std::views::pairwise) {
			hasher.update(std::span{message}.subspan(begin, end - begin));
		}
		if (std::move(hasher).finish() != sha256::digest(message)) {
			return false;
		}
	}
	return true;
}

/** Text and its bytes hash alike. */
[[nodiscard]] auto text_hashes_as_its_bytes(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 200; ++trial) {
		auto const bytes = random_message(random, std::uniform_int_distribution<std::size_t>{0, 700}(random));
		auto text = std::string(bytes.size(), '\0');
		std::ranges::transform(bytes, text.begin(), [](std::byte byte) { return std::bit_cast<char>(byte); });
		if (sha256::digest(std::string_view{text}) != sha256::digest(bytes)) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto hex_round_trips(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 500; ++trial) {
		auto digest = Digest{};
		std::ranges::generate(digest.bytes, [&random] { return std::byte{static_cast<std::uint8_t>(random() & 0xffU)}; });
		auto const lower = sha256::hex(digest.bytes);
		auto upper = std::string{lower.begin(), lower.end()};
		std::ranges::transform(upper, upper.begin(), [](char digit) { return digit >= 'a' && digit <= 'f' ? static_cast<char>(digit - 'a' + 'A') : digit; });
		auto const formatted = std::format("{}", digest);
		if (sha256::parse(std::string_view{lower.data(), lower.size()}) != digest || sha256::parse(upper) != digest ||
		    formatted != std::string_view{lower.data(), lower.size()}) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto hex_refuses_wrong_lengths() -> bool {
	for (auto length = std::size_t{0}; length <= 130; ++length) {
		auto const parsed = sha256::parse(std::string(length, '0'));
		auto const expect_ok = length == 2 * sha256::digest_bytes;
		if (parsed.has_value() != expect_ok || (!expect_ok && parsed.error().failure != sha256::HexFailure::WrongLength)) {
			return false;
		}
	}
	return true;
}

/** Every character outside the hex digits, at every position, is refused; every digit is accepted there. */
[[nodiscard]] auto hex_refuses_every_non_digit() -> bool {
	constexpr auto digits = std::string_view{"0123456789abcdefABCDEF"};
	for (auto position = std::size_t{0}; position < 2 * sha256::digest_bytes; ++position) {
		for (auto code = 0; code < 256; ++code) {
			auto text = std::string(2 * sha256::digest_bytes, '7');
			text.at(position) = std::bit_cast<char>(static_cast<std::uint8_t>(code));
			auto const is_digit = digits.contains(text.at(position));
			auto const parsed = sha256::parse(text);
			if (parsed.has_value() != is_digit || (!is_digit && parsed.error().failure != sha256::HexFailure::NotHex)) {
				return false;
			}
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the standard's text vectors hold at run time", .holds = text_vectors_hold()},
	    engine::testing::Case{.name = "a million 'a' hashes to NIST's digest", .holds = million_a_holds()},
	    engine::testing::Case{.name = "one GiB of NIST's long-message pattern hashes to its digest", .holds = gigabyte_holds()},
	    engine::testing::Case{.name = "run-time digests equal the build's portable digests at every length below 224",
	                          .holds = run_digest_matches_build_digest()},
	    engine::testing::Case{.name = "split points do not change the digest (seed 1)", .holds = split_points_do_not_matter(1)},
	    engine::testing::Case{.name = "split points do not change the digest (seed 2)", .holds = split_points_do_not_matter(2)},
	    engine::testing::Case{.name = "split points do not change the digest (seed 3)", .holds = split_points_do_not_matter(3)},
	    engine::testing::Case{.name = "text hashes as its bytes (seed 4)", .holds = text_hashes_as_its_bytes(4)},
	    engine::testing::Case{.name = "hex round-trips in either case and formats lowercase (seed 5)", .holds = hex_round_trips(5)},
	    engine::testing::Case{.name = "hex of the wrong length is refused", .holds = hex_refuses_wrong_lengths()},
	    engine::testing::Case{.name = "every non-digit character is refused at every position", .holds = hex_refuses_every_non_digit()},
	};
	return engine::testing::report(cases);
}

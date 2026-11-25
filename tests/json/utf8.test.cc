// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.json_model;
import engine.testing;

namespace {

namespace json = engine::json;
namespace model = engine::json_model;

using json::ParseFault;
using Reader = json::Reader<json::request_bounds>;

[[nodiscard]] auto make_reader() -> std::unique_ptr<Reader> {
	auto reader = Reader::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<Reader>(std::move(*reader));
}

[[nodiscard]] auto byte(std::uint32_t value) -> char {
	return std::bit_cast<char>(static_cast<std::uint8_t>(value));
}

/**
 * Raw bytes inside a string read back as themselves exactly when the independent decoder calls them well-formed
 * UTF-8 and none is a quote or a control; when they are refused for their encoding alone, the fault is invalid
 * UTF-8. Backslashes are left out of the search: they begin escapes, which the escape cases cover.
 */
[[nodiscard]] auto agrees(Reader& reader, std::string_view bytes) -> bool {
	auto const special = std::ranges::any_of(bytes, [](char each) { return each == '"' || std::bit_cast<std::uint8_t>(each) < 0x20; });
	auto const well_formed = model::decode_utf8(bytes).has_value();
	auto const read = model::read_model(reader, "\"" + std::string{bytes} + "\"");
	if (well_formed && !special) {
		return read && std::get<std::string>(read->value) == bytes;
	}
	if (read) {
		return false;
	}
	return special || model::fault_of(read.error()) == ParseFault::InvalidUtf8;
}

/** Every one- and two-byte string, and every three-byte string with a lead byte of 0xC0 or more. */
[[nodiscard]] auto short_sequences_exhaustive() -> bool {
	auto reader = make_reader();
	for (auto first = std::uint32_t{0}; first < 256; ++first) {
		if (first == '\\') {
			continue;
		}
		if (!agrees(*reader, std::string{byte(first)})) {
			return false;
		}
		for (auto second = std::uint32_t{0}; second < 256; ++second) {
			if (second == '\\') {
				continue;
			}
			if (!agrees(*reader, std::string{byte(first), byte(second)})) {
				std::println("disagreement at {:02X} {:02X}", first, second);
				return false;
			}
		}
	}
	for (auto first = std::uint32_t{0xC0}; first < 256; ++first) {
		for (auto second = std::uint32_t{0}; second < 256; ++second) {
			for (auto third = std::uint32_t{0}; third < 256; ++third) {
				if (second == '\\' || third == '\\') {
					continue;
				}
				if (!agrees(*reader, std::string{byte(first), byte(second), byte(third)})) {
					std::println("disagreement at {:02X} {:02X} {:02X}", first, second, third);
					return false;
				}
			}
		}
	}
	return true;
}

/** Four-byte sequences: every lead and second byte, with continuation bytes drawn around their boundaries. */
[[nodiscard]] auto long_sequences(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto constexpr edges = std::array<std::uint32_t, 6>{0x7F, 0x80, 0x8F, 0x90, 0xBF, 0xC0};
	for (auto first = std::uint32_t{0xE0}; first < 256; ++first) {
		for (auto second = std::uint32_t{0x70}; second < 0xD0; ++second) {
			for (auto draw = 0; draw < 4; ++draw) {
				auto const third = (rng() % 2) == 0 ? edges[rng() % edges.size()] : static_cast<std::uint32_t>(0x60 + rng() % 0x80);
				auto const fourth = (rng() % 2) == 0 ? edges[rng() % edges.size()] : static_cast<std::uint32_t>(0x60 + rng() % 0x80);
				auto const bytes = std::string{byte(first), byte(second), byte(third), byte(fourth)};
				if (bytes.find('\\') == std::string::npos && !agrees(*reader, bytes)) {
					return false;
				}
			}
		}
	}
	return true;
}

[[nodiscard]] auto unit(std::uint32_t value) -> std::string {
	return std::format("\\u{:04x}", value);
}

/** Every scalar value, spelled as a `\u` escape or surrogate pair, reads as its UTF-8 encoding. */
[[nodiscard]] auto every_escape_reads_as_its_scalar() -> bool {
	auto reader = make_reader();
	for (auto value = std::uint32_t{0}; value <= 0x10FFFF; ++value) {
		if (value >= 0xD800 && value <= 0xDFFF) {
			continue;
		}
		auto const spelled = value < 0x10000 ? unit(value) : unit(0xD800 + ((value - 0x10000) >> 10U)) + unit(0xDC00 + ((value - 0x10000) & 0x3FFU));
		auto const read = model::read_model(*reader, "\"" + spelled + "\"");
		if (!read || std::get<std::string>(read->value) != model::utf8(char32_t{value})) {
			std::println("escape of U+{:04X} misread", value);
			return false;
		}
	}
	return true;
}

/** A surrogate escape that is not the high half of a high-low pair is refused as an invalid string, never replaced. */
[[nodiscard]] auto lone_surrogates_refused(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto const refused = [&reader](std::string const& spelled) {
		auto const read = model::read_model(*reader, "\"" + spelled + "\"");
		return !read && model::fault_of(read.error()) == ParseFault::InvalidString;
	};
	for (auto value = std::uint32_t{0xD800}; value <= 0xDFFF; ++value) {
		if (!refused(unit(value)) || !refused(unit(value) + "x")) {
			return false;
		}
		auto const after = static_cast<std::uint32_t>(rng() % 0x10000);
		// A high half followed by anything but a low half; a low half followed by anything at all.
		if ((value <= 0xDBFF ? (after < 0xDC00 || after > 0xDFFF) : true) && !refused(unit(value) + unit(after))) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "short byte strings agree with the independent UTF-8 decoder", .holds = short_sequences_exhaustive()},
	    engine::testing::Case{.name = "four-byte strings agree with the independent UTF-8 decoder (seed 1)", .holds = long_sequences(1)},
	    engine::testing::Case{.name = "every scalar value's escape reads as its UTF-8", .holds = every_escape_reads_as_its_scalar()},
	    engine::testing::Case{.name = "lone surrogate escapes are refused (seed 2)", .holds = lone_surrogates_refused(2)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.json_model;
import engine.testing;

namespace {

namespace json = engine::json;
namespace model = engine::json_model;

using Reader = json::Reader<json::request_bounds>;

[[nodiscard]] auto make_reader() -> std::unique_ptr<Reader> {
	auto reader = Reader::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<Reader>(std::move(*reader));
}

[[nodiscard]] auto read_number(Reader& reader, std::string_view literal) -> engine::core::Result<model::Model, json::ReadError> {
	return model::read_model(reader, literal);
}

[[nodiscard]] auto bits(double value) -> std::uint64_t {
	return std::bit_cast<std::uint64_t>(value);
}

[[nodiscard]] auto is_real(engine::core::Result<model::Model, json::ReadError> const& read, double expected) -> bool {
	return read && std::holds_alternative<double>(read->value) && bits(std::get<double>(read->value)) == bits(expected);
}

[[nodiscard]] auto is_integer(engine::core::Result<model::Model, json::ReadError> const& read, std::int64_t expected) -> bool {
	return read && std::holds_alternative<std::int64_t>(read->value) && std::get<std::int64_t>(read->value) == expected;
}

/** Every integer within 4096 of either end of the 64-bit range reads exactly; one past either end reads as a real. */
[[nodiscard]] auto integer_edges() -> bool {
	auto reader = make_reader();
	auto constexpr low = std::numeric_limits<std::int64_t>::min();
	auto constexpr high = std::numeric_limits<std::int64_t>::max();
	for (auto offset = std::int64_t{0}; offset < 4096; ++offset) {
		if (!is_integer(read_number(*reader, std::to_string(low + offset)), low + offset) ||
		    !is_integer(read_number(*reader, std::to_string(high - offset)), high - offset)) {
			return false;
		}
	}
	return is_real(read_number(*reader, "9223372036854775808"), 0x1p63) && is_real(read_number(*reader, "-9223372036854775809"), -0x1p63) &&
	       is_real(read_number(*reader, "18446744073709551615"), 0x1p64) && is_integer(read_number(*reader, "-0"), 0) &&
	       is_real(read_number(*reader, "-0.0"), -0.0) && is_real(read_number(*reader, "1e-400"), 0.0) &&
	       !read_number(*reader, "1e400").has_value() && !read_number(*reader, "-1e400").has_value();
}

/** Random 64-bit integers read exactly, in every magnitude. */
[[nodiscard]] auto random_integers(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const value = std::bit_cast<std::int64_t>(rng()) / static_cast<std::int64_t>(std::uint64_t{1} << (rng() % 63));
		if (!is_integer(read_number(*reader, std::to_string(value)), value)) {
			return false;
		}
	}
	return true;
}

/**
 * Integer literals beyond 64 bits read as the nearest double. The oracle is libc++'s correctly rounded
 * `from_chars`, a second implementation of the same rounding.
 */
[[nodiscard]] auto big_integers_round(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto literal = std::string{(rng() % 2) == 0 ? "" : "-"};
		literal += static_cast<char>('1' + rng() % 9);
		for (auto digits = 19 + rng() % 60; digits > 0; --digits) {
			literal += static_cast<char>('0' + rng() % 10);
		}
		auto expected = 0.0;
		std::ignore = std::from_chars(literal.data(), std::to_address(literal.end()), expected);
		if (!is_real(read_number(*reader, literal), expected)) {
			std::println("big integer {} misread", literal);
			return false;
		}
	}
	return true;
}

/** Every finite double reads back from its shortest round-trip text and from its 17-digit scientific text. */
[[nodiscard]] auto doubles_round_trip(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const value = model::random_finite(rng);
		auto shortest = model::number_text(model::Model{.value = value});
		auto digits = std::array<char, 40>{};
		auto const written = std::to_chars(digits.data(), std::to_address(digits.end()), value, std::chars_format::scientific, 16);
		auto const scientific = std::string{digits.data(), written.ptr};
		if (!is_real(read_number(*reader, shortest), value) || !is_real(read_number(*reader, scientific), value)) {
			std::println("double {} misread", shortest);
			return false;
		}
	}
	return true;
}

/** The exact decimal expansion of a double with `places` digits after the point (enough places make it exact). */
[[nodiscard]] auto exact_decimal(double value, int places) -> std::string {
	auto digits = std::array<char, 512>{};
	auto const written = std::to_chars(digits.data(), std::to_address(digits.end()), value, std::chars_format::fixed, places);
	return std::string{digits.data(), written.ptr};
}

/** The sum of two non-negative decimals written with the same number of places. */
[[nodiscard]] auto add_decimals(std::string left, std::string right) -> std::string {
	auto const width = std::max(left.size(), right.size());
	left.insert(0, width - left.size(), '0');
	right.insert(0, width - right.size(), '0');
	auto sum = std::string(width, '0');
	auto carry = 0;
	for (auto index = width; index-- > 0;) {
		if (left[index] == '.') {
			sum[index] = '.';
			continue;
		}
		auto const digit = (left[index] - '0') + (right[index] - '0') + carry;
		sum[index] = static_cast<char>('0' + digit % 10);
		carry = digit / 10;
	}
	if (carry > 0) {
		sum.insert(0, "1");
	}
	while (sum.size() > 1 && sum.front() == '0' && sum[1] != '.') {
		sum.erase(0, 1);
	}
	return sum;
}

/** The decimal one unit in its last place smaller. */
[[nodiscard]] auto decrement_last(std::string decimal) -> std::string {
	for (auto index = decimal.size(); index-- > 0;) {
		if (decimal[index] == '.') {
			continue;
		}
		if (decimal[index] != '0') {
			--decimal[index];
			return decimal;
		}
		decimal[index] = '9';
	}
	return decimal;
}

/**
 * Halfway cases decide correct rounding. For a double `x` and its successor, the midpoint's exact decimal reads as
 * whichever of the two has an even significand, and a hair above or below it reads as the successor or `x`. The
 * midpoint is computed exactly in decimal: `x` plus half its unit in the last place, both printed exactly.
 */
[[nodiscard]] auto halfway_cases_round_to_even(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const exponent = static_cast<int>(rng() % 121) - 60;
		auto const x = std::ldexp(1.0 + static_cast<double>(rng() >> 12U) * 0x1p-52, exponent);
		auto const next = std::nextafter(x, std::numeric_limits<double>::infinity());
		auto const last_bit = std::ilogb(x) - 52;
		auto const places = std::max(0, 1 - last_bit);
		auto const midpoint = add_decimals(exact_decimal(x, places), exact_decimal(std::ldexp(1.0, last_bit - 1), places));
		auto const above = places == 0 ? midpoint + ".1" : midpoint + "1";
		auto const below = places == 0 ? decrement_last(midpoint) + ".9" : decrement_last(midpoint) + "9";
		auto const even = (bits(x) & 1U) == 0 ? x : next;
		auto const negate = (rng() % 2) == 0;
		auto const sign = negate ? std::string{"-"} : std::string{};
		auto const signed_value = [negate](double value) { return negate ? -value : value; };
		auto const exact = places == 0 ? midpoint + ".0" : midpoint;
		if (!is_real(read_number(*reader, sign + exact), signed_value(even)) || !is_real(read_number(*reader, sign + above), signed_value(next)) ||
		    !is_real(read_number(*reader, sign + below), signed_value(x))) {
			std::println("halfway case misread near {} ({})", x, exact);
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "integers at the 64-bit edges read exactly; past them, as reals", .holds = integer_edges()},
	    engine::testing::Case{.name = "random 64-bit integers read exactly (seed 1)", .holds = random_integers(1, 100000)},
	    engine::testing::Case{.name = "integers beyond 64 bits read as the nearest double (seed 2)", .holds = big_integers_round(2, 20000)},
	    engine::testing::Case{.name = "doubles read back from shortest and 17-digit texts (seed 3)", .holds = doubles_round_trip(3, 200000)},
	    engine::testing::Case{.name = "halfway decimals round to even, and a hair off rounds away (seed 4)", .holds = halfway_cases_round_to_even(4, 50000)},
	};
	return engine::testing::report(cases);
}

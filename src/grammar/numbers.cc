// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:numbers;

import std;
import engine.core;
import engine.regex;

namespace engine::grammar {

/**
 * How the engine writes JSON numbers: at most this many digits before the point and after it, and an exponent of at
 * most two digits. Within these every literal is a finite double (below 1e120) and at most 47 characters, inside
 * every reader's number bound, so a number the grammar allows is one every reader reads back.
 */
export inline constexpr std::uint32_t max_integer_digits = 20;
export inline constexpr std::uint32_t max_fraction_digits = 20;
export inline constexpr std::uint32_t max_exponent_digits = 2;

/** An exact decimal: (-1)^negative × digits × 10^exponent, digits without leading zeros ("" for zero). */
struct Decimal {
	bool negative;
	std::string digits;
	std::int64_t exponent;
};

/** Reads a JSON number literal exactly; nothing for text that is no literal. */
[[nodiscard]] auto read_decimal(std::string_view literal) -> std::optional<Decimal> {
	auto out = Decimal{.negative = false, .digits = {}, .exponent = 0};
	auto at = std::size_t{0};
	if (literal.starts_with('-')) {
		out.negative = true;
		++at;
	}
	auto seen = false;
	auto point = false;
	while (at < literal.size()) {
		auto const c = literal[at];
		if (c >= '0' && c <= '9') {
			out.digits.push_back(c);
			if (point) {
				--out.exponent;
			}
			seen = true;
		} else if (c == '.' && !point) {
			point = true;
		} else {
			break;
		}
		++at;
	}
	if (!seen) {
		return std::nullopt;
	}
	if (at < literal.size()) {
		if (literal[at] != 'e' && literal[at] != 'E') {
			return std::nullopt;
		}
		++at;
		auto sign = std::int64_t{1};
		if (at < literal.size() && (literal[at] == '+' || literal[at] == '-')) {
			sign = literal[at] == '-' ? -1 : 1;
			++at;
		}
		auto magnitude = std::int64_t{0};
		auto any = false;
		while (at < literal.size() && literal[at] >= '0' && literal[at] <= '9') {
			// Past a thousand, every bound is beyond the numbers the engine writes either way.
			magnitude = std::min<std::int64_t>(magnitude * 10 + (literal[at] - '0'), 100000);
			any = true;
			++at;
		}
		if (!any || at != literal.size()) {
			return std::nullopt;
		}
		out.exponent += sign * magnitude;
	}
	auto const first = out.digits.find_first_not_of('0');
	out.digits = first == std::string::npos ? std::string{} : out.digits.substr(first);
	while (!out.digits.empty() && out.digits.back() == '0') {
		out.digits.pop_back();
		++out.exponent;
	}
	if (out.digits.empty()) {
		out.negative = false;
		out.exponent = 0;
	}
	return out;
}

/** The integer and fraction digits of a decimal's magnitude, if its magnitude has at most `limit` integer digits. */
struct Split {
	std::string whole;
	std::string fraction;
};

[[nodiscard]] auto split(Decimal const& value) -> std::optional<Split> {
	if (value.digits.empty()) {
		return Split{.whole = "0", .fraction = {}};
	}
	auto const length = core::narrow<std::int64_t>(value.digits.size());
	auto const whole_length = length + value.exponent;
	if (whole_length > std::int64_t{max_integer_digits} + 1 || -value.exponent > 4000) {
		return std::nullopt;
	}
	if (value.exponent >= 0) {
		return Split{.whole = value.digits + std::string(core::narrow<std::size_t>(value.exponent), '0'), .fraction = {}};
	}
	if (whole_length <= 0) {
		return Split{.whole = "0", .fraction = std::string(core::narrow<std::size_t>(-whole_length), '0') + value.digits};
	}
	auto const cut = core::narrow<std::size_t>(whole_length);
	return Split{.whole = value.digits.substr(0, cut), .fraction = value.digits.substr(cut)};
}

[[nodiscard]] auto digit(regex::Store& store, char low, char high) -> regex::Expr {
	return store.bytes(regex::ByteSet::range(std::bit_cast<std::uint8_t>(low), std::bit_cast<std::uint8_t>(high)));
}

[[nodiscard]] auto digits(regex::Store& store, std::uint32_t min, std::uint32_t max) -> regex::Expr {
	return store.repeat(digit(store, '0', '9'), min, max);
}

/** Digit strings of `low.size()` digits between `low` and `high` inclusive, compared as strings of equal length. */
[[nodiscard]] auto same_length(regex::Store& store, std::string_view low, std::string_view high) -> regex::Expr {
	if (low.empty()) {
		return regex::empty;
	}
	if (low.front() == high.front()) {
		return store.concat(store.byte(std::bit_cast<std::uint8_t>(low.front())), same_length(store, low.substr(1), high.substr(1)));
	}
	auto const rest = core::narrow<std::uint32_t>(low.size() - 1);
	// A range whose tails span every digit string is one digit range and free digits; without this the two
	// branches below would each recurse at every digit, exponentially in the length.
	if (low.substr(1).find_first_not_of('0') == std::string_view::npos && high.substr(1).find_first_not_of('9') == std::string_view::npos) {
		return store.concat(digit(store, low.front(), high.front()), digits(store, rest, rest));
	}
	auto const nines = std::string(rest, '9');
	auto const zeros = std::string(rest, '0');
	auto choices = std::vector<regex::Expr>{
	    store.concat(store.byte(std::bit_cast<std::uint8_t>(low.front())), same_length(store, low.substr(1), nines)),
	    store.concat(store.byte(std::bit_cast<std::uint8_t>(high.front())), same_length(store, zeros, high.substr(1))),
	};
	if (high.front() - low.front() > 1) {
		choices.push_back(store.concat(digit(store, static_cast<char>(low.front() + 1), static_cast<char>(high.front() - 1)), digits(store, rest, rest)));
	}
	return store.any_of(choices);
}

/** Decimal digit strings without leading zeros (but "0") for the integers `low` through `high`, both nonnegative, as strings. */
[[nodiscard]] auto natural_range(regex::Store& store, std::string const& low, std::string const& high) -> regex::Expr {
	auto choices = std::vector<regex::Expr>{};
	for (auto length = low.size(); length <= high.size(); ++length) {
		auto const from = length == low.size() ? low : (length == 1 ? std::string{"0"} : "1" + std::string(length - 1, '0'));
		auto const to = length == high.size() ? high : std::string(length, '9');
		choices.push_back(same_length(store, from, to));
	}
	return store.any_of(choices);
}

/** Adds one to a digit string. */
[[nodiscard]] auto increment(std::string value) -> std::string {
	for (auto& c : value | std::views::reverse) {
		if (c != '9') {
			++c;
			return value;
		}
		c = '0';
	}
	return "1" + value;
}

/** Subtracts one from a positive digit string. */
[[nodiscard]] auto decrement(std::string value) -> std::string {
	for (auto& c : value | std::views::reverse) {
		if (c != '0') {
			--c;
			break;
		}
		c = '9';
	}
	auto const first = value.find_first_not_of('0');
	return first == std::string::npos ? std::string{"0"} : value.substr(first);
}

/** A signed integer bound as its sign and magnitude digits. */
struct Integer {
	bool negative;
	std::string magnitude;
};

[[nodiscard]] auto less(Integer const& a, Integer const& b) -> bool {
	auto const magnitude_less = [](std::string const& x, std::string const& y) { return x.size() != y.size() ? x.size() < y.size() : x < y; };
	if (a.negative != b.negative) {
		return a.negative;
	}
	return a.negative ? magnitude_less(b.magnitude, a.magnitude) : magnitude_less(a.magnitude, b.magnitude);
}

/** The integer just above or below a decimal: the least integer ≥ it (`Round::Up`) or the greatest ≤ it. */
enum class Round : std::uint8_t {
	Up,
	Down,
};

[[nodiscard]] auto to_integer(Split const& split, bool negative, Round round) -> Integer {
	auto const fractional = split.fraction.find_first_not_of('0') != std::string::npos;
	// Rounding a magnitude away from zero is up for a positive value and down for a negative one.
	auto const away = fractional && ((round == Round::Up) != negative);
	auto magnitude = away ? increment(split.whole) : split.whole;
	if (magnitude == "0") {
		negative = false;
	}
	return Integer{.negative = negative, .magnitude = std::move(magnitude)};
}

/** The largest magnitude an integer of `max_integer_digits` digits has. */
[[nodiscard]] auto widest() -> std::string {
	auto out = std::string(max_integer_digits, '9');
	return out;
}

/** A bound on a number as the schema writes it. */
export struct Bound {
	std::string literal;
	/** Whether the bound itself is excluded. */
	bool exclusive;
};

/** The integers within the bounds, as JSON writes them (no "-0"), within `max_integer_digits` digits. Nothing when none is. */
[[nodiscard]] auto integer_range(regex::Store& store, std::optional<Bound> const& minimum, std::optional<Bound> const& maximum)
    -> std::optional<regex::Expr> {
	auto low = Integer{.negative = true, .magnitude = widest()};
	auto high = Integer{.negative = false, .magnitude = widest()};
	auto const clamp = [](Integer value) {
		if (value.magnitude.size() > max_integer_digits) {
			value.magnitude = widest();
		}
		return value;
	};
	if (minimum) {
		auto const decimal = read_decimal(minimum->literal).value();
		auto const parts = split(decimal);
		if (!parts) {
			// Beyond every integer the engine writes: none above a huge positive bound, all above a huge negative one.
			if (!decimal.negative) {
				return std::nullopt;
			}
		} else {
			auto bound = to_integer(*parts, decimal.negative, Round::Up);
			auto const integral = parts->fraction.find_first_not_of('0') == std::string::npos;
			if (minimum->exclusive && integral) {
				bound = bound.negative ? Integer{.negative = bound.magnitude != "1", .magnitude = decrement(bound.magnitude)}
				                       : Integer{.negative = false, .magnitude = increment(bound.magnitude)};
			}
			if (bound.magnitude.size() > max_integer_digits && !bound.negative) {
				return std::nullopt;
			}
			low = clamp(bound);
		}
	}
	if (maximum) {
		auto const decimal = read_decimal(maximum->literal).value();
		auto const parts = split(decimal);
		if (!parts) {
			if (decimal.negative) {
				return std::nullopt;
			}
		} else {
			auto bound = to_integer(*parts, decimal.negative, Round::Down);
			auto const integral = parts->fraction.find_first_not_of('0') == std::string::npos;
			if (maximum->exclusive && integral) {
				bound = !bound.negative && bound.magnitude != "0" ? Integer{.negative = false, .magnitude = decrement(bound.magnitude)}
				                                                  : Integer{.negative = true, .magnitude = increment(bound.magnitude)};
			}
			if (bound.magnitude.size() > max_integer_digits && bound.negative) {
				return std::nullopt;
			}
			high = clamp(bound);
		}
	}
	if (less(high, low)) {
		return std::nullopt;
	}
	auto choices = std::vector<regex::Expr>{};
	if (!high.negative) {
		auto const from = low.negative ? std::string{"0"} : low.magnitude;
		choices.push_back(natural_range(store, from, high.magnitude));
	}
	if (low.negative) {
		auto const nearest = high.negative ? high.magnitude : std::string{"1"};
		choices.push_back(store.concat(store.byte('-'), natural_range(store, nearest, low.magnitude)));
	}
	return store.any_of(choices);
}

/** A run of fraction digits `s` (up to `budget` long, possibly empty) with 0.s ≥ 0.f. */
[[nodiscard]] auto fraction_at_least(regex::Store& store, std::string_view f, std::uint32_t budget) -> regex::Expr {
	if (f.find_first_not_of('0') == std::string_view::npos) {
		return digits(store, 0, budget);
	}
	if (budget == 0) {
		return regex::nothing;
	}
	auto choices = std::vector<regex::Expr>{store.concat(store.byte(std::bit_cast<std::uint8_t>(f.front())), fraction_at_least(store, f.substr(1), budget - 1))};
	if (f.front() < '9') {
		choices.push_back(store.concat(digit(store, static_cast<char>(f.front() + 1), '9'), digits(store, 0, budget - 1)));
	}
	return store.any_of(choices);
}

/** A run of fraction digits `s` (up to `budget` long, possibly empty) with 0.s ≤ 0.f. */
[[nodiscard]] auto fraction_at_most(regex::Store& store, std::string_view f, std::uint32_t budget) -> regex::Expr {
	if (f.empty()) {
		return store.repeat(store.byte('0'), 0, budget);
	}
	if (budget == 0) {
		return regex::empty;
	}
	auto choices = std::vector<regex::Expr>{regex::empty, store.concat(store.byte(std::bit_cast<std::uint8_t>(f.front())), fraction_at_most(store, f.substr(1), budget - 1))};
	if (f.front() > '0') {
		choices.push_back(store.concat(digit(store, '0', static_cast<char>(f.front() - 1)), digits(store, 0, budget - 1)));
	}
	return store.any_of(choices);
}

/** A magnitude written `whole[.fraction]`, whole without leading zeros, given its whole and fraction digit languages. */
[[nodiscard]] auto magnitude(regex::Store& store, regex::Expr whole, regex::Expr fraction_digits) -> regex::Expr {
	auto const nonempty = store.both(fraction_digits, digits(store, 1, max_fraction_digits));
	auto const fraction = store.either(store.both(fraction_digits, regex::empty), store.concat(store.byte('.'), nonempty));
	return store.concat(whole, fraction);
}

/** Magnitudes at least (`Round::Up`) or at most (`Round::Down`) the split decimal `bound`. */
[[nodiscard]] auto magnitudes(regex::Store& store, Split const& bound, Round side) -> regex::Expr {
	auto const any_fraction = digits(store, 0, max_fraction_digits);
	auto const whole = store.literal(bound.whole);
	if (side == Round::Up) {
		auto const above = magnitude(store, natural_range(store, increment(bound.whole), widest()), any_fraction);
		auto const equal = magnitude(store, whole, fraction_at_least(store, bound.fraction, max_fraction_digits));
		return store.either(above, equal);
	}
	auto const below = bound.whole == "0" ? regex::nothing : magnitude(store, natural_range(store, "0", decrement(bound.whole)), any_fraction);
	auto const equal = magnitude(store, whole, fraction_at_most(store, bound.fraction, max_fraction_digits));
	return store.either(below, equal);
}

/** Every magnitude, written as the engine writes one without exponent. */
[[nodiscard]] auto any_magnitude(regex::Store& store) -> regex::Expr {
	return magnitude(store, natural_range(store, "0", widest()), digits(store, 0, max_fraction_digits));
}

/** Every magnitude that is zero: "0", "0.0", "0.00", .... */
[[nodiscard]] auto zero_magnitude(regex::Store& store) -> regex::Expr {
	return store.concat(store.byte('0'), store.optional(store.concat(store.byte('.'), store.repeat(store.byte('0'), 1, max_fraction_digits))));
}

/** Numbers as the engine writes them: sign, whole, optional fraction and exponent, no negative zero. */
[[nodiscard]] auto any_number(regex::Store& store) -> regex::Expr {
	auto const exponent = store.optional(store.sequence(std::array{
	    store.bytes(regex::ByteSet::range('E', 'E').united(regex::ByteSet::range('e', 'e'))),
	    store.optional(store.bytes(regex::ByteSet::range('+', '+').united(regex::ByteSet::range('-', '-')))),
	    digits(store, 1, max_exponent_digits),
	}));
	auto const all = any_magnitude(store);
	auto const nonzero = store.both(all, store.complement(zero_magnitude(store)));
	return store.either(store.concat(all, exponent), store.sequence(std::array{store.byte('-'), nonzero, exponent}));
}

/** Whether a decimal has at most `limit` significant digits. */
[[nodiscard]] auto significant_within(Decimal const& value, std::size_t limit) -> bool {
	return value.digits.size() <= limit;
}

/** Why real bounds could not be compiled. */
enum class RealFault : std::uint8_t {
	/** An exclusive bound past 15 significant digits: no double comparison could keep it exact. */
	Precision,
	/** No number the engine writes lies within the bounds. */
	Empty,
};

/** A magnitude written equal to `value`: its digits, then any run of trailing zeros in the fraction. */
[[nodiscard]] auto equal_magnitude(regex::Store& store, Split const& value) -> regex::Expr {
	auto const zeros = store.repeat(store.byte('0'), 0, max_fraction_digits);
	if (value.fraction.empty()) {
		return store.concat(store.literal(value.whole), store.optional(store.concat(store.byte('.'), store.concat(store.byte('0'), zeros))));
	}
	return store.sequence(std::array{store.literal(value.whole), store.byte('.'), store.literal(value.fraction), zeros});
}

/**
 * Reals within the bounds, as the engine writes them, without exponent. An exclusive bound limits the numbers
 * written to 15 significant digits (16 characters of digits and point), where distinct decimals stay distinct
 * doubles, so a reader comparing doubles agrees with the decimal comparison made here.
 */
[[nodiscard]] auto real_range(regex::Store& store, std::optional<Bound> const& minimum, std::optional<Bound> const& maximum)
    -> std::expected<regex::Expr, RealFault> {
	auto const all = any_magnitude(store);
	auto const zero = zero_magnitude(store);
	auto const negative = [&store, zero](regex::Expr magnitudes_expr) {
		return store.concat(store.byte('-'), store.both(magnitudes_expr, store.complement(zero)));
	};
	auto numbers = store.either(all, negative(all));
	auto exclusive = false;
	for (auto const& [bound, side] : {std::pair{minimum, Round::Up}, std::pair{maximum, Round::Down}}) {
		if (!bound) {
			continue;
		}
		auto const decimal = read_decimal(bound->literal).value();
		auto const parts = split(decimal);
		auto const below_all = decimal.negative;
		if (!parts) {
			// Past every number written: a lower bound below them all or an upper bound above them all binds nothing.
			if ((side == Round::Up) == below_all) {
				continue;
			}
			return std::unexpected(RealFault::Empty);
		}
		auto within = regex::nothing;
		if (decimal.digits.empty()) {
			within = side == Round::Up ? all : store.either(store.both(all, zero), negative(all));
		} else if (!decimal.negative) {
			within = side == Round::Up ? magnitudes(store, *parts, Round::Up) : store.either(magnitudes(store, *parts, Round::Down), negative(all));
		} else {
			within = side == Round::Up ? store.either(all, negative(magnitudes(store, *parts, Round::Down))) : negative(magnitudes(store, *parts, Round::Up));
		}
		if (bound->exclusive) {
			if (!significant_within(decimal, 15)) {
				return std::unexpected(RealFault::Precision);
			}
			exclusive = true;
			auto const same = equal_magnitude(store, *parts);
			within = store.both(within, store.complement(decimal.negative ? store.concat(store.byte('-'), same) : same));
		}
		numbers = store.both(numbers, within);
	}
	if (exclusive) {
		auto const digit_or_point = store.bytes(regex::ByteSet::range('0', '9').united(regex::ByteSet::range('.', '.')));
		numbers = store.both(numbers, store.concat(store.optional(store.byte('-')), store.repeat(digit_or_point, 1, 16)));
	}
	return numbers;
}

}

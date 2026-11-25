// SPDX-License-Identifier: AGPL-3.0-only
export module engine.regex:utf8;

import std;
import engine.core;
import :expr;

namespace engine::regex {

/** The code points past the last scalar value. */
export inline constexpr char32_t scalar_end = 0x110000;

/** Code points `low` through `high`, inclusive. */
export struct Range {
	char32_t low;
	char32_t high;

	[[nodiscard]] constexpr auto operator==(Range const&) const -> bool = default;
};

/**
 * A set of Unicode scalar values as sorted, disjoint, non-adjacent ranges. Surrogates are never members: utf-8
 * cannot encode them, so a class of characters never holds them.
 */
export class CharClass {
public:
	[[nodiscard]] static auto of(std::span<Range const> ranges) -> CharClass {
		auto out = CharClass{};
		for (auto const range : ranges) {
			out.add(range);
		}
		return out;
	}

	[[nodiscard]] static auto single(char32_t point) -> CharClass {
		auto const range = std::array{Range{.low = point, .high = point}};
		return of(range);
	}

	[[nodiscard]] static auto all() -> CharClass {
		auto const range = std::array{Range{.low = 0, .high = scalar_end - 1}};
		return of(range);
	}

	auto add(Range range) -> void {
		core::invariant(range.low <= range.high && range.high < scalar_end);
		// Surrogates are cut out of every range on entry.
		if (range.low < 0xd800 && range.high >= 0xd800) {
			insert(Range{.low = range.low, .high = 0xd7ff});
			if (range.high > 0xdfff) {
				insert(Range{.low = 0xe000, .high = range.high});
			}
			return;
		}
		if (range.low >= 0xd800 && range.low <= 0xdfff) {
			if (range.high <= 0xdfff) {
				return;
			}
			range.low = 0xe000;
		}
		insert(range);
	}

	auto add(CharClass const& other) -> void {
		for (auto const range : other.ranges_) {
			insert(range);
		}
	}

	[[nodiscard]] auto complemented() const -> CharClass {
		auto out = CharClass{};
		auto next = char32_t{0};
		for (auto const range : ranges_) {
			if (range.low > next) {
				out.add(Range{.low = next, .high = range.low - 1});
			}
			next = range.high + 1;
		}
		if (next < scalar_end) {
			out.add(Range{.low = next, .high = scalar_end - 1});
		}
		return out;
	}

	[[nodiscard]] auto minus(CharClass const& other) const -> CharClass {
		auto out = complemented();
		out.add(other);
		return out.complemented();
	}

	[[nodiscard]] auto contains(char32_t point) const -> bool {
		return std::ranges::any_of(ranges_, [point](Range range) { return range.low <= point && point <= range.high; });
	}

	[[nodiscard]] auto empty() const -> bool {
		return ranges_.empty();
	}

	[[nodiscard]] auto ranges() const -> std::span<Range const> {
		return ranges_;
	}

	[[nodiscard]] auto operator==(CharClass const&) const -> bool = default;

private:
	auto insert(Range range) -> void {
		auto merged = std::vector<Range>{};
		merged.reserve(ranges_.size() + 1);
		auto placed = false;
		for (auto const existing : ranges_) {
			if (existing.high + 1 < range.low) {
				merged.push_back(existing);
			} else if (range.high + 1 < existing.low) {
				if (!placed) {
					merged.push_back(range);
					placed = true;
				}
				merged.push_back(existing);
			} else {
				range = Range{.low = std::min(range.low, existing.low), .high = std::max(range.high, existing.high)};
			}
		}
		if (!placed) {
			merged.push_back(range);
		}
		ranges_ = std::move(merged);
	}

	std::vector<Range> ranges_;
};

/** The utf-8 encoding of a scalar value. */
export [[nodiscard]] constexpr auto encode(char32_t point) -> std::string {
	auto out = std::string{};
	auto const put = [&out](std::uint32_t byte) { out.push_back(static_cast<char>(static_cast<std::uint8_t>(byte))); };
	auto const value = std::uint32_t{point};
	if (value < 0x80) {
		put(value);
	} else if (value < 0x800) {
		put(0xc0U | (value >> 6U));
		put(0x80U | (value & 0x3fU));
	} else if (value < 0x10000) {
		put(0xe0U | (value >> 12U));
		put(0x80U | ((value >> 6U) & 0x3fU));
		put(0x80U | (value & 0x3fU));
	} else {
		put(0xf0U | (value >> 18U));
		put(0x80U | ((value >> 12U) & 0x3fU));
		put(0x80U | ((value >> 6U) & 0x3fU));
		put(0x80U | (value & 0x3fU));
	}
	return out;
}

[[nodiscard]] constexpr auto encoded_length(char32_t point) -> std::size_t {
	return point < 0x80 ? 1 : point < 0x800 ? 2 : point < 0x10000 ? 3 : 4;
}

/**
 * Appends the byte sequences of scalars `low` through `high`, all of one encoded length: the range is split until
 * every split's encodings, byte by byte, span whole ranges, so each is one sequence of byte sets.
 */
auto add_sequences(Store& store, std::vector<Expr>& out, char32_t low, char32_t high) -> void {
	auto const length = encoded_length(low);
	core::invariant(length == encoded_length(high));
	for (auto suffix = std::size_t{1}; suffix < length; ++suffix) {
		auto const mask = (char32_t{1} << (6U * suffix)) - 1;
		if ((low & ~mask) != (high & ~mask)) {
			if ((low & mask) != 0) {
				add_sequences(store, out, low, low | mask);
				add_sequences(store, out, (low | mask) + 1, high);
				return;
			}
			if ((high & mask) != mask) {
				add_sequences(store, out, low, (high & ~mask) - 1);
				add_sequences(store, out, high & ~mask, high);
				return;
			}
		}
	}
	auto const first = encode(low);
	auto const last = encode(high);
	auto parts = std::vector<Expr>{};
	for (auto index = std::size_t{0}; index < length; ++index) {
		parts.push_back(store.bytes(ByteSet::range(std::bit_cast<std::uint8_t>(first[index]), std::bit_cast<std::uint8_t>(last[index]))));
	}
	out.push_back(store.sequence(parts));
}

/** The language of one character of the class: its members' utf-8 encodings. */
export [[nodiscard]] auto characters(Store& store, CharClass const& set) -> Expr {
	auto sequences = std::vector<Expr>{};
	for (auto const range : set.ranges()) {
		auto low = range.low;
		for (auto const end : {char32_t{0x7f}, char32_t{0x7ff}, char32_t{0xffff}, char32_t{0x10ffff}}) {
			if (low > range.high) {
				break;
			}
			if (low > end) {
				continue;
			}
			add_sequences(store, sequences, low, std::min(range.high, end));
			low = end + 1;
		}
	}
	return store.any_of(sequences);
}

/** Any one character: every utf-8 encoded scalar value. */
export [[nodiscard]] auto any_character(Store& store) -> Expr {
	return characters(store, CharClass::all());
}

/** Any utf-8 text: a run of whole characters. */
export [[nodiscard]] auto any_text(Store& store) -> Expr {
	return store.repeat(any_character(store), 0, unbounded);
}

}

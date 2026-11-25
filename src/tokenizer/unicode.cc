// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:unicode;

import std;
import engine.core;

namespace engine::tokenizer {

/**
 * The general categories the pretokenizer's pattern distinguishes: letters (`\p{L}`), marks (`\p{M}`), numbers
 * (`\p{N}`), whitespace (`\s`), and everything else. The four sets are disjoint.
 */
enum class Class : std::uint8_t {
	Other,
	Letter,
	Mark,
	Number,
	Whitespace,
};

inline constexpr std::uint8_t class_count = 5;

/** A run of codepoints sharing one value, from `start` up to the next run's start. */
template<class V>
struct Run {
	char32_t start;
	V value;
};

/** A canonical composition: `first` followed by `second` composes to `composite`. */
struct Composition {
	char32_t first;
	char32_t second;
	char32_t composite;

	[[nodiscard]] constexpr auto key() const -> std::pair<char32_t, char32_t> {
		return {first, second};
	}
};

/** A full canonical decomposition: `key` decomposes to `values[begin, end)` of its table. */
struct Decomposition {
	char32_t key;
	std::uint32_t begin;
	std::uint32_t end;
};

inline constexpr char32_t scalar_limit = 0x110000;

/**
 * The unicode facts the tokenizer reads, as Hugging Face's own regex engine and normalizer hold them: their unicode
 * versions differ from each other and from the platform's, and only theirs reproduce the reference token ids.
 */
struct Unicode {
	std::vector<Run<Class>> classes;
	std::vector<Run<std::uint8_t>> combining;
	std::vector<Decomposition> decompositions;
	std::vector<char32_t> decomposed;
	std::vector<Composition> compositions;
	/**
	 * Below this codepoint every character is a starter that neither decomposes nor composes with anything, so text
	 * made only of such characters is already in form C. Derived from the tables when they are read.
	 */
	char32_t stable_below;
};

template<class V>
[[nodiscard]] auto lookup(std::vector<Run<V>> const& runs, char32_t point) -> V {
	auto const after = std::ranges::upper_bound(runs, point, {}, &Run<V>::start);
	return std::prev(after)->value;
}

[[nodiscard]] auto class_of(Unicode const& unicode, char32_t point) -> Class {
	return lookup(unicode.classes, point);
}

[[nodiscard]] auto combining_class(Unicode const& unicode, char32_t point) -> std::uint8_t {
	return lookup(unicode.combining, point);
}

// Hangul syllables decompose and compose by arithmetic, not by table.
inline constexpr char32_t hangul_s = 0xac00;
inline constexpr char32_t hangul_l = 0x1100;
inline constexpr char32_t hangul_v = 0x1161;
inline constexpr char32_t hangul_t = 0x11a7;
inline constexpr char32_t hangul_l_count = 19;
inline constexpr char32_t hangul_v_count = 21;
inline constexpr char32_t hangul_t_count = 28;
inline constexpr char32_t hangul_n_count = hangul_v_count * hangul_t_count;
inline constexpr char32_t hangul_s_count = hangul_l_count * hangul_n_count;

[[nodiscard]] constexpr auto is_hangul_syllable(char32_t point) -> bool {
	return point >= hangul_s && point < hangul_s + hangul_s_count;
}

[[nodiscard]] auto decomposition(Unicode const& unicode, char32_t point) -> std::optional<std::span<char32_t const>> {
	auto const found = std::ranges::lower_bound(unicode.decompositions, point, {}, &Decomposition::key);
	if (found == unicode.decompositions.end() || found->key != point) {
		return std::nullopt;
	}
	return std::span{unicode.decomposed}.subspan(found->begin, found->end - found->begin);
}

[[nodiscard]] auto compose(Unicode const& unicode, char32_t first, char32_t second) -> std::optional<char32_t> {
	if (first >= hangul_l && first < hangul_l + hangul_l_count && second >= hangul_v && second < hangul_v + hangul_v_count) {
		return hangul_s + ((first - hangul_l) * hangul_v_count + (second - hangul_v)) * hangul_t_count;
	}
	if (is_hangul_syllable(first) && (first - hangul_s) % hangul_t_count == 0 && second > hangul_t && second < hangul_t + hangul_t_count) {
		return first + (second - hangul_t);
	}
	auto const key = std::pair{first, second};
	auto const found = std::ranges::lower_bound(unicode.compositions, key, {}, &Composition::key);
	if (found == unicode.compositions.end() || found->key() != key) {
		return std::nullopt;
	}
	return found->composite;
}

/** Composes canonically ordered, fully decomposed text (the canonical composition algorithm). */
[[nodiscard]] auto composed(Unicode const& unicode, std::vector<char32_t> text) -> std::vector<char32_t> {
	auto out = std::size_t{0};
	auto starter = std::optional<std::size_t>{};
	// The class of the last character kept after the starter; nullopt while the starter is the last one kept, which
	// is the one position where a character with class 0 may still compose with it.
	auto last = std::optional<std::uint8_t>{};
	for (auto at = std::size_t{0}; at < text.size(); ++at) {
		auto const point = text[at];
		auto const klass = combining_class(unicode, point);
		if (starter && (!last || *last < klass)) {
			if (auto const composite = compose(unicode, text[*starter], point)) {
				text[*starter] = *composite;
				continue;
			}
		}
		if (klass == 0) {
			starter = out;
			last.reset();
		} else if (starter) {
			last = klass;
		}
		text[out] = point;
		++out;
	}
	text.resize(out);
	return text;
}

/** Normalization form C, as Hugging Face's NFC normalizer computes it. */
[[nodiscard]] auto nfc(Unicode const& unicode, std::span<char32_t const> text) -> std::vector<char32_t> {
	if (std::ranges::all_of(text, [&](char32_t point) {
		    return point < unicode.stable_below;
	    })) {
		return {text.begin(), text.end()};
	}
	auto out = std::vector<char32_t>{};
	out.reserve(text.size() + text.size() / 2);
	for (auto const point : text) {
		if (is_hangul_syllable(point)) {
			auto const index = point - hangul_s;
			out.push_back(hangul_l + index / hangul_n_count);
			out.push_back(hangul_v + (index % hangul_n_count) / hangul_t_count);
			if (index % hangul_t_count != 0) {
				out.push_back(hangul_t + index % hangul_t_count);
			}
		} else if (auto const parts = decomposition(unicode, point)) {
			out.insert(out.end(), parts->begin(), parts->end());
		} else {
			out.push_back(point);
		}
	}
	// Canonical ordering: every maximal run of non-starters is sorted by class, keeping equal classes in order.
	auto run = out.begin();
	while (run != out.end()) {
		auto const begin = std::find_if(run, out.end(), [&](char32_t point) {
			return combining_class(unicode, point) != 0;
		});
		auto const end = std::find_if(begin, out.end(), [&](char32_t point) {
			return combining_class(unicode, point) == 0;
		});
		std::stable_sort(begin, end, [&](char32_t left, char32_t right) {
			return combining_class(unicode, left) < combining_class(unicode, right);
		});
		run = end;
	}
	return composed(unicode, std::move(out));
}

}

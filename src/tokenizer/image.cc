// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:image;

import std;
import engine.core;
import :added;
import :bpe;
import :unicode;

namespace engine::tokenizer {

/** What is wrong with a tokenizer image. Every defect is permanent: the image is built into the binary. */
export enum class Defect : std::uint8_t {
	/** Not a tokenizer image. */
	Magic,
	/** A format version this build does not read. */
	Version,
	/** Its vocabulary or merge count is not this model's. */
	Shape,
	/** A unicode table or the token bytes exceed their named maxima. */
	Bound,
	/** The length differs from the one its header implies. */
	Size,
	/** The token table is malformed: an empty or overlong token, or not exactly one token per single byte. */
	Token,
	/** A merge names an unknown token, its result is not the concatenation of its pair, or a pair repeats. */
	Merge,
	/** A unicode property table does not cover every codepoint with ascending runs of valid values. */
	Runs,
	/** The decomposition table is unordered or holds a non-scalar value. */
	Decomposition,
	/** The composition table is unordered or holds a non-scalar value. */
	Composition,
};

export struct [[nodiscard]] LoadError {
	Defect defect;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (defect) {
		case Defect::Magic:
		case Defect::Version:
		case Defect::Shape:
		case Defect::Bound:
		case Defect::Size:
		case Defect::Token:
		case Defect::Merge:
		case Defect::Runs:
		case Defect::Decomposition:
		case Defect::Composition:
			return core::Retry::Permanent;
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(LoadError const&) const -> bool = default;
};

/** The image format version this build reads. */
inline constexpr std::uint32_t image_version = 1;
inline constexpr std::string_view image_magic = "gzqwntok";
inline constexpr std::size_t header_words = 9;
inline constexpr std::size_t header_size = image_magic.size() + header_words * 4;

// Named maxima for the tables the model's shape does not fix, a few times their present size: a corrupt header
// cannot make the reader allocate more than this.
inline constexpr std::size_t max_token_bytes = 256;
inline constexpr std::size_t max_class_runs = 8192;
inline constexpr std::size_t max_combining_runs = 4096;
inline constexpr std::size_t max_decompositions = 8192;
inline constexpr std::size_t max_decomposed = 32768;
inline constexpr std::size_t max_compositions = 4096;

/** Everything the tokenizer reads from its image, checked. */
struct Tables {
	/** Every token's bytes, concatenated in id order; token i ends at ends[i]. */
	std::string bytes;
	std::vector<std::uint32_t> ends;
	/** The token of each single byte. */
	std::array<Id, 256> byte_tokens;
	Merges merges;
	Unicode unicode;

	[[nodiscard]] auto token(Id id) const -> std::string_view {
		auto const begin = id == 0 ? std::uint32_t{0} : ends[id - 1];
		return std::string_view{bytes}.substr(begin, ends[id] - begin);
	}
};

/** Reads little-endian words from the front of the image, in order. */
class Reader {
public:
	explicit Reader(std::span<std::byte const> image) : image_{image} {}

	[[nodiscard]] auto words(std::size_t count) -> std::vector<std::uint32_t> {
		auto out = std::vector<std::uint32_t>{};
		out.reserve(count);
		for (auto index = std::size_t{0}; index < count; ++index) {
			auto value = std::uint32_t{0};
			for (auto shift = 0U; shift < 32U; shift += 8U) {
				value |= std::to_integer<std::uint32_t>(image_[at_]) << shift;
				++at_;
			}
			out.push_back(value);
		}
		return out;
	}

	/** Raw bytes, then padding up to the next word. */
	[[nodiscard]] auto text(std::size_t count) -> std::string {
		auto out = std::string{};
		out.reserve(count);
		for (auto const byte : image_.subspan(at_, count)) {
			out.push_back(std::bit_cast<char>(byte));
		}
		at_ += padded(count);
		return out;
	}

	[[nodiscard]] static constexpr auto padded(std::size_t count) -> std::size_t {
		return (count + 3) / 4 * 4;
	}

private:
	std::span<std::byte const> image_;
	std::size_t at_ = 0;
};

struct Header {
	std::uint32_t tokens;
	std::uint32_t token_bytes;
	std::uint32_t merges;
	std::uint32_t class_runs;
	std::uint32_t combining_runs;
	std::uint32_t decompositions;
	std::uint32_t decomposed;
	std::uint32_t compositions;

	/** The image length this header implies. */
	[[nodiscard]] constexpr auto size() const -> std::size_t {
		auto const words = std::size_t{tokens} + std::size_t{merges} * 3 + class_runs + combining_runs + std::size_t{decompositions} * 2 +
		                   decomposed + std::size_t{compositions} * 3;
		return header_size + words * 4 + Reader::padded(token_bytes);
	}
};

[[nodiscard]] constexpr auto scalar(std::uint32_t value) -> bool {
	return value < scalar_limit && (value < 0xd800 || value >= 0xe000);
}

[[nodiscard]] auto read_header(std::span<std::byte const> image) -> core::Result<Header, LoadError> {
	if (image.size() < header_size) {
		return std::unexpected(LoadError{.defect = Defect::Size});
	}
	auto const magic = Reader{image}.text(image_magic.size());
	if (magic != image_magic) {
		return std::unexpected(LoadError{.defect = Defect::Magic});
	}
	auto const fields = Reader{image.subspan(image_magic.size())}.words(header_words);
	if (fields[0] != image_version) {
		return std::unexpected(LoadError{.defect = Defect::Version});
	}
	auto const header = Header{
	    .tokens = fields[1],
	    .token_bytes = fields[2],
	    .merges = fields[3],
	    .class_runs = fields[4],
	    .combining_runs = fields[5],
	    .decompositions = fields[6],
	    .decomposed = fields[7],
	    .compositions = fields[8],
	};
	if (header.tokens != bpe_vocabulary || header.merges != merge_count) {
		return std::unexpected(LoadError{.defect = Defect::Shape});
	}
	if (header.token_bytes > std::size_t{bpe_vocabulary} * max_token_bytes || header.class_runs > max_class_runs ||
	    header.combining_runs > max_combining_runs || header.decompositions > max_decompositions || header.decomposed > max_decomposed ||
	    header.compositions > max_compositions) {
		return std::unexpected(LoadError{.defect = Defect::Bound});
	}
	if (image.size() != header.size()) {
		return std::unexpected(LoadError{.defect = Defect::Size});
	}
	return header;
}

[[nodiscard]] auto read_tokens(std::vector<std::uint32_t> ends, std::string bytes) -> core::Result<Tables, LoadError> {
	auto tables = Tables{.bytes = std::move(bytes), .ends = std::move(ends), .byte_tokens = {}, .merges = {}, .unicode = {}};
	auto seen = std::array<bool, 256>{};
	auto begin = std::uint32_t{0};
	for (auto id = Id{0}; id < bpe_vocabulary; ++id) {
		auto const end = tables.ends[id];
		if (end <= begin || end - begin > max_token_bytes || end > tables.bytes.size()) {
			return std::unexpected(LoadError{.defect = Defect::Token});
		}
		if (end - begin == 1) {
			auto const byte = std::bit_cast<std::uint8_t>(tables.bytes[begin]);
			if (seen.at(byte)) {
				return std::unexpected(LoadError{.defect = Defect::Token});
			}
			seen.at(byte) = true;
			tables.byte_tokens.at(byte) = id;
		}
		begin = end;
	}
	if (begin != tables.bytes.size() || !std::ranges::all_of(seen, std::identity{})) {
		return std::unexpected(LoadError{.defect = Defect::Token});
	}
	return tables;
}

[[nodiscard]] auto read_merges(Tables const& tables, std::vector<std::uint32_t> const& words) -> core::Result<Merges, LoadError> {
	auto merges = Merges{};
	merges.by_pair.reserve(merge_count);
	for (auto rank = std::uint32_t{0}; rank < merge_count; ++rank) {
		auto const at = std::size_t{rank} * 3;
		auto const merge = Merge{.left = words[at], .right = words[at + 1], .result = words[at + 2], .rank = rank};
		if (merge.left >= bpe_vocabulary || merge.right >= bpe_vocabulary || merge.result >= bpe_vocabulary) {
			return std::unexpected(LoadError{.defect = Defect::Merge});
		}
		auto const left = tables.token(merge.left);
		auto const result = tables.token(merge.result);
		if (!result.starts_with(left) || result.substr(left.size()) != tables.token(merge.right)) {
			return std::unexpected(LoadError{.defect = Defect::Merge});
		}
		merges.by_pair.push_back(merge);
	}
	std::ranges::sort(merges.by_pair, {}, &Merge::pair);
	if (std::ranges::adjacent_find(merges.by_pair, {}, &Merge::pair) != merges.by_pair.end()) {
		return std::unexpected(LoadError{.defect = Defect::Merge});
	}
	return merges;
}

/** Unpacks runs stored as `start << shift | value`, checking they start at 0, ascend and hold values below `limit`. */
template<class V>
[[nodiscard]] auto read_runs(std::vector<std::uint32_t> const& words, unsigned shift, std::uint32_t limit)
    -> core::Result<std::vector<Run<V>>, LoadError> {
	auto runs = std::vector<Run<V>>{};
	runs.reserve(words.size());
	for (auto const word : words) {
		auto const start = char32_t{word >> shift};
		auto const value = word & ((1U << shift) - 1U);
		auto const ascending = runs.empty() ? start == 0 : start > runs.back().start;
		if (!ascending || start >= scalar_limit || value >= limit) {
			return std::unexpected(LoadError{.defect = Defect::Runs});
		}
		runs.push_back(Run<V>{.start = start, .value = static_cast<V>(value)});
	}
	if (runs.empty()) {
		return std::unexpected(LoadError{.defect = Defect::Runs});
	}
	return runs;
}

[[nodiscard]] auto read_decompositions(std::vector<std::uint32_t> const& keys, std::vector<std::uint32_t> const& ends,
                                       std::vector<std::uint32_t> const& values) -> core::Result<std::vector<Decomposition>, LoadError> {
	auto out = std::vector<Decomposition>{};
	out.reserve(keys.size());
	auto begin = std::uint32_t{0};
	for (auto index = std::size_t{0}; index < keys.size(); ++index) {
		auto const ascending = out.empty() || keys[index] > out.back().key;
		if (!ascending || !scalar(keys[index]) || ends[index] <= begin || ends[index] > values.size()) {
			return std::unexpected(LoadError{.defect = Defect::Decomposition});
		}
		out.push_back(Decomposition{.key = keys[index], .begin = begin, .end = ends[index]});
		begin = ends[index];
	}
	if (begin != values.size() || !std::ranges::all_of(values, scalar)) {
		return std::unexpected(LoadError{.defect = Defect::Decomposition});
	}
	return out;
}

[[nodiscard]] auto read_compositions(std::vector<std::uint32_t> const& words) -> core::Result<std::vector<Composition>, LoadError> {
	auto out = std::vector<Composition>{};
	out.reserve(words.size() / 3);
	for (auto index = std::size_t{0}; index + 2 < words.size(); index += 3) {
		auto const composition = Composition{.first = words[index], .second = words[index + 1], .composite = words[index + 2]};
		auto const ascending = out.empty() || composition.key() > out.back().key();
		if (!ascending || !scalar(composition.first) || !scalar(composition.second) || !scalar(composition.composite)) {
			return std::unexpected(LoadError{.defect = Defect::Composition});
		}
		out.push_back(composition);
	}
	return out;
}

/** The smallest codepoint that can take part in normalization: see `Unicode::stable_below`. */
[[nodiscard]] auto stable_below(Unicode const& unicode) -> char32_t {
	auto limit = std::min(hangul_l, hangul_s);
	for (auto const& run : unicode.combining) {
		if (run.value != 0) {
			limit = std::min(limit, run.start);
			break;
		}
	}
	for (auto const& composition : unicode.compositions) {
		limit = std::min(limit, composition.second);
	}
	if (!unicode.decompositions.empty()) {
		limit = std::min(limit, unicode.decompositions.front().key);
	}
	return limit;
}

/** Reads and checks a whole tokenizer image. */
[[nodiscard]] auto read_image(std::span<std::byte const> image) -> core::Result<Tables, LoadError> {
	auto const header = read_header(image);
	if (!header) {
		return std::unexpected(header.error());
	}
	auto reader = Reader{image.subspan(header_size)};
	auto ends = reader.words(header->tokens);
	auto bytes = reader.text(header->token_bytes);
	auto tables = read_tokens(std::move(ends), std::move(bytes));
	if (!tables) {
		return tables;
	}
	auto merges = read_merges(*tables, reader.words(std::size_t{header->merges} * 3));
	if (!merges) {
		return std::unexpected(merges.error());
	}
	tables->merges = std::move(*merges);
	auto classes = read_runs<Class>(reader.words(header->class_runs), 3, class_count);
	auto combining = read_runs<std::uint8_t>(reader.words(header->combining_runs), 8, 256);
	auto const keys = reader.words(header->decompositions);
	auto const key_ends = reader.words(header->decompositions);
	auto decomposed = reader.words(header->decomposed);
	auto decompositions = read_decompositions(keys, key_ends, decomposed);
	auto compositions = read_compositions(reader.words(std::size_t{header->compositions} * 3));
	if (!classes || !combining) {
		return std::unexpected(LoadError{.defect = Defect::Runs});
	}
	if (!decompositions) {
		return std::unexpected(decompositions.error());
	}
	if (!compositions) {
		return std::unexpected(compositions.error());
	}
	tables->unicode = Unicode{
	    .classes = std::move(*classes),
	    .combining = std::move(*combining),
	    .decompositions = std::move(*decompositions),
	    .decomposed = {decomposed.begin(), decomposed.end()},
	    .compositions = std::move(*compositions),
	    .stable_below = 0,
	};
	tables->unicode.stable_below = stable_below(tables->unicode);
	return tables;
}

}

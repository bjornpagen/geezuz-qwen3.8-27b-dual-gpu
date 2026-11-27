// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:tokens;

import std;
import engine.core;
import engine.regex;
import :builder;
import :counting;

namespace engine::grammar {

/** A token id, as the tokenizer numbers them. */
export using Token = std::uint32_t;

/** A token the grammar can allow, with its bytes and its bit in the row layout. */
export struct TokenBytes {
	Token token;
	std::string bytes;
	/** Its bit in the row-order mask: for the engine, card c's row r is bit c * vocab_per_card + r. */
	std::uint32_t row;
};

/** A token the grammar allows only where the output may end, with its row bit. */
export struct EndToken {
	Token token;
	std::uint32_t row;
};

/** The sizes of a mask's two layouts. */
export struct Layout {
	/** Bits in token-id order: one per token id. */
	std::size_t tokens;
	/** Bits in row order. */
	std::size_t rows;
};

/** One token at a trie node: the bits it sets. */
struct Leaf {
	Token token;
	std::uint32_t row;
};

/** A trie node, in depth-first order: the byte into it, its subtree's size (itself included), its tokens. */
struct Node {
	std::uint8_t byte;
	std::uint32_t subtree;
	std::uint32_t leaves_begin;
	std::uint32_t leaves_end;
};

struct Trie {
	std::vector<Node> nodes;
	std::vector<Leaf> leaves;
};

/** A mask's words: token-id order, then row order; bit `i % 64` of word `i / 64` is bit `i`. */
struct Bits {
	std::vector<std::uint64_t> tokens;
	std::vector<std::uint64_t> rows;

	auto set(Leaf leaf) -> void {
		tokens[leaf.token / 64] |= std::uint64_t{1} << (leaf.token % 64U);
		rows[leaf.row / 64] |= std::uint64_t{1} << (leaf.row % 64U);
	}

	auto clear(Leaf leaf) -> void {
		tokens[leaf.token / 64] &= ~(std::uint64_t{1} << (leaf.token % 64U));
		rows[leaf.row / 64] &= ~(std::uint64_t{1} << (leaf.row % 64U));
	}

	auto add(Bits const& other) -> void {
		for (auto index = std::size_t{0}; index < tokens.size(); ++index) {
			tokens[index] |= other.tokens[index];
		}
		for (auto index = std::size_t{0}; index < rows.size(); ++index) {
			rows[index] |= other.rows[index];
		}
	}

	auto clear() -> void {
		std::ranges::fill(tokens, 0);
		std::ranges::fill(rows, 0);
	}

	[[nodiscard]] static auto sized(Layout layout) -> Bits {
		return Bits{.tokens = std::vector<std::uint64_t>((layout.tokens + 63) / 64, 0), .rows = std::vector<std::uint64_t>((layout.rows + 63) / 64, 0)};
	}
};

/**
 * Tokens whose bytes are whole characters of one class: in free text most tokens are, and where the lexer provably
 * takes every such string the whole slice is allowed at once instead of walked.
 */
/** The subtree of a trie under one first byte: its root child, the end of its nodes, and the tokens in it. */
struct Branch {
	std::uint32_t node;
	std::uint32_t end;
	std::vector<Leaf> leaves;
};

/** How many ways a lexeme's length is counted: one per Unit. */
inline constexpr std::size_t unit_count = 2;

/**
 * No token of a band reads whole as units of a kind: the band moves no count, and is never allowed where only counted
 * lexemes are under way.
 */
inline constexpr std::uint32_t no_units = std::numeric_limits<std::uint32_t>::max();

/**
 * The tokens of a slice with one range of lengths, in characters. Near a counted lexeme's bound a long token could
 * cross it where a short one cannot, so the short tokens are still taken whole, the ones past the bound refused
 * whole, and only the few at it walked.
 */
struct Band {
	Trie trie;
	Bits bits;
	/** The band's tokens and every shorter band's: bands clear of a bound come in a run from the shortest. */
	Bits through;
	/** The trie's subtrees by first byte, so a mask can take most of a band whole and walk only the rest. */
	std::vector<Branch> branches;
	/** The most characters of any token in the band: how far one token can move a lexeme's length count. */
	std::uint32_t max_characters;
	/**
	 * Per unit, the fewest units a token of the band completes from a unit's start, among tokens the unit reads
	 * whole: how far it moves a count at least. A token the unit cannot read is never allowed where only counted
	 * lexemes are under way, since the lexer dies with the unit.
	 */
	std::array<std::uint32_t, unit_count> min_units;
};

/** The upper ends of the bands' lengths: one length each while tokens are many, wider past; the last has no end. */
inline constexpr auto band_ends = std::to_array<std::uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 48, 64});

struct Slice {
	/** Recognizes one or more characters of the class, byte by byte. */
	regex::Automaton characters;
	/** By length, shortest first; the last band's `through` is every token of the slice. */
	std::vector<Band> bands;
};

/** A trie's subtrees under the root's children, each with its tokens. */
[[nodiscard]] auto branches_of(Trie const& trie) -> std::vector<Branch> {
	auto branches = std::vector<Branch>{};
	auto const size = core::narrow<std::uint32_t>(trie.nodes.size());
	for (auto node = std::uint32_t{1}; node < size; node += trie.nodes[node].subtree) {
		auto const& child = trie.nodes[node];
		auto branch = Branch{.node = node, .end = node + child.subtree, .leaves = {}};
		for (auto at = node; at < branch.end; ++at) {
			auto const& inner = trie.nodes[at];
			branch.leaves.insert(branch.leaves.end(), trie.leaves.begin() + inner.leaves_begin, trie.leaves.begin() + inner.leaves_end);
		}
		branches.push_back(std::move(branch));
	}
	return branches;
}

/** The utf-8 characters of a text: its bytes that are no continuation byte. */
[[nodiscard]] auto characters_of(std::string_view bytes) -> std::uint32_t {
	return core::narrow<std::uint32_t>(std::ranges::count_if(bytes, [](char c) { return (std::bit_cast<std::uint8_t>(c) & 0xc0U) != 0x80U; }));
}

/** The most utf-8 characters any of the tokens holds. */
[[nodiscard]] auto characters_in(std::span<std::pair<std::string_view, Leaf> const> tokens) -> std::uint32_t {
	auto most = std::uint32_t{0};
	for (auto const& entry : tokens) {
		most = std::max(most, characters_of(entry.first));
	}
	return most;
}

/** How many units a token completes read from a unit's start, unless the unit cannot read it. */
[[nodiscard]] auto units_of(regex::Automaton const& unit, std::string_view bytes) -> std::optional<std::uint32_t> {
	auto state = unit.start();
	auto units = std::uint32_t{0};
	for (auto const c : bytes) {
		state = unit.step(state, std::bit_cast<std::uint8_t>(c));
		if (state == regex::dead) {
			return std::nullopt;
		}
		// As the lexer counts: a unit read whole starts the next.
		if (unit.accepting(state)) {
			state = unit.start();
			++units;
		}
	}
	return units;
}

/** Builds a trie of tokens in depth-first order from (bytes, leaf) pairs. */
[[nodiscard]] auto make_trie(std::vector<std::pair<std::string_view, Leaf>> tokens) -> Trie {
	std::ranges::sort(tokens, {}, [](auto const& entry) { return entry.first; });
	auto trie = Trie{};
	trie.nodes.push_back(Node{.byte = 0, .subtree = 0, .leaves_begin = 0, .leaves_end = 0});
	// The open path from the root: node indices, one per byte of the current prefix.
	auto path = std::vector<std::uint32_t>{0};
	auto previous = std::string_view{};
	auto const close = [&trie](std::uint32_t node) { trie.nodes[node].subtree = core::narrow<std::uint32_t>(trie.nodes.size()) - node; };
	for (auto const& [bytes, leaf] : tokens) {
		auto common = std::size_t{0};
		while (common < bytes.size() && common < previous.size() && bytes[common] == previous[common]) {
			++common;
		}
		while (path.size() > common + 1) {
			close(path.back());
			path.pop_back();
		}
		for (auto at = common; at < bytes.size(); ++at) {
			path.push_back(core::narrow<std::uint32_t>(trie.nodes.size()));
			auto const leaves = core::narrow<std::uint32_t>(trie.leaves.size());
			trie.nodes.push_back(Node{.byte = std::bit_cast<std::uint8_t>(bytes[at]), .subtree = 0, .leaves_begin = leaves, .leaves_end = leaves});
		}
		// Tokens of equal bytes are adjacent after sorting, so a node's leaves are contiguous.
		auto& node = trie.nodes[path.back()];
		if (node.leaves_begin == node.leaves_end) {
			node.leaves_begin = core::narrow<std::uint32_t>(trie.leaves.size());
		}
		trie.leaves.push_back(leaf);
		node.leaves_end = core::narrow<std::uint32_t>(trie.leaves.size());
		previous = bytes;
	}
	while (!path.empty()) {
		close(path.back());
		path.pop_back();
	}
	return trie;
}

/** How many slices a vocabulary has. */
inline constexpr std::size_t slice_count = 2;

/** The slices, as classes of characters: inside a JSON string, and in free text. */
[[nodiscard]] auto slice_classes() -> std::array<regex::CharClass, slice_count> {
	auto json = regex::CharClass::all();
	json = json.minus(regex::CharClass::of(std::to_array<regex::Range>({
	    {.low = 0, .high = 0x1f},
	    {.low = '"', .high = '"'},
	    {.low = '<', .high = '<'},
	    {.low = '\\', .high = '\\'},
	    {.low = 0x7f, .high = 0x7f},
	})));
	auto const text = regex::CharClass::all().minus(regex::CharClass::single('<'));
	return {json, text};
}

/**
 * The tokens a grammar chooses among: their bytes, end tokens, and the trie and slices a mask is computed by. Built
 * once at start from the tokenizer and shared, read-only, by every matcher; a test builds one from any token list.
 */
export class Vocabulary {
public:
	[[nodiscard]] static auto make(std::span<TokenBytes const> tokens, std::span<EndToken const> ends, Layout layout) -> Vocabulary {
		auto out = Vocabulary{};
		out.layout_ = layout;
		out.ends_.assign(ends.begin(), ends.end());
		auto store = regex::Store{std::size_t{1} << 16U};
		auto parts = std::array<std::vector<std::pair<std::string_view, Leaf>>, 3>{};
		auto automata = std::vector<regex::Automaton>{};
		for (auto const& set : slice_classes()) {
			auto built = regex::Automaton::build(store, store.repeat(regex::characters(store, set), 1, regex::unbounded), 4096);
			core::invariant(built.has_value());
			automata.push_back(std::move(*built));
		}
		for (auto const& token : tokens) {
			core::invariant(!token.bytes.empty() && token.token < layout.tokens && token.row < layout.rows);
			auto const leaf = Leaf{.token = token.token, .row = token.row};
			auto const slice = std::ranges::find_if(automata, [&token](regex::Automaton const& automaton) { return automaton.matches(token.bytes); });
			parts.at(core::narrow<std::size_t>(slice - automata.begin())).emplace_back(token.bytes, leaf);
			out.bytes_.emplace(token.token, token.bytes);
		}
		auto units = std::vector<regex::Automaton>{};
		for (auto const unit : {Unit::Character, Unit::JsonCharacter}) {
			core::invariant(std::to_underlying(unit) == units.size());
			auto built = regex::Automaton::build(store, unit_language(store, unit), 4096);
			core::invariant(built.has_value());
			units.push_back(std::move(*built));
		}
		for (auto const index : {std::size_t{0}, std::size_t{1}}) {
			auto banded = std::array<std::vector<std::pair<std::string_view, Leaf>>, band_ends.size() + 1>{};
			for (auto const& entry : parts.at(index)) {
				auto const length = characters_of(entry.first);
				auto const band = std::ranges::find_if(band_ends, [length](std::uint32_t end) { return length <= end; });
				banded.at(core::narrow<std::size_t>(band - band_ends.begin())).push_back(entry);
			}
			auto slice = Slice{.characters = std::move(automata.at(index)), .bands = {}};
			auto through = Bits::sized(layout);
			for (auto const& members : banded) {
				auto band_bits = Bits::sized(layout);
				auto min_units = std::array<std::uint32_t, unit_count>{};
				min_units.fill(no_units);
				for (auto const& entry : members) {
					band_bits.set(entry.second);
					for (auto const [unit, automaton] : std::views::enumerate(units)) {
						if (auto const read = units_of(automaton, entry.first)) {
							auto& least = min_units.at(core::narrow<std::size_t>(unit));
							least = std::min(least, *read);
						}
					}
				}
				through.add(band_bits);
				auto trie = make_trie(members);
				auto branches = branches_of(trie);
				slice.bands.push_back(Band{.trie = std::move(trie),
				                           .bits = std::move(band_bits),
				                           .through = through,
				                           .branches = std::move(branches),
				                           .max_characters = characters_in(members),
				                           .min_units = min_units});
			}
			out.slices_.push_back(std::move(slice));
		}
		out.rest_ = make_trie(parts.at(2));
		return out;
	}

	[[nodiscard]] auto layout() const -> Layout {
		return layout_;
	}

	/** A token's bytes, if the grammar can allow it outside the end. */
	[[nodiscard]] auto bytes(Token token) const -> std::optional<std::string_view> {
		if (auto const found = bytes_.find(token); found != bytes_.end()) {
			return std::string_view{found->second};
		}
		return std::nullopt;
	}

	[[nodiscard]] auto ends() const -> std::span<EndToken const> {
		return ends_;
	}

	[[nodiscard]] auto ends_with(Token token) const -> bool {
		return std::ranges::contains(ends_, token, &EndToken::token);
	}

	[[nodiscard]] auto slices() const -> std::span<Slice const> {
		return slices_;
	}

	[[nodiscard]] auto rest() const -> Trie const& {
		return rest_;
	}

private:
	Vocabulary() = default;

	Layout layout_{};
	std::vector<EndToken> ends_;
	std::unordered_map<Token, std::string> bytes_;
	std::vector<Slice> slices_;
	Trie rest_;
};

}

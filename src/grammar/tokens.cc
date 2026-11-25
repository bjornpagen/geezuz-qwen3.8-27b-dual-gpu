// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:tokens;

import std;
import engine.core;
import engine.regex;

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

struct Slice {
	/** Recognizes one or more characters of the class, byte by byte. */
	regex::Automaton characters;
	Trie trie;
	Bits bits;
	/** The trie's subtrees by first byte, so a mask can take most of a slice whole and walk only the rest. */
	std::vector<Branch> branches;
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
		for (auto const index : {std::size_t{0}, std::size_t{1}}) {
			auto bits = Bits::sized(layout);
			for (auto const& entry : parts.at(index)) {
				bits.set(entry.second);
			}
			auto trie = make_trie(parts.at(index));
			auto branches = branches_of(trie);
			out.slices_.push_back(Slice{.characters = std::move(automata.at(index)),
			                            .trie = std::move(trie),
			                            .bits = std::move(bits),
			                            .branches = std::move(branches)});
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

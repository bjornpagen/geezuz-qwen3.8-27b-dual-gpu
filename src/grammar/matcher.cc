// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:matcher;

import std;
import engine.core;
import engine.regex;
import :compiled;
import :lexer;
import :parser;
import :tokens;

namespace engine::grammar {

export enum class MatchFault : std::uint8_t {
	/** The grammar does not allow the token here: it was not in the mask. */
	Refused,
	/** The output already ended. */
	Ended,
	/** The id names no token the grammar can allow. */
	UnknownToken,
	/** The lexer passed `max_lexer_states`. */
	LexerTooLarge,
	/** A parser row passed `max_row_items`. */
	RowTooLarge,
};

export struct [[nodiscard]] MatchError {
	MatchFault fault;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		// A matcher fails on its own grammar and output: the same tokens fail the same way again.
		switch (fault) {
		case MatchFault::Refused:
		case MatchFault::Ended:
		case MatchFault::UnknownToken:
		case MatchFault::LexerTooLarge:
		case MatchFault::RowTooLarge:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(MatchError const&) const -> bool = default;
};

/**
 * The tokens allowed at one position, in both layouts: bit `t % 64` of word `t / 64` of `tokens` for token id `t`,
 * and the same for row bit `r` of `rows`. A borrow of its matcher, valid until the matcher's next non-const call.
 */
export struct MaskView {
	std::span<std::uint64_t const> tokens;
	std::span<std::uint64_t const> rows;

	[[nodiscard]] auto allows(Token token) const -> bool {
		return ((tokens[token / 64] >> (token % 64U)) & 1U) != 0;
	}

	[[nodiscard]] auto allows_row(std::uint32_t row) const -> bool {
		return ((rows[row / 64] >> (row % 64U)) & 1U) != 0;
	}

	[[nodiscard]] auto count() const -> std::size_t {
		auto total = std::size_t{0};
		for (auto const word : tokens) {
			total += core::narrow<std::size_t>(std::popcount(word));
		}
		return total;
	}
};

/** Whether the output has ended with an end token. */
enum class Phase : std::uint8_t {
	Open,
	Ended,
};

/** Whether every string of a slice is known to keep the lexer going from a lexer state. */
enum class Containment : std::uint8_t {
	Unknown,
	Contained,
	Escapes,
};

/** The most (lexer, slice) state pairs a containment check explores before it answers no. */
inline constexpr std::size_t containment_budget = 64;

/**
 * One request's constrained decoding: its grammar, the committed parse, and the mask at the committed position.
 * Made on the engine thread from the grammar the boundary compiled; advanced by each committed token, rolled back
 * when drafts are rejected.
 */
export class Matcher {
public:
	[[nodiscard]] static auto make(Grammar grammar, Layout layout) -> Matcher {
		return Matcher{std::move(grammar), layout};
	}

	/** Commits one token. On failure nothing changes. */
	[[nodiscard]] auto advance(Vocabulary const& vocabulary, Token token) -> core::Result<void, MatchError> {
		if (phase_ == Phase::Ended) {
			return std::unexpected(MatchError{.fault = MatchFault::Ended});
		}
		if (vocabulary.ends_with(token)) {
			if (!parser_.accepting(grammar_, committed_)) {
				return std::unexpected(MatchError{.fault = MatchFault::Refused});
			}
			history_.push_back(committed_);
			phase_ = Phase::Ended;
			return {};
		}
		auto const bytes = vocabulary.bytes(token);
		if (!bytes) {
			return std::unexpected(MatchError{.fault = MatchFault::UnknownToken});
		}
		auto position = committed_;
		for (auto const letter : *bytes) {
			auto const next = parser_.advance(grammar_, position, std::bit_cast<std::uint8_t>(letter));
			if (auto const fault = health(); fault) {
				return std::unexpected(*fault);
			}
			if (!next) {
				return std::unexpected(MatchError{.fault = MatchFault::Refused});
			}
			position = *next;
		}
		history_.push_back(committed_);
		committed_ = position;
		return {};
	}

	/** Takes back the last `tokens` committed tokens. */
	auto rollback(std::size_t tokens) -> void {
		core::invariant(tokens <= history_.size());
		if (tokens == 0) {
			return;
		}
		committed_ = history_[history_.size() - tokens];
		history_.resize(history_.size() - tokens);
		phase_ = Phase::Open;
		parser_.truncate(committed_.top);
	}

	/** Tokens committed so far, the end token included. */
	[[nodiscard]] auto committed() const -> std::size_t {
		return history_.size();
	}

	/** Whether the output may end here. */
	[[nodiscard]] auto accepting() -> bool {
		return phase_ == Phase::Open && parser_.accepting(grammar_, committed_);
	}

	[[nodiscard]] auto ended() const -> bool {
		return phase_ == Phase::Ended;
	}

	/** The tokens allowed next. After the end token, none. */
	[[nodiscard]] auto mask(Vocabulary const& vocabulary) -> core::Result<MaskView, MatchError> {
		bits_.clear();
		if (phase_ == Phase::Open) {
			if (parser_.accepting(grammar_, committed_)) {
				for (auto const end : vocabulary.ends()) {
					bits_.set(Leaf{.token = end.token, .row = end.row});
				}
			}
			for (auto const [index, one] : std::views::enumerate(vocabulary.slices())) {
				slice(one, core::narrow<std::size_t>(index));
			}
			walk(vocabulary.rest());
		}
		if (auto const fault = health(); fault) {
			return std::unexpected(*fault);
		}
		return MaskView{.tokens = bits_.tokens, .rows = bits_.rows};
	}

	/** The lexer states and parser rows this matcher holds: its memory, for bounds and benchmarks. */
	[[nodiscard]] auto footprint() const -> std::pair<std::size_t, std::size_t> {
		return {lexer_states(), parser_.rows()};
	}

private:
	Matcher(Grammar grammar, Layout layout) : grammar_{std::move(grammar)}, parser_{grammar_}, bits_{Bits::sized(layout)} {
		committed_ = parser_.begin(grammar_);
	}

	[[nodiscard]] auto lexer_states() const -> std::size_t {
		return parser_.lexer_states();
	}

	[[nodiscard]] auto health() -> std::optional<MatchError> {
		if (parser_.lexer().exhaustion() == Exhaustion::Exhausted) {
			return MatchError{.fault = MatchFault::LexerTooLarge};
		}
		if (parser_.health() == Health::RowTooLarge) {
			return MatchError{.fault = MatchFault::RowTooLarge};
		}
		return std::nullopt;
	}

	/** Sets the bits of the tokens of trie nodes [`begin`, `end`) the grammar allows from `from`, starting at a root child. */
	auto walk(Trie const& trie, std::uint32_t begin, std::uint32_t end, Position from) -> void {
		auto const& nodes = trie.nodes;
		frames_.clear();
		frames_.push_back(Frame{.end = end, .position = from});
		auto at = begin;
		while (at < end) {
			while (at >= frames_.back().end) {
				frames_.pop_back();
			}
			auto const& node = nodes[at];
			auto const next = parser_.advance(grammar_, frames_.back().position, node.byte);
			if (!next) {
				at += node.subtree;
				continue;
			}
			for (auto leaf = node.leaves_begin; leaf < node.leaves_end; ++leaf) {
				bits_.set(trie.leaves[leaf]);
			}
			if (node.subtree > 1) {
				frames_.push_back(Frame{.end = at + node.subtree, .position = *next});
			}
			++at;
		}
	}

	auto walk(Trie const& trie) -> void {
		walk(trie, 1, core::narrow<std::uint32_t>(trie.nodes.size()), committed_);
	}

	/**
	 * A slice's tokens at the committed position. When every string of the slice keeps the lexer going, all of them
	 * at once. Otherwise branch by branch under the first byte: a branch whose position after its byte keeps the
	 * lexer going for every rest of the slice is taken whole, the others walked. A position inside a marker (after
	 * `</thi`) escapes only under the few first bytes that continue the marker, so the walk stays small.
	 */
	auto slice(Slice const& slice, std::size_t index) -> void {
		if (contained(slice, index, committed_.lexer, slice.characters.start()) == Containment::Contained) {
			bits_.add(slice.bits);
			return;
		}
		auto taken = std::size_t{0};
		auto total = std::size_t{0};
		auto& kinds = branch_kinds_;
		kinds.assign(slice.branches.size(), BranchKind::Refused);
		for (auto const [number, branch] : std::views::enumerate(slice.branches)) {
			auto const byte = slice.trie.nodes[branch.node].byte;
			total += branch.leaves.size();
			auto const characters = slice.characters.step(slice.characters.start(), byte);
			auto const next = parser_.advance(grammar_, committed_, byte);
			auto& kind = kinds[core::narrow<std::size_t>(number)];
			if (!next) {
				continue;
			}
			// Whole only when the byte stayed inside one lexeme: a lexeme ending or a row pushed is the parser's to judge.
			auto const inside = next->boundary == Boundary::Inside && next->top == committed_.top;
			if (inside && contained(slice, index, next->lexer, characters) == Containment::Contained) {
				kind = BranchKind::Whole;
				taken += branch.leaves.size();
			} else {
				kind = BranchKind::Walked;
			}
		}
		// Taking the slice whole and clearing what is not pays when most of it is allowed.
		auto const bulk = taken * 2 > total;
		if (bulk) {
			bits_.add(slice.bits);
		}
		for (auto const [number, branch] : std::views::enumerate(slice.branches)) {
			auto const kind = kinds[core::narrow<std::size_t>(number)];
			if (kind == BranchKind::Whole) {
				if (!bulk) {
					std::ranges::for_each(branch.leaves, [this](Leaf leaf) { bits_.set(leaf); });
				}
				continue;
			}
			if (bulk) {
				std::ranges::for_each(branch.leaves, [this](Leaf leaf) { bits_.clear(leaf); });
			}
			if (kind == BranchKind::Walked) {
				walk(slice.trie, branch.node, branch.end, committed_);
			}
		}
	}

	/** Whether every rest of the slice's strings keeps the lexer going from a lexer state and a slice state, cached. */
	[[nodiscard]] auto contained(Slice const& slice, std::size_t index, LexerState state, regex::State characters) -> Containment {
		auto& cache = containment_.at(index);
		auto const width = slice.characters.states();
		auto const key = (std::size_t{state} * width) + characters;
		if (cache.size() <= key) {
			cache.resize((lexer_states() + 1) * width, Containment::Unknown);
		}
		if (cache[key] != Containment::Unknown) {
			return cache[key];
		}
		auto const answer = explore(slice, state, characters);
		cache[key] = answer;
		return answer;
	}

	[[nodiscard]] auto explore(Slice const& slice, LexerState start, regex::State characters_start) -> Containment {
		if (characters_start == regex::dead) {
			return Containment::Escapes;
		}
		auto& lexer = parser_.lexer();
		auto seen = std::vector<std::pair<LexerState, regex::State>>{{start, characters_start}};
		for (auto at = std::size_t{0}; at < seen.size(); ++at) {
			auto const [state, characters] = seen[at];
			for (auto byte = 0U; byte < 256U; ++byte) {
				auto const next_characters = slice.characters.step(characters, static_cast<std::uint8_t>(byte));
				if (next_characters == regex::dead) {
					continue;
				}
				auto const next = lexer.step(grammar_, state, static_cast<std::uint8_t>(byte));
				if (next == dead_lexer || lexer.ending(next) == Ending::Lazy) {
					return Containment::Escapes;
				}
				auto const pair = std::pair{next, next_characters};
				if (!std::ranges::contains(seen, pair)) {
					if (seen.size() >= containment_budget) {
						return Containment::Escapes;
					}
					seen.push_back(pair);
				}
			}
		}
		return Containment::Contained;
	}

	struct Frame {
		std::uint32_t end;
		Position position;
	};

	/** How a slice's branch enters a mask. */
	enum class BranchKind : std::uint8_t {
		Refused,
		Whole,
		Walked,
	};

	Grammar grammar_;
	Parser parser_;
	Position committed_{};
	std::vector<Position> history_;
	Phase phase_ = Phase::Open;
	Bits bits_;
	std::vector<Frame> frames_;
	std::vector<BranchKind> branch_kinds_;
	/** Per slice, by lexer state and slice state. */
	std::array<std::vector<Containment>, slice_count> containment_;
};

}

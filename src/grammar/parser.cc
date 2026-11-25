// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:parser;

import std;
import engine.core;
import :builder;
import :compiled;
import :lexer;

namespace engine::grammar {

/** An Earley item: a rule's slot after the dot, and the row where the rule began. */
struct Item {
	std::uint32_t slot;
	std::uint32_t origin;

	[[nodiscard]] constexpr auto operator==(Item const&) const -> bool = default;
};

/** An item of a row whose dot stands before a nonterminal: what completing that nonterminal advances. */
struct Wait {
	Nonterminal nonterminal;
	Item item;
};

/** How a row's lexemes were scanned: at a greedy end, a lazy end, or as the closing after a joined lexeme. */
enum class Scan : std::uint8_t {
	Greedy,
	Lazy,
	Close,
};

/** What scanning produced a row: the lexer state its lexemes ended in, and how. */
struct ScanKey {
	LexerState lexer;
	Scan scan;

	[[nodiscard]] constexpr auto operator==(ScanKey const&) const -> bool = default;
};

/** Whether the output may end at a row. */
enum class Completion : std::uint8_t {
	Open,
	Complete,
};

/**
 * One Earley set. Rows live on a stack; a row's items and waits are ranges of the parser's flat arrays. Rows past
 * the stack's top stay in place after a mask walk moves on, so the next walk or advance that scans the same lexemes
 * from the same parent reuses them instead of computing them again.
 */
struct Row {
	std::uint32_t items_begin;
	std::uint32_t items_end;
	std::uint32_t waits_begin;
	std::uint32_t waits_end;
	LexerState lexer_start;
	Completion completion;
	/** Identity: a stamp unique to this computation, its parent's stamp and the scan that made it. */
	std::uint64_t stamp;
	std::uint64_t parent;
	ScanKey key;
};

/** Whether a position is at a lexeme boundary or inside a lexeme. */
enum class Boundary : std::uint8_t {
	Between,
	Inside,
};

/** A point in the output: the lexer's state, the rows above it on the stack, and whether a lexeme is under way. */
export struct Position {
	LexerState lexer;
	std::uint32_t top;
	Boundary boundary;

	[[nodiscard]] constexpr auto operator==(Position const&) const -> bool = default;
};

/** The most items one row may hold; past it the matcher fails (an ambiguous grammar run away). */
export inline constexpr std::size_t max_row_items = std::size_t{1} << 16U;

/** Whether a parser has stayed within its bounds. */
enum class Health : std::uint8_t {
	Within,
	RowTooLarge,
};

/**
 * The Earley parser over lexemes, advanced a byte at a time. A position names the parse after some output; advancing
 * from a position computes the next and may push rows; any earlier position stays valid as long as the rows below
 * its top are not replaced, which holds for every position a caller keeps (the committed ones, and a walk's
 * ancestors).
 */
class Parser {
public:
	explicit Parser(Grammar const& grammar) : predicted_(grammar.nonterminal_count(), 0), scanned_(grammar.lexeme_count(), 0) {}

	/** The position before any output. */
	[[nodiscard]] auto begin(Grammar const& grammar) -> Position {
		rows_.clear();
		items_.clear();
		waits_.clear();
		auto const index = compute(grammar, std::nullopt, ScanKey{.lexer = dead_lexer, .scan = Scan::Greedy}, {}, std::nullopt);
		return Position{.lexer = rows_[index].lexer_start, .top = index + 1, .boundary = Boundary::Between};
	}

	/** The position after one more byte, or nothing when the grammar refuses it there. */
	[[nodiscard]] auto advance(Grammar const& grammar, Position from, std::uint8_t byte) -> std::optional<Position> {
		auto next = lexer_.step(grammar, from.lexer, byte);
		auto top = from.top;
		if (next == dead_lexer) {
			if (from.boundary == Boundary::Between) {
				return std::nullopt;
			}
			// The lexeme ends before this byte if it can; the byte then starts the next one.
			auto const& info = lexer_.info(from.lexer);
			if (info.greedy.empty()) {
				return std::nullopt;
			}
			top = push(grammar, top, ScanKey{.lexer = from.lexer, .scan = Scan::Greedy}, info.greedy, info.guard) + 1;
			next = lexer_.step(grammar, rows_[top - 1].lexer_start, byte);
			if (next == dead_lexer) {
				return std::nullopt;
			}
		}
		if (lexer_.ending(next) == Ending::Lazy) {
			auto const& info = lexer_.info(next);
			auto inner = std::vector<Lexeme>{};
			auto close = std::optional<Lexeme>{};
			for (auto const lexeme : info.lazy) {
				if (auto const closing = core::alternative<Closing>(grammar.lexeme(lexeme).placement)) {
					inner.push_back(closing->inner);
					close = closing->close;
				}
			}
			if (close) {
				// A lexeme joined to its region's closing matched: the lexeme, then the closing, are scanned.
				std::ranges::sort(inner);
				top = push(grammar, top, ScanKey{.lexer = next, .scan = Scan::Lazy}, inner, info.guard) + 1;
				if (empty(top - 1)) {
					return std::nullopt;
				}
				auto const closes = std::array{*close};
				top = push(grammar, top, ScanKey{.lexer = next, .scan = Scan::Close}, closes, std::nullopt) + 1;
			} else {
				top = push(grammar, top, ScanKey{.lexer = next, .scan = Scan::Lazy}, info.lazy, info.guard) + 1;
			}
			if (empty(top - 1)) {
				return std::nullopt;
			}
			return Position{.lexer = rows_[top - 1].lexer_start, .top = top, .boundary = Boundary::Between};
		}
		return Position{.lexer = next, .top = top, .boundary = Boundary::Inside};
	}

	/** Whether the output may end at a position: its lexeme, if one is under way, ends there and the parse is complete. */
	[[nodiscard]] auto accepting(Grammar const& grammar, Position at) -> bool {
		if (at.boundary == Boundary::Between) {
			return rows_[at.top - 1].completion == Completion::Complete;
		}
		auto const& info = lexer_.info(at.lexer);
		if (info.greedy.empty()) {
			return false;
		}
		auto const index = push(grammar, at.top, ScanKey{.lexer = at.lexer, .scan = Scan::Greedy}, info.greedy, info.guard);
		return rows_[index].completion == Completion::Complete;
	}

	/** Whether a row holds no items: scanning reached it, but no parse goes on from it. */
	[[nodiscard]] auto empty(std::uint32_t row) const -> bool {
		return rows_[row].items_begin == rows_[row].items_end;
	}

	/** Drops every row above `top`, so memory follows the committed output after a rollback. */
	auto truncate(std::uint32_t top) -> void {
		if (rows_.size() <= top) {
			return;
		}
		items_.resize(rows_[top].items_begin);
		waits_.resize(rows_[top].waits_begin);
		rows_.resize(top);
	}

	[[nodiscard]] auto lexer() -> Lexer& {
		return lexer_;
	}

	[[nodiscard]] auto lexer_states() const -> std::size_t {
		return lexer_.states();
	}

	[[nodiscard]] auto health() const -> Health {
		return health_;
	}

	[[nodiscard]] auto rows() const -> std::size_t {
		return rows_.size();
	}

private:
	/** The row scanning `lexemes` makes from the row below `top`, placed at `top`: reused when it is already there. */
	[[nodiscard]] auto push(Grammar const& grammar, std::uint32_t top, ScanKey key, std::span<Lexeme const> lexemes, std::optional<Guard> guard)
	    -> std::uint32_t {
		auto const parent = rows_[top - 1].stamp;
		if (rows_.size() > top && rows_[top].parent == parent && rows_[top].key == key) {
			return top;
		}
		truncate(top);
		return compute(grammar, top - 1, key, lexemes, guard);
	}

	auto add(Item item) -> void {
		auto const packed = (std::uint64_t{item.slot} << 32U) | item.origin;
		if (seen_.insert(packed).second) {
			items_.push_back(item);
		}
	}

	/** Starts a row's items: those of `parent` that scan one of `lexemes`, advanced past it (or the start's predictions). */
	auto kernel(Grammar const& grammar, std::optional<std::uint32_t> parent, std::span<Lexeme const> lexemes, std::uint32_t current) -> void {
		++generation_;
		seen_.clear();
		if (!parent) {
			predict(grammar, grammar.start(), current);
			return;
		}
		for (auto const lexeme : lexemes) {
			scanned_[lexeme.index] = generation_;
		}
		auto const& from = rows_[*parent];
		for (auto at = from.items_begin; at < from.items_end; ++at) {
			auto const item = items_[at];
			if (auto const lexeme = core::alternative<Lexeme>(grammar.slot(item.slot)); lexeme && scanned_[lexeme->index] == generation_) {
				add(Item{.slot = item.slot + 1, .origin = item.origin});
			}
		}
	}

	/** Completes and predicts the items from `items_begin` on, for row number `current`. */
	auto complete(Grammar const& grammar, std::size_t items_begin, std::uint32_t current) -> void {
		for (auto at = items_begin; at < items_.size(); ++at) {
			if (items_.size() - items_begin > max_row_items) {
				health_ = Health::RowTooLarge;
				break;
			}
			auto const item = items_[at];
			auto const& slot = grammar.slot(item.slot);
			if (auto const end = core::alternative<End>(slot)) {
				// Empty completions (origin == current) are covered by advancing over nullable nonterminals on prediction.
				if (item.origin == current) {
					continue;
				}
				auto const lhs = grammar.rule(end->rule).lhs;
				auto const& origin = rows_[item.origin];
				auto const waits = std::span{waits_}.subspan(origin.waits_begin, origin.waits_end - origin.waits_begin);
				auto const first = std::ranges::lower_bound(waits, lhs, {}, &Wait::nonterminal);
				for (auto wait = first; wait != waits.end() && wait->nonterminal == lhs; ++wait) {
					add(Item{.slot = wait->item.slot + 1, .origin = wait->item.origin});
				}
			} else if (auto const nonterminal = core::alternative<Nonterminal>(slot)) {
				predict(grammar, *nonterminal, current);
				if (grammar.nullable(*nonterminal)) {
					add(Item{.slot = item.slot + 1, .origin = item.origin});
				}
			}
		}
	}

	/**
	 * Whether the row after row `row` scans `inner` expects `close`: computed into scratch past the stack and thrown
	 * away. A region's lexeme joined to its closing is offered only where this holds, so the lexer never runs into a
	 * closing the parse cannot take.
	 */
	[[nodiscard]] auto closes_after(Grammar const& grammar, std::uint32_t row, Lexeme inner, Lexeme close) -> bool {
		auto const mark = items_.size();
		auto const scanned = std::array{inner};
		kernel(grammar, row, scanned, row + 1);
		complete(grammar, mark, row + 1);
		auto const expects = std::ranges::any_of(std::span{items_}.subspan(mark), [&](Item const& item) {
			auto const lexeme = core::alternative<Lexeme>(grammar.slot(item.slot));
			return lexeme && *lexeme == close;
		});
		items_.resize(mark);
		return expects;
	}

	/** Computes the row after `parent` scans `lexemes` (or the first row), on top of the stack. */
	[[nodiscard]] auto compute(Grammar const& grammar, std::optional<std::uint32_t> parent, ScanKey key, std::span<Lexeme const> lexemes,
	                           std::optional<Guard> guard) -> std::uint32_t {
		auto const current = core::narrow<std::uint32_t>(rows_.size());
		auto const items_begin = core::narrow<std::uint32_t>(items_.size());
		kernel(grammar, parent, lexemes, current);
		complete(grammar, items_begin, current);
		auto const items_end = core::narrow<std::uint32_t>(items_.size());
		auto const waits_begin = core::narrow<std::uint32_t>(waits_.size());
		auto expected = std::vector<Lexeme>{};
		auto completion = Completion::Open;
		for (auto at = items_begin; at < items_end; ++at) {
			auto const item = items_[at];
			auto const& slot = grammar.slot(item.slot);
			if (auto const nonterminal = core::alternative<Nonterminal>(slot)) {
				waits_.push_back(Wait{.nonterminal = *nonterminal, .item = item});
			} else if (auto const lexeme = core::alternative<Lexeme>(slot)) {
				expected.push_back(*lexeme);
			} else if (item.origin == 0 && grammar.rule(std::get<End>(slot).rule).lhs == grammar.start()) {
				completion = Completion::Complete;
			}
		}
		std::ranges::sort(std::span{waits_}.subspan(waits_begin), {}, &Wait::nonterminal);
		std::ranges::sort(expected);
		auto const [first, last] = std::ranges::unique(expected);
		expected.erase(first, last);
		++stamps_;
		rows_.push_back(Row{
		    .items_begin = items_begin,
		    .items_end = items_end,
		    .waits_begin = waits_begin,
		    .waits_end = core::narrow<std::uint32_t>(waits_.size()),
		    .lexer_start = dead_lexer,
		    .completion = completion,
		    .stamp = stamps_,
		    .parent = parent ? rows_[*parent].stamp : 0,
		    .key = key,
		});
		auto closings = std::vector<Lexeme>{};
		for (auto const lexeme : expected) {
			auto const closing = grammar.closing(lexeme);
			if (closing && closes_after(grammar, current, lexeme, std::get<Closing>(grammar.lexeme(*closing).placement).close)) {
				closings.push_back(*closing);
			}
		}
		rows_.back().lexer_start = lexer_.start(grammar, expected, closings, guard);
		return current;
	}

	auto predict(Grammar const& grammar, Nonterminal nonterminal, std::uint32_t current) -> void {
		if (predicted_[nonterminal.index] == generation_) {
			return;
		}
		predicted_[nonterminal.index] = generation_;
		for (auto const first : grammar.predictions(nonterminal)) {
			add(Item{.slot = first, .origin = current});
		}
	}

	Lexer lexer_;
	std::vector<Row> rows_;
	std::vector<Item> items_;
	std::vector<Wait> waits_;
	std::unordered_set<std::uint64_t> seen_;
	std::vector<std::uint64_t> predicted_;
	std::vector<std::uint64_t> scanned_;
	std::uint64_t generation_ = 0;
	std::uint64_t stamps_ = 0;
	Health health_ = Health::Within;
};

}

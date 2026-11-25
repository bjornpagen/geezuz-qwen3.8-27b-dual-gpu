// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:builder;

import std;
import engine.core;
import engine.regex;
import :error;

namespace engine::grammar {

/** How a lexeme decides where it ends. */
export enum class Mode : std::uint8_t {
	/** At the longest match: it ends when the next byte cannot extend it (or the output ends). */
	Greedy,
	/** At the shortest match: it ends with the byte that first completes it. */
	Lazy,
};

/** A lexeme of a grammar, by index. */
export struct Lexeme {
	std::uint32_t index;

	[[nodiscard]] constexpr auto operator==(Lexeme const&) const -> bool = default;
	[[nodiscard]] constexpr auto operator<=>(Lexeme const&) const = default;
};

/** A nonterminal of a grammar, by index. */
export struct Nonterminal {
	std::uint32_t index;

	[[nodiscard]] constexpr auto operator==(Nonterminal const&) const -> bool = default;
	[[nodiscard]] constexpr auto operator<=>(Nonterminal const&) const = default;
};

export using Symbol = std::variant<Lexeme, Nonterminal>;

/** A span of text in which no guard marker may appear, carried across the lexemes inside it: by index. */
export struct Region {
	std::uint32_t index;

	[[nodiscard]] constexpr auto operator==(Region const&) const -> bool = default;
	[[nodiscard]] constexpr auto operator<=>(Region const&) const = default;
};

/** A lexeme as written: its language, how it ends, and the region whose guard it runs under, if any. */
struct LexemeSpec {
	regex::Expr expr;
	Mode mode;
	std::optional<Region> region;

	[[nodiscard]] constexpr auto operator==(LexemeSpec const&) const -> bool = default;
};

/** A guarded region: the words its text may not contain, and the lexeme that closes it. */
struct RegionSpec {
	regex::Expr forbidden;
	regex::Expr closing;
	Lexeme close;
};

struct RuleSpec {
	Nonterminal lhs;
	std::vector<Symbol> rhs;
};

/**
 * Collects a grammar: lexemes over one expression store, nonterminals and their rules. Frontends write here; `finish`
 * checks and compiles the result into a `Grammar`. Every count is bounded by `Bounds`; past one, the builder records
 * the refusal and `finish` returns it, so frontends need not check each step.
 */
export class Builder {
public:
	explicit Builder(Bounds const& bounds) : bounds_{bounds}, store_{bounds.expr_nodes} {}

	[[nodiscard]] auto store() -> regex::Store& {
		return store_;
	}

	[[nodiscard]] auto bounds() const -> Bounds const& {
		return bounds_;
	}

	/**
	 * An expression built once per grammar and named: the same integer range or text language is asked for by every
	 * parameter of a wide schema, and building it again would cost more than the rest of the parameter.
	 */
	template<class Make>
	[[nodiscard]] auto memo(std::string const& name, Make make) -> regex::Expr {
		if (auto const found = memo_.find(name); found != memo_.end()) {
			return found->second;
		}
		auto const made = make();
		memo_.emplace(name, made);
		return made;
	}

	/** A fresh nonterminal with no rules yet. */
	[[nodiscard]] auto nonterminal() -> Nonterminal {
		auto const out = Nonterminal{.index = core::narrow<std::uint32_t>(nonterminals_)};
		++nonterminals_;
		return out;
	}

	auto rule(Nonterminal lhs, std::vector<Symbol> rhs) -> void {
		symbols_ += rhs.size() + 1;
		if (rules_.size() >= bounds_.rules || symbols_ > bounds_.symbols) {
			overflow(Refusal::GrammarTooLarge);
			return;
		}
		rules_.push_back(RuleSpec{.lhs = lhs, .rhs = std::move(rhs)});
	}

	/**
	 * The symbol matching `expr` as one lexeme. A lexeme never matches the empty string (the lexer could not tell
	 * where an empty match ends); a language that holds it becomes an optional lexeme of its other words.
	 */
	[[nodiscard]] auto lexeme(regex::Expr expr, Mode mode, std::optional<Region> region) -> Symbol {
		if (expr == regex::nothing) {
			return nonterminal();
		}
		if (!store_.nullable(expr)) {
			return raw_lexeme(LexemeSpec{.expr = expr, .mode = mode, .region = region});
		}
		auto const nonempty = store_.both(expr, store_.concat(store_.bytes(regex::ByteSet::range(0, 255)), regex::everything));
		auto const out = nonterminal();
		rule(out, {});
		if (nonempty != regex::nothing) {
			rule(out, {raw_lexeme(LexemeSpec{.expr = nonempty, .mode = mode, .region = region})});
		}
		return out;
	}

	[[nodiscard]] auto literal(std::string_view text) -> Symbol {
		return lexeme(store_.literal(text), Mode::Greedy, std::nullopt);
	}

	/**
	 * A region whose text never contains a word of `forbidden`, closed by the literal `closing`, which must contain
	 * a forbidden word (so the first one ends the region). Every lexeme placed in the region also runs joined to the
	 * closing, so a lexeme that could swallow the closing's first bytes still lets it close.
	 */
	[[nodiscard]] auto region(regex::Expr forbidden, std::string_view closing) -> Region {
		auto const out = Region{.index = core::narrow<std::uint32_t>(regions_.size())};
		auto const closing_expr = store_.literal(closing);
		auto const close = raw_lexeme(LexemeSpec{.expr = closing_expr, .mode = Mode::Greedy, .region = std::nullopt});
		regions_.push_back(RegionSpec{.forbidden = forbidden, .closing = closing_expr, .close = std::get<Lexeme>(close)});
		return out;
	}

	/** `symbol?` */
	[[nodiscard]] auto optional(Symbol symbol) -> Nonterminal {
		auto const out = nonterminal();
		rule(out, {});
		rule(out, {symbol});
		return out;
	}

	/** `symbol*`, left-recursive, which an Earley parser keeps to a constant number of items per position. */
	[[nodiscard]] auto star(Symbol symbol) -> Nonterminal {
		auto const out = nonterminal();
		rule(out, {});
		rule(out, {out, symbol});
		return out;
	}

	/** `symbol+` */
	[[nodiscard]] auto plus(Symbol symbol) -> Nonterminal {
		auto const out = nonterminal();
		rule(out, {symbol});
		rule(out, {out, symbol});
		return out;
	}

	/** `first (separator item)*`-style lists: `symbol{min,max}` with `separator` between items, max `unbounded` for none. */
	[[nodiscard]] auto separated(Symbol item, std::optional<Symbol> separator, std::uint32_t min, std::uint32_t max) -> Nonterminal {
		core::invariant(min <= max && max > 0);
		auto const next = [&](std::vector<Symbol> tail) {
			auto out = std::vector<Symbol>{};
			if (separator) {
				out.push_back(*separator);
			}
			out.push_back(item);
			std::ranges::copy(tail, std::back_inserter(out));
			return out;
		};
		// The items after the first: at least min - 1, at most max - 1.
		auto const later_min = min == 0 ? 0 : min - 1;
		auto rest = Nonterminal{};
		if (max == regex::unbounded) {
			auto const tail = nonterminal();
			rule(tail, {});
			auto repeated = std::vector<Symbol>{tail};
			if (separator) {
				repeated.push_back(*separator);
			}
			repeated.push_back(item);
			rule(tail, std::move(repeated));
			// Left recursion cannot count, so the mandatory items come first.
			rest = tail;
			for (auto count = std::uint32_t{0}; count < later_min; ++count) {
				auto const before = nonterminal();
				rule(before, next({rest}));
				rest = before;
			}
		} else {
			// Optional items nest right to left, so a list of n items costs n rules.
			rest = nonterminal();
			rule(rest, {});
			for (auto count = later_min; count + 1 < max; ++count) {
				auto const before = nonterminal();
				rule(before, {});
				rule(before, next({rest}));
				rest = before;
			}
			for (auto count = std::uint32_t{0}; count < later_min; ++count) {
				auto const before = nonterminal();
				rule(before, next({rest}));
				rest = before;
			}
			if (nonterminals_ > bounds_.rules) {
				overflow(Refusal::GrammarTooLarge);
			}
		}
		auto const out = nonterminal();
		if (min == 0) {
			rule(out, {});
		}
		rule(out, {item, rest});
		return out;
	}

	/** The first refusal recorded while building, if any. */
	[[nodiscard]] auto refused() const -> std::optional<Refusal> {
		if (store_.full()) {
			return Refusal::GrammarTooLarge;
		}
		return refused_;
	}

	[[nodiscard]] auto lexeme_specs() const -> std::span<LexemeSpec const> {
		return lexemes_;
	}

	[[nodiscard]] auto rule_specs() const -> std::span<RuleSpec const> {
		return rules_;
	}

	[[nodiscard]] auto nonterminal_count() const -> std::size_t {
		return nonterminals_;
	}

	/** The lexeme that closes a region. */
	[[nodiscard]] auto region_close(Region region) const -> Symbol {
		return regions_[region.index].close;
	}

	[[nodiscard]] auto region_specs() const -> std::span<RegionSpec const> {
		return regions_;
	}

private:
	[[nodiscard]] auto raw_lexeme(LexemeSpec spec) -> Symbol {
		auto const key = std::tuple{spec.expr.id, spec.mode, spec.region ? spec.region->index : std::numeric_limits<std::uint32_t>::max()};
		if (auto const found = lexeme_index_.find(key); found != lexeme_index_.end()) {
			return Lexeme{.index = found->second};
		}
		if (lexemes_.size() >= bounds_.lexemes) {
			overflow(Refusal::GrammarTooLarge);
			return Lexeme{.index = 0};
		}
		lexemes_.push_back(spec);
		auto const index = core::narrow<std::uint32_t>(lexemes_.size() - 1);
		lexeme_index_.emplace(key, index);
		return Lexeme{.index = index};
	}

	auto overflow(Refusal refusal) -> void {
		if (!refused_) {
			refused_ = refusal;
		}
	}

	Bounds bounds_;
	regex::Store store_;
	std::vector<LexemeSpec> lexemes_;
	std::map<std::tuple<std::uint32_t, Mode, std::uint32_t>, std::uint32_t> lexeme_index_;
	std::vector<RuleSpec> rules_;
	std::vector<RegionSpec> regions_;
	std::map<std::string, regex::Expr> memo_;
	std::size_t nonterminals_ = 0;
	std::size_t symbols_ = 0;
	std::optional<Refusal> refused_;
};

}

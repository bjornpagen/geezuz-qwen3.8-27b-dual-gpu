// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:lexer;

import std;
import engine.core;
import engine.regex;
import :builder;
import :compiled;

namespace engine::grammar {

/** A lexer state, by index into its lexer's table. State 0 is dead: no lexeme can go on. */
using LexerState = std::uint32_t;

inline constexpr LexerState dead_lexer = 0;

/** A region's guards as far as its text has come: free of forbidden words; a prefix of that and the closing. */
struct Guard {
	Region region;
	regex::State text;
	regex::State closed;

	[[nodiscard]] constexpr auto operator==(Guard const&) const -> bool = default;
};

/** One lexeme still possible, and its automaton's state. */
struct Entry {
	Lexeme lexeme;
	regex::State state;

	[[nodiscard]] constexpr auto operator==(Entry const&) const -> bool = default;
	[[nodiscard]] constexpr auto operator<=>(Entry const&) const = default;
};

struct Key {
	std::vector<Entry> entries;
	std::optional<Guard> guard;

	[[nodiscard]] auto operator==(Key const&) const -> bool = default;
};

struct KeyHash {
	[[nodiscard]] auto operator()(Key const& key) const -> std::size_t {
		auto hash = std::uint64_t{0xcbf29ce484222325};
		auto const mix = [&hash](std::uint64_t word) { hash = (hash ^ word) * 0x100000001b3; };
		for (auto const& entry : key.entries) {
			mix((std::uint64_t{entry.lexeme.index} << 32U) | entry.state);
		}
		if (key.guard) {
			mix((std::uint64_t{key.guard->region.index} << 32U) | key.guard->text | (std::uint64_t{1} << 63U));
			mix(key.guard->closed);
		}
		return static_cast<std::size_t>(hash);
	}
};

/** Whether entering a state ends a lazy lexeme there and then. */
enum class Ending : std::uint8_t {
	Open,
	Lazy,
};

struct Info {
	/** The greedy lexemes that would end here if the next byte cannot extend them. */
	std::vector<Lexeme> greedy;
	/** The lazy lexemes that end on entering this state. */
	std::vector<Lexeme> lazy;
	std::optional<Guard> guard;
};

/** The lexer ran past its state bound. */
enum class Exhaustion : std::uint8_t {
	Within,
	Exhausted,
};

/** The most lexer states one matcher builds: 16 MiB of transitions. Past it the matcher fails. */
export inline constexpr std::size_t max_lexer_states = 16384;

[[nodiscard]] auto region_of(Placement const& placement) -> std::optional<Region> {
	if (auto const inside = core::alternative<Inside>(placement)) {
		return inside->region;
	}
	if (auto const closing = core::alternative<Closing>(placement)) {
		return closing->region;
	}
	return std::nullopt;
}

/**
 * The lexer of one matcher: the product of the grammar's lexeme automata, built lazily, one state per set of
 * lexemes still possible with their automaton states, plus the guards of the region the text is in. Transitions
 * are a table filled on first use, so a state seen before costs one load per byte.
 */
class Lexer {
public:
	Lexer() {
		keys_.push_back(Key{.entries = {}, .guard = std::nullopt});
		info_.push_back(Info{.greedy = {}, .lazy = {}, .guard = std::nullopt});
		endings_.push_back(Ending::Open);
		table_.resize(256, dead_lexer);
	}

	/**
	 * The state at a lexeme's start, for the lexemes a parse expects and the region lexemes joined to their closing
	 * it may take, under the guards carried in, if any.
	 */
	[[nodiscard]] auto start(Grammar const& grammar, std::span<Lexeme const> lexemes, std::span<Lexeme const> closings, std::optional<Guard> carried) -> LexerState {
		auto key = Key{.entries = {}, .guard = std::nullopt};
		auto const add = [&](Lexeme lexeme) {
			auto const& compiled = grammar.lexeme(lexeme);
			key.entries.push_back(Entry{.lexeme = lexeme, .state = compiled.automaton.start()});
			if (auto const region = region_of(compiled.placement); region && !key.guard) {
				auto const& guards = grammar.guard(*region);
				key.guard = carried && carried->region == *region ? *carried
				                                                  : Guard{.region = *region, .text = guards.text.start(), .closed = guards.closed.start()};
			}
		};
		std::ranges::for_each(lexemes, add);
		std::ranges::for_each(closings, add);
		std::ranges::sort(key.entries);
		return intern(grammar, std::move(key));
	}

	[[nodiscard]] auto step(Grammar const& grammar, LexerState state, std::uint8_t byte) -> LexerState {
		auto const at = (std::size_t{state} * 256) + byte;
		auto const known = table_[at];
		if (known != unknown) {
			return known;
		}
		auto const next = compute(grammar, state, byte);
		table_[at] = next;
		return next;
	}

	[[nodiscard]] auto ending(LexerState state) const -> Ending {
		return endings_[state];
	}

	[[nodiscard]] auto info(LexerState state) const -> Info const& {
		return info_[state];
	}

	[[nodiscard]] auto exhaustion() const -> Exhaustion {
		return exhaustion_;
	}

	[[nodiscard]] auto states() const -> std::size_t {
		return keys_.size();
	}

private:
	static constexpr LexerState unknown = std::numeric_limits<LexerState>::max();

	[[nodiscard]] auto compute(Grammar const& grammar, LexerState state, std::uint8_t byte) -> LexerState {
		auto const& from = keys_[state];
		auto key = Key{.entries = {}, .guard = std::nullopt};
		if (from.guard) {
			auto const& guards = grammar.guard(from.guard->region);
			key.guard = Guard{.region = from.guard->region,
			                  .text = guards.text.step(from.guard->text, byte),
			                  .closed = guards.closed.step(from.guard->closed, byte)};
		}
		for (auto const entry : from.entries) {
			auto const& compiled = grammar.lexeme(entry.lexeme);
			// A lexeme in a region dies with its guard: the text would hold a forbidden word, or not close.
			if (std::holds_alternative<Inside>(compiled.placement) && key.guard.value().text == regex::dead) {
				continue;
			}
			if (std::holds_alternative<Closing>(compiled.placement) && key.guard.value().closed == regex::dead) {
				continue;
			}
			auto const next = compiled.automaton.step(entry.state, byte);
			if (next != regex::dead) {
				key.entries.push_back(Entry{.lexeme = entry.lexeme, .state = next});
			}
		}
		auto const guarded = std::ranges::any_of(key.entries, [&grammar](Entry const& entry) { return region_of(grammar.lexeme(entry.lexeme).placement).has_value(); });
		if (!guarded) {
			key.guard = std::nullopt;
		}
		return intern(grammar, std::move(key));
	}

	[[nodiscard]] auto intern(Grammar const& grammar, Key key) -> LexerState {
		if (key.entries.empty()) {
			return dead_lexer;
		}
		if (auto const found = index_.find(key); found != index_.end()) {
			return found->second;
		}
		if (keys_.size() >= max_lexer_states) {
			exhaustion_ = Exhaustion::Exhausted;
			return dead_lexer;
		}
		auto info = Info{.greedy = {}, .lazy = {}, .guard = key.guard};
		for (auto const entry : key.entries) {
			auto const& compiled = grammar.lexeme(entry.lexeme);
			if (compiled.automaton.accepting(entry.state)) {
				(compiled.mode == Mode::Lazy ? info.lazy : info.greedy).push_back(entry.lexeme);
			}
		}
		auto const state = core::narrow<LexerState>(keys_.size());
		endings_.push_back(info.lazy.empty() ? Ending::Open : Ending::Lazy);
		info_.push_back(std::move(info));
		index_.emplace(key, state);
		keys_.push_back(std::move(key));
		table_.resize(table_.size() + 256, unknown);
		return state;
	}

	std::vector<Key> keys_;
	std::unordered_map<Key, LexerState, KeyHash> index_;
	std::vector<Info> info_;
	std::vector<Ending> endings_;
	std::vector<LexerState> table_;
	Exhaustion exhaustion_ = Exhaustion::Within;
};

}

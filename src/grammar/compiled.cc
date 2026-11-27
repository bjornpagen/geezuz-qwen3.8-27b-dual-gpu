// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:compiled;

import std;
import engine.core;
import engine.regex;
import :error;
import :builder;
import :strings;
import :counting;

namespace engine::grammar {

/** The end of a rule's right-hand side, naming the rule. */
struct End {
	std::uint32_t rule;

	[[nodiscard]] constexpr auto operator==(End const&) const -> bool = default;
};

/** A position in the flat table of right-hand sides: the symbol after an Earley item's dot, or its rule's end. */
using Slot = std::variant<Lexeme, Nonterminal, End>;

struct Rule {
	Nonterminal lhs;
	/** The slot of its first symbol (or its end, for an empty rule). */
	std::uint32_t first;
};

/** A lexeme outside every region. */
struct Free {};

/** A lexeme inside a region: it dies when the region's text so far holds a forbidden word. */
struct Inside {
	Region region;
};

/**
 * An inside lexeme joined to its region's closing: it lives while the region's text so far is a prefix of
 * forbidden-free text and then the closing, and when it matches, the parser scans `inner` and then `close`.
 */
struct Closing {
	Region region;
	Lexeme inner;
	Lexeme close;
};

using Placement = std::variant<Free, Inside, Closing>;

struct CompiledLexeme {
	regex::Automaton automaton;
	Mode mode;
	Placement placement;
	/** Its length bound, counted beside the automaton, if it has one. */
	std::optional<Counting> counting;
};

/** A region's two guards, run over all of its text: free of forbidden words; a prefix of that and the closing. */
struct GuardAutomata {
	regex::Automaton text;
	regex::Automaton closed;
};

/**
 * A checked, compiled grammar: lexemes as whole automata, regions' guards as automata, rules as a flat slot table
 * with each nonterminal's predictions and nullability. Every rule derives some finite output and is reachable from
 * the start, so a parse can always continue to an end. An owned, immutable value: built once per request at the
 * boundary, moved to the engine, read by each matcher made from it.
 */
export class Grammar {
public:
	[[nodiscard]] static auto compile(Builder builder, Nonterminal start) -> core::Result<Grammar, GrammarError>;

	[[nodiscard]] auto lexeme_count() const -> std::size_t {
		return lexemes_.size();
	}

	[[nodiscard]] auto rule_count() const -> std::size_t {
		return rules_.size();
	}

	/** States across every automaton. */
	[[nodiscard]] auto state_count() const -> std::size_t {
		auto total = std::size_t{0};
		for (auto const& lexeme : lexemes_) {
			total += lexeme.automaton.states();
		}
		for (auto const& guard : guards_) {
			total += guard.text.states() + guard.closed.states();
		}
		return total;
	}

	// The parser's view.
	[[nodiscard]] auto lexeme(Lexeme lexeme) const -> CompiledLexeme const& {
		return lexemes_[lexeme.index];
	}

	/** The automaton of one unit, for a grammar with counted lexemes of that unit. */
	[[nodiscard]] auto unit(Unit unit) const -> regex::Automaton const& {
		return units_.at(std::to_underlying(unit)).value();
	}

	[[nodiscard]] auto remainders(Counting const& counting) const -> Remainders const& {
		return remainders_[counting.table];
	}

	[[nodiscard]] auto guard(Region region) const -> GuardAutomata const& {
		return guards_[region.index];
	}

	/** The lexeme joining an inside lexeme to its region's closing, if it is one. */
	[[nodiscard]] auto closing(Lexeme lexeme) const -> std::optional<Lexeme> {
		return closings_[lexeme.index];
	}

	[[nodiscard]] auto slot(std::uint32_t at) const -> Slot const& {
		return slots_[at];
	}

	[[nodiscard]] auto rule(std::uint32_t index) const -> Rule const& {
		return rules_[index];
	}

	/** The first slots of a nonterminal's rules. */
	[[nodiscard]] auto predictions(Nonterminal nonterminal) const -> std::span<std::uint32_t const> {
		auto const begin = prediction_begin_[nonterminal.index];
		return std::span{predictions_}.subspan(begin, prediction_begin_[nonterminal.index + 1] - begin);
	}

	[[nodiscard]] auto nullable(Nonterminal nonterminal) const -> bool {
		return nullable_[nonterminal.index] != 0;
	}

	[[nodiscard]] auto start() const -> Nonterminal {
		return start_;
	}

	[[nodiscard]] auto nonterminal_count() const -> std::size_t {
		return nullable_.size();
	}

private:
	Grammar() = default;

	std::vector<CompiledLexeme> lexemes_;
	std::vector<GuardAutomata> guards_;
	std::array<std::optional<regex::Automaton>, 2> units_;
	std::vector<Remainders> remainders_;
	std::vector<std::optional<Lexeme>> closings_;
	std::vector<Slot> slots_;
	std::vector<Rule> rules_;
	std::vector<std::uint32_t> predictions_;
	std::vector<std::uint32_t> prediction_begin_;
	std::vector<std::uint8_t> nullable_;
	Nonterminal start_{};
};

/** A rule as a fixed point sees it: its left side holds once every nonterminal it needs holds. */
struct Requirement {
	Nonterminal lhs;
	std::vector<Nonterminal> needs;
};

/**
 * The least fixed point of the requirements: which nonterminals hold. A worklist counts each rule's unmet needs,
 * so the cost is linear in the rules' sizes however the rules are ordered.
 */
[[nodiscard]] auto least_fixed_point(std::size_t nonterminals, std::span<Requirement const> rules) -> std::vector<bool> {
	auto holds = std::vector<bool>(nonterminals, false);
	auto unmet = std::vector<std::size_t>(rules.size());
	auto users = std::vector<std::vector<std::size_t>>(nonterminals);
	auto ready = std::vector<std::size_t>{};
	for (auto const [index, rule] : std::views::enumerate(rules)) {
		auto const at = core::narrow<std::size_t>(index);
		unmet[at] = rule.needs.size();
		for (auto const need : rule.needs) {
			users[need.index].push_back(at);
		}
		if (rule.needs.empty()) {
			ready.push_back(at);
		}
	}
	while (!ready.empty()) {
		auto const lhs = rules[ready.back()].lhs;
		ready.pop_back();
		if (holds[lhs.index]) {
			continue;
		}
		holds[lhs.index] = true;
		for (auto const user : users[lhs.index]) {
			if (--unmet[user] == 0) {
				ready.push_back(user);
			}
		}
	}
	return holds;
}

/** Which nonterminals derive some finite string, given which lexemes match anything. */
[[nodiscard]] auto productive(std::span<RuleSpec const> rules, std::size_t nonterminals, std::vector<bool> const& lexeme_live) -> std::vector<bool> {
	auto requirements = std::vector<Requirement>{};
	for (auto const& rule : rules) {
		auto requirement = Requirement{.lhs = rule.lhs, .needs = {}};
		auto possible = true;
		for (auto const& symbol : rule.rhs) {
			if (auto const lexeme = core::alternative<Lexeme>(symbol)) {
				possible = possible && lexeme_live[lexeme->index];
			} else {
				requirement.needs.push_back(std::get<Nonterminal>(symbol));
			}
		}
		if (possible) {
			requirements.push_back(std::move(requirement));
		}
	}
	return least_fixed_point(nonterminals, requirements);
}

auto Grammar::compile(Builder builder, Nonterminal start) -> core::Result<Grammar, GrammarError> {
	if (auto const refused = builder.refused()) {
		return refuse(*refused, "");
	}
	auto const bounds = builder.bounds();
	auto const all_rules = builder.rule_specs();
	auto& store = builder.store();
	auto out = Grammar{};
	// Every lexeme's automaton first: one whose language is empty makes its rules derive nothing.
	auto states = std::size_t{0};
	auto const budget = [&] { return std::min(bounds.lexeme_states, bounds.states - std::min(states, bounds.states)); };
	auto automata = std::vector<std::optional<regex::Automaton>>{};
	auto countings = std::vector<std::optional<Counting>>{};
	auto lexeme_live = std::vector<bool>{};
	// Tables by language and unit: lexemes that differ only in their bound share one.
	auto tables = std::map<std::pair<std::uint32_t, Unit>, std::optional<std::uint32_t>>{};
	for (auto const& spec : builder.lexeme_specs()) {
		auto automaton = regex::Automaton::build(store, spec.expr, budget());
		if (!automaton) {
			return refuse(Refusal::AutomatonTooLarge, "");
		}
		auto counting = std::optional<Counting>{};
		if (spec.count) {
			auto& unit = out.units_.at(std::to_underlying(spec.count->unit));
			if (!unit) {
				unit = regex::Automaton::build(store, unit_language(store, spec.count->unit), budget()).value();
			}
			auto const key = std::pair{spec.expr.id, spec.count->unit};
			if (!tables.contains(key)) {
				auto table = remainders_of(*automaton, *unit);
				tables[key] = table ? std::optional<std::uint32_t>{core::narrow<std::uint32_t>(out.remainders_.size())} : std::nullopt;
				if (table) {
					out.remainders_.push_back(std::move(*table));
				}
			}
			if (auto const table = tables.at(key)) {
				counting = Counting{.unit = spec.count->unit, .min = spec.count->min, .max = spec.count->max, .table = *table};
			} else {
				// A language whose remaining counts the table cannot hold: the bound goes into the automaton.
				automaton = regex::Automaton::build(store, spec.count->fallback, budget());
				if (!automaton) {
					return refuse(Refusal::AutomatonTooLarge, "");
				}
			}
		}
		states += automaton->states();
		auto const live = counting ? counting->feasible(out.remainders_[counting->table], automaton->start(), out.units_.at(std::to_underlying(counting->unit))->start(), 0)
		                           : automaton->start() != regex::dead;
		lexeme_live.push_back(live);
		automata.emplace_back(std::move(*automaton));
		countings.push_back(counting);
	}
	auto const derives = productive(all_rules, builder.nonterminal_count(), lexeme_live);
	if (!derives[start.index]) {
		return refuse(Refusal::Empty, "");
	}
	// A rule that cannot derive a finite string would lead a parse into a dead end: it is dropped.
	auto by_lhs = std::vector<std::vector<std::size_t>>(builder.nonterminal_count());
	for (auto const [index, rule] : std::views::enumerate(all_rules)) {
		auto const keeps = std::ranges::all_of(rule.rhs, [&](Symbol const& symbol) {
			if (auto const lexeme = core::alternative<Lexeme>(symbol)) {
				return bool{lexeme_live[lexeme->index]};
			}
			return bool{derives[std::get<Nonterminal>(symbol).index]};
		});
		if (keeps) {
			by_lhs[rule.lhs.index].push_back(core::narrow<std::size_t>(index));
		}
	}
	// Only what the start reaches is kept, renumbered densely in the order reached.
	auto nonterminal_map = std::vector<std::optional<std::uint32_t>>(builder.nonterminal_count());
	auto lexeme_map = std::vector<std::optional<std::uint32_t>>(builder.lexeme_specs().size());
	auto order = std::vector<Nonterminal>{start};
	auto used_lexemes = std::vector<Lexeme>{};
	nonterminal_map[start.index] = 0;
	for (auto at = std::size_t{0}; at < order.size(); ++at) {
		for (auto const rule_index : by_lhs[order[at].index]) {
			for (auto const& symbol : all_rules[rule_index].rhs) {
				if (auto const nonterminal = core::alternative<Nonterminal>(symbol)) {
					if (!nonterminal_map[nonterminal->index]) {
						nonterminal_map[nonterminal->index] = core::narrow<std::uint32_t>(order.size());
						order.push_back(*nonterminal);
					}
				} else {
					auto const lexeme = std::get<Lexeme>(symbol);
					if (!lexeme_map[lexeme.index]) {
						lexeme_map[lexeme.index] = core::narrow<std::uint32_t>(used_lexemes.size());
						used_lexemes.push_back(lexeme);
					}
				}
			}
		}
	}
	auto const regions = builder.region_specs();
	for (auto const lexeme : used_lexemes) {
		auto const& spec = builder.lexeme_specs()[lexeme.index];
		auto const placement = spec.region ? Placement{Inside{.region = spec.region.value()}} : Placement{Free{}};
		out.lexemes_.push_back(CompiledLexeme{.automaton = std::move(automata[lexeme.index]).value(), .mode = spec.mode, .placement = placement, .counting = countings[lexeme.index]});
	}
	out.closings_.assign(out.lexemes_.size(), std::nullopt);
	for (auto const [number, lexeme] : std::views::enumerate(used_lexemes)) {
		auto const& spec = builder.lexeme_specs()[lexeme.index];
		if (!spec.region) {
			continue;
		}
		auto const& region = regions[spec.region->index];
		auto const close = lexeme_map[region.close.index];
		if (!close) {
			// The region's closing is used nowhere, so nothing in it can close.
			continue;
		}
		auto automaton = regex::Automaton::build(store, store.concat(spec.expr, region.closing), budget());
		if (!automaton) {
			return refuse(Refusal::AutomatonTooLarge, "");
		}
		states += automaton->states();
		auto const inner = Lexeme{.index = core::narrow<std::uint32_t>(number)};
		out.closings_[inner.index] = Lexeme{.index = core::narrow<std::uint32_t>(out.lexemes_.size())};
		out.lexemes_.push_back(CompiledLexeme{.automaton = std::move(*automaton),
		                                      .mode = Mode::Lazy,
		                                      .placement = Closing{.region = *spec.region, .inner = inner, .close = Lexeme{.index = *close}},
		                                      .counting = std::nullopt});
	}
	out.closings_.resize(out.lexemes_.size(), std::nullopt);
	for (auto const& region : regions) {
		auto const text = avoiding(store, region.forbidden);
		auto guard = regex::Automaton::build(store, text, budget());
		auto closed = regex::Automaton::build(store, store.concat(text, region.closing), budget());
		if (!guard || !closed) {
			return refuse(Refusal::AutomatonTooLarge, "");
		}
		states += guard->states() + closed->states();
		out.guards_.push_back(GuardAutomata{.text = std::move(*guard), .closed = std::move(*closed)});
	}
	if (store.full()) {
		return refuse(Refusal::GrammarTooLarge, "");
	}
	auto const remap = [&](Symbol const& symbol) -> Slot {
		if (auto const nonterminal = core::alternative<Nonterminal>(symbol)) {
			return Nonterminal{.index = *nonterminal_map[nonterminal->index]};
		}
		return Lexeme{.index = *lexeme_map[std::get<Lexeme>(symbol).index]};
	};
	auto per_nonterminal = std::vector<std::vector<std::uint32_t>>(order.size());
	for (auto const [number, old] : std::views::enumerate(order)) {
		for (auto const rule_index : by_lhs[old.index]) {
			auto const rule_number = core::narrow<std::uint32_t>(out.rules_.size());
			auto const first = core::narrow<std::uint32_t>(out.slots_.size());
			out.rules_.push_back(Rule{.lhs = Nonterminal{.index = core::narrow<std::uint32_t>(number)}, .first = first});
			std::ranges::transform(all_rules[rule_index].rhs, std::back_inserter(out.slots_), remap);
			out.slots_.emplace_back(End{.rule = rule_number});
			per_nonterminal[core::narrow<std::size_t>(number)].push_back(first);
		}
	}
	out.prediction_begin_.push_back(0);
	for (auto const& firsts : per_nonterminal) {
		std::ranges::copy(firsts, std::back_inserter(out.predictions_));
		out.prediction_begin_.push_back(core::narrow<std::uint32_t>(out.predictions_.size()));
	}
	// A rule derives the empty string when all its symbols are nonterminals that do.
	auto empties = std::vector<Requirement>{};
	for (auto const& rule : out.rules_) {
		auto requirement = Requirement{.lhs = rule.lhs, .needs = {}};
		auto possible = true;
		for (auto at = rule.first; !std::holds_alternative<End>(out.slots_[at]); ++at) {
			if (auto const nonterminal = core::alternative<Nonterminal>(out.slots_[at])) {
				requirement.needs.push_back(*nonterminal);
			} else {
				possible = false;
			}
		}
		if (possible) {
			empties.push_back(std::move(requirement));
		}
	}
	auto const nullable = least_fixed_point(order.size(), empties);
	out.nullable_.assign(nullable.begin(), nullable.end());
	out.start_ = Nonterminal{.index = 0};
	return out;
}

}

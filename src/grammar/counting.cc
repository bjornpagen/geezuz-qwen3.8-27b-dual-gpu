// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:counting;

import std;
import engine.core;
import engine.regex;
import :builder;

namespace engine::grammar {

/** The language of one unit: prefix-free, so a unit ends exactly where its automaton first accepts. */
[[nodiscard]] auto unit_language(regex::Store& store, Unit unit) -> regex::Expr {
	switch (unit) {
	case Unit::Character:
		return regex::any_character(store);
	case Unit::JsonCharacter: {
		auto const backslash = regex::CharClass::single('\\');
		auto const plain = regex::characters(store, regex::CharClass::all().minus(backslash));
		auto const short_escape = store.concat(store.byte('\\'), regex::characters(store, regex::CharClass::all().minus(regex::CharClass::single('u'))));
		auto const hex = regex::characters(store, regex::CharClass::of(std::to_array<regex::Range>({
		                                              {.low = '0', .high = '9'},
		                                              {.low = 'A', .high = 'F'},
		                                              {.low = 'a', .high = 'f'},
		                                          })));
		auto const long_escape = store.concat(store.literal("\\u"), store.repeat(hex, 4, 4));
		return store.any_of(std::array{plain, short_escape, long_escape});
	}
	}
	std::unreachable();
}

/** Unit counts tracked exactly below this; at and past it, only whether every count from some point on works. */
inline constexpr std::uint32_t horizon = 512;
inline constexpr std::size_t horizon_words = horizon / 64;
inline constexpr std::uint32_t no_tail = std::numeric_limits<std::uint32_t>::max();

/**
 * The numbers of units that can still follow a point inside a counted lexeme and end it: exactly below `horizon`,
 * and from `tail` on every number (`no_tail` when none past the horizon).
 */
struct Remaining {
	std::array<std::uint64_t, horizon_words> near;
	std::uint32_t tail;

	/** Whether some count in [low, high] is one of them. */
	[[nodiscard]] auto reaches(std::uint32_t low, std::uint32_t high) const -> bool {
		if (low > high) {
			return false;
		}
		if (tail != no_tail && high >= tail) {
			return true;
		}
		for (auto at = low; at <= std::min(high, horizon - 1); ++at) {
			if (((near.at(at / 64) >> (at % 64U)) & 1U) != 0) {
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] auto any() const -> bool {
		return tail != no_tail || std::ranges::any_of(near, [](std::uint64_t word) { return word != 0; });
	}
};

/**
 * For every point inside a counted lexeme (a lexeme automaton state and a unit automaton state), the unit counts that
 * can still end it. It depends on the lexeme's language only, so lexemes of one language and different bounds share
 * one table.
 */
struct Remainders {
	std::uint32_t unit_states;
	std::vector<Remaining> remaining;
	/** The largest tail of any point: within it of the bound, a slice of free text is no longer taken whole. */
	std::uint32_t slack;

	[[nodiscard]] auto at(regex::State lexeme, regex::State unit_state) const -> Remaining const& {
		return remaining[(std::size_t{lexeme} * unit_states) + unit_state];
	}
};

/**
 * A counted lexeme's bound, in units of its whole text, and its table. A point with `units` counted so far can go on
 * exactly when some remaining count lands the total in [min, max]; checking that on every byte keeps each mask exact.
 */
struct Counting {
	Unit unit;
	std::uint32_t min;
	std::uint32_t max;
	std::uint32_t table;

	[[nodiscard]] auto feasible(Remainders const& remainders, regex::State lexeme, regex::State unit_state, std::uint32_t units) const -> bool {
		if (units > max) {
			return false;
		}
		return remainders.at(lexeme, unit_state).reaches(min > units ? min - units : 0, max - units);
	}

	[[nodiscard]] auto accepts(std::uint32_t units) const -> bool {
		return min <= units && units <= max;
	}
};

/** One byte's step between points: the next point, and whether it completed a unit. */
struct Edge {
	std::uint32_t to;
	std::uint32_t weight;
};

/**
 * The remaining counts of a lexeme automaton under a unit automaton, or nothing when some point's remaining counts
 * are not of the exact shape `Remaining` holds: a finite set below the horizon, plus everything from a tail on, the
 * tail proven by a point on the way that takes one more unit and returns to itself.
 */
[[nodiscard]] auto remainders_of(regex::Automaton const& lexeme, regex::Automaton const& unit) -> std::optional<Remainders> {
	auto const unit_states = core::narrow<std::uint32_t>(unit.states());
	auto const pairs = lexeme.states() * unit_states;
	auto const index = [unit_states](regex::State state, regex::State unit_state) { return core::narrow<std::uint32_t>((std::size_t{state} * unit_states) + unit_state); };
	// One byte per class both automata tell apart.
	auto representatives = std::vector<std::uint8_t>{};
	auto seen_classes = std::set<std::pair<std::uint32_t, std::uint32_t>>{};
	for (auto byte = 0U; byte < 256U; ++byte) {
		auto const value = static_cast<std::uint8_t>(byte);
		if (seen_classes.insert({lexeme.byte_class(value), unit.byte_class(value)}).second) {
			representatives.push_back(value);
		}
	}
	// The points reachable from the start, and their edges.
	auto edges = std::vector<std::vector<Edge>>(pairs);
	auto reached = std::vector<bool>(pairs, false);
	auto pending = std::vector<std::uint32_t>{index(lexeme.start(), unit.start())};
	reached[pending.back()] = true;
	while (!pending.empty()) {
		auto const at = pending.back();
		pending.pop_back();
		auto const state = core::narrow<regex::State>(at / unit_states);
		auto const unit_state = core::narrow<regex::State>(at % unit_states);
		for (auto const byte : representatives) {
			auto const next = lexeme.step(state, byte);
			auto next_unit = unit.step(unit_state, byte);
			if (next == regex::dead || next_unit == regex::dead) {
				continue;
			}
			auto weight = 0U;
			if (unit.accepting(next_unit)) {
				next_unit = unit.start();
				weight = 1;
			}
			auto const to = index(next, next_unit);
			edges[at].push_back(Edge{.to = to, .weight = weight});
			if (!reached[to]) {
				reached[to] = true;
				pending.push_back(to);
			}
		}
	}
	auto const ends = [&](std::uint32_t at) { return lexeme.accepting(core::narrow<regex::State>(at / unit_states)) && at % unit_states == unit.start(); };
	// The fewest units from each point to an end, and the points that take one unit and return to themselves.
	auto const shortest = [&](std::vector<std::uint32_t> sources_cost) {
		// Costs over reversed edges, from the given starting costs: a small Dijkstra, weights 0 and 1.
		auto into = std::vector<std::vector<Edge>>(pairs);
		for (auto at = std::uint32_t{0}; at < pairs; ++at) {
			for (auto const edge : edges[at]) {
				into[edge.to].push_back(Edge{.to = at, .weight = edge.weight});
			}
		}
		auto queue = std::priority_queue<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::pair<std::uint32_t, std::uint32_t>>, std::greater<>>{};
		for (auto at = std::uint32_t{0}; at < pairs; ++at) {
			if (sources_cost[at] != no_tail) {
				queue.emplace(sources_cost[at], at);
			}
		}
		while (!queue.empty()) {
			auto const [cost, at] = queue.top();
			queue.pop();
			if (cost > sources_cost[at]) {
				continue;
			}
			for (auto const edge : into[at]) {
				if (cost + edge.weight < sources_cost[edge.to]) {
					sources_cost[edge.to] = cost + edge.weight;
					queue.emplace(sources_cost[edge.to], edge.to);
				}
			}
		}
		return sources_cost;
	};
	auto end_cost = std::vector<std::uint32_t>(pairs, no_tail);
	for (auto at = std::uint32_t{0}; at < pairs; ++at) {
		if (reached[at] && ends(at)) {
			end_cost[at] = 0;
		}
	}
	auto const fewest = shortest(end_cost);
	auto loop_cost = std::vector<std::uint32_t>(pairs, no_tail);
	for (auto at = std::uint32_t{0}; at < pairs; ++at) {
		if (!reached[at] || at % unit_states != unit.start() || fewest[at] == no_tail) {
			continue;
		}
		// One unit away: the points within a unit of `at`, then a completing byte back to `at`.
		auto inside = std::vector<std::uint32_t>{at};
		auto within = std::set<std::uint32_t>{at};
		auto returns = false;
		for (auto step = std::size_t{0}; step < inside.size() && !returns; ++step) {
			for (auto const edge : edges[inside[step]]) {
				if (edge.weight == 1) {
					returns = returns || edge.to == at;
				} else if (within.insert(edge.to).second) {
					inside.push_back(edge.to);
				}
			}
		}
		if (returns) {
			loop_cost[at] = fewest[at];
		}
	}
	auto const tails = shortest(loop_cost);
	// Remaining counts below the horizon, by relaxing to a fixed point.
	auto near = std::vector<std::array<std::uint64_t, horizon_words>>(pairs, std::array<std::uint64_t, horizon_words>{});
	for (auto at = std::uint32_t{0}; at < pairs; ++at) {
		if (reached[at] && ends(at)) {
			near[at][0] = 1;
		}
		// Every count from the tail on is proven: seeding them leaves the relaxation only the finite part to find.
		for (auto bit = std::min(tails[at], horizon); bit < horizon; ++bit) {
			near[at][bit / 64] |= std::uint64_t{1} << (bit % 64U);
		}
	}
	for (auto changed = true; changed;) {
		changed = false;
		for (auto at = std::uint32_t{0}; at < pairs; ++at) {
			for (auto const edge : edges[at]) {
				auto incoming = near[edge.to];
				if (edge.weight == 1) {
					auto carry = std::uint64_t{0};
					for (auto& word : incoming) {
						auto const next_carry = word >> 63U;
						word = (word << 1U) | carry;
						carry = next_carry;
					}
				}
				for (auto word = std::size_t{0}; word < horizon_words; ++word) {
					auto const merged = near[at][word] | incoming[word];
					if (merged != near[at][word]) {
						near[at][word] = merged;
						changed = true;
					}
				}
			}
		}
	}
	// A point with no tail must reach no cycle on the way to an end: its counts are then finite and all below the
	// horizon. Points on a cycle are those whose strongly connected component, among the points that can end, has
	// more than one member or an edge to itself.
	auto cyclic = std::vector<bool>(pairs, false);
	{
		auto order = std::vector<std::uint32_t>(pairs, no_tail);
		auto low = std::vector<std::uint32_t>(pairs, 0);
		auto on_stack = std::vector<bool>(pairs, false);
		auto stack = std::vector<std::uint32_t>{};
		auto counter = std::uint32_t{0};
		struct Frame {
			std::uint32_t at;
			std::size_t edge;
		};
		for (auto root = std::uint32_t{0}; root < pairs; ++root) {
			if (!reached[root] || fewest[root] == no_tail || order[root] != no_tail) {
				continue;
			}
			auto frames = std::vector<Frame>{{.at = root, .edge = 0}};
			order[root] = low[root] = counter++;
			stack.push_back(root);
			on_stack[root] = true;
			while (!frames.empty()) {
				auto& frame = frames.back();
				if (frame.edge < edges[frame.at].size()) {
					auto const to = edges[frame.at][frame.edge].to;
					++frame.edge;
					if (fewest[to] == no_tail) {
						continue;
					}
					if (to == frame.at) {
						cyclic[to] = true;
					}
					if (order[to] == no_tail) {
						order[to] = low[to] = counter++;
						stack.push_back(to);
						on_stack[to] = true;
						frames.push_back(Frame{.at = to, .edge = 0});
					} else if (on_stack[to]) {
						low[frame.at] = std::min(low[frame.at], order[to]);
					}
					continue;
				}
				auto const at = frame.at;
				frames.pop_back();
				if (!frames.empty()) {
					low[frames.back().at] = std::min(low[frames.back().at], low[at]);
				}
				if (low[at] == order[at]) {
					auto members = std::vector<std::uint32_t>{};
					while (true) {
						auto const member = stack.back();
						stack.pop_back();
						on_stack[member] = false;
						members.push_back(member);
						if (member == at) {
							break;
						}
					}
					if (members.size() > 1) {
						for (auto const member : members) {
							cyclic[member] = true;
						}
					}
				}
			}
		}
	}
	// Whether each point reaches a cycle among the points that can end.
	auto reaches_cycle = cyclic;
	for (auto changed = true; changed;) {
		changed = false;
		for (auto at = std::uint32_t{0}; at < pairs; ++at) {
			if (reaches_cycle[at] || fewest[at] == no_tail) {
				continue;
			}
			if (std::ranges::any_of(edges[at], [&](Edge const& edge) { return fewest[edge.to] != no_tail && reaches_cycle[edge.to]; })) {
				reaches_cycle[at] = true;
				changed = true;
			}
		}
	}
	// Without a cycle a path to an end visits each point once, so it completes fewer units than there are points.
	auto const reachable = core::narrow<std::uint32_t>(std::ranges::count(reached, true));
	auto out = Remainders{.unit_states = unit_states, .remaining = {}, .slack = 0};
	out.remaining.assign(pairs, Remaining{.near = {}, .tail = no_tail});
	for (auto at = std::uint32_t{0}; at < pairs; ++at) {
		if (!reached[at] || fewest[at] == no_tail) {
			continue;
		}
		auto const tail = tails[at];
		if (tail == no_tail ? reaches_cycle[at] || reachable >= horizon : tail >= horizon) {
			// Counts past the horizon that are not all of them from a tail on: not a shape the table holds.
			return std::nullopt;
		}
		out.remaining[at] = Remaining{.near = near[at], .tail = tail};
		if (tail != no_tail) {
			out.slack = std::max(out.slack, tail);
		}
	}
	return out;
}

}

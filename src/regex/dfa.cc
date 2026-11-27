// SPDX-License-Identifier: AGPL-3.0-only
export module engine.regex:dfa;

import std;
import engine.core;
import :expr;

namespace engine::regex {

/** A state of one automaton. State 0 is dead: it accepts nothing, now or later. */
export using State = std::uint32_t;

export inline constexpr State dead = 0;

/** An automaton passed its state budget while it was being built. */
export struct [[nodiscard]] AutomatonTooLarge {
	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(AutomatonTooLarge const&) const -> bool = default;
};

/**
 * The deterministic automaton of one expression, built whole: its states are the expression's distinct derivatives,
 * its transitions indexed by byte class. Built once when a grammar compiles, so matching never computes a derivative.
 */
export class Automaton {
public:
	/** Explores every derivative of `root`; refuses past `budget` states (the dead state included). */
	[[nodiscard]] static auto build(Store& store, Expr root, std::size_t budget) -> core::Result<Automaton, AutomatonTooLarge> {
		if (auto const text = store.literal_text(root); text && !text->empty()) {
			return chain(*text, budget);
		}
		auto const roots = std::array{root};
		auto out = Automaton{};
		out.classes_ = store.byte_classes(roots);
		out.class_count_ = std::uint32_t{*std::ranges::max_element(out.classes_)} + 1;
		auto representatives = std::vector<std::uint8_t>(out.class_count_);
		for (auto byte = 0U; byte < 256U; ++byte) {
			representatives[out.classes_.at(byte)] = static_cast<std::uint8_t>(byte);
		}
		auto exprs = std::vector<Expr>{nothing};
		auto index = std::unordered_map<std::uint32_t, State>{{nothing.id, dead}};
		auto const intern = [&](Expr expr) -> std::optional<State> {
			if (auto const found = index.find(expr.id); found != index.end()) {
				return found->second;
			}
			if (exprs.size() >= budget) {
				return std::nullopt;
			}
			auto const state = core::narrow<State>(exprs.size());
			exprs.push_back(expr);
			index.emplace(expr.id, state);
			return state;
		};
		auto const start = intern(root);
		if (!start) {
			return std::unexpected(AutomatonTooLarge{});
		}
		out.start_ = *start;
		for (auto state = State{0}; state < exprs.size(); ++state) {
			for (auto const representative : representatives) {
				auto const next = intern(store.derivative(exprs[state], representative));
				if (!next || store.full()) {
					return std::unexpected(AutomatonTooLarge{});
				}
				out.next_.push_back(*next);
			}
			out.accepting_.push_back(store.nullable(exprs[state]) ? 1 : 0);
		}
		out.prune();
		return out;
	}

	[[nodiscard]] auto start() const -> State {
		return start_;
	}

	[[nodiscard]] auto step(State state, std::uint8_t byte) const -> State {
		return next_[(std::size_t{state} * class_count_) + classes_.at(byte)];
	}

	[[nodiscard]] auto accepting(State state) const -> bool {
		return accepting_[state] != 0;
	}

	/** Whether no byte leads anywhere but the dead state. */
	[[nodiscard]] auto finished(State state) const -> bool {
		auto const row = std::span{next_}.subspan(std::size_t{state} * class_count_, class_count_);
		return std::ranges::all_of(row, [](State next) { return next == dead; });
	}

	/** The class of a byte: bytes of one class lead every state to the same state. */
	[[nodiscard]] auto byte_class(std::uint8_t byte) const -> std::uint32_t {
		return classes_.at(byte);
	}

	[[nodiscard]] auto class_count() const -> std::uint32_t {
		return class_count_;
	}

	[[nodiscard]] auto states() const -> std::size_t {
		return accepting_.size();
	}

	/** Whether the automaton matches the whole of `text`. */
	[[nodiscard]] auto matches(std::string_view text) const -> bool {
		auto state = start_;
		for (auto const letter : text) {
			state = step(state, std::bit_cast<std::uint8_t>(letter));
		}
		return accepting(state);
	}

private:
	Automaton() = default;

	/** A literal's automaton, built as the chain it is: names, keys and enum values are most lexemes. */
	[[nodiscard]] static auto chain(std::string_view text, std::size_t budget) -> core::Result<Automaton, AutomatonTooLarge> {
		if (text.size() + 2 > budget) {
			return std::unexpected(AutomatonTooLarge{});
		}
		auto out = Automaton{};
		// Class 0 is every byte the literal does not hold; each byte it holds has a class of its own.
		for (auto const letter : text) {
			auto& assigned = out.classes_.at(std::bit_cast<std::uint8_t>(letter));
			if (assigned == 0) {
				assigned = core::narrow<std::uint8_t>(++out.class_count_);
			}
		}
		++out.class_count_;
		auto const states = text.size() + 2;
		out.next_.assign(states * out.class_count_, dead);
		out.accepting_.assign(states, 0);
		for (auto at = std::size_t{0}; at < text.size(); ++at) {
			auto const state = at + 1;
			out.next_[(state * out.class_count_) + out.classes_.at(std::bit_cast<std::uint8_t>(text[at]))] = core::narrow<State>(state + 1);
		}
		out.accepting_[states - 1] = 1;
		out.start_ = 1;
		return out;
	}

	/**
	 * Sends every transition into a state that can never reach acceptance to the dead state. Intersections and
	 * complements leave such states behind (an expression need not simplify to the empty language to denote it), and
	 * a lexer that entered one could never finish its lexeme.
	 */
	auto prune() -> void {
		auto const count = accepting_.size();
		// Live states are found backwards from the accepting ones over the reversed transitions.
		auto into_begin = std::vector<std::size_t>(count + 1, 0);
		for (auto const next : next_) {
			++into_begin[next + 1];
		}
		std::partial_sum(into_begin.begin(), into_begin.end(), into_begin.begin());
		auto into = std::vector<State>(next_.size());
		auto fill = into_begin;
		for (auto at = std::size_t{0}; at < next_.size(); ++at) {
			into[fill[next_[at]]++] = core::narrow<State>(at / class_count_);
		}
		auto live = std::vector<bool>(count, false);
		auto pending = std::vector<State>{};
		for (auto state = std::size_t{0}; state < count; ++state) {
			if (accepting_[state] != 0) {
				live[state] = true;
				pending.push_back(core::narrow<State>(state));
			}
		}
		while (!pending.empty()) {
			auto const state = pending.back();
			pending.pop_back();
			for (auto at = into_begin[state]; at < into_begin[state + 1]; ++at) {
				if (!live[into[at]]) {
					live[into[at]] = true;
					pending.push_back(into[at]);
				}
			}
		}
		for (auto& next : next_) {
			if (!live[next]) {
				next = dead;
			}
		}
		if (!live[start_]) {
			start_ = dead;
		}
	}

	std::array<std::uint8_t, 256> classes_{};
	std::uint32_t class_count_ = 0;
	State start_ = dead;
	std::vector<State> next_;
	std::vector<std::uint8_t> accepting_;
};

}

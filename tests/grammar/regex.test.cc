// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.regex;
import engine.testing;

namespace {

namespace regex = engine::regex;

/** A regular expression as a tree, for the naive matcher; rendered to text for the engine. */
/** A node of a regex tree, by index into its arena. */
struct Tree {
	std::size_t index;
};

struct Character {
	char32_t point;
};
/** A class of explicit members, possibly negated. */
struct Set {
	std::vector<char32_t> members;
	bool negated;
};
struct Dot {};
struct Sequence {
	std::vector<Tree> parts;
};
struct Choice {
	std::vector<Tree> choices;
};
struct Repeat {
	Tree body;
	std::uint32_t min;
	std::uint32_t max;
};

struct Node {
	std::variant<Character, Set, Dot, Sequence, Choice, Repeat> held;
};

struct Arena {
	std::vector<Node> nodes;

	[[nodiscard]] auto add(Node node) -> Tree {
		nodes.push_back(std::move(node));
		return Tree{.index = nodes.size() - 1};
	}

	[[nodiscard]] auto at(Tree tree) const -> Node const& {
		return nodes[tree.index];
	}
};

inline constexpr auto alphabet = std::to_array<char32_t>({U'a', U'b', U'\n', U'é', U'😀'});

[[nodiscard]] auto encode(char32_t point) -> std::string {
	return regex::encode(point);
}

[[nodiscard]] auto escape(char32_t point) -> std::string {
	if (point == U'\n') {
		return "\\n";
	}
	return encode(point);
}

[[nodiscard]] auto render(Arena const& arena, Tree tree) -> std::string {
	return std::visit(
	    [&arena](auto const& held) -> std::string {
		    using Held = std::remove_cvref_t<decltype(held)>;
		    if constexpr (std::is_same_v<Held, Character>) {
			    return escape(held.point);
		    } else if constexpr (std::is_same_v<Held, Set>) {
			    auto out = std::string{held.negated ? "[^" : "["};
			    for (auto const point : held.members) {
				    out.append(escape(point));
			    }
			    return out + "]";
		    } else if constexpr (std::is_same_v<Held, Dot>) {
			    return ".";
		    } else if constexpr (std::is_same_v<Held, Sequence>) {
			    auto out = std::string{"(?:"};
			    for (auto const& part : held.parts) {
				    out.append(render(arena, part));
			    }
			    return out + ")";
		    } else if constexpr (std::is_same_v<Held, Choice>) {
			    auto out = std::string{"(?:"};
			    for (auto const [index, choice] : std::views::enumerate(held.choices)) {
				    out.append(index == 0 ? "" : "|");
				    out.append(render(arena, choice));
			    }
			    return out + ")";
		    } else {
			    auto const body = "(?:" + render(arena, held.body) + ")";
			    if (held.max == regex::unbounded) {
				    return held.min == 0 ? body + "*" : held.min == 1 ? body + "+" : std::format("{}{{{},}}", body, held.min);
			    }
			    return std::format("{}{{{},{}}}", body, held.min, held.max);
		    }
	    },
	    arena.at(tree).held);
}

/** The positions where a match of `tree` starting at `from` can end: the oracle, by definition. */
[[nodiscard]] auto ends(Arena const& arena, Tree tree, std::u32string_view text, std::size_t from) -> std::set<std::size_t> {
	return std::visit(
	    [&](auto const& held) -> std::set<std::size_t> {
		    using Held = std::remove_cvref_t<decltype(held)>;
		    if constexpr (std::is_same_v<Held, Character>) {
			    return from < text.size() && text[from] == held.point ? std::set{from + 1} : std::set<std::size_t>{};
		    } else if constexpr (std::is_same_v<Held, Set>) {
			    auto const member = from < text.size() && std::ranges::contains(held.members, text[from]);
			    return from < text.size() && member != held.negated ? std::set{from + 1} : std::set<std::size_t>{};
		    } else if constexpr (std::is_same_v<Held, Dot>) {
			    return from < text.size() && text[from] != U'\n' ? std::set{from + 1} : std::set<std::size_t>{};
		    } else if constexpr (std::is_same_v<Held, Sequence>) {
			    auto current = std::set{from};
			    for (auto const& part : held.parts) {
				    auto next = std::set<std::size_t>{};
				    for (auto const at : current) {
					    next.merge(ends(arena, part, text, at));
				    }
				    current = std::move(next);
			    }
			    return current;
		    } else if constexpr (std::is_same_v<Held, Choice>) {
			    auto out = std::set<std::size_t>{};
			    for (auto const& choice : held.choices) {
				    out.merge(ends(arena, choice, text, from));
			    }
			    return out;
		    } else {
			    auto out = std::set<std::size_t>{};
			    auto current = std::set{from};
			    for (auto count = std::uint32_t{0}; !current.empty(); ++count) {
				    if (count >= held.min) {
					    out.insert(current.begin(), current.end());
				    }
				    if (count == held.max || count > text.size() + 1) {
					    break;
				    }
				    auto next = std::set<std::size_t>{};
				    for (auto const at : current) {
					    next.merge(ends(arena, held.body, text, at));
				    }
				    std::erase_if(next, [&out, count, &held](std::size_t at) { return count >= held.min && out.contains(at); });
				    current = std::move(next);
			    }
			    return out;
		    }
	    },
	    arena.at(tree).held);
}

[[nodiscard]] auto random_tree(Arena& arena, std::mt19937_64& random, std::size_t depth) -> Tree {
	auto const pick = [&random](std::size_t count) { return std::uniform_int_distribution<std::size_t>{0, count - 1}(random); };
	auto const leaf = depth == 0 || pick(3) == 0;
	if (leaf) {
		switch (pick(3)) {
		case 0:
			return arena.add(Node{.held = Character{.point = alphabet.at(pick(alphabet.size()))}});
		case 1: {
			auto members = std::vector<char32_t>{};
			for (auto const point : alphabet) {
				if (pick(2) == 0) {
					members.push_back(point);
				}
			}
			if (members.empty()) {
				members.push_back(alphabet.at(0));
			}
			return arena.add(Node{.held = Set{.members = std::move(members), .negated = pick(3) == 0}});
		}
		default:
			return arena.add(Node{.held = Dot{}});
		}
	}
	switch (pick(3)) {
	case 0: {
		auto parts = std::vector<Tree>{};
		for (auto count = pick(3) + 1; count > 0; --count) {
			parts.push_back(random_tree(arena, random, depth - 1));
		}
		return arena.add(Node{.held = Sequence{.parts = std::move(parts)}});
	}
	case 1: {
		auto choices = std::vector<Tree>{};
		for (auto count = pick(3) + 2; count > 0; --count) {
			choices.push_back(random_tree(arena, random, depth - 1));
		}
		return arena.add(Node{.held = Choice{.choices = std::move(choices)}});
	}
	default: {
		auto const min = static_cast<std::uint32_t>(pick(3));
		auto const max = pick(3) == 0 ? regex::unbounded : min + static_cast<std::uint32_t>(pick(3));
		return arena.add(Node{.held = Repeat{.body = random_tree(arena, random, depth - 1), .min = min, .max = max}});
	}
	}
}

/** Every text over the alphabet up to `length` characters. */
[[nodiscard]] auto texts(std::size_t length) -> std::vector<std::u32string> {
	auto out = std::vector<std::u32string>{U""};
	auto frontier = out;
	for (auto size = std::size_t{0}; size < length; ++size) {
		auto next = std::vector<std::u32string>{};
		for (auto const& text : frontier) {
			for (auto const point : alphabet) {
				next.push_back(text + point);
			}
		}
		std::ranges::copy(next, std::back_inserter(out));
		frontier = std::move(next);
	}
	return out;
}

[[nodiscard]] auto utf8(std::u32string_view text) -> std::string {
	auto out = std::string{};
	for (auto const point : text) {
		out.append(encode(point));
	}
	return out;
}

/** For random regexes, the automaton of the parsed text matches exactly the texts the tree's definition matches. */
[[nodiscard]] auto automata_match_the_definition(std::uint64_t seed, std::size_t count) -> bool {
	auto random = engine::testing::generator(seed);
	auto const all = texts(4);
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto arena = Arena{};
		auto const tree = random_tree(arena, random, 4);
		auto const text = render(arena, tree);
		auto store = regex::Store{std::size_t{1} << 20U};
		auto const parsed = regex::parse(store, text, regex::Matching::Whole);
		if (!parsed) {
			std::println("  did not parse: {:?}", text);
			return false;
		}
		auto const automaton = regex::Automaton::build(store, *parsed, 100000);
		if (!automaton) {
			std::println("  too large: {:?}", text);
			return false;
		}
		for (auto const& sample : all) {
			auto const expected = ends(arena, tree, sample, 0).contains(sample.size());
			if (automaton->matches(utf8(sample)) != expected) {
				std::println("  {:?} on {:?}: expected {}", text, utf8(sample), expected);
				return false;
			}
		}
	}
	return true;
}

/** A search pattern matches a text exactly when some substring matches, anchors pinning the ends. */
[[nodiscard]] auto search_matches_substrings(std::uint64_t seed, std::size_t count) -> bool {
	auto random = engine::testing::generator(seed);
	auto const all = texts(3);
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto arena = Arena{};
		auto const tree = random_tree(arena, random, 3);
		auto const start = std::uniform_int_distribution<int>{0, 1}(random) == 1;
		auto const end = std::uniform_int_distribution<int>{0, 1}(random) == 1;
		auto const text = std::string{start ? "^" : ""} + render(arena, tree) + (end ? "$" : "");
		auto store = regex::Store{std::size_t{1} << 20U};
		auto const parsed = regex::parse(store, text, regex::Matching::Search);
		if (!parsed) {
			return false;
		}
		auto const automaton = regex::Automaton::build(store, *parsed, 100000);
		if (!automaton) {
			return false;
		}
		for (auto const& sample : all) {
			auto expected = false;
			for (auto from = std::size_t{0}; from <= sample.size() && !expected; ++from) {
				if (start && from > 0) {
					break;
				}
				auto const found = ends(arena, tree, sample, from);
				expected = end ? found.contains(sample.size()) : !found.empty();
			}
			if (automaton->matches(utf8(sample)) != expected) {
				std::println("  {:?} on {:?}: expected {}", text, utf8(sample), expected);
				return false;
			}
		}
	}
	return true;
}

/** No pattern without a complement matches a byte string that is not utf-8. */
[[nodiscard]] auto only_utf8_matches(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto store = regex::Store{std::size_t{1} << 20U};
	auto const any = regex::Automaton::build(store, *regex::parse(store, "(?s:.*)", regex::Matching::Whole), 1000);
	for (auto index = 0; index < 20000; ++index) {
		auto text = std::string{};
		for (auto length = std::uniform_int_distribution<int>{0, 6}(random); length > 0; --length) {
			text.push_back(static_cast<char>(std::uniform_int_distribution<int>{0, 255}(random)));
		}
		if (any->matches(text) != regex::valid_utf8(text)) {
			return false;
		}
	}
	return true;
}

/** Each class escape means what both dialects agree on, checked over every scalar value. */
[[nodiscard]] auto classes_mean_the_common_dialect() -> bool {
	auto store = regex::Store{std::size_t{1} << 20U};
	auto const build = [&store](std::string_view text) { return *regex::Automaton::build(store, *regex::parse(store, text, regex::Matching::Whole), 10000); };
	auto const digit = build("\\d");
	auto const not_digit = build("\\D");
	auto const word = build("\\w");
	auto const not_word = build("\\W");
	auto const space = build("\\s");
	auto const not_space = build("\\S");
	auto const dot = build(".");
	for (auto point = char32_t{0}; point < regex::scalar_end; ++point) {
		if (point >= 0xd800 && point <= 0xdfff) {
			continue;
		}
		auto const text = encode(point);
		auto const ascii_digit = point >= '0' && point <= '9';
		auto const ascii_word = ascii_digit || (point >= 'a' && point <= 'z') || (point >= 'A' && point <= 'Z') || point == '_';
		auto const either_space = (point >= 9 && point <= 13) || point == ' ' || point == 0x85 || point == 0xa0 || point == 0x1680 ||
		                          (point >= 0x2000 && point <= 0x200a) || point == 0x2028 || point == 0x2029 || point == 0x202f || point == 0x205f ||
		                          point == 0x3000 || point == 0xfeff;
		auto const both_space = either_space && point != 0x85 && point != 0xfeff;
		auto const terminator = point == '\n' || point == '\r' || point == 0x2028 || point == 0x2029;
		if (digit.matches(text) != ascii_digit || not_digit.matches(text) != (point < 0x80 && !ascii_digit) || word.matches(text) != ascii_word ||
		    not_word.matches(text) != (point < 0x80 && !ascii_word) || space.matches(text) != both_space || not_space.matches(text) != !either_space ||
		    dot.matches(text) != !terminator) {
			std::println("  class disagreement at U+{:04X}", std::uint32_t{point});
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto refuses(std::string_view text, regex::Fault fault, regex::Matching matching) -> bool {
	auto store = regex::Store{std::size_t{1} << 16U};
	auto const parsed = regex::parse(store, text, matching);
	if (parsed || parsed.error().fault != fault) {
		std::println("  {:?}: expected fault {}", text, std::to_underlying(fault));
		return false;
	}
	return true;
}

[[nodiscard]] auto refusals_are_named() -> bool {
	using regex::Fault;
	using regex::Matching;
	return refuses("a(?=b)", Fault::Lookaround, Matching::Whole) && refuses("a(?<!b)", Fault::Lookaround, Matching::Whole) &&
	       refuses("a*?", Fault::LazyQuantifier, Matching::Whole) && refuses("(?U)a*", Fault::LazyQuantifier, Matching::Whole) &&
	       refuses("^a", Fault::Anchor, Matching::Whole) && refuses("a$", Fault::Anchor, Matching::Whole) && refuses("a^b", Fault::Anchor, Matching::Search) &&
	       refuses("^a|b$", Fault::Anchor, Matching::Search) && refuses("\\bword", Fault::WordBoundary, Matching::Whole) &&
	       refuses("(a)\\1", Fault::Backreference, Matching::Whole) && refuses("\\p{L}", Fault::UnicodeProperty, Matching::Whole) &&
	       refuses("[[:alpha:]]", Fault::ClassSyntax, Matching::Whole) && refuses("[a&&b]", Fault::ClassSyntax, Matching::Whole) &&
	       refuses("[]a]", Fault::ClassSyntax, Matching::Whole) && refuses("(?x)a", Fault::GroupSyntax, Matching::Whole) &&
	       refuses("a{3,2}", Fault::BadRepetition, Matching::Whole) && refuses("a{200000}", Fault::RepetitionTooLarge, Matching::Whole) &&
	       refuses("a**", Fault::NothingToRepeat, Matching::Whole) && refuses("*a", Fault::NothingToRepeat, Matching::Whole) &&
	       refuses("(a", Fault::UnbalancedGroup, Matching::Whole) && refuses("a)", Fault::UnbalancedGroup, Matching::Whole) &&
	       refuses("[z-a]", Fault::BadClass, Matching::Whole) && refuses("\\q", Fault::BadEscape, Matching::Whole) &&
	       refuses("(?i)é", Fault::UnicodeCase, Matching::Whole) && refuses("(?i)[aé]", Fault::UnicodeCase, Matching::Whole) &&
	       refuses("(?i)\\x{e9}", Fault::UnicodeCase, Matching::Whole);
}

[[nodiscard]] auto case_folding_is_ascii() -> bool {
	auto store = regex::Store{std::size_t{1} << 16U};
	auto const automaton = *regex::Automaton::build(store, *regex::parse(store, "(?i)ab[c-d](?-i)é", regex::Matching::Whole), 1000);
	return automaton.matches("ABcé") && automaton.matches("abDé") && !automaton.matches("ABCÉ");
}

[[nodiscard]] auto automata_refuse_past_their_budget() -> bool {
	auto store = regex::Store{std::size_t{1} << 20U};
	auto const counted = *regex::parse(store, "a{0,5000}", regex::Matching::Whole);
	return !regex::Automaton::build(store, counted, 1000) && regex::Automaton::build(store, counted, 10000).has_value();
}

/** A state from which no accepting state is reachable is pruned to the dead state, so a live state can always finish. */
[[nodiscard]] auto empty_intersections_are_dead() -> bool {
	auto store = regex::Store{std::size_t{1} << 16U};
	auto const a_then_b = *regex::parse(store, "ab", regex::Matching::Whole);
	auto const a_then_c = *regex::parse(store, "ac", regex::Matching::Whole);
	auto const automaton = *regex::Automaton::build(store, store.both(a_then_b, a_then_c), 100);
	return automaton.start() == regex::dead;
}

}

auto main() -> int {
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "automata match their regexes' definitions (seed 1, 300 regexes)", .holds = automata_match_the_definition(1, 300)},
	    {.name = "automata match their regexes' definitions (seed 2, 300 regexes)", .holds = automata_match_the_definition(2, 300)},
	    {.name = "search patterns match substrings, anchors pinning ends (seed 3, 200 regexes)", .holds = search_matches_substrings(3, 200)},
	    {.name = "only utf-8 text matches (seed 4)", .holds = only_utf8_matches(4)},
	    {.name = "class escapes mean what both dialects agree on, at every scalar", .holds = classes_mean_the_common_dialect()},
	    {.name = "unsupported and malformed regexes are refused by name", .holds = refusals_are_named()},
	    {.name = "case folding is ascii", .holds = case_folding_is_ascii()},
	    {.name = "automata refuse past their state budget", .holds = automata_refuse_past_their_budget()},
	    {.name = "an empty intersection is dead from the start", .holds = empty_intersections_are_dead()},
	});
	return engine::testing::report(cases);
}

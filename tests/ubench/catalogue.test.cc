// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.testing;
import engine.ubench;

namespace {

namespace ubench = engine::ubench;
using ubench::Benchmark;

/** Every question the review's table asks has exactly one benchmark or exactly one pending entry, never both. */
[[nodiscard]] auto every_question_is_answered_once() -> bool {
	auto const table = ubench::review();
	auto const rows = table.view();
	auto const once = std::ranges::all_of(rows, [](ubench::Row const& row) {
		auto const found = ubench::answers(row.question);
		return found.benchmarks + found.pending == 1;
	});
	return !rows.empty() && once && ubench::covers(table);
}

/** The table the build embeds is the review's: 22 questions, U1 to U22 in order, each with text in every column. */
[[nodiscard]] auto the_review_table_parses_whole() -> bool {
	auto const table = ubench::review();
	auto const rows = table.view();
	auto numbered = true;
	for (auto index = std::size_t{0}; index < rows.size(); ++index) {
		auto const& row = rows[index];
		numbered = numbered && row.question == static_cast<int>(index) + 1 && !row.measures.empty() && !row.expected.empty() && !row.settles.empty();
	}
	return rows.size() == 22 && numbered;
}

/** A review in the table's format, built from rows `numbers`. */
[[nodiscard]] auto document(std::span<int const> numbers) -> std::string {
	auto text = std::string{"# review\n\n## microbenchmarks for lane `ubench`\n\n| id | measures | expected (source) | settles |\n|---|---|---|---|\n"};
	for (auto const number : numbers) {
		text += std::format("| U{} | something | a value [src] | a decision |\n", number);
	}
	text += "\n## the arithmetic\n\n| U99 | not | in the | table |\n";
	return text;
}

/**
 * Coverage fails for every way the table and the catalogue can disagree: a question dropped from the table (its
 * benchmark or pending entry is then a stray), a question added that nothing answers, and a question listed twice;
 * rows after the section's end do not count.
 */
[[nodiscard]] auto coverage_rejects_every_disagreement(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto whole = std::vector<int>(22);
	std::ranges::iota(whole, 1);
	auto holds = ubench::covers(ubench::review_table(document(whole)));
	for (auto trial = 0; trial < 50; ++trial) {
		auto const victim = std::uniform_int_distribution<std::size_t>{0, whole.size() - 1}(random);
		auto dropped = whole;
		dropped.erase(dropped.begin() + static_cast<std::ptrdiff_t>(victim));
		auto doubled = whole;
		doubled.push_back(whole.at(victim));
		auto extended = whole;
		extended.push_back(23 + static_cast<int>(victim));
		holds = holds && !ubench::covers(ubench::review_table(document(dropped))) && !ubench::covers(ubench::review_table(document(doubled))) &&
		        !ubench::covers(ubench::review_table(document(extended)));
	}
	return holds;
}

/** Every benchmark's name parses back to it; the names are distinct; nothing else parses. */
[[nodiscard]] auto names_round_trip() -> bool {
	auto holds = true;
	for (auto const& listed : ubench::catalogue) {
		holds = holds && ubench::parse(listed.name) == listed.benchmark;
		holds = holds && std::ranges::count(ubench::catalogue, listed.name, &ubench::Entry::name) == 1;
		holds = holds && !ubench::parse(std::string{listed.name} + "_").has_value() && !ubench::parse(listed.name.substr(1)).has_value();
	}
	return holds && !ubench::parse("").has_value() && !ubench::parse("U1").has_value();
}

/**
 * A selection is the named benchmarks in catalogue order, each once, whatever order and repetition the arguments
 * have; no arguments select everything; one unknown name refuses the whole selection.
 */
[[nodiscard]] auto selection_is_a_set_in_catalogue_order(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	auto const everything = ubench::select({});
	holds = holds && everything && everything->size() == ubench::catalogue.size() && std::ranges::is_sorted(*everything);
	for (auto trial = 0; trial < 200; ++trial) {
		auto arguments = std::vector<std::string_view>{};
		auto wanted = std::set<Benchmark>{};
		auto const count = std::uniform_int_distribution<int>{1, 12}(random);
		for (auto pick = 0; pick < count; ++pick) {
			auto const& listed = ubench::catalogue.at(std::uniform_int_distribution<std::size_t>{0, ubench::catalogue.size() - 1}(random));
			arguments.push_back(listed.name);
			wanted.insert(listed.benchmark);
		}
		auto const chosen = ubench::select(arguments);
		holds = holds && chosen && std::ranges::equal(*chosen, wanted);
		arguments.insert(arguments.begin() + static_cast<std::ptrdiff_t>(arguments.size() / 2), "not_a_benchmark");
		auto const refused = ubench::select(arguments);
		holds = holds && !refused && refused.error().name == "not_a_benchmark";
	}
	return holds;
}

/** Pending entries are named once each and say what they wait for. */
[[nodiscard]] auto pending_entries_are_distinct_and_explained() -> bool {
	return std::ranges::all_of(ubench::pending, [](ubench::Pending const& waiting) {
		return !waiting.waits_for.empty() && std::ranges::count(ubench::pending, waiting.name, &ubench::Pending::name) == 1 &&
		       !ubench::parse(waiting.name).has_value();
	});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every question in the review has exactly one benchmark or one pending entry", .holds = every_question_is_answered_once()},
	    engine::testing::Case{.name = "the embedded review's table parses to U1..U22 with every column", .holds = the_review_table_parses_whole()},
	    engine::testing::Case{.name = "coverage fails for a dropped, doubled or unanswered question (seed 1)", .holds = coverage_rejects_every_disagreement(1)},
	    engine::testing::Case{.name = "coverage fails for a dropped, doubled or unanswered question (seed 2)", .holds = coverage_rejects_every_disagreement(2)},
	    engine::testing::Case{.name = "every benchmark name parses back to it and nothing else parses", .holds = names_round_trip()},
	    engine::testing::Case{.name = "a selection is a set in catalogue order, refused whole on an unknown name (seed 3)",
	                          .holds = selection_is_a_set_in_catalogue_order(3)},
	    engine::testing::Case{.name = "pending entries are distinct, explained and not benchmark names", .holds = pending_entries_are_distinct_and_explained()},
	};
	return engine::testing::report(cases);
}

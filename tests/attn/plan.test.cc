// SPDX-License-Identifier: AGPL-3.0-only
//
// The division of attention's work: for any valid cycle and any SM count the assignment computes every group exactly
// once, in contiguous runs, balanced to within one group; the records of a cycle are disjoint and within the workspace
// figure; every appended position is computed once per kv head; the merge's units and the prefill's jobs each cover
// their work exactly once.
import std;
import engine.attn;
import engine.model;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace model = engine::model;

using attn::Count;
using attn::Cycle;

inline constexpr std::array<Count, 7> sm_counts = {1, 2, 7, 41, 82, 83, 200};

/** A valid cycle: some slots live in order, 1 to 5 rows each within the row budget, contexts from empty to the longest. */
[[nodiscard]] auto random_cycle(std::mt19937_64& random) -> Cycle {
	auto cycle = Cycle{.live = 0, .work = {}};
	auto rows_left = model::rows(model::RowBudget::FourTiles);
	auto first_row = 0U;
	for (auto slot = 0U; slot < model::slots; ++slot) {
		if (std::uniform_int_distribution<int>{0, 3}(random) == 0 || rows_left == 0) {
			continue;
		}
		auto const most = static_cast<unsigned>(std::min<Count>(attn::max_rows, rows_left));
		auto const rows = std::uniform_int_distribution<unsigned>{1, most}(random);
		auto const reach = std::array<unsigned, 4>{40, 3000, 40000, static_cast<unsigned>(model::max_positions) - rows};
		auto const limit = reach.at(std::uniform_int_distribution<std::size_t>{0, 3}(random));
		auto const position = std::uniform_int_distribution<unsigned>{0, limit}(random);
		cycle.work[cycle.live] = {.slot = slot, .position = position, .rows = rows, .first_row = first_row};
		++cycle.live;
		first_row += rows;
		rows_left -= rows;
	}
	return cycle;
}

/** A group's place in the cycle's order of work. */
struct Unit {
	Count work;
	Count head;
	Count group;

	[[nodiscard]] constexpr auto operator==(Unit const&) const -> bool = default;
};

[[nodiscard]] auto every_unit(Cycle const& cycle) -> std::vector<Unit> {
	auto units = std::vector<Unit>{};
	for (auto work = Count{0}; work < cycle.live; ++work) {
		for (auto head = Count{0}; head < attn::kv_heads; ++head) {
			for (auto group = Count{0}; group < attn::groups(cycle.work[work]); ++group) {
				units.push_back({.work = work, .head = head, .group = group});
			}
		}
	}
	return units;
}

[[nodiscard]] auto cost(Cycle const& cycle, Unit const& unit) -> Count {
	auto const end = std::min(attn::group_begin(unit.group + 1), attn::chunks(cycle.work[unit.work]));
	return end - attn::group_begin(unit.group);
}

/** Concatenated over the SMs in order, the assignment is exactly the cycle's groups in order, each SM within one group of even. */
[[nodiscard]] auto assignments_cover_once_in_order(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 300; ++trial) {
		auto const cycle = random_cycle(random);
		sound = sound && attn::valid(cycle, model::rows(model::RowBudget::FourTiles));
		auto const expected = every_unit(cycle);
		auto const total = attn::total_chunks(cycle);
		for (auto const sms : sm_counts) {
			auto seen = std::vector<Unit>{};
			for (auto sm = Count{0}; sm < sms; ++sm) {
				auto const mine = attn::assignment(cycle, sm, sms);
				sound = sound && mine.count <= attn::max_items;
				auto load = Count{0};
				for (auto const& item : std::span{mine.item}.first(mine.count)) {
					sound = sound && item.groups > 0;
					for (auto group = Count{item.first_group}; group < Count{item.first_group} + item.groups; ++group) {
						auto const unit = Unit{.work = item.work, .head = item.head, .group = group};
						seen.push_back(unit);
						load += cost(cycle, unit);
					}
				}
				sound = sound && load <= total / sms + 1 + attn::far_group_chunks;
			}
			sound = sound && seen == expected;
		}
	}
	return sound;
}

/** Every record row of a cycle is its own, below the cycle's count, which stays within the workspace figure. */
[[nodiscard]] auto records_are_disjoint_and_bounded(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 200; ++trial) {
		auto const cycle = random_cycle(random);
		auto const rows = attn::record_rows(cycle);
		auto taken = std::vector<bool>(rows, false);
		for (auto const& unit : every_unit(cycle)) {
			for (auto row = Count{0}; row < attn::query_rows(cycle.work[unit.work]); ++row) {
				auto const index = attn::record_row(cycle, unit.work, unit.head, unit.group, row);
				sound = sound && index < rows && !taken.at(index);
				if (index < rows) {
					taken.at(index) = true;
				}
			}
		}
		sound = sound && std::ranges::all_of(taken, std::identity{}) && rows <= attn::max_record_rows(model::RowBudget::FourTiles);
	}
	return sound;
}

/** The cycle's appended positions, by kv head, fall in exactly one SM's items once each, at every SM count. */
[[nodiscard]] auto fresh_rows_are_appended_once(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 200; ++trial) {
		auto const cycle = random_cycle(random);
		for (auto const sms : sm_counts) {
			auto counts = std::map<std::tuple<Count, Count, Count>, int>{};
			for (auto sm = Count{0}; sm < sms; ++sm) {
				auto const mine = attn::assignment(cycle, sm, sms);
				for (auto const& item : std::span{mine.item}.first(mine.count)) {
					auto const& work = cycle.work[item.work];
					auto const range = attn::fresh_positions(work, attn::chunk_range(work, item.first_group, item.groups));
					for (auto position = range.begin; position < range.end; ++position) {
						++counts[{Count{item.work}, Count{item.head}, position}];
					}
				}
			}
			auto expected = std::size_t{0};
			for (auto work = Count{0}; work < cycle.live; ++work) {
				for (auto head = Count{0}; head < attn::kv_heads; ++head) {
					for (auto position = Count{cycle.work[work].position}; position < attn::context(cycle.work[work]); ++position) {
						sound = sound && counts[{work, head, position}] == 1;
						++expected;
					}
				}
			}
			sound = sound && counts.size() == expected;
		}
	}
	return sound;
}

/** The merge's units are every (row, query head) of the cycle, once, in order. */
[[nodiscard]] auto merge_units_cover_the_rows(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 300; ++trial) {
		auto const cycle = random_cycle(random);
		auto unit = Count{0};
		for (auto work = 0U; work < cycle.live; ++work) {
			for (auto token = 0U; token < cycle.work[work].rows; ++token) {
				for (auto head = 0U; head < attn::query_heads; ++head) {
					auto const got = attn::merge_unit(cycle, unit);
					sound = sound && got.work == work && got.token == token && got.query_head == head;
					++unit;
				}
			}
		}
		sound = sound && unit == attn::merge_units(cycle);
	}
	return sound;
}

/** The prefill's jobs over the blocks cover every prompt row of each kv head once, against every group it sees. */
[[nodiscard]] auto prompt_jobs_cover_the_chunk(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 200; ++trial) {
		auto const count = std::uniform_int_distribution<unsigned>{1, 900}(random);
		auto const start = std::uniform_int_distribution<unsigned>{0, 50'000}(random);
		auto const blocks = static_cast<unsigned>(attn::core_blocks);
		auto covered = std::map<std::pair<unsigned, unsigned>, int>{};
		for (auto first = 0U; first < blocks; ++first) {
			auto const jobs = attn::PromptJobs{.slot = 3, .start = start, .rows = count, .first = first, .blocks = blocks};
			for (auto index = 0U; index < jobs.count(); ++index) {
				auto const job = jobs.at(index);
				sound = sound && job.work.slot == 3 && job.first_group == 0 && job.groups == attn::groups(job.work);
				sound = sound && job.work.position == start + job.work.first_row && job.work.rows >= 1 && job.work.rows <= attn::prompt_tile_rows;
				for (auto row = 0U; row < job.work.rows; ++row) {
					++covered[{job.work.first_row + row, job.head}];
				}
			}
		}
		sound = sound && covered.size() == std::size_t{count} * attn::kv_heads;
		sound = sound && std::ranges::all_of(covered, [](auto const& entry) { return entry.second == 1; });
	}
	return sound;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the assignment computes each group once, in order, within one group of even (seed 1)",
	                          .holds = assignments_cover_once_in_order(1)},
	    engine::testing::Case{.name = "the assignment computes each group once, in order, within one group of even (seed 2)",
	                          .holds = assignments_cover_once_in_order(2)},
	    engine::testing::Case{.name = "record rows are disjoint and within the workspace figure (seed 3)", .holds = records_are_disjoint_and_bounded(3)},
	    engine::testing::Case{.name = "every appended position is computed once per kv head (seed 4)", .holds = fresh_rows_are_appended_once(4)},
	    engine::testing::Case{.name = "the merge units are every row and query head once (seed 5)", .holds = merge_units_cover_the_rows(5)},
	    engine::testing::Case{.name = "the prefill jobs cover every prompt row of each kv head once (seed 6)", .holds = prompt_jobs_cover_the_chunk(6)},
	};
	return engine::testing::report(cases);
}

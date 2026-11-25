// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.kv;
import engine.kv_harness;
import engine.testing;

namespace {

namespace kv = engine::kv;
namespace harness = engine::kv_harness;

using harness::Op;
using kv::Count;
using kv::Geometry;

// Two-position blocks in a six-block pool with room for one snapshot: every operation below crosses a block edge,
// fills the pool or meets the cap within a step or two.
constexpr auto small = Geometry{.block_tokens = 2, .pool_blocks = 6, .table_blocks = 4, .snapshot_blocks = 2, .snapshot_capacity = 1};

/** Every operation on two slots, one cache entry and one snapshot, at the lengths that straddle a block edge. */
[[nodiscard]] auto alphabet() -> std::vector<Op> {
	auto ops = std::vector<Op>{};
	for (auto slot = Count{0}; slot < 2; ++slot) {
		auto const other = 1 - slot;
		ops.emplace_back(harness::Open{.slot = slot, .source = harness::Source::Fresh, .pick = 0, .length = 0});
		ops.emplace_back(harness::Open{.slot = slot, .source = harness::Source::Cached, .pick = 0, .length = 0});
		ops.emplace_back(harness::Open{.slot = slot, .source = harness::Source::Fork, .pick = other, .length = 1});
		ops.emplace_back(harness::Open{.slot = slot, .source = harness::Source::Fork, .pick = other, .length = 3});
		for (auto const length : {Count{0}, Count{1}}) {
			for (auto const keep : {Count{0}, Count{2}}) {
				ops.emplace_back(harness::Truncate{.slot = slot, .length = length, .keep = keep});
			}
		}
		ops.emplace_back(harness::Publish{.slot = slot, .length = 1});
		ops.emplace_back(harness::Publish{.slot = slot, .length = 3});
		ops.emplace_back(harness::Close{.slot = slot});
		ops.emplace_back(harness::Save{.slot = slot});
		ops.emplace_back(harness::Load{.snapshot = 0, .slot = slot});
	}
	for (auto const mask : {std::uint8_t{1}, std::uint8_t{2}, std::uint8_t{3}}) {
		for (auto const tokens : {Count{1}, Count{2}}) {
			for (auto const drafts : {Count{0}, Count{2}}) {
				ops.emplace_back(harness::Grow{.slots = mask, .tokens = tokens, .drafts = drafts, .seed = 0});
			}
		}
	}
	ops.emplace_back(harness::Evict{.entry = 0});
	ops.emplace_back(harness::Drop{.snapshot = 0});
	return ops;
}

/** Every sequence of `depth` operations from the alphabet, each run after `prefix` from an empty pool, holds. */
[[nodiscard]] auto every_sequence(std::span<Op const> prefix, Count depth) -> bool {
	auto const ops = alphabet();
	auto choice = std::vector<Count>(depth, 0);
	auto runs = Count{0};
	while (true) {
		auto checked = harness::Harness<small>{};
		for (auto const& op : prefix) {
			checked.run(op);
		}
		for (auto const pick : choice) {
			checked.run(ops[pick]);
		}
		checked.finish();
		runs += 1;
		if (checked.failure()) {
			std::println("sequence {} after the prefix: {}", choice, *checked.failure());
			return false;
		}
		auto position = Count{0};
		while (position < depth && ++choice[position] == ops.size()) {
			choice[position] = 0;
			++position;
		}
		if (position == depth) {
			auto expected = Count{1};
			for (auto level = Count{0}; level < depth; ++level) {
				expected *= ops.size();
			}
			return runs == expected;
		}
	}
}

}

auto main() -> int {
	auto const empty = std::vector<Op>{};
	auto const one_slot = std::vector<Op>{
	    harness::Open{.slot = 0, .source = harness::Source::Fresh, .pick = 0, .length = 0},
	    harness::Grow{.slots = 1, .tokens = 3, .drafts = 0, .seed = 1},
	};
	auto const shared = std::vector<Op>{
	    harness::Open{.slot = 0, .source = harness::Source::Fresh, .pick = 0, .length = 0},
	    harness::Grow{.slots = 1, .tokens = 3, .drafts = 0, .seed = 2},
	    harness::Publish{.slot = 0, .length = 3},
	    harness::Open{.slot = 1, .source = harness::Source::Cached, .pick = 0, .length = 0},
	};
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every sequence of 3 operations from an empty pool holds every invariant",
	        .holds = every_sequence(empty, 3),
	    },
	    engine::testing::Case{
	        .name = "every sequence of 3 operations after one slot holds a block and a half holds every invariant",
	        .holds = every_sequence(one_slot, 3),
	    },
	    engine::testing::Case{
	        .name = "every sequence of 3 operations after two slots share a cached partial block holds every invariant",
	        .holds = every_sequence(shared, 3),
	    },
	};
	return engine::testing::report(cases);
}

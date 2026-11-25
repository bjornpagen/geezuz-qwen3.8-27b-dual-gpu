// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.kv;
import engine.kv_harness;
import engine.testing;

namespace {

namespace kv = engine::kv;
namespace harness = engine::kv_harness;

using kv::Count;
using kv::Geometry;

// Small pools put exhaustion, sharing at the cap, copy-on-write of partial blocks and snapshot refusal within a few
// operations; the model's pool checks the same invariants at the engine's real sizes.
constexpr auto tiny = Geometry{.block_tokens = 4, .pool_blocks = 40, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 5};
constexpr auto unit = Geometry{.block_tokens = 1, .pool_blocks = 24, .table_blocks = 32, .snapshot_blocks = 3, .snapshot_capacity = 2};
constexpr auto wide = Geometry{.block_tokens = 16, .pool_blocks = 12, .table_blocks = 8, .snapshot_blocks = 1, .snapshot_capacity = 3};

struct Outcome {
	bool held;
	harness::Reached reached;
};

/** One swarm run: weights drawn from the seed, then `operations` random operations, then everything given back. */
template<Geometry G>
[[nodiscard]] auto run(std::uint64_t seed, Count operations, harness::Scale scale) -> Outcome {
	auto rng = engine::testing::generator(seed);
	auto weights = harness::Weights{};
	for (auto& weight : weights) {
		weight = rng() % 9;
	}
	// Opening and growing are the only ways out of the empty state, so no run may draw them never.
	weights[0] = std::max(weights[0], Count{1});
	weights[1] = std::max(weights[1], Count{1});
	auto checked = harness::Harness<G>{};
	for (auto step = Count{0}; step < operations; ++step) {
		checked.run(checked.draw(rng, weights, scale));
	}
	checked.finish();
	if (checked.failure()) {
		std::println("seed {}: {}", seed, *checked.failure());
	}
	return {.held = !checked.failure().has_value(), .reached = checked.reached()};
}

/** Every seed in [first, first + runs) holds, and together they reached every outcome the pool can produce. */
template<Geometry G>
[[nodiscard]] auto swarm(std::uint64_t first, Count runs, Count operations, harness::Scale scale) -> bool {
	auto held = true;
	auto total = harness::Reached{};
	for (auto seed = first; seed < first + runs; ++seed) {
		auto const outcome = run<G>(seed, operations, scale);
		held = held && outcome.held;
		total.grown += outcome.reached.grown;
		total.exhausted += outcome.reached.exhausted;
		total.copies += outcome.reached.copies;
		total.snapshots += outcome.reached.snapshots;
		total.refused += outcome.reached.refused;
		total.loads += outcome.reached.loads;
	}
	auto const reached_all =
	    total.grown > 0 && total.exhausted > 0 && total.copies > 0 && total.snapshots > 0 && total.refused > 0 && total.loads > 0;
	if (!reached_all || held) {
		std::println("reached: grown {} exhausted {} copies {} snapshots {} refused {} loads {}", total.grown, total.exhausted,
		             total.copies, total.snapshots, total.refused, total.loads);
	}
	return held && reached_all;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "4-position blocks, 40-block pool: 60 swarm runs of 1500 operations hold every invariant (seeds 1-60)",
	        .holds = swarm<tiny>(1, 60, 1500, {.tokens = 12, .drafts = 5}),
	    },
	    engine::testing::Case{
	        .name = "1-position blocks: 40 swarm runs of 1500 operations hold every invariant (seeds 101-140)",
	        .holds = swarm<unit>(101, 40, 1500, {.tokens = 5, .drafts = 3}),
	    },
	    engine::testing::Case{
	        .name = "16-position blocks, 12-block pool: 40 swarm runs of 1500 operations hold every invariant (seeds 201-240)",
	        .holds = swarm<wide>(201, 40, 1500, {.tokens = 40, .drafts = 8}),
	    },
	    engine::testing::Case{
	        .name = "the model's pool: 3 swarm runs of 300 operations hold every invariant (seeds 301-303)",
	        .holds = swarm<kv::model_geometry>(301, 3, 1000, {.tokens = 12000, .drafts = 64}),
	    },
	};
	return engine::testing::report(cases);
}

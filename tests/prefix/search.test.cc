// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.kv;
import engine.model;
import engine.prefix;
import engine.prefix_world;
import engine.testing;

namespace {

namespace kv = engine::kv;
namespace model = engine::model;
namespace prefix = engine::prefix;
namespace world = engine::prefix_world;

using engine::core::narrow;
using kv::Count;
using kv::Geometry;
using prefix::Point;
using prefix::PointKind;
using prefix::Token;
using world::Phase;
using world::World;

// Small pools put the snapshot cap, eviction under pressure, partial-block points and stalls within a few operations;
// the model's pool checks the same properties at the engine's real sizes.
constexpr auto tiny = Geometry{.block_tokens = 4, .pool_blocks = 26, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 5};
constexpr auto unit = Geometry{.block_tokens = 1, .pool_blocks = 24, .table_blocks = 32, .snapshot_blocks = 3, .snapshot_capacity = 3};
constexpr auto wide = Geometry{.block_tokens = 16, .pool_blocks = 14, .table_blocks = 8, .snapshot_blocks = 1, .snapshot_capacity = 3};

/** The largest quantities one run draws: prompt and extension lengths, prefill chunks and generated runs. */
struct Scale {
	Count longest;
	Count extension;
	Count chunk;
	Count generated;
};

enum class Op : std::uint8_t {
	Open,
	Prefill,
	Generate,
	Close,
	Abandon,
	Evict,
	Peek,
};

inline constexpr Count op_kinds = 7;

using Rng = std::mt19937_64;

[[nodiscard]] auto below(Rng& rng, Count bound) -> Count {
	return bound == 0 ? 0 : narrow<Count>(rng() % bound);
}

[[nodiscard]] auto random_tokens(Rng& rng, Count count) -> std::vector<Token> {
	auto tokens = std::vector<Token>(count);
	std::ranges::generate(tokens, [&] {
		return narrow<Token>(rng() % 3);
	});
	return tokens;
}

/**
 * A prompt as the server sees them: a fresh one, the next turn of a conversation a slot left (its whole sequence and
 * more), a branch off one, or one sharing the start of a prompt a slot is reading now.
 */
template<Geometry G>
[[nodiscard]] auto draw_prompt(Rng& rng, World<G> const& checked, Scale scale) -> std::vector<Token> {
	auto prompt = std::vector<Token>{};
	auto const history = checked.history();
	switch (below(rng, 4)) {
	case 0:
		break;
	case 1:
		if (!history.empty()) {
			prompt = history[history.size() - 1 - below(rng, std::min<Count>(history.size(), 6))];
		}
		break;
	case 2:
		if (!history.empty()) {
			auto const& origin = history[below(rng, history.size())];
			prompt.assign(origin.begin(), origin.begin() + narrow<std::ptrdiff_t>(below(rng, origin.size() + 1)));
		}
		break;
	default: {
		auto const slot = below(rng, model::slots);
		if (checked.phase(slot) != Phase::Free) {
			auto const sequence = checked.sequence(slot);
			prompt.assign(sequence.begin(), sequence.end());
		}
		break;
	}
	}
	prompt.append_range(random_tokens(rng, 1 + below(rng, scale.extension)));
	if (prompt.size() > scale.longest) {
		prompt.resize(scale.longest);
	}
	return prompt;
}

/** Up to four points at random places, the prompt's end often among them, as the chat layer would mark them. */
[[nodiscard]] auto draw_points(Rng& rng, Count length) -> std::vector<Point> {
	auto constexpr kinds = std::array{PointKind::SystemEnd, PointKind::LastUserStart, PointKind::GenerationStart};
	auto points = std::vector<Point>{};
	for (auto count = below(rng, 4); count > 0; --count) {
		points.push_back({.kind = kinds[below(rng, kinds.size())], .position = below(rng, length)});
	}
	if (rng() % 2 == 0) {
		points.push_back({.kind = PointKind::PromptEnd, .position = length - 1});
	}
	return points;
}

template<Geometry G>
auto step(Rng& rng, World<G>& checked, Op op, Scale scale) -> void {
	auto const slot = below(rng, model::slots);
	auto const phase = checked.phase(slot);
	switch (op) {
	case Op::Open:
		if (phase == Phase::Free) {
			auto prompt = draw_prompt(rng, checked, scale);
			auto const points = draw_points(rng, prompt.size());
			std::ignore = checked.open(slot, std::move(prompt), points);
		}
		return;
	case Op::Prefill:
		if (phase == Phase::Reading) {
			std::ignore = checked.prefill(slot, 1 + below(rng, scale.chunk));
		}
		return;
	case Op::Generate: {
		auto const count = 1 + below(rng, scale.generated);
		auto const drafts = below(rng, 4);
		if (phase == Phase::Generating && checked.sequence(slot).size() + count + drafts <= kv::table_positions(G)) {
			std::ignore = checked.generate(slot, count, drafts, rng());
		}
		return;
	}
	case Op::Close:
		if (phase != Phase::Free) {
			checked.close(slot);
		}
		return;
	case Op::Abandon:
		if (phase != Phase::Free) {
			checked.abandon(slot);
		}
		return;
	case Op::Evict:
		std::ignore = checked.evict(1 + below(rng, G.pool_blocks / 4));
		return;
	case Op::Peek: {
		auto const prompt = draw_prompt(rng, checked, scale);
		std::ignore = checked.peek(prompt);
		return;
	}
	}
}

struct Outcome {
	bool held;
	world::Reached reached;
};

/** One swarm run: operation weights drawn from the seed, then `operations` random operations, then everything back. */
template<Geometry G>
[[nodiscard]] auto run(std::uint64_t seed, Count operations, Scale scale) -> Outcome {
	auto rng = engine::testing::generator(seed);
	auto weights = std::array<Count, op_kinds>{};
	for (auto& weight : weights) {
		weight = 1 + below(rng, 8);
	}
	auto checked = World<G>{};
	for (auto done = Count{0}; done < operations && !checked.failure(); ++done) {
		auto pick = below(rng, std::ranges::fold_left(weights, Count{0}, std::plus{}));
		auto kind = Count{0};
		while (pick >= weights[kind]) {
			pick -= weights[kind];
			kind += 1;
		}
		step(rng, checked, static_cast<Op>(kind), scale);
	}
	checked.finish();
	if (checked.failure()) {
		std::println("seed {}: {}", seed, *checked.failure());
	}
	return {.held = !checked.failure().has_value(), .reached = checked.reached()};
}

/** Every seed in [first, first + runs) holds, and together the runs reached the outcomes `wanted` asks for. */
template<Geometry G, class Wanted>
[[nodiscard]] auto swarm(std::uint64_t first, Count runs, Count operations, Scale scale, Wanted wanted) -> bool {
	auto held = true;
	auto total = world::Reached{};
	for (auto seed = first; seed < first + runs; ++seed) {
		auto const outcome = run<G>(seed, operations, scale);
		held = held && outcome.held;
		total.hits += outcome.reached.hits;
		total.tail_hits += outcome.reached.tail_hits;
		total.waits += outcome.reached.waits;
		total.saved += outcome.reached.saved;
		total.present += outcome.reached.present;
		total.refused += outcome.reached.refused;
		total.evicted += outcome.reached.evicted;
		total.stalls += outcome.reached.stalls;
	}
	auto const reached_all = wanted(total);
	if (!reached_all || !held) {
		std::println("reached: hits {} tail hits {} waits {} saved {} present {} refused {} evicted {} stalls {}", total.hits,
		             total.tail_hits, total.waits, total.saved, total.present, total.refused, total.evicted, total.stalls);
	}
	return held && reached_all;
}

/** A geometry with partial blocks, a small snapshot cap and a small pool reaches every outcome. */
[[nodiscard]] auto everything(world::Reached const& total) -> bool {
	return total.hits > 0 && total.tail_hits > 0 && total.waits > 0 && total.saved > 0 && total.present > 0 && total.refused > 0 &&
	       total.evicted > 0 && total.stalls > 0;
}

/** One-position blocks have no partial block for a hit to end in. */
[[nodiscard]] auto everything_but_tails(world::Reached const& total) -> bool {
	return total.hits > 0 && total.waits > 0 && total.saved > 0 && total.present > 0 && total.refused > 0 && total.evicted > 0 &&
	       total.stalls > 0;
}

/** A few hundred operations fill neither the model's pool nor its snapshot share. */
[[nodiscard]] auto sharing(world::Reached const& total) -> bool {
	return total.hits > 0 && total.tail_hits > 0 && total.waits > 0 && total.saved > 0 && total.present > 0 && total.evicted > 0;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "4-position blocks, 26-block pool, 5 snapshots: 60 runs of 800 operations match the reference (seeds 1-60)",
	        .holds = swarm<tiny>(1, 60, 800, {.longest = 56, .extension = 12, .chunk = 9, .generated = 8}, everything),
	    },
	    engine::testing::Case{
	        .name = "1-position blocks, 24-block pool, 3 snapshots of 3 blocks: 40 runs of 800 operations match the reference (seeds 101-140)",
	        .holds = swarm<unit>(101, 40, 800, {.longest = 24, .extension = 6, .chunk = 5, .generated = 4}, everything_but_tails),
	    },
	    engine::testing::Case{
	        .name = "16-position blocks, 14-block pool: 40 runs of 800 operations match the reference (seeds 201-240)",
	        .holds = swarm<wide>(201, 40, 800, {.longest = 100, .extension = 30, .chunk = 20, .generated = 12}, everything),
	    },
	    engine::testing::Case{
	        .name = "the model's pool: 3 runs of 300 operations match the reference (seeds 301-303)",
	        .holds = swarm<kv::model_geometry>(301, 3, 300, {.longest = 6000, .extension = 2000, .chunk = 1500, .generated = 300}, sharing),
	    },
	};
	return engine::testing::report(cases);
}

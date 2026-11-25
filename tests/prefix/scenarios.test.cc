// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.kv;
import engine.prefix;
import engine.prefix_world;
import engine.testing;

namespace {

namespace kv = engine::kv;
namespace prefix = engine::prefix;

using engine::prefix_world::Progress;
using engine::prefix_world::World;
using kv::Count;
using kv::Geometry;
using kv::Slot;
using prefix::Point;
using prefix::PointKind;
using prefix::Token;

constexpr auto tiny = Geometry{.block_tokens = 4, .pool_blocks = 40, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 5};
constexpr auto churn = Geometry{.block_tokens = 4, .pool_blocks = 64, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 8};
constexpr auto capped = Geometry{.block_tokens = 4, .pool_blocks = 40, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 2};
constexpr auto roomy = Geometry{.block_tokens = 4, .pool_blocks = 80, .table_blocks = 32, .snapshot_blocks = 2, .snapshot_capacity = 16};

[[nodiscard]] auto join(std::span<Token const> head, std::span<Token const> tail) -> std::vector<Token> {
	auto joined = std::vector<Token>{head.begin(), head.end()};
	joined.append_range(tail);
	return joined;
}

[[nodiscard]] auto first(std::span<Token const> tokens, Count count) -> std::vector<Token> {
	return {tokens.begin(), tokens.begin() + engine::core::narrow<std::ptrdiff_t>(count)};
}

/** Reads the slot's whole prompt in chunks of `chunk`. */
template<Geometry G>
[[nodiscard]] auto read_all(World<G>& world, Count slot, Count chunk) -> Progress {
	auto progress = Progress::Advanced;
	while (progress == Progress::Advanced) {
		progress = world.prefill(slot, chunk);
	}
	return progress;
}

template<Geometry G>
[[nodiscard]] auto cached(World<G> const& world) -> std::vector<std::vector<Token>> {
	auto sequences = std::vector<std::vector<Token>>{};
	for (auto const& entry : world.cache().entries()) {
		sequences.push_back(entry.tokens);
	}
	std::ranges::sort(sequences);
	return sequences;
}

template<Geometry G>
[[nodiscard]] auto holds(World<G> const& world, std::span<Token const> sequence) -> bool {
	return std::ranges::contains(cached(world), std::vector<Token>{sequence.begin(), sequence.end()});
}

template<Geometry G>
[[nodiscard]] auto clean(World<G>& world) -> bool {
	world.finish();
	if (world.failure()) {
		std::println("  {}", *world.failure());
	}
	return !world.failure().has_value();
}

[[nodiscard]] auto hits_at(prefix::Match const& match, Count length) -> bool {
	return match.hit.has_value() && match.hit->length == length;
}

[[nodiscard]] auto mark_of(prefix::Match const& match) -> std::optional<prefix::Mark> {
	return match.hit.transform([](prefix::Hit const& hit) {
		return hit.mark;
	});
}

/**
 * A closed sequence is hit by a prompt that continues it, at its end; a prompt that branches off is hit at the deepest
 * point before the branch; the same prompt sent again reads its last token; an unrelated prompt misses.
 */
[[nodiscard]] auto hit_and_miss() -> bool {
	auto world = World<tiny>{};
	auto const prompt = std::vector<Token>{0, 1, 2, 0, 1, 2, 1, 1, 0, 2};
	auto const points = std::array{Point{.kind = PointKind::SystemEnd, .position = 6}, Point{.kind = PointKind::PromptEnd, .position = 9}};
	auto const resumed = world.open(0, prompt, points);
	auto const read = read_all(world, 0, 3);
	world.close(0);
	auto const continued = world.peek(join(prompt, std::array<Token, 2>{1, 2}));
	auto const branched = world.peek(join(first(prompt, 7), std::array<Token, 3>{2, 2, 2}));
	auto const again = world.peek(prompt);
	auto const unrelated = world.peek(std::array<Token, 5>{2, 2, 2, 2, 2});
	return resumed == 0 && read == Progress::Read && hits_at(continued, 10) &&
	       mark_of(continued) == prefix::Mark{prefix::SequenceEnd{}} && hits_at(branched, 6) &&
	       mark_of(branched) == prefix::Mark{PointKind::SystemEnd} && hits_at(again, 9) && !unrelated.hit && clean(world);
}

/**
 * A snapshot point inside a block caches the block's first positions: two prompts that diverge right after it both
 * resume there sharing that partial block, and each copies it before writing its own continuation.
 */
[[nodiscard]] auto partial_block_is_shared() -> bool {
	auto world = World<tiny>{};
	auto const system = std::vector<Token>{0, 1, 2, 0, 1, 2};
	auto const at_system = std::array{Point{.kind = PointKind::SystemEnd, .position = 6}};
	std::ignore = world.open(0, join(system, std::array<Token, 2>{2, 2}), at_system);
	std::ignore = read_all(world, 0, 8);
	world.close(0);
	auto const a = world.open(1, join(system, std::array<Token, 2>{1, 1}), at_system);
	auto const b = world.open(2, join(system, std::array<Token, 2>{0, 0}), at_system);
	auto const shared = world.pool().entry(Slot::at(1), 1) == world.pool().entry(Slot::at(2), 1);
	std::ignore = read_all(world, 1, 8);
	std::ignore = read_all(world, 2, 8);
	auto const copied = world.pool().entry(Slot::at(1), 1) != world.pool().entry(Slot::at(2), 1);
	auto const grew = world.generate(1, 5, 2, 11) && world.generate(2, 5, 2, 12);
	world.close(1);
	world.close(2);
	return a == 6 && b == 6 && shared && copied && grew && clean(world);
}

/**
 * The deltanet state is not positional, so a prompt matching a cached sequence deeper than its deepest point resumes
 * at that point; a prompt equal to a cached sequence still reads its own last token.
 */
[[nodiscard]] auto match_is_cut_back_to_a_point() -> bool {
	auto world = World<tiny>{};
	auto const sequence = std::vector<Token>{0, 1, 2, 0, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 1, 1, 2, 0, 2, 1};
	auto const points = std::array{
	    Point{.kind = PointKind::SystemEnd, .position = 6},
	    Point{.kind = PointKind::LastUserStart, .position = 13},
	    Point{.kind = PointKind::PromptEnd, .position = 19},
	};
	std::ignore = world.open(0, sequence, points);
	std::ignore = read_all(world, 0, 5);
	world.abandon(0);
	auto const inside = world.peek(join(first(sequence, 17), std::array<Token, 3>{2, 2, 2}));
	auto const whole = world.peek(sequence);
	auto const short_of_end = world.peek(first(sequence, 19));
	auto const early = world.peek(join(first(sequence, 10), std::array<Token, 2>{2, 2}));
	return hits_at(inside, 13) && hits_at(whole, 19) && hits_at(short_of_end, 13) && hits_at(early, 6) && clean(world);
}

/**
 * Eviction takes resting leaves, oldest first: two branches go before the prefix they share, and a point a live slot
 * resumed from stays however much is asked for, until the slot leaves.
 */
[[nodiscard]] auto eviction_takes_oldest_leaves_first() -> bool {
	auto world = World<tiny>{};
	auto const system = std::vector<Token>{0, 1, 2, 0, 1, 2, 0, 1};
	auto const at_system = std::array{Point{.kind = PointKind::SystemEnd, .position = 8}};
	auto const one = join(system, std::array<Token, 4>{1, 1, 1, 1});
	auto const two = join(system, std::array<Token, 4>{2, 2, 2, 2});
	std::ignore = world.open(0, join(system, std::array<Token, 1>{0}), at_system);
	std::ignore = read_all(world, 0, 16);
	world.abandon(0);
	std::ignore = world.open(1, one, at_system);
	std::ignore = read_all(world, 1, 16);
	world.close(1);
	std::ignore = world.open(1, two, at_system);
	std::ignore = read_all(world, 1, 16);
	world.close(1);
	auto const pinned = world.open(3, join(system, std::array<Token, 5>{0, 1, 2, 0, 1}), at_system);
	auto const all = cached(world) == std::vector<std::vector<Token>>{system, one, two};
	auto const first_out = world.evict(1);
	auto const after_first = cached(world) == std::vector<std::vector<Token>>{system, two};
	std::ignore = world.evict(1);
	auto const after_second = cached(world) == std::vector<std::vector<Token>>{system};
	auto const everything = world.evict(tiny.pool_blocks);
	auto const kept = cached(world) == std::vector<std::vector<Token>>{system} && everything.snapshots == 0;
	world.abandon(3);
	std::ignore = world.evict(1);
	auto const last = cached(world).empty();
	return pinned == 8 && all && first_out.snapshots == 1 && first_out.blocks == 1 + tiny.snapshot_blocks && after_first &&
	       after_second && kept && last && clean(world);
}

/**
 * Turn replay: each turn's prompt is the previous turn's prompt and reply plus a user message and the generation
 * prompt. Closing a turn caches it at its end, so every next turn resumes there and reads only what is new; another
 * conversation under the same system prompt resumes at the system prompt's end.
 */
[[nodiscard]] auto turns_resume_at_the_previous_turn() -> bool {
	auto world = World<roomy>{};
	auto const system = std::vector<Token>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
	auto conversation = system;
	auto resumed_at_turn_end = true;
	for (auto turn = Count{0}; turn < 4; ++turn) {
		auto const user = std::array<Token, 5>{1, engine::core::narrow<Token>(turn % 3), 2, 1, 0};
		auto const generation = std::array<Token, 3>{2, 1, 1};
		auto const prompt = join(join(conversation, user), generation);
		auto const points = std::array{
		    Point{.kind = PointKind::SystemEnd, .position = system.size()},
		    Point{.kind = PointKind::LastUserStart, .position = conversation.size()},
		    Point{.kind = PointKind::GenerationStart, .position = conversation.size() + user.size()},
		    Point{.kind = PointKind::PromptEnd, .position = prompt.size() - 1},
		};
		auto const resumed = world.open(0, prompt, points);
		resumed_at_turn_end = resumed_at_turn_end && resumed == (turn == 0 ? 0 : conversation.size());
		std::ignore = read_all(world, 0, 7);
		std::ignore = world.generate(0, 6, 2, 100 + turn);
		world.close(0);
		conversation = world.history().back();
	}
	auto const other = world.open(1, join(system, std::array<Token, 4>{2, 2, 2, 1}), std::array<Point, 0>{});
	return resumed_at_turn_end && other == system.size() && clean(world);
}

/**
 * Prompts that share an uncached point are prefilled once: while one slot reads toward its system prompt's end, a
 * prompt sharing it is offered that point to wait for; once the point is cached, the prompt resumes there instead. A
 * slot that leaves withdraws the points it had not reached.
 */
[[nodiscard]] auto shared_point_is_prefilled_once() -> bool {
	auto world = World<tiny>{};
	auto const system = std::vector<Token>{0, 1, 2, 0, 1, 2, 0, 1, 2};
	auto const a = join(system, std::array<Token, 3>{0, 0, 1});
	auto const b = join(system, std::array<Token, 3>{1, 2, 2});
	auto const points = std::array{Point{.kind = PointKind::SystemEnd, .position = 9}, Point{.kind = PointKind::PromptEnd, .position = 11}};
	std::ignore = world.open(0, a, points);
	auto const before = world.peek(b);
	std::ignore = world.prefill(0, 5);
	auto const midway = world.peek(b);
	auto const still = before.wait && world.cache().pending(*before.wait);
	std::ignore = world.prefill(0, 10);
	auto const settled = before.wait && !world.cache().pending(*before.wait);
	auto const after = world.peek(b);
	auto const resumed = world.open(1, b, std::array{points[0]});
	auto const c = join(first(a, 11), std::array<Token, 2>{2, 2});
	auto const deeper = world.peek(c);
	world.abandon(0);
	auto const withdrawn = world.peek(c);
	auto const gone = deeper.wait && !world.cache().pending(*deeper.wait);
	auto const waits_at = [](prefix::Match const& match, Count position) {
		return match.wait && match.wait->position == position;
	};
	return !before.hit && waits_at(before, 9) && waits_at(midway, 9) && still && settled && hits_at(after, 9) && !after.wait &&
	       resumed == 9 && hits_at(deeper, 9) && waits_at(deeper, 11) && !withdrawn.wait && gone && clean(world);
}

/**
 * A long generation outlives the other conversations' turns: the point its slot resumed from or saved last stays while
 * their turns churn through the snapshot cap, and is evicted like any other once the slot leaves.
 */
[[nodiscard]] auto busy_slot_point_survives_churn() -> bool {
	auto world = World<churn>{};
	auto const prompt = std::vector<Token>{1, 1, 1, 1, 1, 2};
	auto const guarded = first(prompt, 5);
	std::ignore = world.open(0, prompt, std::array{Point{.kind = PointKind::PromptEnd, .position = 5}});
	std::ignore = read_all(world, 0, 8);
	auto survived = true;
	for (auto turn = Count{0}; turn < 40; ++turn) {
		auto const other = std::vector<Token>{2, engine::core::narrow<Token>(turn % 3), engine::core::narrow<Token>(turn / 3 % 3), engine::core::narrow<Token>(turn / 9 % 3),
		                                      0, 1, 2};
		std::ignore = world.open(1, other, std::array{Point{.kind = PointKind::LastUserStart, .position = 3}});
		std::ignore = read_all(world, 1, 8);
		world.close(1);
		survived = survived && holds(world, guarded);
	}
	auto const churned = world.reached().evicted > 0;
	world.abandon(0);
	for (auto turn = Count{0}; turn < 40; ++turn) {
		auto const other = std::vector<Token>{0, engine::core::narrow<Token>(turn % 3), engine::core::narrow<Token>(turn / 3 % 3), engine::core::narrow<Token>(turn / 9 % 3),
		                                      2, 1, 0};
		std::ignore = world.open(1, other, std::array{Point{.kind = PointKind::LastUserStart, .position = 3}});
		std::ignore = read_all(world, 1, 8);
		world.close(1);
	}
	return survived && churned && !holds(world, guarded) && clean(world);
}

/**
 * At the snapshot cap, a point every snapshot of which is in use is refused; once one rests, the next insertion evicts
 * it, the oldest, and is saved.
 */
[[nodiscard]] auto snapshot_cap_evicts_the_oldest() -> bool {
	auto world = World<capped>{};
	auto const end_of = [](Count length) {
		return std::array{Point{.kind = PointKind::PromptEnd, .position = length - 1}};
	};
	auto const a = std::vector<Token>{0, 0, 1, 1, 2};
	auto const b = std::vector<Token>{1, 1, 2, 2, 0};
	auto const c = std::vector<Token>{2, 2, 0, 0, 1};
	std::ignore = world.open(0, a, end_of(a.size()));
	std::ignore = read_all(world, 0, 8);
	std::ignore = world.open(1, b, end_of(b.size()));
	std::ignore = read_all(world, 1, 8);
	std::ignore = world.open(2, c, end_of(c.size()));
	std::ignore = read_all(world, 2, 8);
	auto const refused = world.reached().refused == 1 && !holds(world, first(c, 4));
	world.abandon(0);
	world.close(2);
	auto const replaced = !holds(world, first(a, 4)) && holds(world, first(b, 4)) && holds(world, c);
	world.close(1);
	return refused && replaced && clean(world);
}

/** In the model's pool, a long first turn is cached and its next turn resumes at its end. */
[[nodiscard]] auto long_turn_in_the_model_pool() -> bool {
	auto world = World<kv::model_geometry>{};
	auto rng = engine::testing::generator(7);
	auto prompt = std::vector<Token>(3000);
	std::ranges::generate(prompt, [&] {
		return engine::core::narrow<Token>(rng() % 5);
	});
	auto const points = std::array{
	    Point{.kind = PointKind::SystemEnd, .position = 700},
	    Point{.kind = PointKind::LastUserStart, .position = 2500},
	    Point{.kind = PointKind::GenerationStart, .position = 2996},
	    Point{.kind = PointKind::PromptEnd, .position = 2999},
	};
	std::ignore = world.open(0, prompt, points);
	std::ignore = read_all(world, 0, 1024);
	std::ignore = world.generate(0, 200, 4, 3);
	world.close(0);
	auto const turn = world.history().back();
	auto const next = join(turn, std::array<Token, 6>{4, 4, 3, 2, 1, 0});
	auto const resumed = world.open(1, next, std::array<Point, 0>{});
	return resumed == turn.size() && clean(world);
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "a continuation hits at the closed end, a branch at the point before it, a resend short of its last token",
	        .holds = hit_and_miss(),
	    },
	    engine::testing::Case{
	        .name = "a point inside a block shares that partial block, and each sharer copies it before writing",
	        .holds = partial_block_is_shared(),
	    },
	    engine::testing::Case{
	        .name = "a match deeper than the deepest point resumes at that point",
	        .holds = match_is_cut_back_to_a_point(),
	    },
	    engine::testing::Case{
	        .name = "eviction takes resting leaves oldest first, prefixes after their branches, never a point in use",
	        .holds = eviction_takes_oldest_leaves_first(),
	    },
	    engine::testing::Case{
	        .name = "four turns of a conversation each resume at the previous turn's end",
	        .holds = turns_resume_at_the_previous_turn(),
	    },
	    engine::testing::Case{
	        .name = "a prompt sharing an uncached point waits for the prefill reaching it, and resumes there",
	        .holds = shared_point_is_prefilled_once(),
	    },
	    engine::testing::Case{
	        .name = "a busy slot's point survives 40 turns of churn through the cap, and goes once the slot leaves",
	        .holds = busy_slot_point_survives_churn(),
	    },
	    engine::testing::Case{
	        .name = "at the snapshot cap a point is refused while all are in use, then evicts the oldest resting one",
	        .holds = snapshot_cap_evicts_the_oldest(),
	    },
	    engine::testing::Case{
	        .name = "the model's pool caches a 3000-token turn and resumes the next turn at its end",
	        .holds = long_turn_in_the_model_pool(),
	    },
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.kv;
import engine.model;
import engine.sched;
import engine.sched_world;
import engine.testing;
import engine.tokenizer;

namespace {

namespace kv = engine::kv;
namespace model = engine::model;
namespace sched = engine::sched;
namespace tokenizer = engine::tokenizer;
namespace world = engine::sched_world;

using engine::core::narrow;
using kv::Count;
using kv::Geometry;
using model::RowBudget;
using sched::Token;

// Pools small enough that a few requests fill them: `roomy` holds eight short conversations and evicts cache under
// them, `tight` pauses running requests and ends ones that cannot fit alone. Chunks of 16 tokens put several chunk
// rounds into every prompt.
constexpr auto roomy = Geometry{.block_tokens = 4, .pool_blocks = 400, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 24};
constexpr auto tight = Geometry{.block_tokens = 4, .pool_blocks = 72, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 8};
constexpr auto small = sched::Tuning{.chunk_tokens = 16, .share = {.numerator = 1, .denominator = 4}};

/** Whether a workload's prompts share a system prefix and continue earlier conversations. */
enum class Sharing : std::uint8_t {
	None,
	Some,
};

enum class When : std::uint8_t {
	/** Between cycles. */
	Boundary,
	/** While the cycle that holds the request runs. */
	InFlight,
};

struct Cancel {
	Count cycle;
	When when;
};

/** A request of a workload, as drawn before the run. */
struct Plan {
	Count arrival;
	world::Script script;
	std::optional<Cancel> cancel;
	/** The earlier request whose finished turn this one's prompt continues. */
	std::optional<Count> follows;
};

using Rng = std::mt19937_64;

[[nodiscard]] auto below(Rng& rng, Count bound) -> Count {
	return narrow<Count>(rng() % bound);
}

[[nodiscard]] auto chance(Rng& rng, Count percent) -> bool {
	return below(rng, 100) < percent;
}

[[nodiscard]] auto tokens(Rng& rng, Count count) -> std::vector<Token> {
	auto drawn = std::vector<Token>(count);
	std::ranges::generate(drawn, [&] {
		return narrow<Token>(below(rng, world::token_range));
	});
	return drawn;
}

/** Marks at random offsets of the prompt; the scheduler only needs them inside it. */
[[nodiscard]] auto marks(Rng& rng, Count length, std::optional<Count> system) -> sched::Marks {
	auto const somewhere = [&] -> std::optional<Count> {
		return chance(rng, 50) ? std::optional{below(rng, length)} : std::nullopt;
	};
	return {.system_end = system, .last_user_start = somewhere(), .generation_start = somewhere()};
}

[[nodiscard]] auto workload(Rng& rng, Sharing sharing, Count longest) -> std::vector<Plan> {
	auto const system = tokens(rng, 20 + below(rng, 30));
	auto plans = std::vector<Plan>{};
	auto const count = 16 + below(rng, 12);
	for (auto index = Count{0}; index < count; ++index) {
		auto prompt = tokens(rng, 1 + below(rng, longest));
		auto shared = std::optional<Count>{};
		if (sharing == Sharing::Some && chance(rng, 40)) {
			prompt.insert(prompt.begin(), system.begin(), system.end());
			shared = system.size();
		}
		auto const length = prompt.size();
		auto const limit = chance(rng, 30) ? sched::Limit{sched::UntilContextFull{}} : sched::Limit{sched::AtMost{.tokens = 1 + below(rng, 160)}};
		auto const arrival = below(rng, 150);
		auto cancel = std::optional<Cancel>{};
		if (chance(rng, 12)) {
			cancel = Cancel{.cycle = arrival + below(rng, 60), .when = chance(rng, 50) ? When::Boundary : When::InFlight};
		}
		auto follows = std::optional<Count>{};
		if (sharing == Sharing::Some && index > 0 && chance(rng, 25)) {
			follows = below(rng, index);
		}
		plans.push_back({
		    .arrival = arrival,
		    .script = {.prompt = std::move(prompt), .marks = marks(rng, length, shared), .limit = limit, .stream = rng(), .rarity = 40 + below(rng, 300)},
		    .cancel = cancel,
		    .follows = follows,
		});
	}
	return plans;
}

struct Outcome {
	std::optional<std::string> failure;
	world::Reached reached;
};

/** A bound no workload here comes near: a run that needs more cycles hangs. */
inline constexpr Count cycle_bound = 40'000;

template<Geometry G, RowBudget B>
[[nodiscard]] auto simulate(tokenizer::Tokenizer const& tokenizer, std::uint64_t seed, Sharing sharing) -> Outcome {
	auto rng = Rng{seed};
	auto plans = workload(rng, sharing, G.pool_blocks > 100 ? 90 : 60);
	auto run = world::World<G, B, small>{seed ^ 0x5eedULL, sharing == Sharing::None ? world::Waits::Impossible : world::Waits::Possible};
	auto handles = std::vector<std::optional<sched::RequestHandle>>(plans.size());
	auto submitted = std::vector<bool>(plans.size());
	auto const submit = [&](Count index) -> std::optional<std::string> {
		auto& plan = plans[index];
		if (plan.follows && handles[*plan.follows]) {
			// A follow-up continues the earlier response's sequence exactly, as a stored response is continued, then adds
			// a new message.
			auto const ended = run.finished(*handles[*plan.follows]);
			auto const stored = ended.and_then([](sched::Finished const& finished) {
				return std::visit(
				    [](auto const& ending) -> std::optional<std::vector<Token>> {
					    if constexpr (requires { ending.sequence; }) {
						    return ending.sequence;
					    } else {
						    return std::nullopt;
					    }
				    },
				    finished.ending);
			});
			if (stored) {
				auto prompt = *stored;
				prompt.append_range(tokens(rng, 1 + below(rng, 20)));
				auto const length = prompt.size();
				plan.script.prompt = std::move(prompt);
				plan.script.marks = {.system_end = plans[*plan.follows].script.marks.system_end, .last_user_start = length - 1, .generation_start = std::nullopt};
			}
		}
		auto const length = plan.script.prompt.size();
		auto const result = run.submit(tokenizer, plan.script);
		submitted[index] = true;
		if (!result) {
			auto const fits = std::visit(
			    [&](auto const& limit) {
				    if constexpr (requires { limit.tokens; }) {
					    return limit.tokens <= sched::room<G>(length);
				    } else {
					    return sched::room<G>(length) > 0;
				    }
			    },
			    plan.script.limit);
			if (fits || !std::holds_alternative<sched::PromptTooLong>(result.error().cause)) {
				return "a request is refused only when its prompt leaves too little room";
			}
			return std::nullopt;
		}
		handles[index] = result->request;
		return std::nullopt;
	};
	// The request to cancel now, if its plan says so.
	auto const due = [&](Count index, Count cycle, When when) -> std::optional<sched::RequestHandle> {
		auto const& cancel = plans[index].cancel;
		if (!cancel || cancel->when != when || cancel->cycle != cycle) {
			return std::nullopt;
		}
		return handles[index];
	};
	for (auto cycle = Count{0}; cycle < cycle_bound; ++cycle) {
		for (auto index = Count{0}; index < plans.size(); ++index) {
			auto const& plan = plans[index];
			// A follow-up waits for its earlier request to be submitted and, unless that was refused, to finish.
			auto const earlier = plan.follows.transform([&](Count follows) {
				return handles[follows];
			});
			auto const ready = !plan.follows || (submitted[*plan.follows] && (!earlier || !*earlier || run.finished(**earlier).has_value()));
			if (!submitted[index] && plan.arrival <= cycle && ready) {
				if (auto const refused = submit(index)) {
					return {.failure = *refused, .reached = run.reached()};
				}
			}
			if (auto const request = due(index, cycle, When::Boundary)) {
				run.cancel(*request);
			}
		}
		if (run.plan()) {
			for (auto index = Count{0}; index < plans.size(); ++index) {
				if (auto const request = due(index, cycle, When::InFlight)) {
					run.cancel(*request);
				}
			}
			run.settle();
		} else if (std::ranges::all_of(submitted, std::identity{}) && run.all_finished()) {
			run.check_outcomes();
			return {.failure = run.failure(), .reached = run.reached()};
		}
		if (run.failure()) {
			return {.failure = run.failure(), .reached = run.reached()};
		}
	}
	return {.failure = "the run ends within the cycle bound (no hang)", .reached = run.reached()};
}

struct Sweep {
	std::optional<std::string> failure;
	std::uint64_t seed;
	world::Reached total;
};

template<Geometry G, RowBudget B>
[[nodiscard]] auto sweep(tokenizer::Tokenizer const& tokenizer, Sharing sharing, std::uint64_t first, std::uint64_t seeds) -> Sweep {
	auto total = world::Reached{};
	for (auto seed = first; seed < first + seeds; ++seed) {
		auto const outcome = simulate<G, B>(tokenizer, seed, sharing);
		if (outcome.failure) {
			return {.failure = outcome.failure, .seed = seed, .total = total};
		}
		auto const& one = outcome.reached;
		total.cycles += one.cycles;
		total.batches += one.batches;
		total.chunks += one.chunks;
		total.shared_cycles += one.shared_cycles;
		total.hits += one.hits;
		total.pauses += one.pauses;
		total.stops += one.stops;
		total.lengths += one.lengths;
		total.cancels += one.cancels;
		total.out_of_pool += one.out_of_pool;
		total.most_live = std::max(total.most_live, one.most_live);
		total.chunk_positions += one.chunk_positions;
	}
	return {.failure = std::nullopt, .seed = first + seeds, .total = total};
}

auto describe(std::string_view name, Sweep const& sweep) -> void {
	auto const& total = sweep.total;
	std::println("{}: {} cycles, {} batches, {} chunks ({} beside decoding), {} hits, {} pauses, {} stops, {} lengths ({} out of pool), "
	             "{} cancels, {} slots at most",
	             name, total.cycles, total.batches, total.chunks, total.shared_cycles, total.hits, total.pauses, total.stops, total.lengths,
	             total.out_of_pool, total.cancels, total.most_live);
	if (sweep.failure) {
		std::println("{}: seed {} fails: {}", name, sweep.seed, *sweep.failure);
	}
}

}

auto main() -> int {
	auto const loaded = tokenizer::load();
	if (!loaded) {
		std::println("the tokenizer image does not load");
		return 1;
	}
	auto const& tokens = *loaded;
	constexpr auto seeds = std::uint64_t{40};
	auto const roomy_three = sweep<roomy, RowBudget::ThreeTiles>(tokens, Sharing::Some, 0, seeds);
	auto const roomy_four = sweep<roomy, RowBudget::FourTiles>(tokens, Sharing::Some, 1000, seeds);
	auto const tight_three = sweep<tight, RowBudget::ThreeTiles>(tokens, Sharing::Some, 2000, seeds);
	auto const tight_alone = sweep<tight, RowBudget::FourTiles>(tokens, Sharing::None, 3000, seeds);
	describe("roomy, 24 rows", roomy_three);
	describe("roomy, 32 rows", roomy_four);
	describe("tight, 24 rows", tight_three);
	describe("tight, 32 rows, nothing shared", tight_alone);
	auto const cases = std::array{
	    engine::testing::Case{.name = "roomy pool, 24 rows, shared prefixes and conversations: every run checks out (seeds 0-39)", .holds = !roomy_three.failure.has_value()},
	    engine::testing::Case{.name = "roomy pool, 32 rows, shared prefixes and conversations: every run checks out (seeds 1000-1039)", .holds = !roomy_four.failure.has_value()},
	    engine::testing::Case{.name = "tight pool, 24 rows, shared prefixes and conversations: every run checks out (seeds 2000-2039)", .holds = !tight_three.failure.has_value()},
	    engine::testing::Case{.name = "tight pool, 32 rows, unrelated prompts: every run checks out (seeds 3000-3039)", .holds = !tight_alone.failure.has_value()},
	    engine::testing::Case{.name = "the runs reach full batches, prefill beside decoding, cache hits, stops, limits and cancellations",
	                          .holds = roomy_three.total.most_live == model::slots && roomy_three.total.shared_cycles > 0 && roomy_three.total.hits > 0 &&
	                              roomy_three.total.stops > 0 && roomy_three.total.lengths > 0 && roomy_three.total.cancels > 0},
	    engine::testing::Case{.name = "a tight pool pauses requests and ends ones it cannot hold alone",
	                          .holds = tight_three.total.pauses > 0 && tight_alone.total.pauses > 0 && tight_three.total.out_of_pool + tight_alone.total.out_of_pool > 0},
	};
	return engine::testing::report(cases);
}

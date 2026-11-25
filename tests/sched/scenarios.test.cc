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

constexpr auto roomy = Geometry{.block_tokens = 4, .pool_blocks = 400, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 24};
constexpr auto snug = Geometry{.block_tokens = 4, .pool_blocks = 56, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 12};
constexpr auto small = sched::Tuning{.chunk_tokens = 16, .share = {.numerator = 1, .denominator = 4}};

// A stream that never ends its turn within any limit used here.
inline constexpr std::uint64_t endless = std::uint64_t{1} << 62U;

template<Geometry G>
using World = world::World<G, RowBudget::ThreeTiles, small>;

[[nodiscard]] auto distinct(Count first, Count count) -> std::vector<Token> {
	auto tokens = std::vector<Token>(count);
	std::ranges::generate(tokens, [&] {
		return narrow<Token>(first++ % world::token_range);
	});
	return tokens;
}

[[nodiscard]] auto script(std::vector<Token> prompt, sched::Marks marks, Count limit, std::uint64_t stream) -> world::Script {
	return {.prompt = std::move(prompt), .marks = marks, .limit = sched::AtMost{.tokens = limit}, .stream = stream, .rarity = endless};
}

/** Runs rounds until the scheduler has nothing left; false on a hang. */
template<class W>
[[nodiscard]] auto drain(W& run) -> bool {
	for (auto round = 0; round < 10'000; ++round) {
		if (!run.step()) {
			return true;
		}
	}
	return false;
}

template<class W>
[[nodiscard]] auto admissions(W const& run) -> std::vector<sched::Admitted> {
	auto found = std::vector<sched::Admitted>{};
	for (auto const& event : run.events()) {
		if (std::holds_alternative<sched::Admitted>(event)) {
			found.push_back(std::get<sched::Admitted>(event));
		}
	}
	return found;
}

template<class W>
[[nodiscard]] auto pauses(W const& run) -> Count {
	return narrow<Count>(std::ranges::count_if(run.events(), [](sched::Event const& event) {
		return std::holds_alternative<sched::Paused>(event);
	}));
}

/** Positions below `limit` each chunk of the run read, summed. */
template<class W>
[[nodiscard]] auto prefilled_below(W const& run, Count limit) -> Count {
	auto total = Count{0};
	for (auto const& cycle : run.cycles()) {
		auto const chunk = std::visit(
		    [](auto const& each) -> std::optional<sched::Chunk> {
			    if constexpr (requires { each.chunk; }) {
				    return each.chunk;
			    } else if constexpr (requires { each.from; }) {
				    return each;
			    } else {
				    return std::nullopt;
			    }
		    },
		    cycle);
		if (chunk) {
			auto const end = chunk->from + chunk->tokens.size();
			total += chunk->from < limit ? std::min(end, limit) - chunk->from : 0;
		}
	}
	return total;
}

/** Three prompts that share a 40-token system block arrive together: the block is read once and the others resume. */
[[nodiscard]] auto shared_prefix_prefills_once(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<roomy>{1, world::Waits::Possible};
	auto const system = distinct(100, 40);
	for (auto index = Count{0}; index < 3; ++index) {
		auto prompt = system;
		prompt.append_range(distinct(500 + (index * 50), 10));
		if (!run.submit(tokens, script(std::move(prompt), {.system_end = 40, .last_user_start = std::nullopt, .generation_start = std::nullopt}, 5, index))) {
			return false;
		}
	}
	if (!drain(run) || run.failure()) {
		return false;
	}
	auto const admitted = admissions(run);
	return admitted.size() == 3 && admitted[0].cached == 0 && admitted[1].cached == 40 && admitted[2].cached == 40 &&
	       prefilled_below(run, 40) == 40;
}

/** The same, without the dedup wait, reads the block three times: the check above can fail. */
[[nodiscard]] auto unshared_prompts_prefill_separately(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<roomy>{2, world::Waits::Possible};
	for (auto index = Count{0}; index < 3; ++index) {
		auto prompt = distinct(100 + (index * 7), 40);
		prompt.append_range(distinct(500 + (index * 50), 10));
		if (!run.submit(tokens, script(std::move(prompt), {.system_end = 40, .last_user_start = std::nullopt, .generation_start = std::nullopt}, 5, index))) {
			return false;
		}
	}
	return drain(run) && !run.failure() && prefilled_below(run, 40) == 120;
}

/**
 * Cached conversations fill the pool; new requests that need their room take it from the cache, and nothing running is
 * paused while the cache can give.
 */
[[nodiscard]] auto cache_gives_way_before_requests(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<snug>{3, world::Waits::Possible};
	for (auto index = Count{0}; index < 4; ++index) {
		if (!run.submit(tokens, script(distinct(200 + (index * 60), 30), {}, 4, index)) || !drain(run)) {
			return false;
		}
	}
	auto const cached = run.scheduler().cache().entries().size();
	auto const free = run.scheduler().pool().free_blocks();
	for (auto index = Count{0}; index < 2; ++index) {
		if (!run.submit(tokens, script(distinct(600 + (index * 80), 40), {}, 30, 10 + index))) {
			return false;
		}
	}
	// Each needs ~20 blocks: more than were free, less than the pool.
	auto const needed = 2 * model::divide_up(40 + 30 + sched::lookahead, snug.block_tokens);
	return drain(run) && !run.failure() && free < needed && cached > 0 && pauses(run) == 0;
}

/** Two requests outgrow a pool that holds one: the later is paused, resumes first, and both report their streams. */
[[nodiscard]] auto latest_request_pauses_and_resumes(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<snug>{4, world::Waits::Impossible};
	for (auto index = Count{0}; index < 3; ++index) {
		if (!run.submit(tokens, script(distinct(100 + (index * 90), 12), {}, 90, 20 + index))) {
			return false;
		}
	}
	if (!drain(run) || run.failure()) {
		return false;
	}
	auto finished = Count{0};
	for (auto const& event : run.events()) {
		if (std::holds_alternative<sched::Finished>(event)) {
			finished += std::get<sched::Finished>(event).usage.generated_tokens == 90 ? 1 : 0;
		}
	}
	return pauses(run) > 0 && finished == 3;
}

/** A queued request cancelled before it runs leaves at the next plan, never admitted. */
[[nodiscard]] auto queued_cancel_leaves(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<roomy>{5, world::Waits::Impossible};
	auto handles = std::vector<sched::RequestHandle>{};
	for (auto index = Count{0}; index <= model::slots; ++index) {
		auto const submitted = run.submit(tokens, script(distinct(100 + (index * 37), 8), {}, 50, index));
		if (!submitted) {
			return false;
		}
		handles.push_back(submitted->request);
	}
	if (!run.step()) {
		return false;
	}
	auto const last = handles.back();
	run.cancel(last);
	if (!run.step()) {
		return false;
	}
	auto const ended = run.finished(last);
	auto const admitted = std::ranges::any_of(admissions(run), [&](sched::Admitted const& each) {
		return each.request == last;
	});
	return ended && ended->ending == sched::Ending::Cancelled && !admitted && drain(run) && !run.failure();
}

/**
 * A request cancelled while its cycle runs keeps its slot until that cycle settles, then leaves with its state cached:
 * everything it reported is in the cache.
 */
[[nodiscard]] auto running_cancel_releases_at_boundary(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<roomy>{6, world::Waits::Possible};
	auto const prompt = distinct(300, 20);
	auto const submitted = run.submit(tokens, script(prompt, {}, 60, 7));
	if (!submitted) {
		return false;
	}
	for (auto round = 0; round < 6; ++round) {
		if (!run.step()) {
			return false;
		}
	}
	if (!run.plan()) {
		return false;
	}
	run.cancel(submitted->request);
	auto const finishing = std::ranges::any_of(run.scheduler().facts().slots, [](sched::SlotFacts const& slot) {
		return std::holds_alternative<sched::Finishing>(slot);
	});
	auto const before = run.finished(submitted->request).has_value();
	run.settle();
	auto const ended = run.finished(submitted->request);
	auto const vacant = std::ranges::all_of(run.scheduler().facts().slots, [](sched::SlotFacts const& slot) {
		return std::holds_alternative<sched::Vacant>(slot);
	});
	auto reported = prompt;
	reported.append_range(run.delivered(submitted->request));
	auto const cached = std::ranges::any_of(run.scheduler().cache().entries(), [&](auto const& entry) {
		return entry.tokens.size() >= reported.size() && std::ranges::starts_with(entry.tokens, reported);
	});
	return finishing && !before && ended && ended->ending == sched::Ending::Cancelled && vacant && cached && !run.delivered(submitted->request).empty() &&
	       !run.failure();
}

/** A request that ends its turn leaves the turn for replay; one cut off by its limit does not. */
[[nodiscard]] auto stopped_turn_kept_for_replay(tokenizer::Tokenizer const& tokens) -> bool {
	auto run = World<roomy>{9, world::Waits::Possible};
	auto prompt = std::vector<Token>{tokenizer::id(tokenizer::Added::im_start)};
	prompt.append_range(distinct(10, 12));
	// A stream that ends its turn at once: every value is a multiple of one.
	auto stopping = world::Script{.prompt = prompt, .marks = {}, .limit = sched::AtMost{.tokens = 10}, .stream = 3, .rarity = 1};
	if (!run.submit(tokens, std::move(stopping)) || !run.submit(tokens, script(prompt, {}, 2, 4)) || !drain(run)) {
		return false;
	}
	return run.scheduler().replay().pending() == 1 && !run.failure();
}

[[nodiscard]] auto refusals(tokenizer::Tokenizer const& tokens) -> bool {
	using Scheduler = sched::BasicScheduler<roomy, RowBudget::ThreeTiles, small>;
	auto scheduler = std::make_unique<Scheduler>();
	auto const most = sched::room<roomy>(1);
	auto const submit = [&](Count length, sched::Limit limit) {
		return scheduler->submit(tokens, {.prompt = distinct(0, length), .marks = {}, .sampling = {}, .limit = limit});
	};
	auto const too_long = submit(most + 1, sched::UntilContextFull{});
	auto const too_many = submit(1, sched::AtMost{.tokens = most + 1});
	auto const exactly = submit(1, sched::AtMost{.tokens = most});
	auto const whole = submit(most, sched::UntilContextFull{});
	auto taken = Count{2};
	auto full = std::optional<sched::SubmitError>{};
	while (!full && taken <= sched::max_requests) {
		auto const more = submit(1, sched::AtMost{.tokens = 1});
		if (!more) {
			full = more.error();
		}
		taken += more ? 1 : 0;
	}
	return !too_long && std::holds_alternative<sched::PromptTooLong>(too_long.error().cause) && !too_many &&
	       std::holds_alternative<sched::PromptTooLong>(too_many.error().cause) && exactly && whole && full &&
	       std::holds_alternative<sched::Full>(full->cause) && full->retry() == engine::core::Retry::Transient && taken == sched::max_requests;
}

}

auto main() -> int {
	auto const loaded = tokenizer::load();
	if (!loaded) {
		std::println("the tokenizer image does not load");
		return 1;
	}
	auto const& tokens = *loaded;
	auto const cases = std::array{
	    engine::testing::Case{.name = "prompts sharing a system block read it once and resume from its snapshot", .holds = shared_prefix_prefills_once(tokens)},
	    engine::testing::Case{.name = "prompts that share nothing read their blocks separately", .holds = unshared_prompts_prefill_separately(tokens)},
	    engine::testing::Case{.name = "the cache gives way to new requests before anything running is paused", .holds = cache_gives_way_before_requests(tokens)},
	    engine::testing::Case{.name = "the latest request pauses when the pool runs out, resumes first, and finishes its stream",
	                          .holds = latest_request_pauses_and_resumes(tokens)},
	    engine::testing::Case{.name = "a queued request cancelled before it runs leaves unadmitted", .holds = queued_cancel_leaves(tokens)},
	    engine::testing::Case{.name = "a running request cancelled mid-cycle leaves at the boundary with its state cached",
	                          .holds = running_cancel_releases_at_boundary(tokens)},
	    engine::testing::Case{.name = "a turn that ends is kept for replay; one cut off is not", .holds = stopped_turn_kept_for_replay(tokens)},
	    engine::testing::Case{.name = "prompts past the context and a full table are refused with their kinds", .holds = refusals(tokens)},
	};
	return engine::testing::report(cases);
}

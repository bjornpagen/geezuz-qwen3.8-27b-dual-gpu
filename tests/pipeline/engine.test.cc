// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.http;
import engine.kv;
import engine.model;
import engine.net;
import engine.pipeline;
import engine.pipeline_doubles;
import engine.sched;
import engine.testing;
import engine.tokenizer;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;
namespace http = engine::http;
namespace kv = engine::kv;
namespace model = engine::model;
namespace net = engine::net;
namespace pipeline = engine::pipeline;
namespace sched = engine::sched;
namespace doubles = engine::pipeline_doubles;
using doubles::Output;
using doubles::Update;
using pipeline::Token;
using namespace std::chrono_literals;

constexpr auto roomy = kv::Geometry{.block_tokens = 4, .pool_blocks = 400, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 24};
constexpr auto tight = kv::Geometry{.block_tokens = 4, .pool_blocks = 72, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 8};
constexpr auto small = sched::Tuning{.chunk_tokens = 16, .share = {.numerator = 1, .denominator = 4}};

using Snapshot = pipeline::Snapshot<std::uint64_t>;

/** One request of a workload: what is asked, and the completion its seed scripts. */
struct Job {
	doubles::Ask ask;
	std::vector<Token> script;
};

[[nodiscard]] auto exchange_of(std::uint64_t tag) -> http::Exchange {
	return http::Exchange{.connection = {.bits = tag}, .sequence = 0};
}

/**
 * The engine on its own thread over the doubles, and the test's side of the mailbox it posts to: the test thread
 * stands where the reactor's owner would, posting submissions and cancellations and draining the engine's updates.
 */
template<kv::Geometry G>
class Bench {
public:
	using Pipeline = pipeline::BasicPipeline<G, model::RowBudget::ThreeTiles, small, doubles::CardsDouble<G>, doubles::FrontDouble>;

	Bench(fake::Machine machine, std::span<Job const> jobs)
	    : driver_{std::move(machine)},
	      poller_{net::Poller::open().value()},
	      mailbox_{std::make_unique<Mailbox>(poller_.waker())},
	      pipeline_{launch(driver_, *mailbox_, jobs)} {}

	Bench(Bench const&) = delete;
	auto operator=(Bench const&) -> Bench& = delete;
	Bench(Bench&&) = delete;
	auto operator=(Bench&&) -> Bench& = delete;
	~Bench() = default;

	auto submit(doubles::Ask ask) -> void {
		auto const tag = ask.tag;
		pipeline_.submit(exchange_of(tag), std::move(ask));
	}

	auto cancel(std::uint64_t tag) -> void {
		pipeline_.cancel(exchange_of(tag));
	}

	/** Drains the engine's updates until `done` holds or `budget` passes. */
	template<class Done>
	[[nodiscard]] auto pump(Done done, std::chrono::milliseconds budget) -> bool {
		auto const deadline = net::Clock::now() + budget;
		for (;;) {
			mailbox_->drain([this](Update&& update) {
				record(std::move(update));
			});
			if (done()) {
				return true;
			}
			if (net::Clock::now() >= deadline) {
				return false;
			}
			std::ignore = poller_.wait_until(std::min(deadline, net::Clock::now() + 5ms));
		}
	}

	[[nodiscard]] auto outputs(std::uint64_t tag) const -> std::vector<Output> {
		auto const found = outputs_.find(tag);
		return found == outputs_.end() ? std::vector<Output>{} : found->second;
	}

	[[nodiscard]] auto ended(std::uint64_t tag) const -> bool {
		auto const found = outputs_.find(tag);
		return found != outputs_.end() && !found->second.empty() &&
		       (std::holds_alternative<doubles::Ended>(found->second.back()) || std::holds_alternative<doubles::Refused>(found->second.back()));
	}

	[[nodiscard]] auto committed(std::uint64_t tag) const -> std::size_t {
		auto const found = outputs_.find(tag);
		return found == outputs_.end() ? 0 : core_count<doubles::Committed>(found->second);
	}

	[[nodiscard]] auto latest() const -> std::optional<Snapshot> const& {
		return latest_;
	}

	/** The most slots any snapshot showed busy at once. */
	[[nodiscard]] auto most_busy() const -> std::size_t {
		return most_busy_;
	}

	/** Stops the engine and reports whether it ended without a failure. */
	[[nodiscard]] auto stop() -> bool {
		return pipeline_.stop().has_value();
	}

private:
	using Mailbox = net::Mailbox<Update, http::max_posted_messages>;

	[[nodiscard]] static auto launch(fake::Fake const& driver, Mailbox& mailbox, std::span<Job const> jobs) -> Pipeline {
		auto devices = device::Devices::open(driver.driver());
		auto tokenizer = engine::tokenizer::load();
		engine::core::invariant(devices.has_value() && tokenizer.has_value());
		auto scripts = std::map<std::uint64_t, std::vector<Token>>{};
		for (auto const& job : jobs) {
			scripts.emplace(job.ask.seed, job.script);
		}
		return Pipeline::start(http::Poster<Update>{mailbox}, std::move(*tokenizer),
		                       doubles::CardsDouble<G>{std::move(*devices), std::move(scripts)}, doubles::FrontDouble{})
		    .value();
	}

	template<class T>
	[[nodiscard]] static auto core_count(std::vector<Output> const& outputs) -> std::size_t {
		return engine::core::narrow<std::size_t>(std::ranges::count_if(outputs, [](Output const& output) {
			return std::holds_alternative<T>(output);
		}));
	}

	auto record(Update update) -> void {
		if (auto const snapshot = engine::core::alternative<Snapshot>(update)) {
			auto const busy = std::ranges::count_if(snapshot->slots, [](pipeline::SlotView<std::uint64_t> const& slot) {
				return !std::holds_alternative<pipeline::Vacant>(slot);
			});
			most_busy_ = std::max(most_busy_, engine::core::narrow<std::size_t>(busy));
			latest_ = *snapshot;
			return;
		}
		for (auto& addressed : std::get<doubles::Batch>(update)) {
			outputs_[addressed.exchange.connection.bits].push_back(std::move(addressed.output));
		}
	}

	fake::Fake driver_;
	net::Poller poller_;
	std::unique_ptr<Mailbox> mailbox_;
	// After the mailbox, so the engine's thread, which posts to it, stops first.
	Pipeline pipeline_;
	std::map<std::uint64_t, std::vector<Output>> outputs_;
	std::optional<Snapshot> latest_;
	std::size_t most_busy_ = 0;
};

/** `count` jobs with distinct random prompts of `prompt` tokens and scripts of up to `longest` tokens. */
[[nodiscard]] auto workload(std::uint64_t seed, std::size_t count, std::size_t prompt, std::size_t longest, sched::Limit limit) -> std::vector<Job> {
	auto random = engine::testing::generator(seed);
	auto jobs = std::vector<Job>{};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto tokens = std::vector<Token>{};
		for (auto position = std::size_t{0}; position < prompt; ++position) {
			tokens.push_back(static_cast<Token>(random() % doubles::script_range));
		}
		auto const job_seed = random();
		jobs.push_back(Job{
		    .ask = {.tag = index + 1, .prompt = std::move(tokens), .seed = job_seed, .limit = limit},
		    .script = doubles::script(job_seed, 1 + (random() % longest)),
		});
	}
	return jobs;
}

/**
 * The outputs a request must have produced: accepted, then each scripted token up to the end of its turn or its
 * limit, then its finish with exact counts and the exact sequence it leaves.
 */
[[nodiscard]] auto expected(Job const& job) -> std::vector<Output> {
	auto const limit = std::visit(
	    [](auto const& chosen) -> std::size_t {
		    if constexpr (requires { chosen.tokens; }) {
			    return chosen.tokens;
		    } else {
			    return std::numeric_limits<std::size_t>::max();
		    }
	    },
	    job.ask.limit);
	auto out = std::vector<Output>{doubles::Accepted{}};
	auto sequence = job.ask.prompt;
	auto stopped = false;
	for (auto const token : job.script) {
		if (sequence.size() - job.ask.prompt.size() == limit) {
			break;
		}
		sequence.push_back(token);
		if (sched::ends_turn(token)) {
			stopped = true;
			break;
		}
		out.emplace_back(doubles::Committed{.token = token});
	}
	auto const usage = sched::Usage{.prompt_tokens = job.ask.prompt.size(), .cached_tokens = 0, .generated_tokens = sequence.size() - job.ask.prompt.size()};
	auto ending = stopped ? pipeline::Ending{sched::Stopped{.sequence = std::move(sequence)}}
	                      : pipeline::Ending{sched::Truncated{.sequence = std::move(sequence)}};
	out.emplace_back(doubles::Ended{.finish = {.ending = std::move(ending), .usage = usage}});
	return out;
}

/** The outputs agree with the expected ones, the cached count aside: the prefix cache decides it. */
[[nodiscard]] auto agrees(std::vector<Output> got, std::vector<Output> const& want) -> bool {
	if (got.empty() || got.size() != want.size()) {
		return false;
	}
	if (auto const ended = engine::core::alternative<doubles::Ended>(got.back())) {
		if (ended->finish.usage.cached_tokens > ended->finish.usage.prompt_tokens) {
			return false;
		}
		ended->finish.usage.cached_tokens = 0;
	}
	return got == want;
}

template<kv::Geometry G>
[[nodiscard]] auto run_all(Bench<G>& bench, std::span<Job const> jobs) -> bool {
	for (auto const& job : jobs) {
		bench.submit(job.ask);
	}
	auto const done = bench.pump(
	    [&] {
		    return std::ranges::all_of(jobs, [&](Job const& job) {
			    return bench.ended(job.ask.tag);
		    });
	    },
	    60s);
	return done && std::ranges::all_of(jobs, [&](Job const& job) {
		       return agrees(bench.outputs(job.ask.tag), expected(job));
	       });
}

/** Eight requests at once stream together, and each commits exactly its script, no token lost or repeated. */
[[nodiscard]] auto eight_streams_commit_their_scripts(std::uint64_t seed) -> bool {
	auto const jobs = workload(seed, model::slots, 24, 80, sched::UntilContextFull{});
	auto bench = std::make_unique<Bench<roomy>>(fake::Machine{.pending_queries = 2}, jobs);
	auto const ran = run_all(*bench, jobs);
	auto const idle = bench->pump(
	    [&] {
		    auto const& latest = bench->latest();
		    return latest && latest->queued == 0 && std::ranges::all_of(latest->slots, [](auto const& slot) {
			           return std::holds_alternative<pipeline::Vacant>(slot);
		           });
	    },
	    10s);
	return ran && idle && bench->most_busy() == model::slots && bench->stop();
}

/** A limit truncates the completion at exactly that many tokens; a turn that ends first stops there. */
[[nodiscard]] auto limits_end_at_length(std::uint64_t seed) -> bool {
	auto jobs = workload(seed, 6, 12, 40, sched::AtMost{.tokens = 15});
	auto bench = std::make_unique<Bench<roomy>>(fake::Machine{}, jobs);
	auto const lengths = std::ranges::count_if(jobs, [](Job const& job) {
		return job.script.size() > 15;
	});
	return run_all(*bench, jobs) && lengths > 0 && lengths < 6 && bench->stop();
}

/** A prompt that leaves no room is refused with its length, and the engine serves the next request as usual. */
[[nodiscard]] auto a_prompt_too_long_is_refused() -> bool {
	auto jobs = workload(41, 2, 10, 20, sched::UntilContextFull{});
	auto const longest = sched::room<roomy>(0);
	jobs[0].ask.prompt.resize(longest, Token{5});
	auto bench = std::make_unique<Bench<roomy>>(fake::Machine{}, jobs);
	bench->submit(jobs[0].ask);
	auto const refused = bench->pump(
	    [&] {
		    return bench->ended(1);
	    },
	    10s);
	auto const served = run_all(*bench, std::span{jobs}.subspan(1));
	return refused && bench->outputs(1) == std::vector<Output>{doubles::Refused{.prompt_tokens = longest}} && served && bench->stop();
}

/**
 * A request cancelled mid-stream says nothing more, never finishes, and gives its slot back: every slot is vacant
 * after the next cycle boundary, and the next request runs as usual.
 */
[[nodiscard]] auto cancellation_frees_the_slot() -> bool {
	auto jobs = workload(57, 2, 16, 1, sched::UntilContextFull{});
	jobs[0].script = doubles::script(jobs[0].ask.seed, 340);
	// Slow cards, so the request is still streaming when the cancellation reaches the engine.
	auto bench = std::make_unique<Bench<roomy>>(fake::Machine{.pending_queries = 3000}, jobs);
	bench->submit(jobs[0].ask);
	auto const streaming = bench->pump(
	    [&] {
		    return bench->committed(1) >= 10;
	    },
	    30s);
	bench->cancel(1);
	auto const vacant = bench->pump(
	    [&] {
		    auto const& latest = bench->latest();
		    return latest && std::ranges::all_of(latest->slots, [](auto const& slot) {
			           return std::holds_alternative<pipeline::Vacant>(slot);
		           });
	    },
	    30s);
	auto const before = bench->outputs(1);
	auto const served = run_all(*bench, std::span{jobs}.subspan(1));
	auto const after = bench->outputs(1);
	auto const tokens = std::ranges::count_if(after, [](Output const& output) {
		return std::holds_alternative<doubles::Committed>(output);
	});
	return streaming && vacant && served && before == after && !bench->ended(1) && tokens < 340 && bench->stop();
}

/**
 * Under a pool too small for every request at once, the latest admitted requests are paused and resumed, and every
 * request still commits exactly its script.
 */
[[nodiscard]] auto preemption_loses_no_token(std::uint64_t seed) -> bool {
	auto const jobs = workload(seed, model::slots, 40, 70, sched::UntilContextFull{});
	auto bench = std::make_unique<Bench<tight>>(fake::Machine{.pending_queries = 1}, jobs);
	auto const ran = run_all(*bench, jobs);
	auto const& latest = bench->latest();
	return ran && latest && latest->counters.preemptions > 0 && bench->stop();
}

/** The same workload, run twice on fresh engines, produces the same outputs for every request. */
[[nodiscard]] auto outputs_are_deterministic(std::uint64_t seed) -> bool {
	auto const jobs = workload(seed, 5, 30, 50, sched::UntilContextFull{});
	auto runs = std::array<std::vector<std::vector<Output>>, 2>{};
	for (auto& run : runs) {
		auto bench = std::make_unique<Bench<tight>>(fake::Machine{.pending_queries = 1}, jobs);
		if (!run_all(*bench, jobs) || !bench->stop()) {
			return false;
		}
		for (auto const& job : jobs) {
			run.push_back(bench->outputs(job.ask.tag));
		}
	}
	return runs[0] == runs[1];
}

}

auto main() -> int {
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "eight streams commit their scripts (seed 1)", .holds = eight_streams_commit_their_scripts(1)},
	    {.name = "eight streams commit their scripts (seed 2)", .holds = eight_streams_commit_their_scripts(2)},
	    {.name = "limits end at length (seed 3)", .holds = limits_end_at_length(3)},
	    {.name = "a prompt too long is refused", .holds = a_prompt_too_long_is_refused()},
	    {.name = "cancellation frees the slot", .holds = cancellation_frees_the_slot()},
	    {.name = "preemption loses no token (seed 4)", .holds = preemption_loses_no_token(4)},
	    {.name = "preemption loses no token (seed 5)", .holds = preemption_loses_no_token(5)},
	    {.name = "outputs are deterministic (seed 6)", .holds = outputs_are_deterministic(6)},
	});
	return engine::testing::report(cases);
}

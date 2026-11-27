// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.grammar;
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
namespace grammar = engine::grammar;
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

[[nodiscard]] auto terminal(Output const& output) -> bool {
	return std::holds_alternative<doubles::Ended>(output) || std::holds_alternative<doubles::Refused>(output) ||
	       std::holds_alternative<doubles::Cancelled>(output) || std::holds_alternative<doubles::Overran>(output);
}

/** Whether the test drains the server's mailbox when the cards break down, or the server is behind then. */
enum class Server : std::uint8_t {
	Draining,
	Behind,
};

/** The verify, counted from zero, at which the cards break down, and the server's state when they do. */
struct Fault {
	std::size_t verify;
	Server server;
};

/**
 * The engine on its own thread over the doubles, and the test's side of the mailbox it posts to: the test thread
 * stands where the reactor's owner would, posting submissions and cancellations and draining the engine's updates.
 */
template<kv::Geometry G, doubles::Rules R, class Gs>
class Bench {
public:
	using Pipeline = pipeline::BasicPipeline<G, model::RowBudget::ThreeTiles, small, doubles::CardsDouble<G, R>, doubles::FrontDouble<Gs>>;

	Bench(fake::Machine machine, std::span<Job const> jobs, std::optional<Fault> fault, Gs grammars, grammar::Vocabulary const& vocabulary)
	    : driver_{std::move(machine)},
	      poller_{net::Poller::open().value()},
	      mailbox_{std::make_unique<Mailbox>(poller_.waker())},
	      pipeline_{launch(driver_, *mailbox_, jobs, fault, std::move(grammars), vocabulary)} {}

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

	/** Whether the request's last output so far is a terminal one. */
	[[nodiscard]] auto answered(std::uint64_t tag) const -> bool {
		auto const found = outputs_.find(tag);
		return found != outputs_.end() && !found->second.empty() && terminal(found->second.back());
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

	/** How many `failed` updates the engine posted. */
	[[nodiscard]] auto failures() const -> std::size_t {
		return failures_;
	}

	/** How many updates the engine posted after a `failed` one. */
	[[nodiscard]] auto after_failure() const -> std::size_t {
		return after_failure_;
	}

	/** Whether the server's mailbox is full; when it is not, an empty batch is posted to it to find out. */
	[[nodiscard]] auto full() -> bool {
		return mailbox_->post(Update{doubles::Batch{}}) == net::Posted::Full;
	}

	/** Stops the engine and returns why it ended. */
	[[nodiscard]] auto stop() -> engine::core::Result<void, pipeline::EngineError<doubles::CardsFailure>> {
		return pipeline_.stop();
	}

private:
	using Mailbox = net::Mailbox<Update, http::max_posted_messages>;

	[[nodiscard]] static auto launch(fake::Fake const& driver, Mailbox& mailbox, std::span<Job const> jobs, std::optional<Fault> fault, Gs grammars,
	                                 grammar::Vocabulary const& vocabulary) -> Pipeline {
		auto devices = device::Devices::open(driver.driver());
		auto tokenizer = engine::tokenizer::load();
		engine::core::invariant(devices.has_value() && tokenizer.has_value());
		auto scripts = std::map<std::uint64_t, std::vector<Token>>{};
		for (auto const& job : jobs) {
			scripts.emplace(job.ask.seed, job.script);
		}
		auto const breakdown = fault.transform([&](Fault const& chosen) {
			return doubles::Breakdown{
			    .verify = chosen.verify,
			    .behind = chosen.server == Server::Behind ? std::optional{http::Poster<Update>{mailbox}} : std::nullopt,
			};
		});
		return Pipeline::start(http::Poster<Update>{mailbox}, std::move(*tokenizer), grammar::Vocabulary{vocabulary},
		                       doubles::CardsDouble<G, R>{std::move(*devices), std::move(scripts), breakdown},
		                       doubles::FrontDouble<Gs>{.grammars = std::move(grammars)})
		    .value();
	}

	template<class T>
	[[nodiscard]] static auto core_count(std::vector<Output> const& outputs) -> std::size_t {
		return engine::core::narrow<std::size_t>(std::ranges::count_if(outputs, [](Output const& output) {
			return std::holds_alternative<T>(output);
		}));
	}

	auto record(Update update) -> void {
		after_failure_ += failures_ > 0 ? 1 : 0;
		std::visit(
		    [this](auto& alternative) {
			    this->take(alternative);
		    },
		    update);
	}

	auto take(Snapshot const& snapshot) -> void {
		auto const busy = std::ranges::count_if(snapshot.slots, [](pipeline::SlotView<std::uint64_t> const& slot) {
			return !std::holds_alternative<pipeline::Vacant>(slot);
		});
		most_busy_ = std::max(most_busy_, engine::core::narrow<std::size_t>(busy));
		latest_ = snapshot;
	}

	auto take(doubles::Batch& batch) -> void {
		for (auto& addressed : batch) {
			outputs_[addressed.exchange.connection.bits].push_back(std::move(addressed.output));
		}
	}

	auto take(doubles::Failed const& /*failed*/) -> void {
		failures_ += 1;
	}

	fake::Fake driver_;
	net::Poller poller_;
	std::unique_ptr<Mailbox> mailbox_;
	// After the mailbox, so the engine's thread, which posts to it, stops first.
	Pipeline pipeline_;
	std::map<std::uint64_t, std::vector<Output>> outputs_;
	std::optional<Snapshot> latest_;
	std::size_t most_busy_ = 0;
	std::size_t failures_ = 0;
	std::size_t after_failure_ = 0;
};

/** A bench whose requests decode under the scripted grammar double. */
template<kv::Geometry G>
using Scripted = Bench<G, doubles::Rules::Scripted, doubles::ScriptedGrammars>;

/** A bench whose requests decode under a real compiled grammar. */
template<kv::Geometry G>
using Compiled = Bench<G, doubles::Rules::Compiled, doubles::CompiledGrammars>;

[[nodiscard]] auto scripted() -> doubles::ScriptedGrammars {
	return doubles::ScriptedGrammars{.everything = doubles::Everything::make()};
}

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
		    .ask = {.tag = index + 1, .prompt = std::move(tokens), .seed = job_seed, .limit = limit, .overrun_at = std::nullopt},
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

template<class B>
[[nodiscard]] auto run_all(B& bench, std::span<Job const> jobs) -> bool {
	for (auto const& job : jobs) {
		bench.submit(job.ask);
	}
	auto const done = bench.pump(
	    [&] {
		    return std::ranges::all_of(jobs, [&](Job const& job) {
			    return bench.answered(job.ask.tag);
		    });
	    },
	    60s);
	return done && std::ranges::all_of(jobs, [&](Job const& job) {
		       return agrees(bench.outputs(job.ask.tag), expected(job));
	       });
}

/** Eight requests at once stream together, and each commits exactly its script, no token lost or repeated. */
[[nodiscard]] auto eight_streams_commit_their_scripts(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto const jobs = workload(seed, model::slots, 24, 80, sched::UntilContextFull{});
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{.pending_queries = 2}, jobs, std::nullopt, scripted(), vocabulary);
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
[[nodiscard]] auto limits_end_at_length(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto jobs = workload(seed, 6, 12, 40, sched::AtMost{.tokens = 15});
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{}, jobs, std::nullopt, scripted(), vocabulary);
	auto const lengths = std::ranges::count_if(jobs, [](Job const& job) {
		return job.script.size() > 15;
	});
	return run_all(*bench, jobs) && lengths > 0 && lengths < 6 && bench->stop();
}

/** A prompt that leaves no room is refused with its length, and the engine serves the next request as usual. */
[[nodiscard]] auto a_prompt_too_long_is_refused(grammar::Vocabulary const& vocabulary) -> bool {
	auto jobs = workload(41, 2, 10, 20, sched::UntilContextFull{});
	auto const longest = sched::room<roomy>(0);
	jobs[0].ask.prompt.resize(longest, Token{5});
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{}, jobs, std::nullopt, scripted(), vocabulary);
	bench->submit(jobs[0].ask);
	auto const refused = bench->pump(
	    [&] {
		    return bench->answered(1);
	    },
	    10s);
	auto const served = run_all(*bench, std::span{jobs}.subspan(1));
	return refused && bench->outputs(1) == std::vector<Output>{doubles::Refused{.prompt_tokens = longest}} && served && bench->stop();
}

/**
 * A request cancelled mid-stream is answered `cancelled`, says nothing after it, never finishes, and gives its slot
 * back: every slot is vacant after the next cycle boundary, and the next request runs as usual.
 */
[[nodiscard]] auto cancellation_frees_the_slot(grammar::Vocabulary const& vocabulary) -> bool {
	auto jobs = workload(57, 2, 16, 1, sched::UntilContextFull{});
	jobs[0].script = doubles::script(jobs[0].ask.seed, 340);
	// Slow cards, so the request is still streaming when the cancellation reaches the engine.
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{.pending_queries = 3000}, jobs, std::nullopt, scripted(), vocabulary);
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
	return streaming && vacant && served && before == after && std::ranges::count_if(after, terminal) == 1 &&
	       after.back() == Output{doubles::Cancelled{}} && tokens < 340 && bench->stop();
}

/** The test thread, as the reactor's owner: submits a job. */
struct Enter {
	std::size_t job;
};

/** The test thread, as the reactor's owner: the request's client left. */
struct Leave {
	std::uint64_t tag;
};

/** The test thread drains the engine's updates for a while. */
struct Idle {
	std::chrono::milliseconds budget;
};

/** The test thread drains the engine's updates until the request has said this many outputs, or briefly. */
struct Watch {
	std::uint64_t tag;
	std::size_t outputs;
};

using Step = std::variant<Enter, Leave, Idle, Watch>;

/** A family of schedules: how many jobs, how large, how slow the cards, and how dense the posts. */
struct Shape {
	std::size_t jobs;
	std::size_t longest_prompt;
	std::size_t longest_script;
	/** One job in this many has a prompt that leaves no room, and is refused. */
	std::uint64_t refuse_one_in;
	std::size_t most_pending_queries;
	/** One step in this many drains the engine's updates; the rest post. */
	std::uint64_t drain_one_in;
};

/**
 * Jobs of the shape, with prompts of random length, some too long. The schedule only keeps the harness's own rules:
 * each request is submitted once, in tag order, and its client leaves only after it was submitted, sometimes twice;
 * when, relative to the engine, is left to the generator and the threads.
 */
[[nodiscard]] auto plan(std::mt19937_64& random, Shape const& shape) -> std::pair<std::vector<Job>, std::vector<Step>> {
	auto jobs = std::vector<Job>{};
	for (auto index = std::size_t{0}; index < shape.jobs; ++index) {
		auto const length = random() % shape.refuse_one_in == 0 ? sched::room<roomy>(0) : 1 + (random() % shape.longest_prompt);
		auto tokens = std::vector<Token>(length);
		std::ranges::generate(tokens, [&] {
			return static_cast<Token>(random() % doubles::script_range);
		});
		auto const job_seed = random();
		jobs.push_back(Job{
		    .ask = {.tag = index + 1, .prompt = std::move(tokens), .seed = job_seed, .limit = sched::UntilContextFull{}, .overrun_at = std::nullopt},
		    .script = doubles::script(job_seed, 1 + (random() % shape.longest_script)),
		});
	}
	auto steps = std::vector<Step>{};
	auto open = std::vector<std::uint64_t>{};
	auto next = std::size_t{0};
	while (next < jobs.size() || !open.empty()) {
		if (random() % shape.drain_one_in == 0) {
			if (open.empty() || random() % 2 == 0) {
				steps.emplace_back(Idle{.budget = std::chrono::milliseconds{random() % 6}});
			} else {
				steps.emplace_back(Watch{.tag = open[random() % open.size()], .outputs = 1 + (random() % 12)});
			}
			continue;
		}
		if (next < jobs.size() && (open.empty() || random() % 2 == 0)) {
			steps.emplace_back(Enter{.job = next});
			open.push_back(jobs[next].ask.tag);
			next += 1;
			continue;
		}
		auto const chosen = open.begin() + engine::core::narrow<std::ptrdiff_t>(random() % open.size());
		if (random() % 2 == 0) {
			steps.emplace_back(Leave{.tag = *chosen});
			// Left open, the request's client may leave again: a repeated cancel is answered by nothing.
			if (random() % 4 == 0) {
				continue;
			}
		}
		open.erase(chosen);
	}
	return {std::move(jobs), std::move(steps)};
}

/** What a request says when its client stays: refused, or its whole expected answer. */
[[nodiscard]] auto whole(Job const& job) -> std::vector<Output> {
	if (job.ask.prompt.size() == sched::room<roomy>(0)) {
		return {doubles::Refused{.prompt_tokens = job.ask.prompt.size()}};
	}
	return expected(job);
}

/** Whether a request's client stayed for its answer, or left. */
enum class Client : std::uint8_t {
	Stayed,
	Left,
};

/**
 * A request's outputs end in exactly one terminal output: the whole answer, or, when its client left before the
 * answer was done, `cancelled` after a prefix of the answer short of its terminal output (empty while it waited).
 */
[[nodiscard]] auto answered_once(std::vector<Output> const& got, Job const& job, Client client) -> bool {
	if (got.empty() || std::ranges::count_if(got, terminal) != 1 || !terminal(got.back())) {
		return false;
	}
	auto const want = whole(job);
	if (agrees(got, want)) {
		return true;
	}
	auto const said = std::span{got}.first(got.size() - 1);
	return client == Client::Left && got.back() == Output{doubles::Cancelled{}} && said.size() < want.size() &&
	       std::ranges::equal(said, std::span{want}.first(said.size()));
}

/** Takes one step of a schedule, as the reactor's owner would. */
struct Stepper {
	Scripted<roomy>& bench;
	std::span<Job const> jobs;
	std::set<std::uint64_t>& left;

	auto operator()(Enter const& enter) const -> void {
		bench.submit(jobs[enter.job].ask);
	}

	auto operator()(Leave const& leave) const -> void {
		bench.cancel(leave.tag);
		left.insert(leave.tag);
	}

	auto operator()(Idle const& idle) const -> void {
		std::ignore = bench.pump(
		    [] {
			    return false;
		    },
		    idle.budget);
	}

	auto operator()(Watch const& watch) const -> void {
		std::ignore = bench.pump(
		    [&] {
			    return bench.outputs(watch.tag).size() >= watch.outputs;
		    },
		    30ms);
	}
};

/** Follows a schedule of the shape, drawn from `seed`, then checks every request's answer and that the engine ends idle. */
[[nodiscard]] auto follow(grammar::Vocabulary const& vocabulary, std::uint64_t seed, Shape const& shape) -> bool {
	auto random = engine::testing::generator(seed);
	auto const [jobs, steps] = plan(random, shape);
	auto const pending = random() % (shape.most_pending_queries + 1);
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{.pending_queries = pending}, jobs, std::nullopt, scripted(), vocabulary);
	auto left = std::set<std::uint64_t>{};
	for (auto const& step : steps) {
		std::visit(Stepper{.bench = *bench, .jobs = jobs, .left = left}, step);
	}
	auto const answered = bench->pump(
	    [&] {
		    return std::ranges::all_of(jobs, [&](Job const& job) {
			    return bench->answered(job.ask.tag);
		    });
	    },
	    60s);
	auto const idle = bench->pump(
	    [&] {
		    auto const& latest = bench->latest();
		    return latest && latest->queued == 0 && std::ranges::all_of(latest->slots, [](auto const& slot) {
			           return std::holds_alternative<pipeline::Vacant>(slot);
		           });
	    },
	    10s);
	auto const once = std::ranges::all_of(jobs, [&](Job const& job) {
		return answered_once(bench->outputs(job.ask.tag), job, left.contains(job.ask.tag) ? Client::Left : Client::Stayed);
	});
	return answered && idle && once && bench->failures() == 0 && bench->stop();
}

/** The few-job family: long prompts read in several chunks, slow cards, frequent drains. */
constexpr auto few = Shape{.jobs = 10, .longest_prompt = 60, .longest_script = 40, .refuse_one_in = 6, .most_pending_queries = 400, .drain_one_in = 3};

/** The flood family: more requests at once than the scheduler has records, so the latest wait for one. */
constexpr auto flood = Shape{.jobs = 400, .longest_prompt = 8, .longest_script = 6, .refuse_one_in = 50, .most_pending_queries = 0, .drain_one_in = 400};

static_assert(flood.jobs > sched::max_requests, "the flood outgrows the scheduler's records");

/**
 * Every submitted request yields exactly one terminal output, whatever the order of submissions, cancellations and
 * finishes: searched over generated schedules, with the cancellation landing on requests prefilling, decoding,
 * finishing, finished or refused. The engine ends idle and posts no failure.
 */
[[nodiscard]] auto every_request_is_answered_once(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	return follow(vocabulary, seed, few);
}

/** The same, under a flood. */
[[nodiscard]] auto a_flood_is_answered_once(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	return follow(vocabulary, seed, flood);
}

/**
 * A request still waiting for a scheduler record when its client leaves is answered `cancelled` alone and at once:
 * the scheduler's records are all held by requests far from done on slow cards, so the last one submitted must wait.
 * A request the scheduler holds is answered `cancelled` only once it lets the request go, at a cycle boundary.
 */
[[nodiscard]] auto a_waiting_request_is_cancelled_at_once(grammar::Vocabulary const& vocabulary) -> bool {
	auto jobs = workload(71, sched::max_requests + 4, 4, 1, sched::UntilContextFull{});
	for (auto& job : jobs) {
		job.script = doubles::script(job.ask.seed, 300);
	}
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{.pending_queries = 3000}, jobs, std::nullopt, scripted(), vocabulary);
	for (auto const& job : jobs) {
		bench->submit(job.ask);
	}
	auto const last = jobs.back().ask.tag;
	auto const first = jobs.front().ask.tag;
	auto const accepted = bench->pump(
	    [&] {
		    return bench->outputs(first).size() >= 3;
	    },
	    30s);
	bench->cancel(last);
	bench->cancel(first);
	auto const answered = bench->pump(
	    [&] {
		    return bench->answered(last) && bench->answered(first);
	    },
	    30s);
	auto const head = bench->outputs(first);
	return accepted && answered && bench->outputs(last) == std::vector<Output>{doubles::Cancelled{}} &&
	       head.front() == Output{doubles::Accepted{}} && head.back() == Output{doubles::Cancelled{}} &&
	       std::ranges::count_if(head, terminal) == 1 && bench->stop();
}

/**
 * Cards that break down mid-batch make the engine post `failed` exactly once, as its last update, and return the
 * cards' error from `stop`. When the server is behind at that moment, the failure waits for it rather than being
 * dropped.
 */
[[nodiscard]] auto a_breakdown_fails_closed(grammar::Vocabulary const& vocabulary, std::uint64_t seed, Server server) -> bool {
	auto random = engine::testing::generator(seed);
	auto const jobs = workload(random(), model::slots, 24, 60, sched::UntilContextFull{});
	auto const fault = Fault{.verify = random() % 4, .server = server};
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{.pending_queries = random() % 4}, jobs, fault, scripted(), vocabulary);
	for (auto const& job : jobs) {
		bench->submit(job.ask);
	}
	auto behind = true;
	if (server == Server::Behind) {
		// Nothing drains the mailbox until the cards have filled it, so the engine's first post of its failure is
		// refused; the engine is then retrying while the test sleeps.
		auto const deadline = net::Clock::now() + 30s;
		while (!bench->full() && net::Clock::now() < deadline) {
			std::this_thread::sleep_until(net::Clock::now() + 1ms);
		}
		behind = bench->full();
		std::this_thread::sleep_until(net::Clock::now() + 20ms);
	}
	auto const failed = bench->pump(
	    [&] {
		    return bench->failures() > 0;
	    },
	    30s);
	auto const stopped = bench->stop();
	std::ignore = bench->pump(
	    [] {
		    return true;
	    },
	    0ms);
	return behind && failed && !stopped && stopped.error() == pipeline::EngineError<doubles::CardsFailure>{.cause = doubles::CardsFailure{.device = doubles::breakdown_error()}} &&
	       bench->failures() == 1 && bench->after_failure() == 0;
}

/**
 * Under a pool too small for every request at once, the latest admitted requests are paused and resumed, and every
 * request still commits exactly its script.
 */
[[nodiscard]] auto preemption_loses_no_token(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto const jobs = workload(seed, model::slots, 40, 70, sched::UntilContextFull{});
	auto bench = std::make_unique<Scripted<tight>>(fake::Machine{.pending_queries = 1}, jobs, std::nullopt, scripted(), vocabulary);
	auto const ran = run_all(*bench, jobs);
	auto const& latest = bench->latest();
	return ran && latest && latest->counters.preemptions > 0 && bench->stop();
}

/** The same workload, run twice on fresh engines, produces the same outputs for every request. */
[[nodiscard]] auto outputs_are_deterministic(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto const jobs = workload(seed, 5, 30, 50, sched::UntilContextFull{});
	auto runs = std::array<std::vector<std::vector<Output>>, 2>{};
	for (auto& run : runs) {
		auto bench = std::make_unique<Scripted<tight>>(fake::Machine{.pending_queries = 1}, jobs, std::nullopt, scripted(), vocabulary);
		if (!run_all(*bench, jobs) || !bench->stop()) {
			return false;
		}
		for (auto const& job : jobs) {
			run.push_back(bench->outputs(job.ask.tag));
		}
	}
	return runs[0] == runs[1];
}

/** The output grammar of a request for free text without reasoning or tools, as the boundary compiles it. */
[[nodiscard]] auto plain_text() -> grammar::Grammar {
	return grammar::compile(grammar::OutputRequest{
	                            .reasoning = grammar::Reasoning::Direct,
	                            .tools = {},
	                            .tool_choice = grammar::Calls::None,
	                            .parallel = grammar::Parallel::Many,
	                            .format = grammar::PlainText{},
	                        })
	    .value();
}

/**
 * A completion the grammar allows, drawn at random: at least `length` tokens, each one the mask allows at its
 * position (and below the cards double's script range), then the end of the turn where the grammar accepts it.
 */
[[nodiscard]] auto walk(grammar::Grammar const& compiled, grammar::Vocabulary const& vocabulary, std::mt19937_64& random, std::size_t length)
    -> std::vector<Token> {
	auto matcher = grammar::Matcher::make(compiled, grammar::engine_layout);
	auto tokens = std::vector<Token>{};
	while (tokens.size() < length || !matcher.accepting()) {
		auto const view = matcher.mask(vocabulary).value();
		auto allowed = std::vector<Token>{};
		for (auto token = Token{0}; token < doubles::script_range; ++token) {
			if (view.allows(token)) {
				allowed.push_back(token);
			}
		}
		engine::core::invariant(!allowed.empty());
		auto const token = allowed[random() % allowed.size()];
		engine::core::invariant(matcher.advance(vocabulary, token).has_value());
		tokens.push_back(token);
	}
	tokens.push_back(engine::tokenizer::id(engine::tokenizer::Added::im_end));
	return tokens;
}

/**
 * Eight streams under a real compiled grammar, its matcher and the engine's vocabulary: every verify row's mask, in
 * each card's row order, allows the token the completion commits there, and each request commits exactly its walk.
 */
[[nodiscard]] auto a_compiled_grammar_masks_every_row(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const compiled = plain_text();
	auto jobs = workload(random(), model::slots, 16, 1, sched::UntilContextFull{});
	for (auto& job : jobs) {
		job.script = walk(compiled, vocabulary, random, 5 + (random() % 40));
	}
	auto bench = std::make_unique<Compiled<roomy>>(fake::Machine{.pending_queries = 1}, jobs, std::nullopt,
	                                                doubles::CompiledGrammars{.compiled = compiled}, vocabulary);
	return run_all(*bench, jobs) && bench->failures() == 0 && bench->stop();
}

/**
 * A request whose grammar outgrows its bounds says a prefix of its completion, then `overran` with the fault as its
 * one terminal output, and frees its slot; the requests beside it run as usual.
 */
[[nodiscard]] auto an_overrun_grammar_ends_its_request_alone(grammar::Vocabulary const& vocabulary, std::uint64_t seed) -> bool {
	auto jobs = workload(seed, 4, 12, 30, sched::UntilContextFull{});
	jobs[0].script = doubles::script(jobs[0].ask.seed, 80);
	jobs[0].ask.overrun_at = 6;
	auto bench = std::make_unique<Scripted<roomy>>(fake::Machine{.pending_queries = 1}, jobs, std::nullopt, scripted(), vocabulary);
	for (auto const& job : jobs) {
		bench->submit(job.ask);
	}
	auto const answered = bench->pump(
	    [&] {
		    return std::ranges::all_of(jobs, [&](Job const& job) {
			    return bench->answered(job.ask.tag);
		    });
	    },
	    60s);
	auto const overran = bench->outputs(jobs[0].ask.tag);
	auto const want = expected(jobs[0]);
	auto const said = std::span{overran}.first(overran.empty() ? 0 : overran.size() - 1);
	auto const prefix = said.size() <= 1 + 6 && std::ranges::equal(said, std::span{want}.first(std::min(said.size(), want.size())));
	auto const others = std::ranges::all_of(std::span{jobs}.subspan(1), [&](Job const& job) {
		return agrees(bench->outputs(job.ask.tag), expected(job));
	});
	return answered && !overran.empty() && overran.back() == Output{doubles::Overran{.fault = {.fault = grammar::MatchFault::LexerTooLarge}}} &&
	       std::ranges::count_if(overran, terminal) == 1 && prefix && others && bench->failures() == 0 && bench->stop();
}

}

auto main() -> int {
	// Built once: every engine of the run takes a copy, as each would build its own.
	auto const tokenizer = engine::tokenizer::load();
	if (!tokenizer) {
		return 1;
	}
	auto const vocabulary = grammar::engine_vocabulary(*tokenizer);
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "eight streams commit their scripts (seed 1)", .holds = eight_streams_commit_their_scripts(vocabulary, 1)},
	    {.name = "eight streams commit their scripts (seed 2)", .holds = eight_streams_commit_their_scripts(vocabulary, 2)},
	    {.name = "limits end at length (seed 3)", .holds = limits_end_at_length(vocabulary, 3)},
	    {.name = "a prompt too long is refused", .holds = a_prompt_too_long_is_refused(vocabulary)},
	    {.name = "cancellation frees the slot", .holds = cancellation_frees_the_slot(vocabulary)},
	    {.name = "preemption loses no token (seed 4)", .holds = preemption_loses_no_token(vocabulary, 4)},
	    {.name = "preemption loses no token (seed 5)", .holds = preemption_loses_no_token(vocabulary, 5)},
	    {.name = "outputs are deterministic (seed 6)", .holds = outputs_are_deterministic(vocabulary, 6)},
	    {.name = "every request is answered once (seed 7)", .holds = every_request_is_answered_once(vocabulary, 7)},
	    {.name = "every request is answered once (seed 8)", .holds = every_request_is_answered_once(vocabulary, 8)},
	    {.name = "every request is answered once (seed 9)", .holds = every_request_is_answered_once(vocabulary, 9)},
	    {.name = "every request is answered once (seed 10)", .holds = every_request_is_answered_once(vocabulary, 10)},
	    {.name = "every request is answered once (seed 11)", .holds = every_request_is_answered_once(vocabulary, 11)},
	    {.name = "every request is answered once (seed 12)", .holds = every_request_is_answered_once(vocabulary, 12)},
	    {.name = "a flood is answered once (seed 13)", .holds = a_flood_is_answered_once(vocabulary, 13)},
	    {.name = "a flood is answered once (seed 14)", .holds = a_flood_is_answered_once(vocabulary, 14)},
	    {.name = "a waiting request is cancelled at once", .holds = a_waiting_request_is_cancelled_at_once(vocabulary)},
	    {.name = "a breakdown fails closed (seed 16, server draining)", .holds = a_breakdown_fails_closed(vocabulary, 16, Server::Draining)},
	    {.name = "a breakdown fails closed (seed 17, server draining)", .holds = a_breakdown_fails_closed(vocabulary, 17, Server::Draining)},
	    {.name = "a breakdown fails closed (seed 18, server behind)", .holds = a_breakdown_fails_closed(vocabulary, 18, Server::Behind)},
	    {.name = "a breakdown fails closed (seed 19, server behind)", .holds = a_breakdown_fails_closed(vocabulary, 19, Server::Behind)},
	    {.name = "a compiled grammar masks every row (seed 20)", .holds = a_compiled_grammar_masks_every_row(vocabulary, 20)},
	    {.name = "a compiled grammar masks every row (seed 21)", .holds = a_compiled_grammar_masks_every_row(vocabulary, 21)},
	    {.name = "an overrun grammar ends its request alone (seed 22)", .holds = an_overrun_grammar_ends_its_request_alone(vocabulary, 22)},
	});
	return engine::testing::report(cases);
}

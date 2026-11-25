// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sched_world;

import std;
import engine.core;
import engine.kv;
import engine.kv_fake;
import engine.model;
import engine.prefix;
import engine.sampling;
import engine.schedule;
import engine.sched;
import engine.tokenizer;

/**
 * The oracle side of the scheduler tests: a fake engine and a checker around the real scheduler, its real pool and
 * prefix cache. The fake engine runs each planned cycle the way the cards would, writing every position's kv through
 * the content-checking fake device and naming each slot's deltanet state by the tokens it has read. What a request
 * generates is a fixed function of the request and the token's index, and how many drafts a row accepts is drawn at
 * random, so whatever the batching, pausing and resuming, each request must report exactly its scripted stream. After
 * every plan and settle the world checks the cycle (the row budget, every decoding slot in it, chunks stopping at
 * snapshot points, the tokens fed), the events (each request admitted, generating and ending in order, the paused one
 * always the latest admitted), and the device (every live slot and every cached point reads back exactly, blocks are
 * conserved).
 */
namespace engine::sched_world {

using kv::Count;
using kv::Geometry;
using kv::Slot;
using kv_fake::Cell;
using kv_fake::Content;
using kv_fake::Kv;
using sched::RequestHandle;
using sched::Token;

inline constexpr std::uint64_t empty_chain = 1469598103934665603ULL;

[[nodiscard]] constexpr auto chain_step(std::uint64_t chain, Token token) -> std::uint64_t {
	return (chain ^ (std::uint64_t{token} + 7)) * 1099511628211ULL;
}

export [[nodiscard]] auto chain_of(std::span<Token const> tokens) -> std::uint64_t {
	return std::ranges::fold_left(tokens, empty_chain, chain_step);
}

// Positions past a slot's accepted rows hold kv it later overwrites: a chain no real sequence reaches.
inline constexpr std::uint64_t draft_mark = 0x9e37'79b9'7f4a'7c15ULL;

inline constexpr Token end_of_turn = tokenizer::id(tokenizer::Added::im_end);

/** Ordinary tokens the workloads draw: byte-level ids, none of them markup. */
export inline constexpr Token token_range = 1000;

[[nodiscard]] constexpr auto mix(std::uint64_t value) -> std::uint64_t {
	value += 0x9e37'79b9'7f4a'7c15ULL;
	value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
	value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
	return value ^ (value >> 31U);
}

/** One request as a workload defines it; `stream` names what it generates, `rarity` how seldom it ends its turn. */
export struct Script {
	std::vector<Token> prompt;
	sched::Marks marks;
	sched::Limit limit;
	std::uint64_t stream;
	std::uint64_t rarity;
};

/** Token `index` the request generates. */
export [[nodiscard]] auto generated(Script const& script, Count index) -> Token {
	auto const value = mix(script.stream ^ mix(index));
	return value % script.rarity == 0 ? end_of_turn : core::narrow<Token>(value % token_range);
}

/** What a run reached, so a test can insist the interesting outcomes happened at all. */
export struct Reached {
	Count cycles;
	Count batches;
	Count chunks;
	Count shared_cycles;
	Count hits;
	Count pauses;
	Count stops;
	Count lengths;
	Count cancels;
	Count out_of_pool;
	Count most_live;
	Count chunk_positions;
};

enum class Fate : std::uint8_t {
	Running,
	Cancelled,
};

struct Tracked {
	Script script;
	Count prompt_tokens;
	Count limit;
	std::uint64_t arrival;
	std::vector<Token> delivered;
	std::optional<sched::Finished> finished;
	Fate fate;
};

/** Facts one chunk round was chosen among, for the alternation check. */
struct Round {
	bool earliest;
};

template<class... F>
struct Overloaded : F... {
	using F::operator()...;
};

/** Whether prompts of a workload can share a snapshot point, so that one waits for another's prefill. */
export enum class Waits : std::uint8_t {
	Possible,
	/** No two prompts share a prefix: a paused request is always the next admitted. */
	Impossible,
};

export template<Geometry G, model::RowBudget B, sched::Tuning T>
class World {
public:
	using Scheduler = sched::BasicScheduler<G, B, T>;

	World(std::uint64_t seed, Waits waits)
	    : waits_{waits}, scheduler_{std::make_unique<Scheduler>()}, device_{std::make_unique<kv_fake::FakeDevice<G>>()},
	      rng_{seed} {}

	auto submit(tokenizer::Tokenizer const& tokenizer, Script script) -> core::Result<sched::Submitted, sched::SubmitError> {
		auto submitted = scheduler_->submit(tokenizer, {
		                                                    .prompt = script.prompt,
		                                                    .marks = script.marks,
		                                                    .sampling = sampling::Params{sampling::Greedy{}},
		                                                    .limit = script.limit,
		                                                });
		if (!submitted) {
			return submitted;
		}
		expect(submitted->prompt_tokens == script.prompt.size(), "a prompt nothing replays keeps its length");
		auto const limit = std::visit(Overloaded{
		                                  [&](sched::UntilContextFull) {
			                                  return sched::room<G>(script.prompt.size());
		                                  },
		                                  [](sched::AtMost at_most) {
			                                  return at_most.tokens;
		                                  },
		                              },
		                              script.limit);
		arrivals_ += 1;
		tracked_.emplace(submitted->request.bits, Tracked{
		                                              .script = std::move(script),
		                                              .prompt_tokens = submitted->prompt_tokens,
		                                              .limit = limit,
		                                              .arrival = arrivals_,
		                                              .delivered = {},
		                                              .finished = std::nullopt,
		                                              .fate = Fate::Running,
		                                          });
		return submitted;
	}

	auto cancel(RequestHandle request) -> void {
		if (auto const found = tracked_.find(request.bits); found != tracked_.end() && !found->second.finished) {
			found->second.fate = Fate::Cancelled;
		}
		scheduler_->cancel(request);
	}

	/** Plans the next cycle and checks it; false when the scheduler had nothing to run. */
	[[nodiscard]] auto plan() -> bool {
		auto planned = scheduler_->plan(*device_);
		observe(planned.events);
		if (!planned.cycle) {
			expect(live_.empty(), "no cycle only when no request is live");
			return false;
		}
		check_cycle(*planned.cycle);
		flight_ = std::move(planned.cycle);
		return true;
	}

	/** Runs the planned cycle on the fake engine and settles it. */
	auto settle() -> void {
		core::invariant(flight_.has_value());
		auto const done = run(*flight_);
		account(done);
		flight_.reset();
		observe(scheduler_->settle(*device_, done));
		check_state();
	}

	/** One whole round. */
	[[nodiscard]] auto step() -> bool {
		if (!plan()) {
			return false;
		}
		settle();
		return true;
	}

	[[nodiscard]] auto in_flight() const -> std::optional<sched::Cycle> const& {
		return flight_;
	}

	[[nodiscard]] auto finished(RequestHandle request) const -> std::optional<sched::Finished> {
		auto const found = tracked_.find(request.bits);
		core::invariant(found != tracked_.end());
		return found->second.finished;
	}

	[[nodiscard]] auto delivered(RequestHandle request) const -> std::span<Token const> {
		auto const found = tracked_.find(request.bits);
		core::invariant(found != tracked_.end());
		return found->second.delivered;
	}

	[[nodiscard]] auto all_finished() const -> bool {
		return std::ranges::all_of(tracked_, [](auto const& entry) {
			return entry.second.finished.has_value();
		});
	}

	/** Every event observed, in order. */
	[[nodiscard]] auto events() const -> std::span<sched::Event const> {
		return events_;
	}

	/** Every cycle run, in order. */
	[[nodiscard]] auto cycles() const -> std::span<sched::Cycle const> {
		return cycles_;
	}

	[[nodiscard]] auto scheduler() const -> Scheduler const& {
		return *scheduler_;
	}

	[[nodiscard]] auto failure() const -> std::optional<std::string> const& {
		return failure_;
	}

	[[nodiscard]] auto reached() const -> Reached {
		return reached_;
	}

	/** Every request finished as its script and fate allow; checked once a run is over. */
	auto check_outcomes() -> void {
		for (auto const& [bits, tracked] : tracked_) {
			expect(tracked.finished.has_value(), "every request finishes or is cancelled");
			if (tracked.fate == Fate::Running) {
				expect(!tracked.finished || tracked.finished->ending != sched::Ending::Cancelled, "only a cancelled request ends cancelled");
			}
		}
		expect(scheduler_->facts().queued == 0, "nothing is left queued");
	}

private:
	auto expect(bool holds, std::string_view what) -> void {
		if (!holds && !failure_) {
			failure_ = std::string{what};
		}
	}

	[[nodiscard]] auto tracked(RequestHandle request) -> Tracked& {
		auto const found = tracked_.find(request.bits);
		core::invariant(found != tracked_.end());
		return found->second;
	}

	/** The token at `position` of the request's sequence: its prompt, then its scripted stream. */
	[[nodiscard]] auto expected(RequestHandle request, Count position) -> Token {
		auto const& entry = tracked(request);
		return position < entry.prompt_tokens ? entry.script.prompt[position] : generated(entry.script, position - entry.prompt_tokens);
	}

	[[nodiscard]] auto expected_chain(RequestHandle request, Count length) -> std::uint64_t {
		auto chain = empty_chain;
		for (auto position = Count{0}; position < length; ++position) {
			chain = chain_step(chain, expected(request, position));
		}
		return chain;
	}

	/** The snapshot points of a request's prompt, as the scheduler marks them. */
	[[nodiscard]] auto points(RequestHandle request) -> std::vector<Count> {
		auto const& entry = tracked(request);
		auto result = std::vector<Count>{entry.prompt_tokens - 1};
		for (auto const mark : {entry.script.marks.system_end, entry.script.marks.last_user_start, entry.script.marks.generation_start}) {
			if (mark && *mark > 0) {
				result.push_back(*mark);
			}
		}
		return result;
	}

	auto observe(std::span<sched::Event const> events) -> void {
		for (auto const& event : events) {
			events_.push_back(event);
			std::visit(Overloaded{
			               [&](sched::Admitted const& admitted) {
				               auto& entry = tracked(admitted.request);
				               expect(!entry.finished && !std::ranges::contains(live_, admitted.request.bits, &Live::bits),
				                      "an admitted request is queued and unfinished");
				               admissions_ += 1;
				               live_.push_back({.bits = admitted.request.bits, .stamp = admissions_});
				               if (waits_ == Waits::Impossible && resumes_) {
					               expect(*resumes_ == admitted.request.bits, "a paused request resumes from the head of the queue");
				               }
				               resumes_.reset();
				               reached_.hits += admitted.cached > 0 ? 1 : 0;
				               reached_.most_live = std::max(reached_.most_live, live_.size());
				               expect(live_.size() <= model::slots, "at most one request per slot");
			               },
			               [&](sched::Generated const& generated_tokens) {
				               auto& entry = tracked(generated_tokens.request);
				               expect(!entry.finished && entry.fate == Fate::Running, "only a running request reports tokens");
				               expect(!generated_tokens.tokens.empty(), "a report carries tokens");
				               for (auto const token : generated_tokens.tokens) {
					               expect(token == generated(entry.script, entry.delivered.size()), "a request reports exactly its stream, nothing lost or repeated");
					               expect(!sched::ends_turn(token), "the token that ends a turn is not reported");
					               entry.delivered.push_back(token);
				               }
				               expect(entry.delivered.size() <= entry.limit, "a request reports at most its limit");
			               },
			               [&](sched::Paused const& paused) {
				               auto const found = std::ranges::find(live_, paused.request.bits, &Live::bits);
				               expect(found != live_.end(), "a paused request was running");
				               if (found == live_.end()) {
					               return;
				               }
				               expect(std::ranges::all_of(live_, [&](Live const& other) {
					                      return other.stamp <= found->stamp;
				                      }),
				                      "the pool pauses the latest-admitted request");
				               live_.erase(found);
				               reached_.pauses += 1;
				               resumes_ = paused.request.bits;
			               },
			               [&](sched::Finished const& finished) {
				               auto& entry = tracked(finished.request);
				               expect(!entry.finished, "a request finishes once");
				               auto const found = std::ranges::find(live_, finished.request.bits, &Live::bits);
				               auto const alone = found != live_.end() && live_.size() == 1;
				               if (found != live_.end()) {
					               live_.erase(found);
				               }
				               entry.finished = finished;
				               if (resumes_ == finished.request.bits) {
					               resumes_.reset();
				               }
				               auto const delivered = entry.delivered.size();
				               expect(finished.usage.prompt_tokens == entry.prompt_tokens, "usage counts the prompt as run");
				               switch (finished.ending) {
				               case sched::Ending::Stop:
					               reached_.stops += 1;
					               expect(finished.usage.generated_tokens == delivered + 1 && generated(entry.script, delivered) == end_of_turn &&
					                          finished.usage.generated_tokens <= entry.limit,
					                      "a request stops exactly at its stream's end of turn, within its limit");
					               break;
				               case sched::Ending::Length:
					               reached_.lengths += 1;
					               expect(finished.usage.generated_tokens == delivered, "a request cut short reports every token it counts");
					               if (delivered < entry.limit) {
						               reached_.out_of_pool += 1;
						               expect(alone, "a request ends short of its limit only when the pool cannot hold it alone");
					               }
					               break;
				               case sched::Ending::Cancelled:
					               reached_.cancels += 1;
					               expect(entry.fate == Fate::Cancelled, "only a cancelled request ends cancelled");
					               expect(finished.usage.generated_tokens == delivered, "a cancelled request counts what it reported");
					               break;
				               }
			               },
			           },
			           event);
		}
	}

	auto check_cycle(sched::Cycle const& cycle) -> void {
		cycles_.push_back(cycle);
		auto const facts = scheduler_->facts();
		auto const check_batch = [&](sched::Batch const& batch) {
			auto const rows = batch.rows.size();
			expect(rows >= 1 && batch.depth == schedule::shape(rows, B), "the batch's depth is the fixed allocation for its slots");
			expect(rows * (1 + model::drafts(batch.depth)) <= model::rows(B), "a cycle stays within the row budget");
			auto decoding = Count{0};
			for (auto const& slot : facts.slots) {
				decoding += std::holds_alternative<sched::Decoding>(slot) ? 1 : 0;
			}
			expect(decoding == rows, "every decoding slot runs in every cycle");
			auto previous = std::optional<Count>{};
			for (auto const& row : batch.rows) {
				expect(!previous || *previous < row.slot.index(), "rows are in slot order, one per slot");
				previous = row.slot.index();
				auto const& at = facts.slots[row.slot.index()];
				auto const decoding_slot = std::holds_alternative<sched::Decoding>(at) && std::get<sched::Decoding>(at).request == row.request;
				expect(decoding_slot, "a row is a decoding slot's");
				if (!decoding_slot) {
					continue;
				}
				auto const& state = std::get<sched::Decoding>(at);
				expect(row.position + 1 == state.prompt_tokens + state.generated_tokens, "a row is its slot's last committed token");
				expect(row.token == expected(row.request, row.position), "a row feeds the sequence's own token");
			}
		};
		auto const check_chunk = [&](sched::Chunk const& chunk) {
			auto const& at = facts.slots[chunk.slot.index()];
			auto const reading = std::holds_alternative<sched::Prefilling>(at) && std::get<sched::Prefilling>(at).request == chunk.request;
			expect(reading, "a chunk is a prefilling slot's");
			if (!reading) {
				return;
			}
			auto const& state = std::get<sched::Prefilling>(at);
			auto const end = chunk.from + chunk.tokens.size();
			expect(!chunk.tokens.empty() && chunk.tokens.size() <= T.chunk_tokens, "a chunk reads at most the chunk size");
			expect(chunk.from == state.prefilled_tokens && end <= state.prompt_tokens, "a chunk continues where its slot stopped");
			expect((chunk.read == sched::Read::Last) == (end == state.prompt_tokens), "only the chunk reading the sequence's end samples");
			for (auto const point : points(chunk.request)) {
				expect(point <= chunk.from || point >= end, "a chunk stops at every snapshot point");
			}
			for (auto const [offset, token] : std::views::enumerate(chunk.tokens)) {
				expect(token == expected(chunk.request, chunk.from + core::narrow<Count>(offset)), "a chunk feeds the sequence's own tokens");
			}
			reached_.chunk_positions += chunk.tokens.size();
			// The round's order: every second round takes the longest-waiting prompt.
			auto earliest = true;
			for (auto const& slot : facts.slots) {
				if (std::holds_alternative<sched::Prefilling>(slot)) {
					earliest = earliest && tracked(std::get<sched::Prefilling>(slot).request).arrival >= tracked(chunk.request).arrival;
				}
			}
			if (last_round_ && !last_round_->earliest) {
				expect(earliest, "of two consecutive chunk rounds one goes to the longest-waiting prompt");
			}
			last_round_ = Round{.earliest = earliest};
		};
		std::visit(Overloaded{
		               [&](sched::Batch const& batch) {
			               check_batch(batch);
		               },
		               [&](sched::Chunk const& chunk) {
			               expect(std::ranges::none_of(facts.slots,
			                                           [](sched::SlotFacts const& slot) {
				                                           return std::holds_alternative<sched::Decoding>(slot);
			                                           }),
			                      "a chunk runs alone only when nothing decodes");
			               check_chunk(chunk);
		               },
		               [&](sched::Both const& both) {
			               check_batch(both.batch);
			               check_chunk(both.chunk);
			               expect(std::ranges::none_of(both.batch.rows,
			                                           [&](sched::Row const& row) {
				                                           return row.slot == both.chunk.slot;
			                                           }),
			                      "the chunk's slot is not decoding");
			               // Before this chunk the decoding slots had their time back for every earlier one.
			               auto const repaid = core::narrow<std::int64_t>(T.share.denominator - T.share.numerator) * window_prefill.count();
			               expect(repaid <= core::narrow<std::int64_t>(T.share.numerator) * window_decode.count(),
			                      "while slots decode, prefill takes at most its share of the time");
		               },
		           },
		           cycle);
	}

	/** Writes positions [from, from + tokens) of the slot's sequence from `chain`; returns the chain after them. */
	auto write(Slot slot, Count from, std::span<Token const> tokens, std::uint64_t chain) -> std::uint64_t {
		for (auto const [offset, token] : std::views::enumerate(tokens)) {
			chain = chain_step(chain, token);
			device_->write(slot, from + core::narrow<Count>(offset), Cell{Kv{.chain = chain}});
		}
		return chain;
	}

	/** Writes kv the slot later overwrites (rejected drafts, the next cycle's drafts) at [from, until). */
	auto scribble(Slot slot, Count from, Count until, std::uint64_t chain) -> void {
		for (auto position = from; position < until; ++position) {
			device_->write(slot, position, Cell{Kv{.chain = chain ^ draft_mark}});
		}
	}

	[[nodiscard]] auto run_batch(sched::Batch const& batch) -> sched::BatchDone {
		auto done = sched::BatchDone{.rows = {}, .took = std::chrono::milliseconds{20}};
		for (auto const& row : batch.rows) {
			auto const& entry = tracked(row.request);
			expect(device_->state(row.slot) == expected_chain(row.request, row.position), "a decoding slot's state is its sequence's");
			auto const accepted = 1 + core::narrow<Count>(rng_() % (1 + model::drafts(batch.depth)));
			auto result = sched::Accepted{.tokens = {}, .count = accepted};
			auto read = std::vector<Token>{row.token};
			for (auto index = Count{0}; index < accepted; ++index) {
				auto const token = generated(entry.script, row.position + 1 + index - entry.prompt_tokens);
				result.tokens[index] = token;
				if (index + 1 < accepted) {
					read.push_back(token);
				}
			}
			auto const chain = write(row.slot, row.position, read, device_->state(row.slot));
			scribble(row.slot, row.position + accepted, row.position + sched::lookahead, chain);
			device_->set_state(row.slot, chain);
			done.rows.push_back(result);
		}
		return done;
	}

	[[nodiscard]] auto run_chunk(sched::Chunk const& chunk) -> sched::ChunkDone {
		auto const start = chunk.from == 0 ? empty_chain : device_->state(chunk.slot);
		expect(start == expected_chain(chunk.request, chunk.from), "a prefilling slot's state is its sequence's");
		auto const chain = write(chunk.slot, chunk.from, chunk.tokens, start);
		device_->set_state(chunk.slot, chain);
		auto const end = chunk.from + chunk.tokens.size();
		auto sampled = std::optional<Token>{};
		if (chunk.read == sched::Read::Last) {
			auto const& entry = tracked(chunk.request);
			sampled = generated(entry.script, end - entry.prompt_tokens);
			scribble(chunk.slot, end, end - 1 + sched::lookahead, chain);
		}
		// About 1,600 prompt tokens a second.
		return {.sampled = sampled, .took = std::chrono::microseconds{625} * core::narrow<std::int64_t>(chunk.tokens.size())};
	}

	[[nodiscard]] auto run(sched::Cycle const& cycle) -> sched::Done {
		return std::visit(Overloaded{
		                      [&](sched::Batch const& batch) -> sched::Done {
			                      return run_batch(batch);
		                      },
		                      [&](sched::Chunk const& chunk) -> sched::Done {
			                      return run_chunk(chunk);
		                      },
		                      [&](sched::Both const& both) -> sched::Done {
			                      auto batch = run_batch(both.batch);
			                      return sched::BothDone{.batch = std::move(batch), .chunk = run_chunk(both.chunk)};
		                      },
		                  },
		                  cycle);
	}

	auto account(sched::Done const& done) -> void {
		reached_.cycles += 1;
		std::visit(Overloaded{
		               [&](sched::BatchDone const& batch) {
			               reached_.batches += 1;
			               window_decode += batch.took;
		               },
		               [&](sched::ChunkDone const&) {
			               reached_.chunks += 1;
			               window_prefill = {};
			               window_decode = {};
		               },
		               [&](sched::BothDone const& both) {
			               reached_.batches += 1;
			               reached_.chunks += 1;
			               reached_.shared_cycles += 1;
			               window_prefill += both.chunk.took;
			               window_decode += both.batch.took;
		               },
		           },
		           done);
	}

	[[nodiscard]] auto reads_back(kv::BlockIndex index, Count offset, std::uint64_t chain) const -> bool {
		auto const& content = device_->content(index);
		return std::holds_alternative<std::vector<Cell>>(content) && std::get<std::vector<Cell>>(content)[offset] == Cell{Kv{.chain = chain}};
	}

	auto check_state() -> void {
		expect(device_->violations().empty(), "the device sees only well-formed commands and writes through set table entries");
		auto const& pool = scheduler_->pool();
		auto const facts = scheduler_->facts();
		auto blocks = std::set<std::uint64_t>{};
		for (auto index = Count{0}; index < model::slots; ++index) {
			auto const slot = Slot::at(index);
			for (auto entry = Count{0}; entry < pool.table_size(slot); ++entry) {
				blocks.insert(pool.entry(slot, entry).bits);
			}
			auto const live = std::visit(Overloaded{
			                                 [](sched::Vacant) -> std::optional<std::pair<RequestHandle, Count>> {
				                                 return std::nullopt;
			                                 },
			                                 [](sched::Finishing) -> std::optional<std::pair<RequestHandle, Count>> {
				                                 return std::nullopt;
			                                 },
			                                 [](sched::Prefilling const& reading) -> std::optional<std::pair<RequestHandle, Count>> {
				                                 return std::pair{reading.request, reading.prefilled_tokens};
			                                 },
			                                 [](sched::Decoding const& writing) -> std::optional<std::pair<RequestHandle, Count>> {
				                                 return std::pair{writing.request, writing.prompt_tokens + writing.generated_tokens - 1};
			                                 },
			                             },
			                             facts.slots[index]);
			if (!live) {
				expect(std::holds_alternative<sched::Finishing>(facts.slots[index]) || pool.table_size(slot) == 0, "a vacant slot holds no blocks");
				continue;
			}
			auto const [request, read] = *live;
			expect(pool.table_size(slot) * G.block_tokens >= read, "a slot's table covers what it has read");
			auto chain = empty_chain;
			for (auto position = Count{0}; position < read && position < pool.table_size(slot) * G.block_tokens; ++position) {
				chain = chain_step(chain, expected(request, position));
				auto const block = pool.index(pool.entry(slot, position / G.block_tokens));
				expect(device_->entry(slot, position / G.block_tokens) == block, "the device's table follows the pool's");
				expect(reads_back(block, position % G.block_tokens, chain), "every live slot reads back its own sequence");
			}
			if (read > 0) {
				expect(device_->state(slot) == chain, "every live slot's state is the state after what it read");
			}
		}
		auto const entries = scheduler_->cache().entries();
		for (auto const& entry : entries) {
			auto chain = empty_chain;
			for (auto const [position, token] : std::views::enumerate(entry.tokens)) {
				chain = chain_step(chain, token);
				auto const at = core::narrow<Count>(position);
				auto const handle = entry.blocks[at / G.block_tokens];
				blocks.insert(handle.bits);
				expect(reads_back(pool.index(handle), at % G.block_tokens, chain), "every cached prefix reads back exactly");
			}
			auto const parts = pool.snapshot_blocks(entry.snapshot);
			expect(parts.has_value() && device_->content((*parts)[0]) == Content{kv_fake::Part{.state = chain, .part = 0}},
			       "every cached snapshot is the state after its prefix");
		}
		expect(pool.snapshots() == entries.size(), "every snapshot in the pool is a cached point's");
		expect(pool.free_blocks() + blocks.size() + entries.size() * G.snapshot_blocks == G.pool_blocks,
		       "blocks are conserved: free, kv and snapshot make the pool");
		expect(facts.pool.used_blocks + facts.pool.cached_blocks + pool.free_blocks() == G.pool_blocks, "the status accounts for every block");
	}

	struct Live {
		std::uint64_t bits;
		std::uint64_t stamp;
	};

	Waits waits_;
	/** The request paused last, while no admission has followed. */
	std::optional<std::uint64_t> resumes_;
	std::unique_ptr<Scheduler> scheduler_;
	std::unique_ptr<kv_fake::FakeDevice<G>> device_;
	std::mt19937_64 rng_;
	std::map<std::uint64_t, Tracked> tracked_;
	std::vector<Live> live_;
	std::optional<sched::Cycle> flight_;
	std::vector<sched::Event> events_;
	std::vector<sched::Cycle> cycles_;
	std::optional<Round> last_round_;
	std::chrono::nanoseconds window_prefill{0};
	std::chrono::nanoseconds window_decode{0};
	std::uint64_t arrivals_ = 0;
	std::uint64_t admissions_ = 0;
	std::optional<std::string> failure_;
	Reached reached_{};
};

}

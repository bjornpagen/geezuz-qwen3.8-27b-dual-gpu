// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sched:scheduler;

import std;
import engine.core;
import engine.handle;
import engine.kv;
import engine.model;
import engine.prefix;
import engine.sampling;
import engine.schedule;
import engine.tokenizer;
import :cycle;
import :replay;
import :request;
import :status;

namespace engine::sched {

/** The most tokens a request may generate after a prompt of `prompt` tokens in a pool of geometry `G`. */
export template<kv::Geometry G>
[[nodiscard]] constexpr auto room(Count prompt) -> Count {
	// A decoding slot's committed row sits at the sequence's last position and the cycle writes `lookahead` positions
	// from it, so a sequence of n tokens needs n - 1 + lookahead positions of the table.
	auto const longest = kv::table_positions(G) + 1 - lookahead;
	return prompt < longest ? longest - prompt : 0;
}

/** A plan for the cards, or none when no request is left; and what happened to requests while planning it. */
export struct Planned {
	std::optional<Cycle> cycle;
	std::vector<Event> events;
};

/** Whether the caller still wants the request run. */
enum class Wanted : std::uint8_t {
	Run,
	End,
};

/** Why a request ends; the `Ending` it reports carries what it leaves. */
enum class Reason : std::uint8_t {
	Stop,
	Length,
	Cancelled,
};

/** A request the scheduler holds, queued or in a slot. */
struct Request {
	/** The prompt, then every token the slot committed after it. */
	std::vector<Token> context;
	Count prompt_tokens;
	prefix::Points points;
	sampling::Rule rule;
	/** Tokens it may generate. */
	Count limit;
	/** Positions its state was cached through when it was last paused, if the pool took the snapshot. */
	std::optional<Count> parked;
	/** Positions its first admission resumed from the cache; absent until then. */
	std::optional<Count> cached;
	/** Its place in arrival order. */
	std::uint64_t arrival;
	/** Its place in admission order, latest admission; absent until admitted. */
	std::optional<std::uint64_t> admitted;
	/** Blocks its slot held when it was last paused: most of them stay cached, which is no room it can count on. */
	Count held;
	Wanted wanted;
};

/** A slot no request holds; its table is empty. */
struct Idle {};

/** A slot reading its request's context: the state holds [0, read), and read < context.size(). */
struct Reading {
	RequestHandle request;
	Count read;
};

/** A slot generating: the state holds every position of the context but its last, the committed row. */
struct Writing {
	RequestHandle request;
};

/** A slot whose request was cancelled; it holds [0, read) until the next cycle boundary releases it. */
struct Leaving {
	RequestHandle request;
	Count read;
};

using Occupancy = std::variant<Idle, Reading, Writing, Leaving>;

/** Tokens the request generated, the one that ended its turn included: a row never commits past the end. */
[[nodiscard]] auto generated(Request const& request) -> Count {
	return request.context.size() - request.prompt_tokens;
}

template<class... F>
struct Overloaded : F... {
	using F::operator()...;
};

// Enough that no request record exhausts its generations in any lifetime of the server (2^48 requests per record).
inline constexpr std::uint8_t generation_bits = 48;

/**
 * The scheduler: S slots over one shared kv pool and its prefix cache, driven as a deterministic state machine. It
 * decides; the pipeline acts. Each round is `plan`, which admits queued requests, releases cancelled ones, makes the
 * pool room for the cycle (evicting cache first, then pausing the latest-admitted request) and returns the cycle; the
 * cards run it; then `settle` reads its outcome back, commits tokens and ends finished requests. Every kv command goes
 * to the device lent to the call, which applies them in order around the cycles. Time enters only as the durations
 * the cards report, so the same inputs make the same decisions.
 *
 * `G` is the pool's geometry, `B` the row budget of the kernel that runs the cycles, `T` the prefill share.
 *
 * Large (the pool and the cache): the owner keeps it behind a `std::unique_ptr`.
 */
export template<kv::Geometry G, model::RowBudget B, Tuning T>
class BasicScheduler {
	static_assert(sound(T));
	static_assert(room<G>(1) > 0, "the table holds a one-token prompt and its cycle");

	// Decode time owed per unit of prefill time: the rest of the share over the share.
	static constexpr auto repaid = core::narrow<std::int64_t>(T.share.denominator - T.share.numerator);
	static constexpr auto given = core::narrow<std::int64_t>(T.share.numerator);

public:
	BasicScheduler() : pool_{std::make_unique<kv::BasicPool<G>>()}, cache_{std::make_unique<prefix::BasicCache<G>>()} {}

	BasicScheduler(BasicScheduler const&) = delete;
	auto operator=(BasicScheduler const&) -> BasicScheduler& = delete;
	BasicScheduler(BasicScheduler&&) noexcept = default;
	auto operator=(BasicScheduler&&) noexcept -> BasicScheduler& = default;
	~BasicScheduler() = default;

	/** Queues a request, its prompt first replayed with the turns this scheduler generated. */
	[[nodiscard]] auto submit(tokenizer::Tokenizer const& tokenizer, Submission submission) -> core::Result<Submitted, SubmitError> {
		core::invariant(!submission.prompt.empty());
		auto replayed = replay_.apply(tokenizer, std::move(submission.prompt), submission.marks);
		auto const length = replayed.prompt.size();
		auto const most = room<G>(length);
		auto const limit = std::visit(Overloaded{
		                                  [&](UntilContextFull) {
			                                  return most;
		                                  },
		                                  [&](AtMost at_most) {
			                                  core::invariant(at_most.tokens > 0);
			                                  return at_most.tokens <= most ? at_most.tokens : 0;
		                                  },
		                              },
		                              submission.limit);
		if (limit == 0) {
			return std::unexpected(SubmitError{.cause = PromptTooLong{.prompt_tokens = length}});
		}
		auto marked = std::vector<prefix::Point>{};
		auto const mark = [&](std::optional<Count> position, prefix::PointKind kind) {
			if (position) {
				marked.push_back({.kind = kind, .position = *position});
			}
		};
		mark(replayed.marks.system_end, prefix::PointKind::SystemEnd);
		mark(replayed.marks.last_user_start, prefix::PointKind::LastUserStart);
		mark(replayed.marks.generation_start, prefix::PointKind::GenerationStart);
		marked.push_back({.kind = prefix::PointKind::PromptEnd, .position = length - 1});
		arrivals_ += 1;
		auto const handle = requests_.insert(Request{
		    .context = std::move(replayed.prompt),
		    .prompt_tokens = length,
		    .points = prefix::Points::of(length, marked),
		    .rule = sampling::rule(submission.sampling, 0),
		    .limit = limit,
		    .parked = std::nullopt,
		    .cached = std::nullopt,
		    .arrival = arrivals_,
		    .admitted = std::nullopt,
		    .held = 0,
		    .wanted = Wanted::Run,
		});
		if (!handle) {
			return std::unexpected(SubmitError{.cause = Full{}});
		}
		queue_.push_back(*handle);
		return Submitted{.request = *handle, .prompt_tokens = length};
	}

	/**
	 * Ends a request by its caller's decision. A queued one leaves at the next `plan`; a running one keeps its slot
	 * until the next cycle boundary, its state cached for the conversation's next request, and reports what that
	 * cycle committed. Either then reports `Finished`, `Cancelled` unless that cycle ended the request anyway. A stale handle (the request already finished) is a
	 * no-op: a cancellation races the request's own end.
	 */
	auto cancel(RequestHandle handle) -> void {
		auto request = requests_.try_get(handle);
		if (!request) {
			return;
		}
		request->wanted = Wanted::End;
		auto const slot = slot_of(handle);
		if (!slot) {
			return;
		}
		auto const read = read_of(*slot);
		slots_[slot->index()] = Leaving{.request = handle, .read = read};
	}

	/** The next cycle. Call it only when no cycle is in flight. */
	template<kv::Device<G> D>
	[[nodiscard]] auto plan(D& device) -> Planned {
		core::invariant(!flight_.has_value());
		auto events = std::vector<Event>{};
		release_ended(device, events);
		auto cycle = std::optional<Cycle>{};
		while (true) {
			admit(device, events);
			cycle = build(device, events);
			// A cycle comes from every live slot, so without one no slot is live: a request alone that the pool could
			// not hold ended, and the queue may admit the next.
			if (cycle || queue_.empty()) {
				break;
			}
		}
		if (cycle && carries_chunk(*cycle)) {
			rounds_ += 1;
		}
		flight_ = cycle;
		return {.cycle = std::move(cycle), .events = std::move(events)};
	}

	/** Reads back the cycle `plan` returned: `done` is its outcome, the alternative matching the cycle's. */
	template<kv::Device<G> D>
	[[nodiscard]] auto settle(D& device, Done const& done) -> std::vector<Event> {
		core::invariant(flight_.has_value());
		auto const cycle = *std::exchange(flight_, std::nullopt);
		auto events = std::vector<Event>{};
		std::visit(Overloaded{
		               [&](Batch const& batch, BatchDone const& outcome) {
			               settle_batch(device, batch, outcome, events);
			               owed_ = std::max(std::chrono::nanoseconds{0}, owed_ - outcome.took);
		               },
		               [&](Chunk const& chunk, ChunkDone const& outcome) {
			               settle_chunk(device, chunk, outcome, events);
			               // Nothing decoded, so no decoding slot is owed time.
			               owed_ = std::chrono::nanoseconds{0};
		               },
		               [&](Both const& both, BothDone const& outcome) {
			               settle_batch(device, both.batch, outcome.batch, events);
			               settle_chunk(device, both.chunk, outcome.chunk, events);
			               auto const owed = owed_ + outcome.chunk.took * repaid / given;
			               owed_ = std::max(std::chrono::nanoseconds{0}, owed - outcome.batch.took);
		               },
		               [](auto const&, auto const&) {
			               // The cards ran the cycle they were given, so the outcome has its shape.
			               core::invariant(false);
		               },
		           },
		           cycle, done);
		return events;
	}

	[[nodiscard]] auto facts() const -> Facts {
		auto facts = Facts{
		    .slots = {},
		    .queued = queue_.size(),
		    .pool = {},
		    .counters = counters_,
		};
		for (auto const [index, occupancy] : std::views::enumerate(slots_)) {
			facts.slots[core::narrow<std::size_t>(index)] = std::visit(
			    Overloaded{
			        [](Idle) -> SlotFacts {
				        return Vacant{};
			        },
			        [&](Reading const& reading) -> SlotFacts {
				        auto const& request = look(reading.request);
				        return Prefilling{
				            .request = reading.request,
				            .prompt_tokens = request.context.size(),
				            .cached_tokens = request.cached.value_or(0),
				            .prefilled_tokens = reading.read,
				        };
			        },
			        [&](Writing const& writing) -> SlotFacts {
				        auto const& request = look(writing.request);
				        return Decoding{
				            .request = writing.request,
				            .prompt_tokens = request.prompt_tokens,
				            .cached_tokens = request.cached.value_or(0),
				            .generated_tokens = generated(request),
				        };
			        },
			        [](Leaving const& ending) -> SlotFacts {
				        return Finishing{.request = ending.request};
			        },
			    },
			    occupancy);
		}
		auto const cached = pool_->held_only_blocks() + pool_->snapshots() * G.snapshot_blocks;
		facts.pool = {.blocks = G.pool_blocks, .used_blocks = G.pool_blocks - pool_->free_blocks() - cached, .cached_blocks = cached};
		return facts;
	}

	[[nodiscard]] auto pool() const -> kv::BasicPool<G> const& {
		return *pool_;
	}

	[[nodiscard]] auto cache() const -> prefix::BasicCache<G> const& {
		return *cache_;
	}

	[[nodiscard]] auto replay() const -> Replay const& {
		return replay_;
	}

private:
	using Requests = handle::Table<Request, RequestHandle, max_requests, generation_bits>;

	[[nodiscard]] auto look(RequestHandle handle) const -> Request const& {
		auto const found = requests_.try_get(handle);
		core::invariant(found.has_value());
		return *found;
	}

	[[nodiscard]] auto look(RequestHandle handle) -> Request& {
		return requests_.get(handle);
	}

	[[nodiscard]] static auto carries_chunk(Cycle const& cycle) -> bool {
		return !std::holds_alternative<Batch>(cycle);
	}

	[[nodiscard]] auto slot_of(RequestHandle handle) const -> std::optional<kv::Slot> {
		for (auto index = Count{0}; index < model::slots; ++index) {
			if (occupant(kv::Slot::at(index)) == handle) {
				return kv::Slot::at(index);
			}
		}
		return std::nullopt;
	}

	[[nodiscard]] auto occupant(kv::Slot slot) const -> std::optional<RequestHandle> {
		return std::visit(Overloaded{
		                      [](Idle) -> std::optional<RequestHandle> {
			                      return std::nullopt;
		                      },
		                      [](auto const& held) -> std::optional<RequestHandle> {
			                      return held.request;
		                      },
		                  },
		                  slots_[slot.index()]);
	}

	/** Positions of the slot's context its kv and state hold. */
	[[nodiscard]] auto read_of(kv::Slot slot) const -> Count {
		return std::visit(Overloaded{
		                      [](Idle) {
			                      return Count{0};
		                      },
		                      [](Reading const& reading) {
			                      return reading.read;
		                      },
		                      [&](Writing const& writing) {
			                      return look(writing.request).context.size() - 1;
		                      },
		                      [](Leaving const& ending) {
			                      return ending.read;
		                      },
		                  },
		                  slots_[slot.index()]);
	}

	[[nodiscard]] auto live() const -> Count {
		return core::narrow<Count>(std::ranges::count_if(slots_, [](Occupancy const& occupancy) {
			return !std::holds_alternative<Idle>(occupancy);
		}));
	}

	/**
	 * The slot leaves its request: everything its state has read is cached at its end, so the conversation's next
	 * request (or this one, resumed) starts there. The positions cached, if the pool took them.
	 */
	template<kv::Device<G> D>
	[[nodiscard]] auto vacate(D& device, kv::Slot slot) -> std::optional<Count> {
		auto const read = read_of(slot);
		auto const handle = occupant(slot);
		core::invariant(handle.has_value());
		auto const& request = look(*handle);
		slots_[slot.index()] = Idle{};
		if (read == 0) {
			cache_->abandon(*pool_, slot);
			return std::nullopt;
		}
		if (auto const kept = cache_->close(device, *pool_, slot, std::span{request.context}.first(read)); !kept) {
			counters_.snapshots_refused += 1;
			return std::nullopt;
		}
		return read;
	}

	[[nodiscard]] static auto usage(Request const& request) -> Usage {
		return {.prompt_tokens = request.prompt_tokens, .cached_tokens = request.cached.value_or(0), .generated_tokens = generated(request)};
	}

	/** What a cancelled request leaves: the sequence whose state was cached through `cached` positions, if any. */
	[[nodiscard]] static auto cancelled(Request const& request, std::optional<Count> cached) -> Ending {
		if (!cached) {
			return Cancelled{.state = Dropped{}};
		}
		return Cancelled{.state = Kept{.sequence = {request.context.begin(), request.context.begin() + core::narrow<std::ptrdiff_t>(*cached)}}};
	}

	/**
	 * Ends a request in a slot: its slot is vacated, a turn it ended is kept for replay, and it reports `Finished`
	 * with the sequence it leaves.
	 */
	template<kv::Device<G> D>
	auto finish(D& device, kv::Slot slot, Reason reason, std::vector<Event>& events) -> void {
		auto const handle = occupant(slot);
		core::invariant(handle.has_value());
		auto const cached = vacate(device, slot);
		auto& request = look(*handle);
		auto const counts = usage(request);
		auto ending = [&] -> Ending {
			switch (reason) {
			case Reason::Stop:
				replay_.finished(std::span{request.context}.first(request.prompt_tokens), std::span{request.context}.subspan(request.prompt_tokens));
				return Stopped{.sequence = std::move(request.context)};
			case Reason::Length:
				return Truncated{.sequence = std::move(request.context)};
			case Reason::Cancelled:
				return cancelled(request, cached);
			}
			std::unreachable();
		}();
		events.emplace_back(Finished{.request = *handle, .ending = std::move(ending), .usage = counts});
		requests_.release(*handle);
	}

	/** A cycle boundary: cancelled requests leave, queued or running. */
	template<kv::Device<G> D>
	auto release_ended(D& device, std::vector<Event>& events) -> void {
		for (auto index = Count{0}; index < model::slots; ++index) {
			if (std::holds_alternative<Leaving>(slots_[index])) {
				finish(device, kv::Slot::at(index), Reason::Cancelled, events);
			}
		}
		std::erase_if(queue_, [&](RequestHandle handle) {
			auto const& request = look(handle);
			if (request.wanted == Wanted::Run) {
				return false;
			}
			// A paused request's state was cached when it left its slot.
			events.emplace_back(Finished{.request = handle, .ending = cancelled(request, request.parked), .usage = usage(request)});
			requests_.release(handle);
			return true;
		});
	}

	/**
	 * Whether the pool plausibly holds the request beside the running ones: free blocks and what the cache could give
	 * back cover its whole sequence, one cycle's lookahead and a block of growth per running request. Optimistic by
	 * design: a request that turns out not to fit is paused again, and the running ones keep going meanwhile.
	 */
	[[nodiscard]] auto fits(Request const& request) const -> bool {
		// Each live slot pins at most one snapshot, so at least this many rest and are the cache's to give back.
		auto const resting = pool_->snapshots() - std::min(pool_->snapshots(), live());
		auto const available = pool_->free_blocks() + pool_->held_only_blocks() + resting * G.snapshot_blocks;
		auto const need = model::divide_up(request.context.size() + lookahead, G.block_tokens) + live() + request.held;
		return need <= available;
	}

	/**
	 * Admits queued requests in order while a slot is free. A request whose prompt shares a snapshot point another
	 * slot is still prefilling waits for it and lets the ones behind it go first; otherwise the first that does not fit
	 * beside the running ones stops admission, so a long request is not overtaken forever.
	 */
	template<kv::Device<G> D>
	auto admit(D& device, std::vector<Event>& events) -> void {
		auto position = Count{0};
		while (position < queue_.size() && live() < model::slots) {
			auto const handle = queue_[position];
			auto& request = look(handle);
			auto match = cache_->lookup(request.context);
			if (match.wait) {
				position += 1;
				continue;
			}
			if (live() > 0 && !fits(request)) {
				return;
			}
			queue_.erase(queue_.begin() + core::narrow<std::ptrdiff_t>(position));
			auto const free = std::ranges::find_if(std::views::iota(Count{0}, model::slots), [&](Count index) {
				return std::holds_alternative<Idle>(slots_[index]);
			});
			auto const slot = kv::Slot::at(*free);
			auto const from = match.hit ? match.hit->length : 0;
			if (match.hit) {
				cache_->resume(device, *pool_, slot, *match.hit);
			}
			cache_->announce(slot, request.context, request.points, from);
			admissions_ += 1;
			request.admitted = admissions_;
			request.cached = request.cached.value_or(from);
			slots_[slot.index()] = Reading{.request = handle, .read = from};
			events.emplace_back(Admitted{.request = handle, .slot = slot, .cached = from});
		}
	}

	/** The live request admitted last. */
	[[nodiscard]] auto newest() const -> kv::Slot {
		auto best = std::optional<kv::Slot>{};
		auto stamp = std::uint64_t{0};
		for (auto index = Count{0}; index < model::slots; ++index) {
			auto const handle = occupant(kv::Slot::at(index));
			if (!handle) {
				continue;
			}
			auto const admitted = look(*handle).admitted;
			core::invariant(admitted.has_value());
			if (!best || *admitted > stamp) {
				best = kv::Slot::at(index);
				stamp = *admitted;
			}
		}
		core::invariant(best.has_value());
		return *best;
	}

	/** The pool ran out: the slot's request leaves with its state cached and waits at the head of the queue. */
	template<kv::Device<G> D>
	auto pause(D& device, kv::Slot slot, std::vector<Event>& events) -> void {
		auto const handle = occupant(slot);
		core::invariant(handle.has_value());
		auto const held = pool_->table_size(slot);
		auto& request = look(*handle);
		request.parked = vacate(device, slot);
		request.held = held;
		queue_.push_front(*handle);
		counters_.preemptions += 1;
		events.emplace_back(Paused{.request = *handle});
	}

	/**
	 * The prefill chunk this cycle runs, if any: when nothing decodes, prompts go in back to back; while slots decode,
	 * a chunk runs only once they have had their share of time back. Rounds alternate between the shortest remaining
	 * prompt first, so short prompts arriving together start together, and the longest waiting first, so a long prompt
	 * keeps at least every second chunk while short turns keep arriving. A chunk stops at the prompt's next snapshot
	 * point, so the cache's insert lands exactly there.
	 */
	[[nodiscard]] auto select_chunk(Count decoding) const -> std::optional<Chunk> {
		if (decoding > 0 && owed_ > std::chrono::nanoseconds{0}) {
			return std::nullopt;
		}
		auto best = std::optional<kv::Slot>{};
		auto const key = [&](kv::Slot slot) {
			auto const& reading = std::get<Reading>(slots_[slot.index()]);
			auto const& request = look(reading.request);
			auto const remaining = request.context.size() - reading.read;
			auto const order = (rounds_ + 1) % 2 == 1 ? remaining : request.arrival;
			return std::pair{order, *request.admitted};
		};
		for (auto index = Count{0}; index < model::slots; ++index) {
			if (!std::holds_alternative<Reading>(slots_[index])) {
				continue;
			}
			auto const slot = kv::Slot::at(index);
			if (!best || key(slot) < key(*best)) {
				best = slot;
			}
		}
		if (!best) {
			return std::nullopt;
		}
		auto const& reading = std::get<Reading>(slots_[best->index()]);
		auto const& request = look(reading.request);
		auto const ahead = request.points.after(reading.read);
		auto end = std::min(request.context.size(), reading.read + T.chunk_tokens);
		if (!ahead.empty()) {
			end = std::min(end, ahead.front().position);
		}
		return Chunk{
		    .slot = *best,
		    .request = reading.request,
		    .from = reading.read,
		    .tokens = {request.context.begin() + core::narrow<std::ptrdiff_t>(reading.read), request.context.begin() + core::narrow<std::ptrdiff_t>(end)},
		    .read = end == request.context.size() ? Read::Last : Read::Part,
		    .rule = request.rule,
		};
	}

	/** Positions a chunk writes: a last chunk is followed by the slot's first decode cycle's drafts. */
	[[nodiscard]] static auto extent(Chunk const& chunk) -> kv::Extend {
		auto const end = chunk.from + chunk.tokens.size();
		return {.slot = chunk.slot, .length = chunk.from, .until = chunk.read == Read::Last ? end - 1 + lookahead : end};
	}

	/** Makes the extends writable, evicting cached blocks first when the pool is short. */
	template<kv::Device<G> D>
	[[nodiscard]] auto ensured(D& device, std::span<kv::Extend const> extends) -> bool {
		auto done = pool_->ensure(device, extends);
		if (done) {
			return true;
		}
		std::ignore = cache_->make_room(*pool_, done.error().needed);
		return pool_->ensure(device, extends).has_value();
	}

	/**
	 * The cycle of every decoding slot and the chosen chunk, with the pool made room for both. When it cannot be, the
	 * latest-admitted request is paused and the cycle chosen again; a request alone that the pool cannot hold ends.
	 */
	template<kv::Device<G> D>
	[[nodiscard]] auto build(D& device, std::vector<Event>& events) -> std::optional<Cycle> {
		while (true) {
			auto rows = std::vector<Row>{};
			auto extends = std::vector<kv::Extend>{};
			for (auto index = Count{0}; index < model::slots; ++index) {
				if (!std::holds_alternative<Writing>(slots_[index])) {
					continue;
				}
				auto const handle = std::get<Writing>(slots_[index]).request;
				auto const& request = look(handle);
				auto const position = request.context.size() - 1;
				rows.push_back({
				    .slot = kv::Slot::at(index),
				    .request = handle,
				    .position = position,
				    .token = request.context.back(),
				    .rule = request.rule,
				    // Capped at the batch's depth once it is known.
				    .most = request.limit - generated(request),
				});
				extends.push_back({.slot = kv::Slot::at(index), .length = position, .until = position + lookahead});
			}
			auto chunk = select_chunk(rows.size());
			if (chunk) {
				extends.push_back(extent(*chunk));
			}
			if (extends.empty()) {
				return std::nullopt;
			}
			if (ensured(device, extends)) {
				if (rows.empty()) {
					return Cycle{std::move(*chunk)};
				}
				auto batch = Batch{.depth = schedule::shape(rows.size(), B), .rows = std::move(rows)};
				for (auto& row : batch.rows) {
					row.most = std::min(row.most, 1 + model::drafts(batch.depth));
				}
				if (!chunk) {
					return Cycle{std::move(batch)};
				}
				return Cycle{Both{.batch = std::move(batch), .chunk = std::move(*chunk)}};
			}
			if (live() == 1) {
				finish(device, newest(), Reason::Length, events);
				continue;
			}
			pause(device, newest(), events);
		}
	}

	/**
	 * Hands the request the tokens just appended to its context: an end-of-turn token, only ever the last, is counted
	 * but not reported. Why the request ends, if these tokens ended it.
	 */
	[[nodiscard]] auto deliver(RequestHandle handle, std::span<Token const> tokens, std::vector<Event>& events) -> std::optional<Reason> {
		auto const& request = look(handle);
		core::invariant(generated(request) <= request.limit && std::ranges::none_of(tokens.first(tokens.size() - 1), ends_turn));
		auto const stops = ends_turn(tokens.back());
		auto reported = std::vector<Token>{tokens.begin(), tokens.end() - (stops ? 1 : 0)};
		if (!reported.empty()) {
			events.emplace_back(Generated{.request = handle, .tokens = std::move(reported)});
		}
		if (stops) {
			return Reason::Stop;
		}
		return generated(request) == request.limit ? std::optional{Reason::Length} : std::nullopt;
	}

	/**
	 * The slot committed `tokens`: they are reported, and the request ends if they ended it, or if it was cancelled
	 * while they were being made.
	 */
	template<kv::Device<G> D>
	auto commit(D& device, kv::Slot slot, RequestHandle handle, std::span<Token const> tokens, std::vector<Event>& events) -> void {
		auto const reason = deliver(handle, tokens, events);
		if (std::holds_alternative<Leaving>(slots_[slot.index()])) {
			finish(device, slot, reason.value_or(Reason::Cancelled), events);
		} else if (reason) {
			finish(device, slot, *reason, events);
		}
	}

	template<kv::Device<G> D>
	auto settle_batch(D& device, Batch const& batch, BatchDone const& done, std::vector<Event>& events) -> void {
		core::invariant(done.rows.size() == batch.rows.size());
		counters_.cycles += 1;
		counters_.drafted_tokens += batch.rows.size() * model::drafts(batch.depth);
		for (auto const [row, accepted] : std::views::zip(batch.rows, done.rows)) {
			core::invariant(accepted.count >= 1 && accepted.count <= row.most);
			counters_.accepted_tokens += accepted.count - 1;
			auto& request = look(row.request);
			core::invariant(request.context.size() == row.position + 1);
			request.context.append_range(accepted.view());
			if (auto leaving = core::alternative<Leaving>(slots_[row.slot.index()])) {
				leaving->read = request.context.size() - 1;
			}
			commit(device, row.slot, row.request, accepted.view(), events);
		}
	}

	template<kv::Device<G> D>
	auto settle_chunk(D& device, Chunk const& chunk, ChunkDone const& done, std::vector<Event>& events) -> void {
		core::invariant(done.sampled.has_value() == (chunk.read == Read::Last));
		counters_.prefill_chunks += 1;
		counters_.prefill_tokens += chunk.tokens.size();
		auto const read = chunk.from + chunk.tokens.size();
		auto& request = look(chunk.request);
		auto leaving = core::alternative<Leaving>(slots_[chunk.slot.index()]);
		if (leaving) {
			leaving->read = read;
		} else {
			auto const ahead = request.points.after(chunk.from);
			if (!ahead.empty() && ahead.front().position == read) {
				auto const saved = cache_->insert(device, *pool_, chunk.slot, std::span{request.context}.first(read), prefix::Mark{ahead.front().kind});
				counters_.snapshots_refused += saved ? 0 : 1;
			}
		}
		if (chunk.read == Read::Part) {
			if (leaving) {
				finish(device, chunk.slot, Reason::Cancelled, events);
			} else {
				slots_[chunk.slot.index()] = Reading{.request = chunk.request, .read = read};
			}
			return;
		}
		request.context.push_back(*done.sampled);
		if (!leaving) {
			slots_[chunk.slot.index()] = Writing{.request = chunk.request};
		}
		commit(device, chunk.slot, chunk.request, std::span{request.context}.last(1), events);
	}

	std::unique_ptr<kv::BasicPool<G>> pool_;
	std::unique_ptr<prefix::BasicCache<G>> cache_;
	Requests requests_;
	/** Waiting requests in admission order: arrivals at the back, paused requests at the front. */
	std::deque<RequestHandle> queue_;
	std::array<Occupancy, model::slots> slots_{};
	/** The cycle the cards are running: planned and not yet settled. */
	std::optional<Cycle> flight_;
	Replay replay_;
	/** Decode time the decoding slots are owed for the prefill time they gave up. */
	std::chrono::nanoseconds owed_{0};
	/** Chunks run so far; its parity picks the next round's order. */
	std::uint64_t rounds_ = 0;
	std::uint64_t arrivals_ = 0;
	std::uint64_t admissions_ = 0;
	Counters counters_{};
};

/** The engine's scheduler, for the row budget of the kernel that runs its cycles. */
export template<model::RowBudget B>
using Scheduler = BasicScheduler<kv::model_geometry, B, tuning>;

}

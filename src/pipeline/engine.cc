// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline:engine;

import std;
import engine.card_scheduler;
import engine.core;
import engine.exec;
import engine.grammar;
import engine.http;
import engine.kv;
import engine.model;
import engine.net;
import engine.sched;
import engine.tokenizer;
import :masks;
import :outbox;
import :seams;

namespace engine::pipeline {

/** A request the api's service hands the engine. */
export template<class Submission>
struct Submit {
	http::Exchange exchange;
	Submission submission;
};

/** The client of a request is gone or too slow: the request ends, answered `cancelled`. */
export struct Cancel {
	http::Exchange exchange;

	[[nodiscard]] constexpr auto operator==(Cancel const&) const -> bool = default;
};

/** What the reactor's thread posts to the engine's. */
export template<class Submission>
using Intake = std::variant<Submit<Submission>, Cancel>;

export inline constexpr std::size_t intake_capacity = http::max_posted_messages;

// A connection carries one request at a time and posts at most one submit and one cancel for it, so the intake never
// holds more than two messages per connection, and a full intake is a broken service.
static_assert(intake_capacity >= 2 * http::max_connections, "the intake holds every connection's submit and cancel");

// Every request the service owes an answer is one connection's, so the scheduler's records hold them all.
static_assert(http::max_connections <= sched::max_requests, "the scheduler holds a request per connection");

export template<class Submission>
using Inbox = card_scheduler::Inbox<Intake<Submission>, intake_capacity>;

/** Rendered, and waiting for a request record of the scheduler: every one holds a request that has not finished. */
struct Waiting {
	sched::Submission request;
};

/** The client is there, and the request goes on as its grammar allows. */
struct Listening {};

/**
 * The client left: the request ends at the next cycle boundary and is then answered `cancelled`. The cycle in flight
 * may still report its tokens, and may end the request anyway; none of that is posted.
 */
struct Left {};

/**
 * The request's grammar outgrew its bounds: it is sampled under an end-of-turn mask, ends at the next cycle boundary,
 * and is then answered `overran` with the fault. Nothing more of it is posted before that.
 */
struct Overran {
	grammar::MatchError fault;
};

using Fate = std::variant<Listening, Left, Overran>;

/** In the scheduler: queued, running or paused. */
struct Scheduled {
	sched::RequestHandle request;
	Fate fate;
};

/** A request the engine holds, from its submission until the scheduler reports it finished. */
template<class F>
struct Live {
	http::Exchange exchange;
	F::Tag tag;
	std::variant<Waiting, Scheduled> place;
	F::Grammar grammar;
	F::Reply reply;
};

template<class... F>
struct Overloaded : F... {
	using F::operator()...;
};

/** How long the engine waits before it posts again to a full server mailbox: the reactor drains it every turn. */
inline constexpr auto repost_after = std::chrono::milliseconds{1};

/**
 * The longest a failed engine waits between posts of its failure: it backs off from `repost_after` while the server
 * stays behind, and stays prompt enough for an owner that stops it.
 */
inline constexpr auto most_repost_after = std::chrono::milliseconds{32};

/**
 * The engine's thread: one owner of the scheduler, the cards, and every request's grammar state and reply. Each round
 * takes what the reactor posted, plans a cycle (the pool's commands go to the cards as the scheduler issues them),
 * runs it, settles it, and turns the scheduler's events into the api's outputs, posted as one batch.
 *
 * A cycle is one sender per step (draft, verify, prefill), waited on this thread's own loop, which polls the cards'
 * events; the work a round can do without the cards, taking and preparing new requests, runs on the same loop while
 * the cards compute. A batch's masks are made between its `draft` and its `verify`.
 */
export template<kv::Geometry G, model::RowBudget B, sched::Tuning T, Cards<G> C, class F>
    requires Front<F, typename C::Error>
class Loop {
	using Scheduler = sched::BasicScheduler<G, B, T>;
	using Output = F::Output;
	using Tag = F::Tag;
	using Request = Live<F>;
	using Failure = EngineError<typename C::Error>;

public:
	Loop(http::Poster<typename F::Update> poster, tokenizer::Tokenizer tokenizer, grammar::Vocabulary vocabulary, C cards, F front)
	    : scheduler_{std::make_unique<Scheduler>()},
	      tokenizer_{std::move(tokenizer)},
	      cards_{std::move(cards)},
	      front_{std::move(front)},
	      vocabulary_{std::make_unique<grammar::Vocabulary>(std::move(vocabulary))},
	      outbox_{poster} {
		masks_.rows.reserve(model::rows(B));
		lives_.reserve(http::max_connections + sched::max_requests);
	}

	/**
	 * Serves until the inbox is stopping, or until the cards or the thread's wait fail. A failure is posted to the
	 * server as the engine's last update before it is returned.
	 */
	[[nodiscard]] auto run(Inbox<typename F::Submission>& inbox) -> core::Result<void, Failure> {
		auto served = serve(inbox);
		if (!served) {
			fail(served.error(), inbox);
		}
		return served;
	}

private:
	[[nodiscard]] auto serve(Inbox<typename F::Submission>& inbox) -> core::Result<void, Failure> {
		while (!inbox.stopping()) {
			take(inbox);
			admit_waiting();
			if (post() == Sent::Held) {
				// No cycle adds to what the server has not taken yet: what is held stays one round's outputs.
				if (auto waited = inbox.wait(net::Clock::now() + repost_after); !waited) {
					return std::unexpected(Failure{.cause = waited.error()});
				}
				continue;
			}
			auto planned = scheduler_->plan(cards_);
			observe(std::move(planned.events));
			if (!planned.cycle) {
				// Ended requests freed records for those waiting: they go to the scheduler before anything sleeps.
				if (std::ranges::any_of(lives_, &Loop::waiting)) {
					continue;
				}
				auto const until = post() == Sent::Held ? std::optional{net::Clock::now() + repost_after} : std::nullopt;
				if (auto waited = inbox.wait(until); !waited) {
					return std::unexpected(Failure{.cause = waited.error()});
				}
				continue;
			}
			auto done = run_cycle(*planned.cycle, inbox);
			if (!done) {
				return std::unexpected(done.error());
			}
			observe(scheduler_->settle(cards_, *done));
		}
		return {};
	}

	/**
	 * The failed engine's last posts: what it said before the failure, then the failure, posted again with a growing
	 * pause while the server's mailbox is full. Nothing is posted after them.
	 */
	auto fail(Failure const& error, Inbox<typename F::Submission>& inbox) -> void {
		gather();
		outbox_.close(front_.failed(error));
		auto pause = repost_after;
		// A stopping owner learns the failure from `stop`, and may be the reactor's own thread, which drains nothing
		// while it joins this one.
		while (outbox_.flush() == Sent::Held && !inbox.stopping()) {
			// The inbox's wait also ends at once when the owner stops the engine. A wait that cannot wait leaves no
			// way to pace the retries, and the owner learns the failure from `stop` all the same.
			if (!inbox.wait(net::Clock::now() + pause)) {
				return;
			}
			pause = std::min(2 * pause, most_repost_after);
		}
	}

	[[nodiscard]] static auto waiting(Request const& live) -> bool {
		return std::holds_alternative<Waiting>(live.place);
	}

	/** Whether the request's client still waits for its terminal output. */
	[[nodiscard]] static auto owed(Request const& live) -> bool {
		auto const held = core::alternative<Scheduled>(live.place);
		return !held || !std::holds_alternative<Left>(held->fate);
	}

	[[nodiscard]] static auto holds(Request const& live, sched::RequestHandle request) -> bool {
		auto const held = core::alternative<Scheduled>(live.place);
		return held && held->request == request;
	}

	[[nodiscard]] auto live_of(sched::RequestHandle request) -> Request& {
		auto const found = std::ranges::find_if(lives_, [&](Request const& live) {
			return holds(live, request);
		});
		// The engine keeps every request the scheduler holds until the scheduler reports it finished.
		core::invariant(found != lives_.end());
		return *found;
	}

	[[nodiscard]] auto tag_of(sched::RequestHandle request) const -> Tag const& {
		auto const found = std::ranges::find_if(lives_, [&](Request const& live) {
			return holds(live, request);
		});
		core::invariant(found != lives_.end());
		return found->tag;
	}

	/** Everything the reactor posted so far. */
	auto take(Inbox<typename F::Submission>& inbox) -> void {
		inbox.drain([this](Intake<typename F::Submission>&& message) {
			std::visit(Overloaded{
			               [this](Submit<typename F::Submission> submit) {
				               receive(submit.exchange, std::move(submit.submission));
			               },
			               [this](Cancel const& cancel) {
				               drop(cancel.exchange);
			               },
			           },
			           std::move(message));
		});
	}

	auto receive(http::Exchange exchange, F::Submission submission) -> void {
		auto prepared = front_.prepare(tokenizer_, std::move(submission));
		lives_.push_back(Request{
		    .exchange = exchange,
		    .tag = std::move(prepared.tag),
		    .place = Waiting{.request = std::move(prepared.request)},
		    .grammar = std::move(prepared.grammar),
		    .reply = std::move(prepared.reply),
		});
		admit_waiting();
	}

	/**
	 * Hands waiting requests to the scheduler, oldest first, while it has request records free. The engine holds every
	 * request the scheduler does, so it counts the records in use itself and never meets a full scheduler.
	 */
	auto admit_waiting() -> void {
		auto const in_use = std::ranges::count_if(lives_, [](Request const& live) {
			return !waiting(live);
		});
		auto free = sched::max_requests - core::narrow<std::size_t>(in_use);
		auto index = std::size_t{0};
		while (index < lives_.size() && free > 0) {
			auto& live = lives_[index];
			auto const held = core::alternative<Waiting>(live.place);
			if (!held) {
				index += 1;
				continue;
			}
			auto submitted = scheduler_->submit(tokenizer_, std::move(held->request));
			if (submitted) {
				live.place = Scheduled{.request = submitted->request, .fate = Listening{}};
				round_.push_back({.exchange = live.exchange, .output = front_.accepted()});
				free -= 1;
				index += 1;
				continue;
			}
			auto const too_long = core::alternative<sched::PromptTooLong>(submitted.error().cause);
			// The engine counted a free record, so only the prompt can be refused.
			core::invariant(too_long.has_value());
			round_.push_back({.exchange = live.exchange, .output = front_.refused(*too_long)});
			lives_.erase(lives_.begin() + core::narrow<std::ptrdiff_t>(index));
		}
	}

	/**
	 * The client is gone. A waiting request is forgotten and answered `cancelled` at once; a scheduled one ends at the
	 * next cycle boundary and is answered `cancelled` when the scheduler finishes it, so every request the engine
	 * still holds is one the service counts as outstanding.
	 */
	auto drop(http::Exchange exchange) -> void {
		auto const found = std::ranges::find_if(lives_, [&](Request const& live) {
			return live.exchange == exchange && owed(live);
		});
		// A request answered before its client left (finished or refused), or whose client already left, is owed nothing more.
		if (found == lives_.end()) {
			return;
		}
		if (auto held = core::alternative<Scheduled>(found->place)) {
			scheduler_->cancel(held->request);
			held->fate = Left{};
			return;
		}
		round_.push_back({.exchange = exchange, .output = front_.cancelled()});
		lives_.erase(found);
	}

	auto observe(std::vector<sched::Event> events) -> void {
		for (auto& event : events) {
			std::visit(Overloaded{
			               [](sched::Admitted const& /*admitted*/) {},
			               [this](sched::Generated const& generated) {
				               produce(live_of(generated.request), generated.tokens);
			               },
			               [](sched::Paused const& /*paused*/) {},
			               [this](sched::Finished& finished) {
				               auto const request = finished.request;
				               end(live_of(request), std::move(finished));
				               std::erase_if(lives_, [&](Request const& live) {
					               return holds(live, request);
				               });
			               },
			           },
			           event);
		}
	}

	auto say(Request const& live, std::vector<Output> outputs) -> void {
		for (auto& output : outputs) {
			round_.push_back({.exchange = live.exchange, .output = std::move(output)});
		}
	}

	/** The request's committed tokens, in order: its grammar advances by each, and its reply turns each into outputs. */
	auto produce(Request& live, std::span<Token const> tokens) -> void {
		if (!std::holds_alternative<Listening>(std::get<Scheduled>(live.place).fate)) {
			return;
		}
		for (auto const token : tokens) {
			if (auto const taken = live.grammar.advance(*vocabulary_, token); !taken) {
				// The cards sample under the request's masks, so a committed token can only fail the grammar's bounds.
				core::invariant(outgrown(taken.error()));
				overrun(live, taken.error());
				return;
			}
			say(live, live.reply.feed(tokenizer_, token));
		}
	}

	/** The request's grammar outgrew its bounds: it ends at the next cycle boundary, answered `overran`. */
	auto overrun(Request& live, grammar::MatchError fault) -> void {
		auto& held = std::get<Scheduled>(live.place);
		scheduler_->cancel(held.request);
		held.fate = Overran{.fault = fault};
	}

	auto end(Request& live, sched::Finished finished) -> void {
		auto const& fate = std::get<Scheduled>(live.place).fate;
		// The cycle in flight when the client left, or when the grammar overran, may have ended the request anyway: its
		// answer is the one its fate decided.
		if (std::holds_alternative<Left>(fate)) {
			round_.push_back({.exchange = live.exchange, .output = front_.cancelled()});
			return;
		}
		if (auto const overran = core::alternative<Overran>(fate)) {
			round_.push_back({.exchange = live.exchange, .output = front_.overran(overran->fault)});
			return;
		}
		std::visit(Overloaded{
		               [](sched::Cancelled& /*cancelled*/) {
			               // Only the engine cancels a request, and only when its client left or its grammar overran.
			               core::invariant(false);
		               },
		               [&]<class E>(E& ended) {
			               say(live, live.reply.end(tokenizer_, Finish{.ending = Ending{std::move(ended)}, .usage = finished.usage}));
		               },
		           },
		           finished.ending);
	}

	/** Hands the round's outputs to the outbox as one batch. */
	auto gather() -> void {
		if (!round_.empty()) {
			outbox_.add(front_.batch(std::exchange(round_, {})));
		}
	}

	/** Posts the round's outputs as one batch, and the status when it changed. */
	[[nodiscard]] auto post() -> Sent {
		gather();
		auto snapshot = this->snapshot();
		if (snapshot != reported_) {
			outbox_.report(front_.status(snapshot));
			reported_ = std::move(snapshot);
		}
		return outbox_.flush();
	}

	/** Waits on this thread for a step of the cycle, taking the reactor's posts while the cards work. */
	template<class S>
	[[nodiscard]] auto await(S step, Inbox<typename F::Submission>& inbox) {
		auto waited = ex::wait<typename C::Error>(ex::when_all(std::move(step), ex::just() | ex::then([this, &inbox] {
			                                                                            take(inbox);
		                                                                            })));
		// Nothing asks a wait of the engine's to stop, and the cards' senders are never stopped.
		core::invariant(waited.has_value());
		using Value = std::tuple_element_t<0, std::remove_cvref_t<decltype(**waited)>>;
		if (!*waited) {
			return core::Result<Value, Failure>{std::unexpected(Failure{.cause = std::move(*waited).error()})};
		}
		return core::Result<Value, Failure>{std::get<0>(std::move(**waited))};
	}

	/**
	 * Each verify row's mask: the request's grammar after its committed tokens, then after each draft in turn. A draft
	 * the grammar refuses is excluded by the mask before it, so the cards reject it and no row after it commits; those
	 * rows repeat the last mask, as do the rows past a row's drafts, which read nothing. The grammar is rolled back to
	 * the committed tokens afterwards. A request whose grammar outgrows its bounds on the way has all its rows masked
	 * to the end of the turn.
	 */
	auto fill(sched::Batch const& batch, Drafts const& drafts) -> void {
		auto const depth = model::drafts(batch.depth);
		masks_.rows.clear();
		masks_.rows.resize(batch.rows.size() * (1 + depth));
		for (auto const [signed_index, row] : std::views::enumerate(batch.rows)) {
			auto const index = core::narrow<std::size_t>(signed_index);
			auto const held = drafts.rows[index].view();
			// The cards hold at most the batch's depth of drafts for a row.
			core::invariant(held.size() <= depth);
			auto const masks = std::span{masks_.rows}.subspan(index * (1 + depth), 1 + depth);
			auto& live = live_of(row.request);
			if (std::holds_alternative<Overran>(std::get<Scheduled>(live.place).fate)) {
				std::ranges::fill(masks, RowMask::ending());
			} else if (auto const fault = fill_row(live.grammar, held, masks)) {
				overrun(live, *fault);
				std::ranges::fill(masks, RowMask::ending());
			}
		}
	}

	/** One row's masks, the grammar rolled back after; the fault when the grammar outgrew its bounds on the way. */
	[[nodiscard]] auto fill_row(F::Grammar& state, std::span<Token const> held, std::span<RowMask> masks)
	    -> std::optional<grammar::MatchError> {
		auto taken = std::size_t{0};
		auto fault = std::optional<grammar::MatchError>{};
		for (auto const [signed_at, mask] : std::views::enumerate(masks)) {
			auto const at = core::narrow<std::size_t>(signed_at);
			// A row past the first reads its draft, if the cards hold one and every draft before it was taken.
			if (at > 0) {
				auto const step = at - 1 < held.size() && taken == at - 1
				                      ? std::optional{state.advance(*vocabulary_, held[at - 1])}
				                      : std::nullopt;
				if (!step || !*step) {
					if (step && outgrown(step->error())) {
						fault = step->error();
						break;
					}
					mask = masks[at - 1];
					continue;
				}
				taken += 1;
			}
			auto const view = state.mask(*vocabulary_);
			if (!view) {
				// A mask fails only when its grammar outgrows its bounds.
				core::invariant(outgrown(view.error()));
				fault = view.error();
				break;
			}
			mask = RowMask::of(*view);
		}
		state.rollback(taken);
		return fault;
	}

	/** The mask a chunk that reaches the end of its sequence samples under: its grammar's, or the end of the turn's. */
	[[nodiscard]] auto first_mask(Request& live) -> RowMask {
		if (std::holds_alternative<Overran>(std::get<Scheduled>(live.place).fate)) {
			return RowMask::ending();
		}
		auto const view = live.grammar.mask(*vocabulary_);
		if (view) {
			return RowMask::of(*view);
		}
		// A mask fails only when its grammar outgrows its bounds.
		core::invariant(outgrown(view.error()));
		overrun(live, view.error());
		return RowMask::ending();
	}

	[[nodiscard]] auto run_batch(sched::Batch const& batch, Inbox<typename F::Submission>& inbox) -> core::Result<sched::BatchDone, Failure> {
		auto drafted = await(cards_.draft(sched::Batch{batch}), inbox);
		if (!drafted) {
			return std::unexpected(drafted.error());
		}
		// The cards draft every row of the batch.
		core::invariant(drafted->rows.size() == batch.rows.size());
		fill(batch, *drafted);
		auto verified = await(cards_.verify(std::move(masks_)), inbox);
		if (!verified) {
			return std::unexpected(verified.error());
		}
		masks_ = std::move(verified->masks);
		return std::move(verified->done);
	}

	[[nodiscard]] auto run_chunk(sched::Chunk const& chunk, Inbox<typename F::Submission>& inbox) -> core::Result<sched::ChunkDone, Failure> {
		// A chunk that reaches the end of its sequence samples the request's next token, under its grammar's mask.
		if (chunk.read == sched::Read::Last) {
			return await(cards_.prefill(Prefill{Sampling{.chunk = chunk, .mask = first_mask(live_of(chunk.request))}}), inbox);
		}
		return await(cards_.prefill(Prefill{Reading{.chunk = chunk}}), inbox);
	}

	[[nodiscard]] auto run_cycle(sched::Cycle const& cycle, Inbox<typename F::Submission>& inbox) -> core::Result<sched::Done, Failure> {
		return std::visit(Overloaded{
		                      [&](sched::Batch const& batch) {
			                      return run_batch(batch, inbox).transform([](sched::BatchDone done) {
				                      return sched::Done{std::move(done)};
			                      });
		                      },
		                      [&](sched::Chunk const& chunk) {
			                      return run_chunk(chunk, inbox).transform([](sched::ChunkDone done) {
				                      return sched::Done{done};
			                      });
		                      },
		                      [&](sched::Both const& both) -> core::Result<sched::Done, Failure> {
			                      auto batch = run_batch(both.batch, inbox);
			                      if (!batch) {
				                      return std::unexpected(batch.error());
			                      }
			                      return run_chunk(both.chunk, inbox).transform([&](sched::ChunkDone chunk) {
				                      return sched::Done{sched::BothDone{.batch = std::move(*batch), .chunk = chunk}};
			                      });
		                      },
		                  },
		                  cycle);
	}

	/** The scheduler's facts, each running request named by its tag. */
	[[nodiscard]] auto snapshot() const -> Snapshot<Tag> {
		auto const facts = scheduler_->facts();
		auto out = Snapshot<Tag>{
		    .slots = {},
		    .queued = facts.queued + core::narrow<std::uint64_t>(std::ranges::count_if(lives_, &Loop::waiting)),
		    .pool = facts.pool,
		    .counters = facts.counters,
		};
		for (auto const [index, slot] : std::views::enumerate(facts.slots)) {
			out.slots[core::narrow<std::size_t>(index)] = std::visit(
			    Overloaded{
			        [](sched::Vacant) -> SlotView<Tag> {
				        return Vacant{};
			        },
			        // A cancelled request's slot runs nothing more; the cycle boundary releases it.
			        [](sched::Finishing const&) -> SlotView<Tag> {
				        return Vacant{};
			        },
			        [this](sched::Prefilling const& prefilling) -> SlotView<Tag> {
				        return Prefilling<Tag>{
				            .tag = tag_of(prefilling.request),
				            .prompt_tokens = prefilling.prompt_tokens,
				            .cached_tokens = prefilling.cached_tokens,
				            .prefilled_tokens = prefilling.prefilled_tokens,
				        };
			        },
			        [this](sched::Decoding const& decoding) -> SlotView<Tag> {
				        return Decoding<Tag>{
				            .tag = tag_of(decoding.request),
				            .prompt_tokens = decoding.prompt_tokens,
				            .cached_tokens = decoding.cached_tokens,
				            .generated_tokens = decoding.generated_tokens,
				        };
			        },
			    },
			    slot);
		}
		return out;
	}

	std::unique_ptr<Scheduler> scheduler_;
	tokenizer::Tokenizer tokenizer_;
	C cards_;
	F front_;
	// Too large to hold per request: every matcher borrows this one.
	std::unique_ptr<grammar::Vocabulary> vocabulary_;
	Masks masks_;
	Outbox<typename F::Update> outbox_;
	/** Every request the engine holds, in arrival order: waiting ones go to the scheduler oldest first. */
	std::vector<Request> lives_;
	/** This round's outputs, posted as one batch. */
	std::vector<Addressed<Output>> round_;
	/** The snapshot posted last; absent before the first. */
	std::optional<Snapshot<Tag>> reported_;
};

}

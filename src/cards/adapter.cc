// SPDX-License-Identifier: AGPL-3.0-only
//
// The cards as the pipeline drives them: decode's resident loops and prefill's chunks over one kv pool. A batch is
// `begin` (its drafts known at once, from what the cards hold), then the grammar's masks published and the cycle's
// answer polled; a chunk yields the loops (decode applies the pool's queued commands first), runs on both cards'
// streams, relaunches the loops and, at a prompt's end, samples its first token under its mask. Every wait is a sender
// polled on the engine thread's own loop, so the thread takes the reactor's posts while the cards compute.
export module engine.cards:adapter;

import std;
import engine.card_scheduler;
import engine.core;
import engine.decode;
import engine.device;
import engine.exec;
import engine.kv;
import engine.model;
import engine.pipeline;
import engine.prefill;
import engine.sched;

namespace engine::cards {

namespace ex = engine::exec;

using Clock = std::chrono::steady_clock;

static_assert(std::same_as<decltype(pipeline::RowMask{}.card(model::Card::Zero)), decode::CardMask>, "the pipeline's row masks are what the cards read");
static_assert(pipeline::max_drafts == decode::max_drafts, "a pipeline draft row holds every draft the cards keep");

/** How long teardown waits for the loops to end: a cycle takes milliseconds, so loops silent this long are gone. */
inline constexpr auto teardown_patience = std::chrono::seconds{10};

/** How often teardown asks whether the loops have ended. */
inline constexpr auto teardown_poll = std::chrono::microseconds{50};

/**
 * The drafts a prompt's end has the mtp head propose: the most any cycle verifies. A later batch verifies the fewer of
 * these and its own depth, so drafting fewer here could only shorten a first cycle.
 */
inline constexpr model::DraftDepth prompt_end_depth = model::max_draft_depth;

[[nodiscard]] auto as_decode(device::DeviceError error) -> decode::DecodeError {
	return decode::DecodeError{.reason = std::move(error)};
}

/** Prefill's view of the memories decode owns. */
template<decode::Catalogs C>
[[nodiscard]] auto memories(decode::Engine<C> const& engine) -> prefill::Memories {
	auto const card = [&](model::Card which) {
		auto const lent = engine.memories(which);
		return prefill::CardMemories{
		    .image = lent.image,
		    .pool = lent.pool,
		    .tables = lent.tables,
		    .states = lent.states,
		    .rotations = lent.rotations,
		    .prompt = lent.prompt_row,
		};
	};
	return {.embedding = engine.memories(model::Card::Zero).embedding, .cards = {card(model::Card::Zero), card(model::Card::One)}};
}

template<decode::Catalogs C>
[[nodiscard]] auto streams(decode::Engine<C> const& engine) -> prefill::CardStreams {
	auto const zero = engine.card(model::Card::Zero);
	auto const one = engine.card(model::Card::One);
	return {prefill::CardStream{.context = zero.context, .stream = zero.stream}, prefill::CardStream{.context = one.context, .stream = one.stream}};
}

/** The pipeline's draft rows from what the cards verify per row: a fresh row verifies none. */
[[nodiscard]] auto drafts_of(decode::Pending const& pending) -> pipeline::Drafts {
	auto drafts = pipeline::Drafts{};
	drafts.rows.reserve(pending.rows.size());
	for (auto const row : std::views::iota(std::size_t{0}, pending.rows.size())) {
		auto const held = pending.drafts(row);
		auto drafted = pipeline::DraftRow{.tokens = {}, .count = held.size()};
		std::ranges::copy(held, drafted.tokens.begin());
		drafts.rows.push_back(drafted);
	}
	return drafts;
}

/** The prompt's last position as a decoding row: its sampled token is the one the row commits. */
[[nodiscard]] auto prompt_row(sched::Chunk const& chunk) -> sched::Row {
	core::invariant(!chunk.tokens.empty());
	return {
	    .slot = chunk.slot,
	    .request = chunk.request,
	    .position = chunk.from + chunk.tokens.size() - 1,
	    .token = chunk.tokens.back(),
	    .rule = chunk.rule,
	    .most = 1,
	};
}

/** An outcome known when the sender is made, delivered by its first poll. */
template<class T>
struct Ready {
	std::optional<core::Result<T, decode::DecodeError>> outcome;

	[[nodiscard]] auto operator()() -> std::optional<core::Result<T, decode::DecodeError>> {
		return std::exchange(outcome, std::nullopt);
	}
};

/**
 * The loops yield the cards to a chunk, polled: decode applies the pool's queued commands first, since the chunk reads
 * the tables and states they set, then the loops exit and their kernels finish.
 */
template<decode::Catalogs C>
struct Yield {
	std::reference_wrapper<decode::Engine<C>> engine;

	[[nodiscard]] auto operator()() -> std::optional<core::Result<void, decode::DecodeError>> {
		return engine.get().try_stop();
	}
};

/** A batch's answer, polled: its masks were published (or refused, which fails it at once). */
template<decode::Catalogs C>
struct Answer {
	std::reference_wrapper<decode::Engine<C>> engine;
	decode::Pending pending;
	pipeline::Masks masks;
	std::optional<decode::DecodeError> refused;
	Clock::time_point started;

	[[nodiscard]] auto operator()() -> std::optional<core::Result<pipeline::Verified, decode::DecodeError>> {
		if (refused) {
			return std::unexpected(*refused);
		}
		auto answer = engine.get().try_complete(pending);
		if (!answer || !*answer) {
			return answer.transform([](auto const& failed) -> core::Result<pipeline::Verified, decode::DecodeError> {
				return std::unexpected(failed.error());
			});
		}
		auto done = std::move(**answer);
		done.took = Clock::now() - started;
		return pipeline::Verified{.done = std::move(done), .masks = std::move(masks)};
	}
};

/**
 * A chunk's last step, polled: the loops relaunched and, for a chunk that ends its prompt, the prompt's end begun
 * under its mask and its sampled token collected.
 */
template<decode::Catalogs C>
struct Tail {
	std::reference_wrapper<decode::Engine<C>> engine;
	pipeline::Prefill work;
	std::optional<decode::Pending> pending;
	Clock::time_point started;

	[[nodiscard]] auto operator()() -> std::optional<core::Result<sched::ChunkDone, decode::DecodeError>> {
		auto& cards = engine.get();
		if (pending) {
			auto answer = cards.try_complete_prompt_end(*pending);
			if (answer && *answer) {
				(*answer)->took = Clock::now() - started;
			}
			return answer;
		}
		if (auto relaunched = cards.start(); !relaunched) {
			return std::unexpected(relaunched.error());
		}
		auto const sampling = core::alternative<pipeline::Sampling>(work);
		if (!sampling) {
			return sched::ChunkDone{.sampled = std::nullopt, .took = Clock::now() - started};
		}
		auto begun = cards.begin_prompt_end(prompt_row(sampling->chunk), prompt_end_depth);
		if (!begun) {
			return std::unexpected(begun.error());
		}
		auto const& mask = sampling->mask;
		auto published = cards.publish(
		    *begun,
		    [&mask](std::size_t /*index*/, model::Card card) -> decode::CardMask {
			    return mask.card(card);

		    },
		    begun->mask_count());
		if (!published) {
			return std::unexpected(published.error());
		}
		pending = std::move(*begun);
		return std::nullopt;
	}
};

/** Each poll the cards hand out is a poll the card scheduler watches. */
static_assert(card_scheduler::Poll<Ready<pipeline::Drafts>> && card_scheduler::Poll<Yield<decode::ModelCatalogs>> &&
              card_scheduler::Poll<Answer<decode::ModelCatalogs>> && card_scheduler::Poll<Tail<decode::ModelCatalogs>>);

/**
 * Both cards, owned: the decode engine (behind a pointer, so the senders' borrows of it survive a move of the cards)
 * and prefill's chunks. `Error` is every way the cards fail, as the pipeline reports it.
 *
 * Every step is a sender polled on the waiting thread's loop; submitting a ticket and launching the loops return at
 * once.
 */
export template<decode::Catalogs C>
class BasicCards {
public:
	using Error = decode::DecodeError;

	/**
	 * Takes an opened decode engine whose loops have not started: loads prefill's programs and zeroes its workspaces on
	 * the engine's streams, then starts the loops behind them.
	 */
	[[nodiscard]] static auto assemble(decode::Engine<C> opened) -> core::Result<BasicCards, Error> {
		auto engine = std::make_unique<decode::Engine<C>>(std::move(opened));
		auto chunks = prefill::Chunks::create(streams(*engine));
		if (!chunks) {
			return std::unexpected(as_decode(chunks.error()));
		}
		if (auto started = engine->start(); !started) {
			return std::unexpected(started.error());
		}
		return BasicCards{std::move(engine), std::make_unique<prefill::Chunks>(std::move(*chunks))};
	}

	BasicCards(BasicCards&&) noexcept = default;
	auto operator=(BasicCards&&) noexcept -> BasicCards& = default;
	BasicCards(BasicCards const&) = delete;
	auto operator=(BasicCards const&) -> BasicCards& = delete;

	~BasicCards() {
		if (!engine_) {
			return;
		}
		// The loops read every allocation the engine holds, so they end before it does, from wherever serving left
		// them. Loops that never answer are abandoned at `give_up`, which decode reports as an error; nothing is left to
		// tell of it here, since the failure that ended serving has been reported already.
		auto const give_up = Clock::now() + teardown_patience;
		auto ended = engine_->try_shutdown(give_up);
		while (!ended) {
			std::this_thread::sleep_until(Clock::now() + teardown_poll);
			ended = engine_->try_shutdown(give_up);
		}
	}

	auto apply(decode::PoolCommand const& command) -> void {
		engine_->apply(command);
	}

	/** Starts the batch's cycle; completes at once with the drafts each row verifies, which `begin` knows. */
	[[nodiscard]] auto draft(sched::Batch batch) {
		started_ = Clock::now();
		auto begun = engine_->begin(std::move(batch));
		auto drafted = begun.transform([](decode::Pending const& pending) {
			return drafts_of(pending);
		});
		if (begun) {
			pending_ = std::move(*begun);
		}
		return card_scheduler::polled(Ready<pipeline::Drafts>{.outcome = std::move(drafted)});
	}

	/** Publishes the batch's masks, then completes with its outcome and the masks' buffer once both cards answer. */
	[[nodiscard]] auto verify(pipeline::Masks masks) {
		core::invariant(pending_.has_value());
		auto answer =
		    Answer<C>{.engine = *engine_, .pending = *std::exchange(pending_, std::nullopt), .masks = {}, .refused = std::nullopt, .started = started_};
		answer.masks = std::move(masks);
		auto const& rows = answer.masks.rows;
		auto published = engine_->publish(
		    answer.pending,
		    [&rows](std::size_t index, model::Card card) -> decode::CardMask {
			    return rows.at(index).card(card);
		    },
		    rows.size());
		if (!published) {
			answer.refused = std::move(published).error();
		}
		return card_scheduler::polled(std::move(answer));
	}

	/**
	 * Runs one prefill chunk: the pool's queued commands reach the cards, the loops yield, the chunk runs on both cards,
	 * the loops come back, and a chunk that ends its prompt samples the first token under its mask.
	 */
	[[nodiscard]] auto prefill(pipeline::Prefill work) {
		auto const chunk = std::visit(
		    [](auto const& chosen) {
			    return chosen.chunk;
		    },
		    work);
		auto tail = Tail<C>{.engine = *engine_, .work = std::move(work), .pending = std::nullopt, .started = Clock::now()};
		return card_scheduler::polled(Yield<C>{.engine = *engine_}) |
		       ex::let_value([&engine = *engine_, &chunks = *chunks_, chunk] {
			       auto ran = chunks.run(memories(engine), chunk, streams(engine)).transform_error([](device::DeviceError failed) {
				       return as_decode(std::move(failed));
			       });
			       return card_scheduler::polled(Ready<prefill::Finished>{.outcome = std::move(ran)});
		       }) |
		       ex::let_value([](prefill::Finished& finished) {
			       return std::move(finished) | ex::let_error([](device::DeviceError failed) {
				              return ex::just_error(as_decode(std::move(failed)));
			              });
		       }) |
		       ex::let_value([last = std::move(tail)] mutable {
			       return card_scheduler::polled(std::move(last));
		       });
	}

private:
	BasicCards(std::unique_ptr<decode::Engine<C>> engine, std::unique_ptr<prefill::Chunks> chunks)
	    : engine_{std::move(engine)}, chunks_{std::move(chunks)} {}

	std::unique_ptr<decode::Engine<C>> engine_;
	std::unique_ptr<prefill::Chunks> chunks_;
	/** The batch begun by `draft` and not yet verified. */
	std::optional<decode::Pending> pending_;
	Clock::time_point started_;
};

/** The cards over the model's images. */
export using Cards = BasicCards<decode::ModelCatalogs>;

}

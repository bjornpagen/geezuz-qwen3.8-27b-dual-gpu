// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline:seams;

import std;
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

/**
 * What the pipeline asks of the parts it composes but does not hold: the cards (decode and prefill over one kv pool),
 * each request's grammar state, and the api in front of it. Each is a concept, so the product's parts and the tests'
 * doubles plug in alike.
 */
namespace engine::pipeline {

namespace ex = engine::exec;

/** The most drafts a row carries into a cycle. */
export inline constexpr std::size_t max_drafts = model::drafts(model::max_draft_depth);

/**
 * One batch row's drafts, as the cards hold them: `count` tokens, at most the batch's depth, fewer when the mtp head
 * has drafted less for the row (none for a row fresh from its prompt).
 */
export struct DraftRow {
	std::array<Token, max_drafts> tokens;
	std::size_t count;

	[[nodiscard]] auto view() const -> std::span<Token const> {
		core::invariant(count <= max_drafts);
		return std::span{tokens}.first(count);
	}

	[[nodiscard]] auto operator==(DraftRow const&) const -> bool = default;
};

/** What the mtp head proposed for a batch: one row per batch row, in the batch's order. */
export struct Drafts {
	std::vector<DraftRow> rows;

	[[nodiscard]] auto operator==(Drafts const&) const -> bool = default;
};

/** A verified batch: its outcome, and the masks buffer handed back for the next cycle. */
export struct Verified {
	sched::BatchDone done;
	Masks masks;
};

/** A prefill chunk that reads on: nothing is sampled. */
export struct Reading {
	sched::Chunk chunk;
};

/** A prefill chunk that reaches the end of its sequence: its last row is sampled under `mask`. */
export struct Sampling {
	sched::Chunk chunk;
	RowMask mask;
};

export using Prefill = std::variant<Reading, Sampling>;

/**
 * The cards, as the pipeline drives them: one owner of both cards' decode and prefill, applying the pool's commands
 * in order before any cycle that reads them.
 *
 * - `draft(batch)`: starts the batch's cycle; completes with the drafts its rows carry, once the host can know them.
 * - `verify(masks)`: hands the cycle each verify row's mask; completes with the batch's outcome and the buffer.
 * - `prefill(chunk)`: runs one prefill chunk on both cards; completes with its outcome.
 *
 * The masks are computed between `draft` and `verify`, while the cards stream the cycle's weights; a row's verify rows
 * past its drafts repeat its last mask. Every sender may fail with the cards' own `Error`, which reaches the engine's
 * owner whole; none is stopped. A sender borrows the cards (as a scheduler borrows its context): the pipeline waits
 * for each before it calls the cards again.
 */
export template<class C, kv::Geometry G>
concept Cards = std::movable<C> && kv::Device<C, G> && core::Classified<typename C::Error> && requires(C& cards, sched::Batch batch, Masks masks, Prefill prefill) {
	{ cards.draft(std::move(batch)) } -> ex::sender;
	{ cards.verify(std::move(masks)) } -> ex::sender;
	{ cards.prefill(std::move(prefill)) } -> ex::sender;
};

/**
 * One request's parser state over its compiled grammar, as `grammar::Matcher` is: advancing by a token, taking back
 * the last tokens taken (a draft's prefix, tried for its mask), and the mask at the committed position, a borrow of
 * the state valid until its next non-const call. Each reads the engine's one vocabulary, borrowed for the call. On a
 * failed advance nothing changes: `Refused`, `Ended` and `UnknownToken` say the token cannot come there,
 * `LexerTooLarge` and `RowTooLarge` that the grammar outgrew its bounds.
 */
export template<class S>
concept GrammarState = std::move_constructible<S> &&
                       requires(S& state, grammar::Vocabulary const& vocabulary, Token token, std::size_t count) {
	                       { state.advance(vocabulary, token) } -> std::same_as<core::Result<void, grammar::MatchError>>;
	                       { state.rollback(count) } -> std::same_as<void>;
	                       { state.mask(vocabulary) } -> std::same_as<core::Result<grammar::MaskView, grammar::MatchError>>;
                       };

static_assert(GrammarState<grammar::Matcher>);

/** Whether a matcher's fault is its grammar outgrowing its bounds, rather than a token that cannot come there. */
export [[nodiscard]] constexpr auto outgrown(grammar::MatchError error) -> bool {
	switch (error.fault) {
	case grammar::MatchFault::LexerTooLarge:
	case grammar::MatchFault::RowTooLarge:
		return true;
	case grammar::MatchFault::Refused:
	case grammar::MatchFault::Ended:
	case grammar::MatchFault::UnknownToken:
		return false;
	}
	std::unreachable();
}

/**
 * Why generation ended, with the exact sequence the request leaves (the prompt as it ran, then every generated token),
 * which a stored response keeps: `Stopped` when the model ended its turn (the end-of-turn token last), `Truncated`
 * when the budget, the context or the pool ran out. A request whose client left has none: it is answered `cancelled`.
 */
export using Ending = std::variant<sched::Stopped, sched::Truncated>;

/** A request's end: why, and its counts (generated tokens include the one that ended the turn). */
export struct Finish {
	Ending ending;
	sched::Usage usage;

	[[nodiscard]] auto operator==(Finish const&) const -> bool = default;
};

/**
 * One request as the api's adapter prepared it on the engine's thread: `tag` names it in status snapshots, `request`
 * is its rendered prompt with sampling and budget, `grammar` its parser state over the grammar the boundary compiled,
 * and `reply` what turns its tokens into the api's outputs.
 */
export template<class Tag, class Grammar, class Reply>
struct Prepared {
	Tag tag;
	sched::Submission request;
	Grammar grammar;
	Reply reply;
};

/** One output addressed to the exchange that asked for it. */
export template<class Output>
struct Addressed {
	http::Exchange exchange;
	Output output;

	[[nodiscard]] auto operator==(Addressed const&) const -> bool = default;
};

export struct Vacant {
	[[nodiscard]] constexpr auto operator==(Vacant const&) const -> bool = default;
};

/** A slot reading a prompt: `prefilled` of `prompt_tokens` positions done, the first `cached_tokens` from cache. */
export template<class Tag>
struct Prefilling {
	Tag tag;
	std::uint64_t prompt_tokens;
	std::uint64_t cached_tokens;
	std::uint64_t prefilled_tokens;

	[[nodiscard]] auto operator==(Prefilling const&) const -> bool = default;
};

export template<class Tag>
struct Decoding {
	Tag tag;
	std::uint64_t prompt_tokens;
	std::uint64_t cached_tokens;
	std::uint64_t generated_tokens;

	[[nodiscard]] auto operator==(Decoding const&) const -> bool = default;
};

export template<class Tag>
using SlotView = std::variant<Vacant, Prefilling<Tag>, Decoding<Tag>>;

/** What the engine is doing: the scheduler's facts, with each running request named by its tag. */
export template<class Tag>
struct Snapshot {
	std::array<SlotView<Tag>, model::slots> slots;
	/** Requests accepted and not yet in a slot. */
	std::uint64_t queued;
	sched::PoolFacts pool;
	sched::Counters counters;

	[[nodiscard]] auto operator==(Snapshot const&) const -> bool = default;
};

/**
 * Turns a request's committed tokens into the api's outputs, in order: `feed` takes each token the request commits
 * (the token that ends its turn is never fed), `end` its finish, after which nothing more is fed.
 */
export template<class R, class Output>
concept Reply = std::movable<R> && requires(R& reply, tokenizer::Tokenizer const& tokenizer, Token token, Finish finish) {
	{ reply.feed(tokenizer, token) } -> std::same_as<std::vector<Output>>;
	{ reply.end(tokenizer, finish) } -> std::same_as<std::vector<Output>>;
};

/** Why the engine stopped serving: the cards failed with their own error, or its thread could not wait for work. */
export template<core::Classified CardError>
struct [[nodiscard]] EngineError {
	std::variant<CardError, net::NetError> cause;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& failure) {
			    return failure.retry();
		    },
		    cause);
	}

	[[nodiscard]] auto operator==(EngineError const&) const -> bool = default;
};

/**
 * The api in front of the engine: what a submission becomes on the engine's thread (rendered, tokenized, its grammar's
 * parser started), the outputs that answer it, and the updates the engine posts to the api's service.
 *
 * Every submission is answered by exactly one terminal output, the last one addressed to its exchange: `refused`
 * alone; or `accepted`, what its reply makes of its tokens, then its reply's `end`; or `cancelled` once its client
 * left and the engine let go of it (at once while it waited for the scheduler, else at the next cycle boundary), after
 * whatever was said before the cancel arrived (nothing, or `accepted` and some of its reply's outputs). So the
 * requests without a terminal output are exactly those the engine holds. A cancel that arrives after the request's
 * terminal output is answered by nothing. A request whose grammar outgrew its bounds is sampled under an end-of-turn
 * mask from then on and answered `overran` with the matcher's fault, after what its reply said before, once the
 * scheduler lets it go.
 *
 * An engine that fails posts what it said before the failure, then `failed` as its last update, retried until the
 * server's mailbox takes it or the engine is stopped: no request it has not answered will be. An engine that is
 * stopped posts nothing more.
 */
export template<class F, class CardError>
concept Front = std::movable<F> && std::movable<typename F::Submission> && std::movable<typename F::Output> &&
                std::movable<typename F::Update> && std::regular<typename F::Tag> && GrammarState<typename F::Grammar> &&
                Reply<typename F::Reply, typename F::Output> &&
                requires(F& front, tokenizer::Tokenizer const& tokenizer, F::Submission submission,
                         sched::PromptTooLong refused, std::vector<Addressed<typename F::Output>> outputs,
                         Snapshot<typename F::Tag> snapshot, EngineError<CardError> const& error, grammar::MatchError fault) {
	                { front.prepare(tokenizer, std::move(submission)) }
	                    -> std::same_as<Prepared<typename F::Tag, typename F::Grammar, typename F::Reply>>;
	                { front.accepted() } -> std::same_as<typename F::Output>;
	                { front.refused(refused) } -> std::same_as<typename F::Output>;
	                { front.cancelled() } -> std::same_as<typename F::Output>;
	                { front.overran(fault) } -> std::same_as<typename F::Output>;
	                { front.batch(std::move(outputs)) } -> std::same_as<typename F::Update>;
	                { front.status(std::move(snapshot)) } -> std::same_as<typename F::Update>;
	                { front.failed(error) } -> std::same_as<typename F::Update>;
                };

}

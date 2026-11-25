// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sched:cycle;

import std;
import engine.core;
import engine.kv;
import engine.model;
import engine.sampling;
import :request;

namespace engine::sched {

/** Tokens one slot's row can commit in a cycle: its committed token's successor and one per accepted draft. */
export inline constexpr Count max_row_tokens = 1 + model::drafts(model::max_draft_depth);

/**
 * Positions from a decoding slot's committed row that one cycle may write kv into: its verify rows reach
 * `max_row_tokens - 1` past it, and the mtp head, drafting the next cycle from the last accepted token (up to
 * `max_row_tokens` past), writes its own kv up to `max_row_tokens - 2` further. The next cycle's depth is not known
 * when it drafts, so the deepest shape bounds both.
 */
export inline constexpr Count lookahead = 2 * max_row_tokens - 1;

/** How prompt prefill shares the cards with decoding. */
export struct Tuning {
	/** Prompt tokens one prefill chunk reads at most. */
	Count chunk_tokens;
	/** The share of the cards' time prefill takes while slots decode. */
	model::Share share;
};

export [[nodiscard]] constexpr auto sound(Tuning tuning) -> bool {
	return tuning.chunk_tokens > 0 && tuning.share.numerator > 0 && tuning.share.numerator <= tuning.share.denominator;
}

/**
 * 2,048 tokens is about a second of prefill at the planned ~1,600 tok/s: a decoding slot stalls one chunk at most,
 * the cycle loop yields once (~10 µs) per chunk, and the scheduler regains every second for admissions and
 * cancellations. Smaller chunks re-read the cached context more often, which slows deep prompts. A quarter of the time
 * keeps decoding streams at three quarters of their speed while a prompt takes four times as long as it would alone.
 */
export inline constexpr Tuning tuning = {.chunk_tokens = 2048, .share = {.numerator = 1, .denominator = 4}};

static_assert(sound(tuning));

/** One decoding slot's work in a cycle: its committed row, the token at `position`, and its drafts. */
export struct Row {
	kv::Slot slot;
	RequestHandle request;
	/** The slot's kv and deltanet state hold positions [0, position); the cycle reads this row first. */
	Count position;
	Token token;
	sampling::Rule rule;
	/**
	 * Tokens the row may commit, 1 to `1 + drafts(depth)`: what is left of the request's limit. Acceptance also stops
	 * at an end-of-turn token, which is then the row's last. So a row never commits past its request's end, and the
	 * state of a finished request has read exactly its sequence but the last token.
	 */
	Count most;
};

/**
 * The decoding slots of a cycle, in slot order, all at one draft depth: the fixed allocation of the row budget for
 * this many live slots, so `rows.size() * (1 + drafts(depth))` never exceeds the budget.
 */
export struct Batch {
	model::DraftDepth depth;
	std::vector<Row> rows;
};

/** Whether a chunk reads the last token of the slot's sequence. */
export enum class Read : std::uint8_t {
	/** More of the sequence follows; nothing is sampled. */
	Part,
	/** The last row's logits are sampled with `rule` (the slot's first token) and the mtp head drafts from it. */
	Last,
};

/**
 * One slot reads `tokens` at positions [from, from + size). At `from == 0` its deltanet state starts empty; otherwise
 * it is the state after [0, from), resumed from the cache or left by the slot's previous chunk.
 */
export struct Chunk {
	kv::Slot slot;
	RequestHandle request;
	Count from;
	std::vector<Token> tokens;
	Read read;
	sampling::Rule rule;
};

/** A decode cycle, then a prefill chunk of a slot outside it. */
export struct Both {
	Batch batch;
	Chunk chunk;
};

/**
 * What the cards run next. The kv commands the scheduler issued while planning it are applied before it; those it
 * issues while settling it, after it.
 */
export using Cycle = std::variant<Batch, Chunk, Both>;

/**
 * What one row committed: the token after its committed one, then the token after each accepted draft, in order;
 * `count` is the accepted drafts plus one, at most the row's `most`, and only the last may end the turn. The slot's
 * state then holds the row and its accepted drafts: every committed token but the last.
 */
export struct Accepted {
	std::array<Token, max_row_tokens> tokens;
	Count count;

	[[nodiscard]] auto view() const -> std::span<Token const> {
		core::invariant(count <= max_row_tokens);
		return std::span{tokens}.first(count);
	}

	[[nodiscard]] constexpr auto operator==(Accepted const&) const -> bool = default;
};

/** A batch's outcome: one `Accepted` per row, in the batch's order, and the time the cards spent on it. */
export struct BatchDone {
	std::vector<Accepted> rows;
	std::chrono::nanoseconds took;

	[[nodiscard]] auto operator==(BatchDone const&) const -> bool = default;
};

/** A chunk's outcome: the token its last row sampled when it was `Read::Last`, and the time it took. */
export struct ChunkDone {
	std::optional<Token> sampled;
	std::chrono::nanoseconds took;

	[[nodiscard]] constexpr auto operator==(ChunkDone const&) const -> bool = default;
};

export struct BothDone {
	BatchDone batch;
	ChunkDone chunk;

	[[nodiscard]] auto operator==(BothDone const&) const -> bool = default;
};

/** What the cards read back for a cycle: the alternative matching the cycle's. */
export using Done = std::variant<BatchDone, ChunkDone, BothDone>;

}

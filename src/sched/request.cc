// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sched:request;

import std;
import engine.core;
import engine.kv;
import engine.sampling;
import engine.tokenizer;

namespace engine::sched {

export using kv::Count;
export using Token = tokenizer::Id;

/** A generational reference to one request the scheduler holds; stale from the moment it reports `Finished`. */
export struct RequestHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(RequestHandle const&) const -> bool = default;
};

/**
 * Requests queued and running at once. The server holds at most one request per connection, so its connection bound
 * is what fills this; the pipeline checks the two against each other at compile time.
 */
export inline constexpr Count max_requests = 256;

/** The completion runs until the turn ends or the prompt and completion fill the context. */
export struct UntilContextFull {
	[[nodiscard]] constexpr auto operator==(UntilContextFull const&) const -> bool = default;
};

/** At most `tokens` completion tokens, one or more. */
export struct AtMost {
	Count tokens;

	[[nodiscard]] constexpr auto operator==(AtMost const&) const -> bool = default;
};

export using Limit = std::variant<UntilContextFull, AtMost>;

/**
 * The prompt offsets the chat format marks for deltanet snapshots, each the offset of an `<|im_start|>`: the first
 * block after the system block, the last user query, the generation prompt. A conversation without one of them has
 * no such point. The point before the prompt's last token is a fact of the token list, and the scheduler adds it.
 */
export struct Marks {
	std::optional<Count> system_end;
	std::optional<Count> last_user_start;
	std::optional<Count> generation_start;

	[[nodiscard]] constexpr auto operator==(Marks const&) const -> bool = default;
};

/** One chat completion as rendered: the prompt ends with the generation prompt. */
export struct Submission {
	std::vector<Token> prompt;
	Marks marks;
	sampling::Params sampling;
	Limit limit;
};

/** A queued request, and its prompt's length after turn replay: what usage reports as prompt tokens. */
export struct Submitted {
	RequestHandle request;
	Count prompt_tokens;

	[[nodiscard]] constexpr auto operator==(Submitted const&) const -> bool = default;
};

/** The prompt leaves fewer than the completion's minimum, or `AtMost` asks for more than the context has room for. */
export struct PromptTooLong {
	Count prompt_tokens;

	[[nodiscard]] constexpr auto operator==(PromptTooLong const&) const -> bool = default;
};

/** Every request record is taken. */
export struct Full {
	[[nodiscard]] constexpr auto operator==(Full const&) const -> bool = default;
};

export struct [[nodiscard]] SubmitError {
	std::variant<PromptTooLong, Full> cause;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return std::holds_alternative<Full>(cause) ? core::Retry::Transient : core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(SubmitError const&) const -> bool = default;
};

/**
 * The model ended its turn. `sequence` is the request's whole sequence: the prompt as it ran (after turn replay),
 * then every generated token, the end-of-turn token last. The cache closed it without that last token: the state after
 * every token but the last is cached at a sequence-end point (unless the pool refused the snapshot or later evicts it),
 * so a request whose prompt continues this sequence resumes there and reads the last token first, as for a prompt's
 * end. A stored response keeps exactly these tokens.
 */
export struct Stopped {
	std::vector<Token> sequence;

	[[nodiscard]] auto operator==(Stopped const&) const -> bool = default;
};

/**
 * The limit, the context or the pool ran out. `sequence` and what the cache holds are as for `Stopped`, without an
 * end-of-turn token; a request the pool ended while still reading its prompt has only what it read cached.
 */
export struct Truncated {
	std::vector<Token> sequence;

	[[nodiscard]] auto operator==(Truncated const&) const -> bool = default;
};

/** The cache holds the state after every token of `sequence`, which a request continuing it resumes from. */
export struct Kept {
	std::vector<Token> sequence;

	[[nodiscard]] auto operator==(Kept const&) const -> bool = default;
};

/** Nothing of the request was cached: it had read nothing, or the pool refused the snapshot. */
export struct Dropped {
	[[nodiscard]] constexpr auto operator==(Dropped const&) const -> bool = default;
};

/** Its caller ended it (the client left); its state was kept if it had any and the pool took it. */
export struct Cancelled {
	std::variant<Kept, Dropped> state;

	[[nodiscard]] auto operator==(Cancelled const&) const -> bool = default;
};

/** Why a request ended, with the sequence it leaves. */
export using Ending = std::variant<Stopped, Truncated, Cancelled>;

/** Counts for a finished request: the generated tokens include the token that ended the turn. */
export struct Usage {
	Count prompt_tokens;
	Count cached_tokens;
	Count generated_tokens;

	[[nodiscard]] constexpr auto operator==(Usage const&) const -> bool = default;
};

/** The request took a slot, resuming from `cached` positions of the prefix cache. */
export struct Admitted {
	RequestHandle request;
	kv::Slot slot;
	Count cached;

	[[nodiscard]] constexpr auto operator==(Admitted const&) const -> bool = default;
};

/** The request's next committed tokens, in order. The token that ends a turn is never among them. */
export struct Generated {
	RequestHandle request;
	std::vector<Token> tokens;

	[[nodiscard]] auto operator==(Generated const&) const -> bool = default;
};

/** The pool ran out: the request left its slot with its state cached and waits at the head of the queue. */
export struct Paused {
	RequestHandle request;

	[[nodiscard]] constexpr auto operator==(Paused const&) const -> bool = default;
};

/** The request's last event; its handle is stale from here on. */
export struct Finished {
	RequestHandle request;
	Ending ending;
	Usage usage;

	[[nodiscard]] auto operator==(Finished const&) const -> bool = default;
};

export using Event = std::variant<Admitted, Generated, Paused, Finished>;

/** The model's end-of-generation tokens, as its generation config lists them. */
export [[nodiscard]] constexpr auto ends_turn(Token token) -> bool {
	return token == tokenizer::id(tokenizer::Added::im_end) || token == tokenizer::id(tokenizer::Added::endoftext);
}

}

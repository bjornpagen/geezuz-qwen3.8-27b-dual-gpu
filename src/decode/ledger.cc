// SPDX-License-Identifier: AGPL-3.0-only
//
// What the cards hold for each slot between cycles, as the host keeps track of it. The kernel keeps a slot's drafts
// (and the distributions they were drawn from) in its book and the draft rows' deltanet inputs in its replay ring, so
// the host sends only the committed token. A batch row continues what the cards hold exactly when it is the same
// request at the position and token the last cycle left; any other row (fresh from prefill, resumed from a snapshot,
// or rewound) starts with no drafts and nothing to replay, its stored state covering exactly its positions.
export module engine.decode:ledger;

import std;
import engine.core;
import engine.decode_device;
import engine.deltanet;
import engine.kv;
import engine.model;
import engine.sched;

namespace engine::decode {

export using sched::Token;
export inline constexpr std::size_t max_drafts = decode_device::max_drafts;

/** A slot's drafts, the first `drafted` meaningful. */
export using DraftRow = std::array<Token, max_drafts>;

/** What the cards hold for a slot after a cycle. */
export struct Continuation {
	sched::RequestHandle request;
	/** Where the next committed token sits, and the token: what the slot's next row must bring to continue. */
	std::uint64_t position;
	Token token;
	DraftRow drafts;
	std::uint32_t drafted;
	/** The deltanet's stored state and the accepted drafts waiting in its ring. */
	deltanet::Settled state;

	[[nodiscard]] auto operator==(Continuation const&) const -> bool = default;
};

/** One batch row's share of a cycle: what the descriptor carries, and what the ledger settles it with. */
export struct Planned {
	decode_device::SlotRun run;
	deltanet::Running state;
	DraftRow drafts;
};

[[nodiscard]] constexpr auto continues(std::optional<Continuation> const& held, sched::Row const& row) -> bool {
	return held && held->request == row.request && held->position == row.position && held->token == row.token;
}

export class Ledger {
public:
	/** The row's plan at `depth`: its book's drafts and its pending replays if it continues the slot, none otherwise. */
	[[nodiscard]] auto plan(sched::Row const& row, model::DraftDepth depth) const -> Planned {
		return plan_with(row, core::narrow<std::uint32_t>(model::drafts(depth)));
	}

	/** The row's plan with no drafts, its pending replays still taken: a logits ticket runs each committed token alone. */
	[[nodiscard]] auto plan_alone(sched::Row const& row) const -> Planned {
		return plan_with(row, 0);
	}

	/**
	 * The row's tokens from the cards' outcome, at most the row's `most` and through its first end-of-turn token, and what
	 * the slot holds after it. A row cut short commits fewer tokens than the cards accepted: the drafts they proposed
	 * continue the longer sequence, so the slot keeps none, and its replay takes only the drafts it committed.
	 */
	[[nodiscard]] auto settle(sched::Row const& row, Planned const& planned, decode_device::SlotOutcome const& outcome, model::DraftDepth depth)
	    -> sched::Accepted {
		// The kernel accepts at most the drafts it was given: a larger count is a broken card, not an input.
		core::invariant(outcome.accepted <= planned.run.drafts);
		core::invariant(row.most >= 1);
		auto accepted = sched::Accepted{.tokens = {}, .count = std::min<std::size_t>(outcome.accepted + 1, row.most)};
		// An end-of-turn token is the row's last: cutting the verdict there keeps the distribution of every token before it.
		auto const ending = std::ranges::find_if(std::span{outcome.committed}.first(accepted.count), sched::ends_turn);
		accepted.count = std::min<std::size_t>(accepted.count, static_cast<std::size_t>(ending - std::span{outcome.committed}.begin()) + 1);
		auto const cut = accepted.count < outcome.accepted + 1;
		std::ranges::copy(std::span{outcome.committed}.first(accepted.count), accepted.tokens.begin());
		auto next = Continuation{
		    .request = row.request,
		    .position = row.position + accepted.count,
		    .token = outcome.committed[accepted.count - 1],
		    .drafts = {},
		    .drafted = cut ? 0U : core::narrow<std::uint32_t>(model::drafts(depth)),
		    .state = deltanet::settle(planned.state, core::narrow<std::uint32_t>(accepted.count - 1)),
		};
		std::ranges::copy(std::span{outcome.drafts}.first(next.drafted), next.drafts.begin());
		slots_.at(row.slot.index()) = next;
		return accepted;
	}

	/**
	 * The slot after a prompt's end at `depth`: prefill left its state covering the prompt through `position`, and the
	 * cards sampled the token after it and drafted from it. Returns the sampled token.
	 */
	auto start(kv::Slot slot, sched::RequestHandle request, std::uint64_t position, decode_device::SlotOutcome const& outcome,
	           model::DraftDepth depth) -> Token {
		core::invariant(outcome.accepted == 0);
		auto next = Continuation{
		    .request = request,
		    .position = position + 1,
		    .token = outcome.committed[0],
		    .drafts = {},
		    .drafted = core::narrow<std::uint32_t>(model::drafts(depth)),
		    .state = deltanet::settled(deltanet::Committed{.tokens = position + 1}),
		};
		std::ranges::copy(std::span{outcome.drafts}.first(next.drafted), next.drafts.begin());
		slots_.at(slot.index()) = next;
		return next.token;
	}

	/**
	 * The accepted drafts the slot's stored state has yet to take, if any, now counted as taken: the cards replay them
	 * before a snapshot, which must hold the state after every token but the slot's last.
	 */
	[[nodiscard]] auto materialize(kv::Slot slot) -> std::optional<std::uint32_t> {
		auto& held = slots_.at(slot.index());
		if (!held || held->state.pending == 0) {
			return std::nullopt;
		}
		auto const pending = held->state.pending;
		held->state = {.committed = held->state.committed + pending, .pending = 0};
		return pending;
	}

	/** The slot's state was rewritten on the cards (a snapshot loaded into it): nothing of the last cycle carries on. */
	auto forget(kv::Slot slot) -> void {
		slots_.at(slot.index()) = std::nullopt;
	}

	[[nodiscard]] auto held(kv::Slot slot) const -> std::optional<Continuation> const& {
		return slots_.at(slot.index());
	}

private:
	/** The row's plan verifying at most `most` of its book's drafts. */
	[[nodiscard]] auto plan_with(sched::Row const& row, std::uint32_t most) const -> Planned {
		auto const& held = slots_.at(row.slot.index());
		auto const continued = continues(held, row) ? held : std::nullopt;
		auto const drafts = continued.has_value() ? std::min<std::uint32_t>(continued->drafted, most) : 0U;
		auto const stored = continued.has_value() ? continued->state : deltanet::settled(deltanet::Committed{.tokens = row.position});
		auto const state = deltanet::run(stored, drafts);
		core::invariant(deltanet::first_position(state) == row.position);
		auto planned = Planned{
		    .run =
		        {
		            .slot = core::narrow<unsigned>(row.slot.index()),
		            .position = core::narrow<unsigned>(row.position),
		            .token = row.token,
		            .drafts = core::narrow<unsigned char>(drafts),
		            .replays = core::narrow<unsigned char>(state.replays),
		            .unused = {},
		            .rule = row.rule,
		        },
		    .state = state,
		    .drafts = {},
		};
		if (continued.has_value()) {
			std::ranges::copy(std::span{continued->drafts}.first(drafts), planned.drafts.begin());
		}
		return planned;
	}

	std::array<std::optional<Continuation>, model::slots> slots_{};
};

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's side of the resident decode loops on both cards: each card's ring and mask buffer, the ledger of what the
// cards hold per slot, and the pool's commands waiting for the next ticket. A cycle is three steps. `begin` posts the
// cycle descriptor to both cards, whose kernels start streaming the body at once; it hands back the drafts each row
// verifies, which the host needs for the grammar's masks. `publish` writes those masks into the cards' mask buffers
// (each card's half in its own row order) and releases them; the kernels acquire them only right before they rank the
// lm_head's rows, so the host has the whole body (~15 ms) to compute them. `complete` collects both cards' answers,
// which must agree, and settles the ledger.
export module engine.decode:session;

import std;
import engine.core;
import engine.decode_device;
import engine.deltanet;
import engine.device;
import engine.kv;
import engine.model;
import engine.sched;
import :commands;
import :error;
import :ledger;

namespace engine::decode {

using decode_device::Descriptor;
using decode_device::Result;

/** Each card's ring of tickets. */
export using Ring = device::Ring<Descriptor, Result, decode_device::ring_slots>;
/** Each card's mask buffer: two halves by ticket parity, each a mask per verify row in the card's row order. */
export using MaskBuffer = device::MappedArray<decode_device::Masks, 1>;

/** Bits of one card's row-order mask, in the 64-bit words the host's masks hold. */
export inline constexpr std::size_t card_words = model::vocab_per_card / 64;

static_assert(card_words * 2 == decode_device::mask_words, "a host word is two of the card's words, low first");

/** One verify row's mask: each card's rows in its own lm_head row order, bit r of word r / 64 allowing row r. */
export using CardMasks = std::array<std::array<std::uint64_t, card_words>, model::cards>;

/** The masks of a cycle's verify rows: `masks(i)` is batch row r's verify row v at i = r (1 + drafts(depth)) + v. */
export template<class M>
concept MaskRows = requires(M const& masks, std::size_t index) {
	{ masks(index) } -> std::convertible_to<CardMasks const&>;
};

/**
 * A ticket the cards are running: a decode cycle over a batch, or a prompt's end over its one row. It carries the ticket
 * both cards hold, what it runs, and each row's plan.
 */
export struct Pending {
	std::uint64_t ticket;
	decode_device::Work work;
	model::DraftDepth depth;
	/** Masks each row takes: a cycle's every verify row at the full depth, a prompt's end its one row. */
	std::size_t width;
	std::vector<sched::Row> rows;
	std::vector<Planned> planned;

	/** The drafts batch row `row` verifies after its committed token: the prefixes its masks condition on. */
	[[nodiscard]] auto drafts(std::size_t row) const -> std::span<Token const> {
		auto const& plan = planned.at(row);
		return std::span{plan.drafts}.first(plan.run.drafts);
	}

	/** Masks the ticket takes, as `MaskRows` indexes them. */
	[[nodiscard]] auto mask_count() const -> std::size_t {
		return rows.size() * width;
	}
};

/** Where each card's kernel finds the host's memories. */
export struct HostAddresses {
	std::uint64_t ring;
	std::uint64_t masks;
};

[[nodiscard]] constexpr auto card_index(model::Card card) -> std::size_t {
	return std::to_underlying(card);
}

/** Polls `ready` until it yields a value or `deadline` passes. */
template<class Ready>
[[nodiscard]] auto poll_until(std::chrono::steady_clock::time_point deadline, Ready ready) -> decltype(ready()) {
	constexpr auto interval = std::chrono::microseconds{20};
	for (;;) {
		if (auto value = ready()) {
			return value;
		}
		auto const now = std::chrono::steady_clock::now();
		if (now >= deadline) {
			return std::nullopt;
		}
		std::this_thread::sleep_until(std::min(now + interval, deadline));
	}
}

[[nodiscard]] constexpr auto same_outcome(decode_device::SlotOutcome const& left, decode_device::SlotOutcome const& right) -> bool {
	return left.accepted == right.accepted && std::ranges::equal(left.committed, right.committed) && std::ranges::equal(left.drafts, right.drafts);
}

export class Session {
public:
	[[nodiscard]] static auto create(device::Devices const& devices) -> core::Result<Session, DecodeError> {
		auto zero = Ring::create(devices.context(model::Card::Zero));
		auto one = zero.and_then([&](Ring const&) {
			return Ring::create(devices.context(model::Card::One));
		});
		auto masks_zero = one.and_then([&](Ring const&) {
			return devices.context(model::Card::Zero).allocate_mapped<decode_device::Masks, 1>();
		});
		auto masks_one = masks_zero.and_then([&](MaskBuffer const&) {
			return devices.context(model::Card::One).allocate_mapped<decode_device::Masks, 1>();
		});
		if (!masks_one) {
			return fail(masks_one.error());
		}
		return Session{
		    {std::move(*zero), std::move(*one)},
		    {std::move(*masks_zero), std::move(*masks_one)},
		};
	}

	[[nodiscard]] auto addresses(model::Card card) const -> HostAddresses {
		return {.ring = rings_.at(card_index(card)).device_address(), .masks = masks_.at(card_index(card)).device_address()};
	}

	/**
	 * Queues one of the pool's commands for the next ticket. A loaded snapshot ends what the slot carried on; a saved one
	 * is preceded by the replay of the slot's pending drafts, so it holds the state after every token but the last.
	 */
	auto apply(PoolCommand const& command) -> void {
		if (auto const load = core::alternative<kv::LoadSnapshot<kv::model_geometry>>(command)) {
			ledger_.forget(load->slot);
		}
		if (auto const save = core::alternative<kv::SaveSnapshot<kv::model_geometry>>(command)) {
			if (auto const pending = ledger_.materialize(save->slot)) {
				commands_.push_back({.action = decode_device::Action::Replay,
				                     .slot = core::narrow<unsigned char>(save->slot.index()),
				                     .wave = 0,
				                     .first = *pending,
				                     .second = 0,
				                     .third = 0});
			}
		}
		std::ranges::copy(lower(command), std::back_inserter(commands_));
	}

	/** Applies every queued command now, in command-only tickets. */
	[[nodiscard]] auto flush(std::chrono::steady_clock::time_point deadline) -> core::Result<void, DecodeError> {
		for (auto const& ticket : take_commands()) {
			if (auto applied = run_apply(ticket, deadline); !applied) {
				return applied;
			}
		}
		return {};
	}

	/**
	 * Starts the batch's cycle on both cards, after the queued commands: all but the last ticket's worth go ahead in
	 * command-only tickets, the rest ride with the cycle.
	 */
	[[nodiscard]] auto begin(sched::Batch batch, std::chrono::steady_clock::time_point deadline) -> core::Result<Pending, DecodeError> {
		if (auto refused = check(batch)) {
			return fail(*refused);
		}
		auto pending = Pending{
		    .ticket = 0, .work = decode_device::Work::Cycle, .depth = batch.depth, .width = 1 + model::drafts(batch.depth), .rows = std::move(batch.rows), .planned = {}};
		pending.planned.reserve(pending.rows.size());
		for (auto const& row : pending.rows) {
			pending.planned.push_back(ledger_.plan(row, pending.depth));
		}
		return post(std::move(pending), deadline);
	}

	/**
	 * Starts a prompt's end on both cards: the lm_head over its last row (whose final norm prefill left in the prompt
	 * row), the sample under the row's one mask, and the mtp head's drafts from it at `depth`. `prompt` is the prompt's
	 * last row: its slot, request, position, token and rule.
	 */
	[[nodiscard]] auto begin_prompt_end(sched::Row prompt, model::DraftDepth depth, std::chrono::steady_clock::time_point deadline)
	    -> core::Result<Pending, DecodeError> {
		if (prompt.position + sched::lookahead > model::max_positions) {
			return fail(Refused{.reason = Refusal::PastContext});
		}
		auto const run = decode_device::SlotRun{
		    .slot = core::narrow<unsigned>(prompt.slot.index()),
		    .position = core::narrow<unsigned>(prompt.position),
		    .token = prompt.token,
		    .drafts = 0,
		    .replays = 0,
		    .unused = {},
		    .rule = prompt.rule,
		};
		auto const state = deltanet::run(deltanet::settled(deltanet::Committed{.tokens = prompt.position + 1}), 0);
		return post(Pending{.ticket = 0,
		                    .work = decode_device::Work::PromptEnd,
		                    .depth = depth,
		                    .width = 1,
		                    .rows = {prompt},
		                    .planned = {Planned{.run = run, .state = state, .drafts = {}}}},
		            deadline);
	}

	/** The prompt's sampled token, which both cards must agree on; the slot then continues from it with its drafts. */
	[[nodiscard]] auto complete_prompt_end(Pending pending, std::chrono::steady_clock::time_point deadline)
	    -> core::Result<sched::ChunkDone, DecodeError> {
		core::invariant(pending.work == decode_device::Work::PromptEnd && pending.rows.size() == 1);
		auto answers = agreed(pending, deadline);
		if (!answers) {
			return std::unexpected(answers.error());
		}
		auto const& row = pending.rows.front();
		auto const sampled = ledger_.start(row.slot, row.request, row.position, answers->slots[0], pending.depth);
		return sched::ChunkDone{.sampled = sampled, .took = took(*answers)};
	}

	/**
	 * Writes each verify row's mask into both cards' mask buffers and releases them for the cycle. Masks of draft rows
	 * past a row's drafts are not read. A mask that allows nothing would leave nothing to sample, and is refused.
	 */
	template<MaskRows M>
	[[nodiscard]] auto publish(Pending const& pending, M const& masks, std::size_t count) -> core::Result<void, DecodeError> {
		if (running_ != pending.ticket) {
			return fail(Refused{.reason = Refusal::WrongTicket});
		}
		if (count != pending.mask_count()) {
			return fail(Refused{.reason = Refusal::MaskCount});
		}
		auto const width = 1 + model::drafts(pending.depth);
		auto const half = decode_device::mask_half(pending.ticket);
		auto device_row = std::size_t{0};
		for (auto const [index, plan] : std::views::enumerate(pending.planned)) {
			for (auto verify = std::size_t{0}; verify <= plan.run.drafts; ++verify) {
				CardMasks const& row = masks((core::narrow<std::size_t>(index) * width) + verify);
				auto const allows = std::ranges::any_of(row, [](auto const& card) {
					return std::ranges::any_of(card, [](std::uint64_t word) {
						return word != 0;
					});
				});
				if (!allows) {
					return fail(Refused{.reason = Refusal::EmptyMask});
				}
				for (auto const card : device::card_pair) {
					auto& words = masks_.at(card_index(card)).host()[0].half[half].words[device_row];
					for (auto const [word, bits] : std::views::enumerate(row.at(card_index(card)))) {
						auto const at = core::narrow<std::size_t>(word) * 2;
						words[at] = static_cast<unsigned>(bits & 0xffff'ffffU);
						words[at + 1] = static_cast<unsigned>(bits >> 32U);
					}
				}
				++device_row;
			}
		}
		for (auto& buffer : masks_) {
			decode_device::release_masks(buffer.host()[0].flag[half], pending.ticket);
		}
		return {};
	}

	/** Both cards' answers to the cycle, which must agree, as the batch's outcome; the ledger moves on. */
	[[nodiscard]] auto complete(Pending pending, std::chrono::steady_clock::time_point deadline) -> core::Result<sched::BatchDone, DecodeError> {
		core::invariant(pending.work == decode_device::Work::Cycle);
		auto answer = agreed(pending, deadline);
		if (!answer) {
			return std::unexpected(answer.error());
		}
		auto const& zero = *answer;
		auto done = sched::BatchDone{.rows = {}, .took = took(zero)};
		done.rows.reserve(pending.rows.size());
		for (auto const [index, row] : std::views::enumerate(pending.rows)) {
			auto const at = core::narrow<std::size_t>(index);
			done.rows.push_back(ledger_.settle(row, pending.planned.at(at), std::span{zero.slots}[at], pending.depth));
		}
		return done;
	}

	/** Asks both loops to exit and waits for their acknowledgements. */
	[[nodiscard]] auto stop(std::chrono::steady_clock::time_point deadline) -> core::Result<void, DecodeError> {
		core::invariant(!running_.has_value());
		std::array<std::uint64_t, model::cards> tickets{};
		for (auto const card : device::card_pair) {
			auto& ring = rings_.at(card_index(card));
			auto const ticket = poll_until(deadline, [&] {
				return ring.try_exit();
			});
			if (!ticket) {
				return fail(TimedOut{.waiting = Waiting::Submit, .ticket = 0});
			}
			tickets.at(card_index(card)) = ticket->value;
		}
		for (auto const card : device::card_pair) {
			auto& ring = rings_.at(card_index(card));
			auto const ticket = tickets.at(card_index(card));
			auto const answer = poll_until(deadline, [&] {
				return ring.try_collect();
			});
			if (!answer) {
				return fail(TimedOut{.waiting = Waiting::Answer, .ticket = ticket});
			}
			if (!std::holds_alternative<device::Exited>(*answer)) {
				return fail(LoopGone{.ticket = ticket});
			}
		}
		return {};
	}

	/**
	 * Fresh rings for the next launch of the loops, after `stop`: a ring that carried an exit ticket takes no more, and
	 * the new loops count tickets from one. The mask buffers' flags go back to zero so no stale release passes for a new
	 * ticket.
	 */
	[[nodiscard]] auto renew(device::Devices const& devices) -> core::Result<void, DecodeError> {
		core::invariant(!running_.has_value());
		auto zero = Ring::create(devices.context(model::Card::Zero));
		auto one = zero.and_then([&](Ring const&) {
			return Ring::create(devices.context(model::Card::One));
		});
		if (!one) {
			return fail(one.error());
		}
		rings_ = {std::move(*zero), std::move(*one)};
		for (auto& buffer : masks_) {
			for (auto& flag : buffer.host()[0].flag) {
				decode_device::release_masks(flag, 0);
			}
		}
		return {};
	}

	/** The card's mask buffer as the host wrote it. */
	[[nodiscard]] auto masks(model::Card card) const -> decode_device::Masks const& {
		return masks_.at(card_index(card)).host()[0];
	}

	[[nodiscard]] auto ledger() const -> Ledger const& {
		return ledger_;
	}

private:
	/** The answers of both cards to `pending`, once both are in and agree on every row. */
	[[nodiscard]] auto agreed(Pending const& pending, std::chrono::steady_clock::time_point deadline) -> core::Result<Result, DecodeError> {
		auto answers = collect(pending.ticket, deadline);
		if (!answers) {
			return std::unexpected(answers.error());
		}
		running_ = std::nullopt;
		auto const& [zero, one] = *answers;
		for (auto index = std::size_t{0}; index < pending.rows.size(); ++index) {
			if (!same_outcome(std::span{zero.slots}[index], std::span{one.slots}[index])) {
				return fail(CardsDisagree{.ticket = pending.ticket});
			}
		}
		return zero;
	}

	[[nodiscard]] static auto took(Result const& answer) -> std::chrono::nanoseconds {
		return std::chrono::nanoseconds{core::narrow<std::int64_t>(answer.finished - answer.started)};
	}

	/** Posts `pending` to both cards after the queued commands: all but the last ticket's worth go ahead on their own. */
	[[nodiscard]] auto post(Pending pending, std::chrono::steady_clock::time_point deadline) -> core::Result<Pending, DecodeError> {
		core::invariant(!running_.has_value());
		auto cut = take_commands();
		auto last = cut.empty() ? std::vector<decode_device::Command>{} : std::move(cut.back());
		if (!cut.empty()) {
			cut.pop_back();
		}
		for (auto const& ticket : cut) {
			if (auto applied = run_apply(ticket, deadline); !applied) {
				return std::unexpected(applied.error());
			}
		}
		auto& descriptor = *descriptor_;
		descriptor = Descriptor{};
		descriptor.work = pending.work;
		descriptor.depth = pending.depth;
		descriptor.live = core::narrow<unsigned char>(pending.rows.size());
		descriptor.commands = core::narrow<unsigned>(last.size());
		for (auto const [index, plan] : std::views::enumerate(pending.planned)) {
			std::span{descriptor.slots}[core::narrow<std::size_t>(index)] = plan.run;
		}
		std::ranges::copy(last, std::span{descriptor.command}.begin());
		auto ticket = submit(descriptor, deadline);
		if (!ticket) {
			return std::unexpected(ticket.error());
		}
		pending.ticket = *ticket;
		running_ = *ticket;
		return pending;
	}

	Session(std::array<Ring, model::cards> rings, std::array<MaskBuffer, model::cards> masks)
	    : rings_{std::move(rings)}, masks_{std::move(masks)}, descriptor_{std::make_unique<Descriptor>()} {}

	[[nodiscard]] static auto check(sched::Batch const& batch) -> std::optional<Refused> {
		if (batch.rows.empty()) {
			return Refused{.reason = Refusal::NoRows};
		}
		if (batch.rows.size() * (1 + model::drafts(batch.depth)) > decode_device::max_rows) {
			return Refused{.reason = Refusal::OverBudget};
		}
		auto const ascending = std::ranges::adjacent_find(batch.rows, [](sched::Row const& left, sched::Row const& right) {
			return left.slot.index() >= right.slot.index();
		});
		if (ascending != batch.rows.end()) {
			return Refused{.reason = Refusal::SlotOrder};
		}
		auto const past = std::ranges::any_of(batch.rows, [](sched::Row const& row) {
			return row.position + sched::lookahead > model::max_positions;
		});
		if (past) {
			return Refused{.reason = Refusal::PastContext};
		}
		return std::nullopt;
	}

	[[nodiscard]] auto take_commands() -> std::vector<std::vector<decode_device::Command>> {
		auto cut = tickets(waves(commands_));
		commands_.clear();
		return cut;
	}

	[[nodiscard]] auto run_apply(std::span<decode_device::Command const> commands, std::chrono::steady_clock::time_point deadline)
	    -> core::Result<void, DecodeError> {
		auto& descriptor = *descriptor_;
		descriptor = Descriptor{};
		descriptor.work = decode_device::Work::Apply;
		descriptor.commands = core::narrow<unsigned>(commands.size());
		std::ranges::copy(commands, std::span{descriptor.command}.begin());
		return submit(descriptor, deadline).and_then([&](std::uint64_t ticket) {
			return collect(ticket, deadline).transform([](auto const&) {});
		});
	}

	/** Posts `descriptor` to both cards; both rings hand out tickets in lockstep, so one number names it on both. */
	[[nodiscard]] auto submit(Descriptor const& descriptor, std::chrono::steady_clock::time_point deadline) -> core::Result<std::uint64_t, DecodeError> {
		std::array<std::uint64_t, model::cards> tickets{};
		for (auto const card : device::card_pair) {
			auto& ring = rings_.at(card_index(card));
			auto const ticket = poll_until(deadline, [&] {
				return ring.try_submit(descriptor);
			});
			if (!ticket) {
				return fail(TimedOut{.waiting = Waiting::Submit, .ticket = 0});
			}
			tickets.at(card_index(card)) = ticket->value;
		}
		core::invariant(tickets[0] == tickets[1]);
		return tickets[0];
	}

	/** Both cards' results for `ticket`, the oldest each has outstanding. */
	[[nodiscard]] auto collect(std::uint64_t ticket, std::chrono::steady_clock::time_point deadline)
	    -> core::Result<std::array<Result, model::cards>, DecodeError> {
		std::array<Result, model::cards> results{};
		for (auto const card : device::card_pair) {
			auto& ring = rings_.at(card_index(card));
			auto const answer = poll_until(deadline, [&] {
				return ring.try_collect();
			});
			if (!answer) {
				return fail(TimedOut{.waiting = Waiting::Answer, .ticket = ticket});
			}
			auto const ran = core::alternative<device::Ran<Result>>(*answer);
			if (!ran) {
				return fail(LoopGone{.ticket = ticket});
			}
			core::invariant(ran->ticket.value == ticket);
			results.at(card_index(card)) = ran->result;
		}
		return results;
	}

	std::array<Ring, model::cards> rings_;
	std::array<MaskBuffer, model::cards> masks_;
	/** Descriptor staging: 8.7 KB, built in place for each ticket. */
	std::unique_ptr<Descriptor> descriptor_;
	Ledger ledger_;
	std::vector<decode_device::Command> commands_;
	/** The ticket begun and not yet completed. */
	std::optional<std::uint64_t> running_;
};

}

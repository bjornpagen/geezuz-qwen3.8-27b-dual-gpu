// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's side of the resident decode loops on both cards: each card's ring and mask buffer, the ledger of what the
// cards hold per slot, and the pool's commands waiting for the next ticket. A cycle is three steps. `begin` posts the
// cycle descriptor to both cards, whose kernels start streaming the body at once; it hands back the drafts each row
// verifies, which the host needs for the grammar's masks. `publish` writes those masks into the cards' mask buffers
// (each card's half in its own row order) and releases them; the kernels acquire them only right before they rank the
// lm_head's rows, so the host has the whole body (~15 ms) to compute them. `try_complete` polls for both cards'
// answers, which must agree, and settles the ledger once both are in; it never blocks, so the pipeline polls it on its
// run loop. A logits ticket (`begin_logits`, `try_complete_logits`) runs the same verify pass and hands back the
// lm_head's logits in token order instead of a verdict, for the determinism harness and the rulers.
export module engine.decode:session;

import std;
import engine.core;
import engine.decode_device;
import engine.deltanet;
import engine.device;
import engine.kv;
import engine.model;
import engine.sched;
import engine.vocabulary;
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
/** Each card's logits buffer: two halves by ticket parity, each a logits ticket's rows in the card's row order. */
export using LogitsMemory = device::MappedArray<decode_device::LogitsBuffer, 1>;

static_assert(model::card_mask_words * 2 == decode_device::mask_words, "a host word is two of the card's words, low first");

/** One card's mask of one verify row, borrowed: the card's rows in its own lm_head row order, bit r of word r / 64 allowing row r. */
export using CardMask = std::span<std::uint64_t const, model::card_mask_words>;

/**
 * The masks of a cycle's verify rows: `masks(i, card)` borrows card `card`'s mask of batch row r's verify row v at
 * i = r (1 + drafts(depth)) + v. The session copies it into the card's mask buffer and keeps no reference.
 */
export template<class M>
concept MaskRows = requires(M const& masks, std::size_t index, model::Card card) {
	{ masks(index, card) } -> std::convertible_to<CardMask>;
};

/**
 * A ticket the cards are running: a decode cycle over a batch, a prompt's end over its one row, or a logits ticket over
 * a batch's committed tokens. It carries the ticket both cards hold, what it runs, and each row's plan.
 */
export struct Pending {
	std::uint64_t ticket;
	decode_device::Work work;
	model::DraftDepth depth;
	/** Masks each row takes: a cycle's every verify row at the full depth, a prompt's end its one row, a logits ticket none. */
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
	std::uint64_t logits;
};

/**
 * A logits ticket's outcome: each batch row's logits over the sampled vocabulary in token order (padding rows
 * excluded), both cards' halves put back together: what the reference's dumps and golden hashes take.
 */
export struct Logits {
	std::vector<std::vector<float>> rows;
};

[[nodiscard]] constexpr auto card_index(model::Card card) -> std::size_t {
	return std::to_underlying(card);
}

/**
 * The first place the cards' answers to `ticket` differ, over its rows (whose slots are `slots`, in row order): which
 * slot, what of its outcome, and each card's word there. None when they agree.
 */
[[nodiscard]] auto disagreement(std::uint64_t ticket, std::span<kv::Slot const> slots, Result const& zero, Result const& one) -> std::optional<CardsDisagree> {
	for (auto const [row, slot] : std::views::enumerate(slots)) {
		auto const& left = std::span{zero.slots}[core::narrow<std::size_t>(row)];
		auto const& right = std::span{one.slots}[core::narrow<std::size_t>(row)];
		auto const differ = [&](Disagreement what, std::span<unsigned const> lefts, std::span<unsigned const> rights) -> std::optional<CardsDisagree> {
			auto const [at_left, at_right] = std::ranges::mismatch(lefts, rights);
			if (at_left == lefts.end()) {
				return std::nullopt;
			}
			return CardsDisagree{.ticket = ticket,
			                     .slot = core::narrow<std::uint32_t>(slot.index()),
			                     .what = what,
			                     .at = core::narrow<std::uint32_t>(at_left - lefts.begin()),
			                     .zero = *at_left,
			                     .one = *at_right};
		};
		auto found = differ(Disagreement::Accepted, std::span{&left.accepted, 1}, std::span{&right.accepted, 1});
		if (!found) {
			found = differ(Disagreement::Committed, left.committed, right.committed);
		}
		if (!found) {
			found = differ(Disagreement::Drafted, left.drafts, right.drafts);
		}
		if (found) {
			return found;
		}
	}
	return std::nullopt;
}

/** What a ticket asks of the loops: apply commands, run the begun cycle or prompt end, or exit. */
enum class Ask : std::uint8_t {
	Apply,
	Begun,
	Exit,
};

/** A ticket waiting for room in the rings: what it asks, and for `Apply` its commands. */
struct Outgoing {
	Ask ask;
	std::vector<decode_device::Command> commands;
};

/** A ticket both rings carry, waiting for both cards' answers. */
struct Awaited {
	Ask ask;
	std::uint64_t ticket;
	std::array<std::optional<Result>, model::cards> answers;
};

/** The loops serve tickets and none is begun. */
struct Idle {};

/**
 * Whether a begun ticket's masks have been released to the cards, which wait for them before they rank; a logits
 * ticket ranks nothing and takes none.
 */
enum class Masking : std::uint8_t {
	Unpublished,
	Published,
	Unmasked,
};

/**
 * A cycle or prompt end is begun: its ticket, the slots it runs, its verify rows, whether its masks are out, and both
 * cards' answers once in.
 */
struct Ticketed {
	std::uint64_t ticket;
	decode_device::Work work;
	std::vector<kv::Slot> slots;
	std::size_t verify_rows;
	Masking masking;
	std::optional<std::array<Result, model::cards>> answers;
};

/** The exit tickets are queued; the loops end once both are answered. A failure found on the way is reported then. */
struct Stopping {
	std::optional<DecodeError> trouble;
};

/** Both loops answered their exit: the cards run nothing of the engine's. A failure found on the way is kept. */
struct Stopped {
	std::optional<DecodeError> trouble;
};

/** The loops failed or were given up on; nothing more is sent to them. */
struct Broken {
	DecodeError error;
};

/** Where the loops stand. The begun ticket exists only while ticketed, so no call can find one where none is. */
using LoopState = std::variant<Idle, Ticketed, Stopping, Stopped, Broken>;

/** Whether the cards can be trusted with the queued commands: not after they disagreed. */
enum class Health : std::uint8_t {
	Healthy,
	Suspect,
};

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
		auto logits_zero = masks_one.and_then([&](MaskBuffer const&) {
			return devices.context(model::Card::Zero).allocate_mapped<decode_device::LogitsBuffer, 1>();
		});
		auto logits_one = logits_zero.and_then([&](LogitsMemory const&) {
			return devices.context(model::Card::One).allocate_mapped<decode_device::LogitsBuffer, 1>();
		});
		if (!logits_one) {
			return fail(logits_one.error());
		}
		return Session{
		    {std::move(*zero), std::move(*one)},
		    {std::move(*masks_zero), std::move(*masks_one)},
		    {std::move(*logits_zero), std::move(*logits_one)},
		};
	}

	[[nodiscard]] auto addresses(model::Card card) const -> HostAddresses {
		return {.ring = rings_.at(card_index(card)).device_address(),
		        .masks = masks_.at(card_index(card)).device_address(),
		        .logits = logits_.at(card_index(card)).device_address()};
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

	/**
	 * Starts the batch's cycle on both cards, after the queued commands: all but the last ticket's worth go ahead in
	 * command-only tickets, the rest ride with the cycle. Submits what the rings have room for and returns at once;
	 * `try_complete` submits the rest as room frees.
	 */
	[[nodiscard]] auto begin(sched::Batch batch) -> core::Result<Pending, DecodeError> {
		if (auto refused = check(batch)) {
			return fail(*refused);
		}
		auto pending = Pending{
		    .ticket = 0, .work = decode_device::Work::Cycle, .depth = batch.depth, .width = 1 + model::drafts(batch.depth), .rows = std::move(batch.rows), .planned = {}};
		pending.planned.reserve(pending.rows.size());
		for (auto const& row : pending.rows) {
			pending.planned.push_back(ledger_.plan(row, pending.depth));
		}
		return post(std::move(pending));
	}

	/**
	 * Starts a logits ticket on both cards over the batch's rows, after the queued commands: each row's committed token
	 * runs through the verify pass exactly as a cycle runs it, with no drafts, and the lm_head's logits come back in
	 * place of a verdict. Returns at once, like `begin`.
	 *
	 * Afterwards each row's slot holds its trunk state exactly as after a cycle that committed the row's token with no
	 * drafts: its kv at the row's position written, its deltanet state through it, its pending replays taken. Nothing is
	 * sampled and no mtp pass runs, so the slot keeps no drafts and the mtp head's kv at that position is not written;
	 * the ledger forgets the slot, which plans its next row as one whose state covers exactly its position, as after
	 * prefill. Teacher forcing walks a prompt by giving each next ticket the next token at the next position. The mtp
	 * kv left unwritten only lowers a later cycle's acceptance, never changes a committed token: verification is exact.
	 */
	[[nodiscard]] auto begin_logits(sched::Batch batch) -> core::Result<Pending, DecodeError> {
		if (auto refused = check(batch)) {
			return fail(*refused);
		}
		auto pending = Pending{
		    .ticket = 0, .work = decode_device::Work::Logits, .depth = batch.depth, .width = 0, .rows = std::move(batch.rows), .planned = {}};
		pending.planned.reserve(pending.rows.size());
		for (auto const& row : pending.rows) {
			pending.planned.push_back(ledger_.plan_alone(row));
		}
		return post(std::move(pending));
	}

	/**
	 * Starts a prompt's end on both cards: the lm_head over its last row (whose final norm prefill left in the prompt
	 * row), the sample under the row's one mask, and the mtp head's drafts from it at `depth`. `prompt` is the prompt's
	 * last row: its slot, request, position, token and rule. Returns at once, like `begin`.
	 */
	[[nodiscard]] auto begin_prompt_end(sched::Row prompt, model::DraftDepth depth) -> core::Result<Pending, DecodeError> {
		if (prompt.position + sched::lookahead > model::max_positions) {
			return fail(Refused{.reason = Refusal::PastContext, .ticket = std::nullopt});
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
		                    .planned = {Planned{.run = run, .state = state, .drafts = {}}}});
	}

	/**
	 * Writes each verify row's mask into both cards' mask buffers and releases them for the ticket. Masks of draft rows
	 * past a row's drafts are not read. A mask that allows nothing would leave nothing to sample, and is refused.
	 */
	template<MaskRows M>
	[[nodiscard]] auto publish(Pending const& pending, M const& masks, std::size_t count) -> core::Result<void, DecodeError> {
		// A logits ticket ranks nothing: there is no mask for it to take.
		core::invariant(pending.work != decode_device::Work::Logits);
		if (!begun(pending)) {
			return fail(Refused{.reason = Refusal::WrongTicket, .ticket = pending.ticket});
		}
		if (count != pending.mask_count()) {
			return fail(Refused{.reason = Refusal::MaskCount, .ticket = pending.ticket});
		}
		auto const half = decode_device::mask_half(pending.ticket);
		auto device_row = std::size_t{0};
		for (auto const [index, plan] : std::views::enumerate(pending.planned)) {
			for (auto verify = std::size_t{0}; verify <= plan.run.drafts; ++verify) {
				auto const mask = (core::narrow<std::size_t>(index) * pending.width) + verify;
				auto const row = std::array<CardMask, model::cards>{CardMask{masks(mask, model::Card::Zero)}, CardMask{masks(mask, model::Card::One)}};
				auto const allows = std::ranges::any_of(row, [](CardMask card) {
					return std::ranges::any_of(card, [](std::uint64_t word) {
						return word != 0;
					});
				});
				if (!allows) {
					return fail(Refused{.reason = Refusal::EmptyMask, .ticket = pending.ticket});
				}
				for (auto const card : device::card_pair) {
					auto& words = std::span{std::span{masks_.at(card_index(card)).host()[0].half}[half].words}[device_row];
					for (auto const [word, bits] : std::views::enumerate(row.at(card_index(card)))) {
						auto const at = core::narrow<std::size_t>(word) * 2;
						std::span{words}[at] = static_cast<unsigned>(bits & 0xffff'ffffU);
						std::span{words}[at + 1] = static_cast<unsigned>(bits >> 32U);
					}
				}
				++device_row;
			}
		}
		for (auto& buffer : masks_) {
			decode_device::release_masks(std::span{buffer.host()[0].flag}[half], pending.ticket);
		}
		std::get<Ticketed>(state_).masking = Masking::Published;
		return {};
	}

	/**
	 * The batch's outcome once both cards have answered every ticket ahead of it and the cycle, and agree; nullopt
	 * while they have not. The ledger moves on with the outcome. Never blocks: the pipeline polls it on its run loop.
	 */
	[[nodiscard]] auto try_complete(Pending const& pending) -> std::optional<core::Result<sched::BatchDone, DecodeError>> {
		core::invariant(pending.work == decode_device::Work::Cycle);
		return try_agreed(pending).transform([&](core::Result<Result, DecodeError> const& answer) -> core::Result<sched::BatchDone, DecodeError> {
			if (!answer) {
				return std::unexpected(answer.error());
			}
			auto done = sched::BatchDone{.rows = {}, .took = took(*answer)};
			done.rows.reserve(pending.rows.size());
			for (auto const [index, row] : std::views::enumerate(pending.rows)) {
				auto const at = core::narrow<std::size_t>(index);
				done.rows.push_back(ledger_.settle(row, pending.planned.at(at), std::span{answer->slots}[at], pending.depth));
			}
			return done;
		});
	}

	/** The prompt's sampled token, as `try_complete` for a cycle; the slot then continues from it with its drafts. */
	[[nodiscard]] auto try_complete_prompt_end(Pending const& pending) -> std::optional<core::Result<sched::ChunkDone, DecodeError>> {
		core::invariant(pending.work == decode_device::Work::PromptEnd && pending.rows.size() == 1);
		return try_agreed(pending).transform([&](core::Result<Result, DecodeError> const& answer) -> core::Result<sched::ChunkDone, DecodeError> {
			if (!answer) {
				return std::unexpected(answer.error());
			}
			auto const& row = pending.rows.front();
			auto const sampled = ledger_.start(row.slot, row.request, row.position, answer->slots[0], pending.depth);
			return sched::ChunkDone{.sampled = sampled, .took = took(*answer)};
		});
	}

	/**
	 * A logits ticket's logits once both cards have answered every ticket ahead of it and it, nullopt while they have
	 * not. Each card holds its own half of the vocabulary, so there is nothing for the two to agree on; the slots are
	 * forgotten, as `begin_logits` says. Never blocks.
	 */
	[[nodiscard]] auto try_complete_logits(Pending const& pending) -> std::optional<core::Result<Logits, DecodeError>> {
		core::invariant(pending.work == decode_device::Work::Logits && begun(pending));
		if (auto failed = pump(); failed) {
			state_ = Broken{.error = *failed};
			return std::unexpected(*failed);
		}
		auto const& ticketed = std::get<Ticketed>(state_);
		if (!ticketed.answers) {
			return std::nullopt;
		}
		auto const half = decode_device::logits_half(pending.ticket);
		auto logits = Logits{.rows = {}};
		logits.rows.reserve(pending.rows.size());
		for (auto row = std::size_t{0}; row < pending.rows.size(); ++row) {
			auto& tokens = logits.rows.emplace_back(model::sampled_vocabulary);
			for (auto const card : device::card_pair) {
				auto const& values = std::span{std::span{logits_.at(card_index(card)).host()[0].half}[half].rows}[row];
				for (auto const [card_row, token] : std::views::enumerate(row_tokens_.at(card_index(card)))) {
					// A padding row scores no token the tokenizer has.
					if (token < model::sampled_vocabulary) {
						tokens[token] = std::span{values}[core::narrow<std::size_t>(card_row)];
					}
				}
			}
		}
		for (auto const& row : pending.rows) {
			ledger_.forget(row.slot);
		}
		state_ = Idle{};
		return logits;
	}

	/**
	 * Stops both loops with nothing begun, the one way the cards are left consistent for prefill: the first call queues
	 * every queued pool command as command-only tickets and the exit tickets behind them, so a prefill that follows the
	 * stop sees every block table and slot state the scheduler set up (cards caught disagreeing get no more commands).
	 * nullopt until both loops have answered the exit; then the outcome, as often as it is asked. Never blocks.
	 */
	[[nodiscard]] auto try_stop() -> std::optional<core::Result<void, DecodeError>> {
		core::invariant(!std::holds_alternative<Ticketed>(state_));
		if (std::holds_alternative<Idle>(state_)) {
			queue_exit(health_, std::nullopt);
		}
		return try_shutdown(std::chrono::steady_clock::time_point::max());
	}

	/**
	 * Ends the loops from wherever they stand, for teardown: a begun ticket is collected (its outcome discarded, its
	 * slots forgotten), the queued commands are sent only if the cards have not been caught disagreeing, then the exits;
	 * stopped loops are an immediate success, broken ones their failure. Past `give_up` the loops are abandoned with
	 * `Abandoned`. nullopt while it is still waiting. Never blocks.
	 */
	[[nodiscard]] auto try_shutdown(std::chrono::steady_clock::time_point give_up) -> std::optional<core::Result<void, DecodeError>> {
		if (auto const stopped = core::alternative<Stopped>(state_)) {
			return stopped->trouble ? core::Result<void, DecodeError>{std::unexpected(*stopped->trouble)} : core::Result<void, DecodeError>{};
		}
		if (auto const broken = core::alternative<Broken>(state_)) {
			return std::unexpected(broken->error);
		}
		auto trouble = std::optional<DecodeError>{};
		if (auto const ticketed = core::alternative<Ticketed>(state_)) {
			// Masks never published would leave the cards waiting before they rank: release every row allowing every
			// token, so the ticket runs out. Its outcome is discarded.
			if (ticketed->masking == Masking::Unpublished) {
				release_all(ticketed->ticket, ticketed->verify_rows);
				ticketed->masking = Masking::Published;
			}
			if (auto failed = pump(); failed) {
				return broke(*failed);
			}
			if (!ticketed->answers) {
				return give_up_at(give_up, ticketed->ticket);
			}
			auto const& [zero, one] = *ticketed->answers;
			// A logits ticket's answers carry no verdict to compare: each card holds its own half of the logits.
			auto const disagree = ticketed->work == decode_device::Work::Logits ? std::nullopt : disagreement(ticketed->ticket, ticketed->slots, zero, one);
			if (disagree) {
				health_ = Health::Suspect;
				trouble = DecodeError{.reason = *disagree};
			}
			// The cards ran the ticket; the host never settled it, so what they hold for its slots is unknown here.
			for (auto const slot : ticketed->slots) {
				ledger_.forget(slot);
			}
			state_ = Idle{};
		}
		if (std::holds_alternative<Idle>(state_)) {
			queue_exit(health_, std::move(trouble));
		}
		auto stopped = step_stopping();
		if (!stopped) {
			// The oldest ticket still unanswered, or the next to go out while the rings are full.
			return give_up_at(give_up, awaited_.empty() ? submitted_ + 1 : awaited_.front().ticket);
		}
		return stopped;
	}

	/**
	 * Fresh rings for the next launch of the loops, after a stop: a ring that carried an exit ticket takes no more, and
	 * the new loops count tickets from one. The mask buffers' flags go back to zero so no stale release passes for a new
	 * ticket.
	 */
	[[nodiscard]] auto renew(device::Devices const& devices) -> core::Result<void, DecodeError> {
		core::invariant(std::holds_alternative<Stopped>(state_) && outgoing_.empty() && awaited_.empty());
		auto zero = Ring::create(devices.context(model::Card::Zero));
		auto one = zero.and_then([&](Ring const&) {
			return Ring::create(devices.context(model::Card::One));
		});
		if (!one) {
			return fail(one.error());
		}
		rings_ = {std::move(*zero), std::move(*one)};
		submitted_ = 0;
		state_ = Idle{};
		health_ = Health::Healthy;
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
	Session(std::array<Ring, model::cards> rings, std::array<MaskBuffer, model::cards> masks, std::array<LogitsMemory, model::cards> logits)
	    : rings_{std::move(rings)},
	      masks_{std::move(masks)},
	      logits_{std::move(logits)},
	      row_tokens_{vocabulary::row_tokens(model::Card::Zero), vocabulary::row_tokens(model::Card::One)},
	      descriptor_{std::make_unique<Descriptor>()},
	      begun_descriptor_{std::make_unique<Descriptor>()} {}

	[[nodiscard]] static auto took(Result const& answer) -> std::chrono::nanoseconds {
		return std::chrono::nanoseconds{core::narrow<std::int64_t>(answer.finished - answer.started)};
	}

	[[nodiscard]] auto begun(Pending const& pending) const -> bool {
		auto const ticketed = core::alternative<Ticketed>(state_);
		return ticketed.has_value() && ticketed->ticket == pending.ticket;
	}

	/** Verify rows a ticket runs: every row's committed token and its drafts. */
	[[nodiscard]] static auto verify_rows(Pending const& pending) -> std::size_t {
		auto rows = std::size_t{0};
		for (auto const& plan : pending.planned) {
			rows += 1U + plan.run.drafts;
		}
		return rows;
	}

	/** Releases `rows` masks allowing every row of both cards for `ticket`: what a ticket abandoned before its masks runs on. */
	auto release_all(std::uint64_t ticket, std::size_t rows) -> void {
		auto const half = decode_device::mask_half(ticket);
		for (auto& buffer : masks_) {
			auto& words = std::span{buffer.host()[0].half}[half].words;
			for (auto row = std::size_t{0}; row < rows; ++row) {
				std::ranges::fill(std::span{words}[row], ~0U);
			}
			decode_device::release_masks(std::span{buffer.host()[0].flag}[half], ticket);
		}
	}

	/** The loops are broken by `error`: it is kept, and returned to every later shutdown. */
	[[nodiscard]] auto broke(DecodeError error) -> std::optional<core::Result<void, DecodeError>> {
		state_ = Broken{.error = error};
		return std::unexpected(std::move(error));
	}

	[[nodiscard]] auto give_up_at(std::chrono::steady_clock::time_point give_up, std::uint64_t ticket) -> std::optional<core::Result<void, DecodeError>> {
		if (std::chrono::steady_clock::now() < give_up) {
			return std::nullopt;
		}
		return broke(DecodeError{.reason = Abandoned{.ticket = ticket, .kernels = Kernels::MayRun}});
	}

	/** Queues the exits, behind every queued command when the cards are healthy, and none otherwise. */
	auto queue_exit(Health health, std::optional<DecodeError> trouble) -> void {
		auto cut = take_commands();
		if (health == Health::Healthy) {
			for (auto& ticket : cut) {
				outgoing_.push_back({.ask = Ask::Apply, .commands = std::move(ticket)});
			}
		}
		outgoing_.push_back({.ask = Ask::Exit, .commands = {}});
		state_ = Stopping{.trouble = std::move(trouble)};
	}

	/** One step of a stop: done once every queued ticket, the exits last, is answered. */
	[[nodiscard]] auto step_stopping() -> std::optional<core::Result<void, DecodeError>> {
		if (auto failed = pump(); failed) {
			return broke(*failed);
		}
		if (!outgoing_.empty() || !awaited_.empty()) {
			return std::nullopt;
		}
		auto trouble = std::get<Stopping>(state_).trouble;
		state_ = Stopped{.trouble = trouble};
		if (trouble) {
			return std::unexpected(*trouble);
		}
		return core::Result<void, DecodeError>{};
	}

	/**
	 * Queues `pending` behind the queued commands (all but the last ticket's worth go ahead on their own) and submits
	 * what fits. Tickets are numbered in submission order on both rings, so the begun ticket's number is known now.
	 */
	[[nodiscard]] auto post(Pending pending) -> core::Result<Pending, DecodeError> {
		// Broken loops take nothing more: the failure is returned again, and only a shutdown ends them.
		if (auto const broken = core::alternative<Broken>(state_)) {
			return std::unexpected(broken->error);
		}
		core::invariant(std::holds_alternative<Idle>(state_) && outgoing_.empty());
		auto cut = take_commands();
		auto last = cut.empty() ? std::vector<decode_device::Command>{} : std::move(cut.back());
		if (!cut.empty()) {
			cut.pop_back();
		}
		for (auto& ticket : cut) {
			outgoing_.push_back({.ask = Ask::Apply, .commands = std::move(ticket)});
		}
		auto& descriptor = *begun_descriptor_;
		descriptor = Descriptor{};
		descriptor.work = pending.work;
		descriptor.depth = pending.depth;
		descriptor.live = core::narrow<unsigned char>(pending.rows.size());
		descriptor.commands = core::narrow<unsigned>(last.size());
		for (auto const [index, plan] : std::views::enumerate(pending.planned)) {
			std::span{descriptor.slots}[core::narrow<std::size_t>(index)] = plan.run;
		}
		std::ranges::copy(last, std::span{descriptor.command}.begin());
		outgoing_.push_back({.ask = Ask::Begun, .commands = {}});
		pending.ticket = submitted_ + outgoing_.size();
		auto slots = std::vector<kv::Slot>{};
		for (auto const& row : pending.rows) {
			slots.push_back(row.slot);
		}
		state_ = Ticketed{.ticket = pending.ticket,
		                  .work = pending.work,
		                  .slots = std::move(slots),
		                  .verify_rows = verify_rows(pending),
		                  .masking = pending.work == decode_device::Work::Logits ? Masking::Unmasked : Masking::Unpublished,
		                  .answers = std::nullopt};
		if (auto failed = pump(); failed) {
			state_ = Broken{.error = *failed};
			return std::unexpected(*failed);
		}
		return pending;
	}

	/**
	 * Both cards' answers to the begun ticket, once every ticket ahead of it has been answered and both are in; nullopt
	 * while any is missing. They must agree on every row; cards that do not are suspect from then on.
	 */
	[[nodiscard]] auto try_agreed(Pending const& pending) -> std::optional<core::Result<Result, DecodeError>> {
		core::invariant(begun(pending));
		if (auto failed = pump(); failed) {
			state_ = Broken{.error = *failed};
			return std::unexpected(*failed);
		}
		auto& ticketed = std::get<Ticketed>(state_);
		if (!ticketed.answers) {
			return std::nullopt;
		}
		auto const [zero, one] = *ticketed.answers;
		auto const disagree = disagreement(pending.ticket, ticketed.slots, zero, one);
		state_ = Idle{};
		if (disagree) {
			health_ = Health::Suspect;
			return fail(*disagree);
		}
		return core::Result<Result, DecodeError>{zero};
	}

	/**
	 * One non-blocking step of the ticket pipeline: submits queued tickets while both rings have room, then collects
	 * whatever both cards have answered, oldest first. An answer to the begun ticket is kept in the ticketed state.
	 * Returns the failure, if a loop answered a ticket with its exit or a result where an exit was due.
	 */
	[[nodiscard]] auto pump() -> std::optional<DecodeError> {
		while (!outgoing_.empty() && std::ranges::all_of(rings_, [](Ring const& ring) { return ring.in_flight() < decode_device::ring_slots; })) {
			auto const next = std::move(outgoing_.front());
			outgoing_.pop_front();
			auto const ticket = submit(next);
			awaited_.push_back({.ask = next.ask, .ticket = ticket, .answers = {}});
		}
		while (!awaited_.empty()) {
			auto& oldest = awaited_.front();
			for (auto const card : device::card_pair) {
				auto& answer = oldest.answers.at(card_index(card));
				if (answer) {
					continue;
				}
				auto const collected = rings_.at(card_index(card)).try_collect();
				if (!collected) {
					continue;
				}
				// An exit ticket is answered by the exit acknowledgement, every other by a result.
				auto const exited = std::holds_alternative<device::Exited>(*collected);
				if (exited != (oldest.ask == Ask::Exit)) {
					outgoing_.clear();
					awaited_.clear();
					return DecodeError{.reason = LoopGone{.ticket = oldest.ticket}};
				}
				answer = exited ? Result{} : std::get<device::Ran<Result>>(*collected).result;
			}
			auto const& [zero, one] = oldest.answers;
			if (!zero || !one) {
				return std::nullopt;
			}
			if (auto const ticketed = core::alternative<Ticketed>(state_); ticketed && oldest.ask == Ask::Begun) {
				ticketed->answers = std::array{*zero, *one};
			}
			awaited_.pop_front();
		}
		return std::nullopt;
	}

	/** Submits one ticket to both rings, which have room; both hand out the same number. */
	[[nodiscard]] auto submit(Outgoing const& next) -> std::uint64_t {
		auto const descriptor = [&] -> Descriptor const& {
			if (next.ask == Ask::Begun) {
				return *begun_descriptor_;
			}
			auto& apply = *descriptor_;
			apply = Descriptor{};
			apply.work = decode_device::Work::Apply;
			apply.commands = core::narrow<unsigned>(next.commands.size());
			std::ranges::copy(next.commands, std::span{apply.command}.begin());
			return apply;
		};
		std::array<std::uint64_t, model::cards> tickets{};
		for (auto const card : device::card_pair) {
			auto& ring = rings_.at(card_index(card));
			auto const ticket = next.ask == Ask::Exit ? ring.try_exit() : ring.try_submit(descriptor());
			if (!ticket) {
				// Not reached: a ticket is submitted only while both rings have room.
				core::invariant(false);
				std::unreachable();
			}
			tickets.at(card_index(card)) = ticket->value;
		}
		++submitted_;
		core::invariant(tickets[0] == tickets[1] && tickets[0] == submitted_);
		return tickets[0];
	}

	[[nodiscard]] static auto check(sched::Batch const& batch) -> std::optional<Refused> {
		if (batch.rows.empty()) {
			return Refused{.reason = Refusal::NoRows, .ticket = std::nullopt};
		}
		if (batch.rows.size() * (1 + model::drafts(batch.depth)) > decode_device::max_rows) {
			return Refused{.reason = Refusal::OverBudget, .ticket = std::nullopt};
		}
		auto const ascending = std::ranges::adjacent_find(batch.rows, [](sched::Row const& left, sched::Row const& right) {
			return left.slot.index() >= right.slot.index();
		});
		if (ascending != batch.rows.end()) {
			return Refused{.reason = Refusal::SlotOrder, .ticket = std::nullopt};
		}
		auto const past = std::ranges::any_of(batch.rows, [](sched::Row const& row) {
			return row.position + sched::lookahead > model::max_positions;
		});
		if (past) {
			return Refused{.reason = Refusal::PastContext, .ticket = std::nullopt};
		}
		return std::nullopt;
	}

	[[nodiscard]] auto take_commands() -> std::vector<std::vector<decode_device::Command>> {
		auto cut = tickets(waves(commands_));
		commands_.clear();
		return cut;
	}

	std::array<Ring, model::cards> rings_;
	std::array<MaskBuffer, model::cards> masks_;
	std::array<LogitsMemory, model::cards> logits_;
	/** Each card's lm_head row-to-token table, for putting a logits ticket's halves back in token order. */
	std::array<std::vector<vocabulary::Token>, model::cards> row_tokens_;
	/** Descriptor staging, 8.7 KB each: a command-only ticket's, built as it is submitted, and the begun ticket's. */
	std::unique_ptr<Descriptor> descriptor_;
	std::unique_ptr<Descriptor> begun_descriptor_;
	Ledger ledger_;
	/** Pool commands queued for the next ticket. */
	std::vector<decode_device::Command> commands_;
	/** Tickets waiting for room in the rings, and tickets waiting for answers, each in submission order. */
	std::deque<Outgoing> outgoing_;
	std::deque<Awaited> awaited_;
	/** Tickets submitted to each ring so far: the last one's number. */
	std::uint64_t submitted_ = 0;
	LoopState state_ = Idle{};
	Health health_ = Health::Healthy;
};

}

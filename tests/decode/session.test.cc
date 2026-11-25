// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's side of the resident loops against the driver double and a fake loop per card: the card's half of each
// ring served on its own thread, as the persistent kernel serves it. The fake loop is a test double of the kernel's
// protocol, not of its arithmetic: it applies command tickets by logging them, and for a cycle it waits until the host
// has released the cycle's masks (as the kernel does before ranking), then answers from those masks: each verify row
// commits the lowest token both cards' masks allow, a slot accepts a number of drafts its position decides, and its next
// drafts follow from its last token. Both fakes compute that answer from both cards' mask buffers, so they agree unless
// a test makes one of them lie.
import std;
import engine.core;
import engine.decode;
import engine.decode_device;
import engine.device;
import engine.driver_double;
import engine.kv;
import engine.model;
import engine.sampling;
import engine.sched;
import engine.schedule;
import engine.testing;
import engine.vocabulary;

namespace {

namespace decode = engine::decode;
namespace dd = engine::decode_device;
namespace device = engine::device;
namespace fake = engine::driver_double;
namespace kv = engine::kv;
namespace model = engine::model;
namespace sched = engine::sched;
namespace vocabulary = engine::vocabulary;

using model::Card;

auto const deadline_after = [] {
	return std::chrono::steady_clock::now() + std::chrono::seconds{30};
};

/** Polls `poll` until it yields an outcome, as the pipeline's run loop does; a poll that never does fails as abandoned. */
template<class Poll>
[[nodiscard]] auto settled(Poll poll) -> std::remove_cvref_t<decltype(*poll())> {
	auto const deadline = deadline_after();
	for (;;) {
		if (auto outcome = poll()) {
			return std::move(*outcome);
		}
		if (std::chrono::steady_clock::now() >= deadline) {
			return std::unexpected(decode::DecodeError{.reason = decode::Abandoned{.ticket = 0, .kernels = decode::Kernels::MayRun}});
		}
		std::this_thread::yield();
	}
}

[[nodiscard]] constexpr auto rule() -> engine::sampling::Rule {
	return {.seed = 7,
	        .filter = {.temperature = 1.0, .top_p = 1.0, .min_p = 0.0, .top_k = 1, .nucleus = engine::sampling::Nucleus::Off},
	        .request = 1};
}

/** The lowest token a verify row's masks allow, reading both cards' halves in their row order. */
[[nodiscard]] auto lowest_allowed(std::array<std::reference_wrapper<dd::Masks const>, 2> const& masks, std::size_t half, std::size_t row) -> unsigned {
	auto lowest = std::numeric_limits<unsigned>::max();
	for (auto const card : device::card_pair) {
		auto const& words = std::span{std::span{masks.at(std::to_underlying(card)).get().half}[half].words}[row];
		for (auto const [index, word] : std::views::enumerate(std::span{words})) {
			for (auto bits = word; bits != 0; bits &= bits - 1) {
				auto const card_row = (static_cast<std::size_t>(index) * 32) + static_cast<std::size_t>(std::countr_zero(bits));
				lowest = std::min(lowest, vocabulary::token(card, card_row));
			}
		}
	}
	return lowest;
}

/** What the fake loops saw, for the test to check. */
struct Seen {
	std::vector<std::vector<dd::Command>> applied;
	std::vector<dd::Descriptor> cycles;
};

/** Whether a fake loop answers truly, or claims one more accepted draft than the truth allows. */
enum class Honesty : std::uint8_t { Truthful, Lying };

/** The fake loop of one card. */
struct FakeLoop {
	std::array<std::reference_wrapper<dd::Masks const>, 2> masks;
	std::reference_wrapper<Seen> seen;
	Honesty honesty;
	std::uint64_t ticket = 0;

	[[nodiscard]] auto operator()(dd::Descriptor const& descriptor) -> dd::Result {
		++ticket;
		auto result = dd::Result{};
		if (descriptor.work == dd::Work::Apply) {
			seen.get().applied.emplace_back(std::span{descriptor.command}.first(descriptor.commands).begin(),
			                                std::span{descriptor.command}.first(descriptor.commands).end());
			return result;
		}
		seen.get().cycles.push_back(descriptor);
		auto const half = dd::mask_half(ticket);
		while (dd::released_masks(std::span{masks[0].get().flag}[half]) < ticket || dd::released_masks(std::span{masks[1].get().flag}[half]) < ticket) {
			std::this_thread::yield();
		}
		auto row = std::size_t{0};
		for (auto index = std::size_t{0}; index < descriptor.live; ++index) {
			auto const& run = std::span{descriptor.slots}[index];
			auto& outcome = std::span{result.slots}[index];
			outcome.accepted = (run.position % (run.drafts + 1U)) + (honesty == Honesty::Lying ? 1U : 0U);
			for (auto verify = std::size_t{0}; verify <= run.drafts; ++verify) {
				if (verify <= outcome.accepted && verify < dd::max_committed) {
					std::span{outcome.committed}[verify] = lowest_allowed(masks, half, row + verify);
				}
			}
			for (auto draft = std::size_t{0}; draft < dd::max_drafts; ++draft) {
				std::span{outcome.drafts}[draft] = std::span{outcome.committed}[std::min<std::size_t>(outcome.accepted, dd::max_committed - 1)] + 1 +
				                                   static_cast<unsigned>(draft);
			}
			row += 1U + run.drafts;
		}
		result.started = 1000 * ticket;
		result.finished = result.started + 500;
		return result;
	}
};

using Kernel = fake::RingKernel<dd::Descriptor, dd::Result, dd::ring_slots, FakeLoop>;

/** One verify row's masks, each card's in its own row order: what the grammar hands the pipeline. */
using RowMasks = std::array<std::array<std::uint64_t, model::card_mask_words>, model::cards>;

/** A row's mask allowing exactly `tokens`, in each card's row order. */
[[nodiscard]] auto mask_of(std::span<decode::Token const> tokens) -> RowMasks {
	auto masks = RowMasks{};
	for (auto const token : tokens) {
		auto const place = vocabulary::place(token);
		masks.at(std::to_underlying(place.card)).at(place.row / 64) |= std::uint64_t{1} << (place.row % 64);
	}
	return masks;
}


/** How a scenario leaves the loops: shut down cleanly after it, or failed, as it checked itself. */
enum class Ending : std::uint8_t { Clean, Failed };

/** Runs `scenario(session, seen)` with both cards' fake loops serving, then shuts both loops down, ending as `ending`. */
template<class Scenario>
[[nodiscard]] auto with_loops_ending(Honesty card_one, Ending ending, Scenario scenario) -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto devices = device::Devices::open(driver.driver());
	if (!devices) {
		return false;
	}
	auto session = decode::Session::create(*devices);
	if (!session) {
		return false;
	}
	auto seen = std::array<Seen, 2>{};
	auto const masks = std::array{std::cref(session->masks(Card::Zero)), std::cref(session->masks(Card::One))};
	auto const zero = Kernel{session->addresses(Card::Zero).ring, FakeLoop{.masks = masks, .seen = seen[0], .honesty = Honesty::Truthful}};
	auto const one = Kernel{session->addresses(Card::One).ring, FakeLoop{.masks = masks, .seen = seen[1], .honesty = card_one}};
	auto const held = scenario(*session, seen);
	auto const ended = settled([&] { return session->try_shutdown(deadline_after()); });
	return held && ended.has_value() == (ending == Ending::Clean) && driver.violations().empty();
}

/** Runs `scenario(session, seen)` with both cards' fake loops serving, then shuts both loops down cleanly. */
template<class Scenario>
[[nodiscard]] auto with_loops(Honesty card_one, Scenario scenario) -> bool {
	return with_loops_ending(card_one, Ending::Clean, std::move(scenario));
}

[[nodiscard]] auto random_mask(std::mt19937_64& random) -> std::vector<decode::Token> {
	auto tokens = std::vector<decode::Token>(1 + (random() % 3));
	std::ranges::generate(tokens, [&] { return static_cast<decode::Token>(random() % model::sampled_vocabulary); });
	return tokens;
}

/** One cycle's masks: the allowed tokens per mask index, and the masks they make. */
struct CycleMasks {
	std::vector<std::vector<decode::Token>> allowed;
	std::vector<RowMasks> masks;

	[[nodiscard]] auto operator()(std::size_t index, Card card) const -> decode::CardMask {
		return masks.at(index).at(std::to_underlying(card));
	}
};

[[nodiscard]] auto masks_for(std::size_t count, std::mt19937_64& random) -> CycleMasks {
	auto made = CycleMasks{};
	for (auto index = std::size_t{0}; index < count; ++index) {
		made.allowed.push_back(random_mask(random));
		made.masks.push_back(mask_of(made.allowed.back()));
	}
	return made;
}

/** A batch of the slots in `live`, each row continuing what the cards hold for it when `continuing`, else fresh. */
[[nodiscard]] auto batch_of(decode::Session const& session, std::span<std::size_t const> live, std::mt19937_64& random) -> sched::Batch {
	auto const depth = engine::schedule::shape(live.size(), dd::budget);
	auto batch = sched::Batch{.depth = depth, .rows = {}};
	for (auto const index : live) {
		auto const slot = kv::Slot::at(index);
		auto const& held = session.ledger().held(slot);
		auto const continuing = held.has_value() && random() % 4 != 0;
		batch.rows.push_back(sched::Row{
		    .slot = slot,
		    .request = continuing ? held->request : sched::RequestHandle{.bits = index},
		    .position = continuing ? held->position : 64 + (random() % 4096),
		    .token = continuing ? held->token : static_cast<decode::Token>(random() % 1000),
		    .rule = rule(),
		    .most = 1 + model::drafts(depth),
		});
	}
	return batch;
}

/**
 * Random runs of cycles: each row's outcome is what the fake loop derives from the masks the host published (so the
 * masks reached both cards in their row orders), the drafts a continuing row verifies are the ones the cards proposed
 * for it, and its replays are the drafts its last cycle accepted.
 */
[[nodiscard]] auto cycles_run_and_settle(std::uint64_t seed) -> bool {
	return with_loops(Honesty::Truthful, [seed](decode::Session& session, std::array<Seen, 2>& seen) {
		auto random = engine::testing::generator(seed);
		for (auto cycle = 0; cycle < 40; ++cycle) {
			auto live = std::vector<std::size_t>{};
			for (auto index = std::size_t{0}; index < model::slots; ++index) {
				if (random() % 2 == 0) {
					live.push_back(index);
				}
			}
			if (live.empty()) {
				live.push_back(random() % model::slots);
			}
			auto const batch = batch_of(session, live, random);
			auto expected_drafts = std::vector<std::vector<decode::Token>>{};
			auto expected_replays = std::vector<std::uint32_t>{};
			for (auto const& row : batch.rows) {
				auto const& held = session.ledger().held(row.slot);
				auto const continues = held && held->request == row.request && held->position == row.position && held->token == row.token;
				auto const count = continues ? std::min<std::size_t>(held->drafted, model::drafts(batch.depth)) : 0;
				expected_drafts.emplace_back();
				if (continues) {
					expected_drafts.back() = std::vector<decode::Token>(std::from_range, std::span{held->drafts}.first(count));
				}
				expected_replays.push_back(continues ? held->state.pending : 0U);
			}
			auto pending = session.begin(batch);
			if (!pending) {
				return false;
			}
			for (auto row = std::size_t{0}; row < batch.rows.size(); ++row) {
				if (!std::ranges::equal(pending->drafts(row), expected_drafts[row])) {
					return false;
				}
			}
			auto const masks = masks_for(pending->mask_count(), random);
			if (!session.publish(*pending, masks, masks.masks.size())) {
				return false;
			}
			auto const width = 1 + model::drafts(batch.depth);
			auto const rows = batch.rows;
			auto planned = pending->planned;
			auto done = settled([&] { return session.try_complete(*pending); });
			if (!done || done->rows.size() != rows.size() || done->took != std::chrono::nanoseconds{500}) {
				return false;
			}
			auto const& descriptor = seen[0].cycles.back();
			for (auto row = std::size_t{0}; row < rows.size(); ++row) {
				auto const drafts = planned[row].run.drafts;
				auto const accepted = rows[row].position % (drafts + 1U);
				auto const& got = done->rows[row];
				if (got.count != accepted + 1 || std::span{descriptor.slots}[row].replays != expected_replays[row]) {
					return false;
				}
				for (auto verify = std::size_t{0}; verify < got.count; ++verify) {
					if (got.tokens.at(verify) != std::ranges::min(masks.allowed.at((row * width) + verify))) {
						return false;
					}
				}
			}
		}
		return seen[0].cycles.size() == 40 && seen[1].cycles.size() == 40;
	});
}

/** A card answering differently from its peer fails the cycle rather than settling either answer. */
[[nodiscard]] auto cards_must_agree() -> bool {
	return with_loops(Honesty::Lying, [](decode::Session& session, std::array<Seen, 2>& /*seen*/) {
		auto random = engine::testing::generator(5);
		auto const live = std::array{std::size_t{2}};
		auto pending = session.begin(batch_of(session, live, random));
		if (!pending) {
			return false;
		}
		auto const masks = masks_for(pending->mask_count(), random);
		if (!session.publish(*pending, masks, masks.masks.size())) {
			return false;
		}
		auto const ticket = pending->ticket;
		auto done = settled([&] { return session.try_complete(*pending); });
		auto const disagree = done ? std::nullopt : engine::core::alternative<decode::CardsDisagree>(done.error().reason);
		// Card one claims one more accepted draft than card zero.
		return disagree && disagree->ticket == ticket && disagree->slot == 2 && disagree->what == decode::Disagreement::Accepted &&
		       disagree->one == disagree->zero + 1 && !session.ledger().held(kv::Slot::at(2));
	});
}

[[nodiscard]] auto refused(auto const& result, decode::Refusal reason) -> bool {
	auto const refusal = result ? std::nullopt : engine::core::alternative<decode::Refused>(result.error().reason);
	return refusal && refusal->reason == reason;
}

/** Masks for another ticket, of the wrong count, or allowing nothing are refused, and the cycle still completes. */
[[nodiscard]] auto bad_masks_are_refused() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& /*seen*/) {
		auto random = engine::testing::generator(6);
		auto const live = std::array{std::size_t{0}, std::size_t{5}};
		auto pending = session.begin(batch_of(session, live, random));
		if (!pending) {
			return false;
		}
		auto masks = masks_for(pending->mask_count(), random);
		auto const stale = decode::Pending{.ticket = pending->ticket + 1, .work = pending->work, .depth = pending->depth, .width = pending->width, .rows = pending->rows, .planned = pending->planned};
		auto const wrong_ticket = refused(session.publish(stale, masks, masks.masks.size()), decode::Refusal::WrongTicket);
		auto const wrong_count = refused(session.publish(*pending, masks, masks.masks.size() - 1), decode::Refusal::MaskCount);
		auto const kept = masks.masks.front();
		masks.masks.front() = RowMasks{};
		auto const empty = refused(session.publish(*pending, masks, masks.masks.size()), decode::Refusal::EmptyMask);
		masks.masks.front() = kept;
		auto const published = session.publish(*pending, masks, masks.masks.size()).has_value();
		auto const done = settled([&] { return session.try_complete(*pending); }).has_value();
		return wrong_ticket && wrong_count && empty && published && done;
	});
}

/** Batches the cards cannot run are refused before anything is posted. */
[[nodiscard]] auto bad_batches_are_refused() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto const row = [](std::size_t slot, std::uint64_t position) {
			return sched::Row{.slot = kv::Slot::at(slot), .request = {.bits = slot}, .position = position, .token = 1, .rule = rule(), .most = 1};
		};
		auto const empty = refused(session.begin(sched::Batch{.depth = model::DraftDepth::One, .rows = {}}), decode::Refusal::NoRows);
		auto const order = refused(session.begin(sched::Batch{.depth = model::DraftDepth::One, .rows = {row(3, 10), row(1, 10)}}),
		                           decode::Refusal::SlotOrder);
		auto const twice = refused(session.begin(sched::Batch{.depth = model::DraftDepth::One, .rows = {row(3, 10), row(3, 10)}}),
		                           decode::Refusal::SlotOrder);
		auto rows = std::vector<sched::Row>{};
		for (auto slot = std::size_t{0}; slot < model::slots; ++slot) {
			rows.push_back(row(slot, 10));
		}
		auto const budget = refused(session.begin(sched::Batch{.depth = model::DraftDepth::Four, .rows = rows}), decode::Refusal::OverBudget);
		auto const past = refused(session.begin(sched::Batch{.depth = model::DraftDepth::One, .rows = {row(0, model::max_positions - 2)}}),
		                          decode::Refusal::PastContext);
		return empty && order && twice && budget && past && seen[0].cycles.empty() && seen[1].cycles.empty();
	});
}

/**
 * The pool's commands reach both cards in order before the cycle: whole tickets of commands go ahead, the rest ride
 * with the cycle; a loaded snapshot ends what its slot carried on, so its next row verifies no drafts.
 */
[[nodiscard]] auto commands_go_ahead() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto random = engine::testing::generator(8);
		auto const live = std::array{std::size_t{4}};
		// A first cycle, so slot 4 holds drafts.
		auto first = session.begin(batch_of(session, live, random));
		if (!first) {
			return false;
		}
		auto const first_masks = masks_for(first->mask_count(), random);
		if (!session.publish(*first, first_masks, first_masks.masks.size()) || !settled([&] { return session.try_complete(*first); })) {
			return false;
		}
		auto lowered = std::vector<dd::Command>{};
		for (auto index = 0; index < 700; ++index) {
			auto const command = decode::PoolCommand{kv::SetTableEntry{
			    .slot = kv::Slot::at(static_cast<std::size_t>(index) % model::slots), .entry = static_cast<std::uint64_t>(index), .block = 3}};
			session.apply(command);
			std::ranges::copy(decode::lower(command), std::back_inserter(lowered));
		}
		auto load = kv::LoadSnapshot<kv::model_geometry>{.slot = kv::Slot::at(4), .blocks = {}};
		std::ranges::fill(load.blocks, 9);
		session.apply(load);
		std::ranges::copy(decode::lower(load), std::back_inserter(lowered));
		auto const held = session.ledger().held(kv::Slot::at(4));
		auto pending = session.begin(batch_of(session, live, random));
		if (!pending || held.has_value() || pending->planned.front().run.drafts != 0) {
			return false;
		}
		auto const masks = masks_for(pending->mask_count(), random);
		if (!session.publish(*pending, masks, masks.masks.size()) || !settled([&] { return session.try_complete(*pending); })) {
			return false;
		}
		for (auto const& card : seen) {
			auto arrived = std::vector<dd::Command>{};
			for (auto const& ticket : card.applied) {
				arrived.insert(arrived.end(), ticket.begin(), ticket.end());
			}
			auto const& cycle = card.cycles.back();
			auto const riding = std::span{cycle.command}.first(cycle.commands);
			arrived.insert(arrived.end(), riding.begin(), riding.end());
			auto const same = std::ranges::equal(arrived, lowered, [](dd::Command const& left, dd::Command const& right) {
				return left.action == right.action && left.slot == right.slot && left.first == right.first && left.second == right.second;
			});
			if (!same || card.applied.size() != 1 || card.applied.front().size() != dd::max_commands) {
				return false;
			}
		}
		return true;
	});
}

/**
 * A prompt's end samples its one row under its one mask and drafts from it; the slot then continues from the sampled
 * token, its first cycle verifying those drafts with nothing to replay.
 */
[[nodiscard]] auto prompt_end_starts_decoding() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto random = engine::testing::generator(9);
		auto const prompt = sched::Row{.slot = kv::Slot::at(6), .request = {.bits = 3}, .position = 200, .token = 17, .rule = rule(), .most = 1};
		auto pending = session.begin_prompt_end(prompt, model::DraftDepth::Three);
		if (!pending || pending->mask_count() != 1 || !pending->drafts(0).empty()) {
			return false;
		}
		// The loops wait for the masks before they rank, so nothing can be answered yet: the poll says so and returns.
		if (session.try_complete_prompt_end(*pending).has_value()) {
			return false;
		}
		auto const masks = masks_for(1, random);
		if (!session.publish(*pending, masks, 1)) {
			return false;
		}
		auto const done = settled([&] { return session.try_complete_prompt_end(*pending); });
		auto const sampled = std::ranges::min(masks.allowed.front());
		auto const& ran = seen[0].cycles.back();
		if (!done || done->sampled != sampled || ran.work != dd::Work::PromptEnd || ran.live != 1 || ran.slots[0].drafts != 0 ||
		    ran.slots[0].position != 200) {
			return false;
		}
		auto const next = sched::Row{.slot = prompt.slot, .request = prompt.request, .position = 201, .token = sampled, .rule = rule(), .most = 4};
		auto cycle = session.begin(sched::Batch{.depth = model::DraftDepth::Three, .rows = {next}});
		if (!cycle) {
			return false;
		}
		auto const drafted = std::array{sampled + 1, sampled + 2, sampled + 3};
		auto const continued = std::ranges::equal(cycle->drafts(0), drafted) && cycle->planned.front().run.replays == 0;
		auto const cycle_masks = masks_for(cycle->mask_count(), random);
		auto const finished = session.publish(*cycle, cycle_masks, cycle_masks.masks.size()).has_value() &&
		                      settled([&] { return session.try_complete(*cycle); }).has_value();
		return continued && finished;
	});
}

/** Runs cycles of one slot until its stored state has drafts pending; whether it got there. */
[[nodiscard]] auto run_until_pending(decode::Session& session, std::size_t slot, std::mt19937_64& random) -> bool {
	auto const live = std::array{slot};
	for (auto cycle = 0; cycle < 50; ++cycle) {
		auto const& held = session.ledger().held(kv::Slot::at(slot));
		if (held && held->state.pending > 0) {
			return true;
		}
		auto batch = batch_of(session, live, random);
		if (held) {
			// Continue the slot, so its drafts are verified and some accepted.
			batch.rows.front().request = held->request;
			batch.rows.front().position = held->position;
			batch.rows.front().token = held->token;
		}
		auto pending = session.begin(batch);
		if (!pending) {
			return false;
		}
		auto const masks = masks_for(pending->mask_count(), random);
		if (!session.publish(*pending, masks, masks.masks.size()) || !settled([&] { return session.try_complete(*pending); })) {
			return false;
		}
	}
	return false;
}

/**
 * A snapshot of a slot with drafts pending in its replay ring is preceded by their replay, alone in its wave, so the
 * snapshot holds the state after every token but the last; the slot then has nothing pending.
 */
[[nodiscard]] auto snapshots_take_the_pending_drafts() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto random = engine::testing::generator(10);
		if (!run_until_pending(session, 1, random)) {
			return false;
		}
		auto const pending = session.ledger().held(kv::Slot::at(1))->state.pending;
		auto save = kv::SaveSnapshot<kv::model_geometry>{.slot = kv::Slot::at(1), .blocks = {}};
		std::ranges::fill(save.blocks, 5);
		session.apply(save);
		if (session.ledger().held(kv::Slot::at(1))->state.pending != 0 || !settled([&] { return session.try_stop(); })) {
			return false;
		}
		auto const& applied = seen[0].applied.back();
		auto const replay = applied.front();
		return replay.action == dd::Action::Replay && replay.slot == 1 && replay.first == pending && replay.wave == 0 &&
		       applied.size() == 1 + save.blocks.size() && applied[1].action == dd::Action::SaveSnapshotPart && applied[1].wave == 1;
	});
}


/** Queues `count` table entries for the pool and returns them as the cards should see them, in order. */
[[nodiscard]] auto queue_entries(decode::Session& session, int count, std::uint64_t first) -> std::vector<dd::Command> {
	auto lowered = std::vector<dd::Command>{};
	for (auto index = 0; index < count; ++index) {
		auto const command = decode::PoolCommand{kv::SetTableEntry{
		    .slot = kv::Slot::at(static_cast<std::size_t>(index) % model::slots), .entry = first + static_cast<std::uint64_t>(index), .block = 2}};
		session.apply(command);
		std::ranges::copy(decode::lower(command), std::back_inserter(lowered));
	}
	return lowered;
}

/** Whether both cards applied exactly `lowered`, in order, in command-only tickets. */
[[nodiscard]] auto applied_exactly(std::array<Seen, 2> const& seen, std::span<dd::Command const> lowered) -> bool {
	return std::ranges::all_of(seen, [&](Seen const& card) {
		auto arrived = std::vector<dd::Command>{};
		for (auto const& ticket : card.applied) {
			arrived.insert(arrived.end(), ticket.begin(), ticket.end());
		}
		return std::ranges::equal(arrived, lowered, [](dd::Command const& left, dd::Command const& right) {
			return left.action == right.action && left.slot == right.slot && left.first == right.first && left.second == right.second;
		});
	});
}

/** Runs one cycle of `slots` to its outcome. */
[[nodiscard]] auto run_cycle(decode::Session& session, std::span<std::size_t const> slots, std::mt19937_64& random) -> bool {
	auto pending = session.begin(batch_of(session, slots, random));
	if (!pending) {
		return false;
	}
	auto const masks = masks_for(pending->mask_count(), random);
	return session.publish(*pending, masks, masks.masks.size()).has_value() && settled([&] { return session.try_complete(*pending); }).has_value();
}

/**
 * Commands queued before a stop reach both cards before the loops exit, in order, in command-only tickets: a prefill
 * that follows the stop sees every block table entry the scheduler set.
 */
[[nodiscard]] auto stops_apply_queued_commands() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto const lowered = queue_entries(session, 700, 0);
		auto const stopped = settled([&] { return session.try_stop(); });
		return stopped.has_value() && applied_exactly(seen, lowered) && seen[0].applied.size() == 2 && seen[0].cycles.empty();
	});
}

/** A shutdown from idle loops sends the queued commands, then the exits. */
[[nodiscard]] auto shutdown_from_idle() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto random = engine::testing::generator(11);
		auto const live = std::array{std::size_t{7}};
		if (!run_cycle(session, live, random)) {
			return false;
		}
		auto const lowered = queue_entries(session, 40, 100);
		auto const ended = settled([&] { return session.try_shutdown(deadline_after()); });
		return ended.has_value() && applied_exactly(seen, lowered) && seen[1].cycles.size() == 1;
	});
}

/**
 * A shutdown with a cycle begun and its masks never published releases masks allowing everything, so the cycle runs
 * out; its outcome is discarded and its slots forgotten, and the commands queued meanwhile still reach the cards.
 */
[[nodiscard]] auto shutdown_from_a_begun_cycle() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		auto random = engine::testing::generator(12);
		auto const live = std::array{std::size_t{1}, std::size_t{3}};
		if (!run_cycle(session, live, random) || !session.ledger().held(kv::Slot::at(1))) {
			return false;
		}
		auto const pending = session.begin(batch_of(session, live, random));
		if (!pending) {
			return false;
		}
		auto const lowered = queue_entries(session, 30, 200);
		auto const ended = settled([&] { return session.try_shutdown(deadline_after()); });
		return ended.has_value() && seen[0].cycles.size() == 2 && seen[1].cycles.size() == 2 && applied_exactly(seen, lowered) &&
		       !session.ledger().held(kv::Slot::at(1)) && !session.ledger().held(kv::Slot::at(3));
	});
}

/** A shutdown of stopped loops succeeds at once and sends nothing. */
[[nodiscard]] auto shutdown_from_stopped() -> bool {
	return with_loops(Honesty::Truthful, [](decode::Session& session, std::array<Seen, 2>& seen) {
		if (!settled([&] { return session.try_stop(); })) {
			return false;
		}
		auto const ended = session.try_shutdown(std::chrono::steady_clock::now());
		return ended.has_value() && ended->has_value() && seen[0].applied.empty() && seen[0].cycles.empty();
	});
}

/**
 * A shutdown that finds the cards disagreeing on the begun cycle fails with what they disagreed on: the queued commands
 * are not sent to suspect cards, the exits still are, and every later shutdown returns the same failure.
 */
[[nodiscard]] auto shutdown_fails_on_disagreeing_cards() -> bool {
	return with_loops_ending(
	    Honesty::Lying, Ending::Failed,
	    [](decode::Session& session, std::array<Seen, 2>& seen) {
		    auto random = engine::testing::generator(13);
		    auto const live = std::array{std::size_t{5}};
		    auto pending = session.begin(batch_of(session, live, random));
		    if (!pending) {
			    return false;
		    }
		    auto const masks = masks_for(pending->mask_count(), random);
		    if (!session.publish(*pending, masks, masks.masks.size())) {
			    return false;
		    }
		    std::ignore = queue_entries(session, 20, 300);
		    auto const ended = settled([&] { return session.try_shutdown(deadline_after()); });
		    auto const disagree = ended ? std::nullopt : engine::core::alternative<decode::CardsDisagree>(ended.error().reason);
		    auto const again = session.try_shutdown(std::chrono::steady_clock::now());
		    return disagree && disagree->ticket == pending->ticket && disagree->slot == 5 && again && !*again && again->error() == ended.error() &&
		           seen[0].applied.empty() && seen[1].applied.empty() && !session.ledger().held(kv::Slot::at(5));
	    });
}

/** Loops that never answer are abandoned at the shutdown's deadline, naming the ticket that went unanswered. */
[[nodiscard]] auto silent_loops_are_abandoned() -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto devices = device::Devices::open(driver.driver());
	if (!devices) {
		return false;
	}
	auto session = decode::Session::create(*devices);
	if (!session) {
		return false;
	}
	auto random = engine::testing::generator(14);
	auto const live = std::array{std::size_t{0}};
	auto pending = session->begin(batch_of(*session, live, random));
	if (!pending) {
		return false;
	}
	auto const ended = session->try_shutdown(std::chrono::steady_clock::now());
	if (!ended || *ended || ended->error() != decode::DecodeError{.reason = decode::Abandoned{.ticket = pending->ticket, .kernels = decode::Kernels::MayRun}}) {
		return false;
	}
	// Abandoned loops take nothing more: a later batch fails with the same failure.
	auto const later = session->begin(batch_of(*session, live, random));
	return !later && later.error() == ended->error();
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "cycles settle from the published masks and the cards' drafts (seed 1)", .holds = cycles_run_and_settle(1)},
	    Case{.name = "cycles settle from the published masks and the cards' drafts (seed 2)", .holds = cycles_run_and_settle(2)},
	    Case{.name = "cards that disagree fail the cycle", .holds = cards_must_agree()},
	    Case{.name = "masks for another ticket, of the wrong count, or allowing nothing are refused", .holds = bad_masks_are_refused()},
	    Case{.name = "batches the cards cannot run are refused before anything is posted", .holds = bad_batches_are_refused()},
	    Case{.name = "pool commands reach the cards in order before the cycle", .holds = commands_go_ahead()},
	    Case{.name = "a prompt's end samples under its mask and its drafts start the slot's decoding", .holds = prompt_end_starts_decoding()},
	    Case{.name = "a snapshot is preceded by the replay of the slot's pending drafts", .holds = snapshots_take_the_pending_drafts()},
	    Case{.name = "a stop applies the queued commands before the loops exit", .holds = stops_apply_queued_commands()},
	    Case{.name = "a shutdown from idle loops sends the queued commands, then the exits", .holds = shutdown_from_idle()},
	    Case{.name = "a shutdown runs out a begun cycle with unpublished masks and forgets its slots", .holds = shutdown_from_a_begun_cycle()},
	    Case{.name = "a shutdown of stopped loops succeeds at once", .holds = shutdown_from_stopped()},
	    Case{.name = "a shutdown that finds the cards disagreeing fails, skips the commands and still exits", .holds = shutdown_fails_on_disagreeing_cards()},
	    Case{.name = "loops that never answer are abandoned at the deadline, and take nothing more", .holds = silent_loops_are_abandoned()},
	};
	return engine::testing::report(cases);
}

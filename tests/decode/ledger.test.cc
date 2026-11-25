// SPDX-License-Identifier: AGPL-3.0-only
//
// The ledger of what the cards hold per slot, over random runs of cycles against a model of the protocol: a row that
// continues its slot (same request, at the position and token the last cycle left) verifies the drafts the cards
// drafted for it, up to the cycle's depth, and replays the drafts the last cycle accepted; any other row, and any row
// after its slot's state was loaded from a snapshot, verifies none and replays none. The outcome's tokens are the
// row's accepted tokens, and the slot moves past them.
import std;
import engine.decode;
import engine.decode_device;
import engine.kv;
import engine.model;
import engine.sampling;
import engine.sched;
import engine.testing;
import engine.tokenizer;

namespace {

namespace decode = engine::decode;
namespace dd = engine::decode_device;
namespace kv = engine::kv;
namespace model = engine::model;
namespace sched = engine::sched;

constexpr auto depths = std::array{model::DraftDepth::One, model::DraftDepth::Two, model::DraftDepth::Three, model::DraftDepth::Four};

[[nodiscard]] constexpr auto rule() -> engine::sampling::Rule {
	return {.seed = 7,
	        .filter = {.temperature = 1.0, .top_p = 1.0, .min_p = 0.0, .top_k = 1, .nucleus = engine::sampling::Nucleus::Off},
	        .request = 1};
}

/** The tokens that end a turn, as the model's generation config lists them. */
constexpr auto ends = std::array{engine::tokenizer::id(engine::tokenizer::Added::im_end), engine::tokenizer::id(engine::tokenizer::Added::endoftext)};

/** The model: what the cards hold for one slot, as the protocol says, kept apart from the ledger. */
struct Held {
	sched::RequestHandle request;
	std::uint64_t position;
	decode::Token token;
	std::uint32_t drafted;
	std::uint32_t accepted;
	decode::DraftRow drafts;
};

/** A run of cycles on one slot: continuing rows, fresh rows and snapshot loads, in random order. */
[[nodiscard]] auto run(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto ledger = decode::Ledger{};
	auto const slot = kv::Slot::at(random() % model::slots);
	auto held = std::optional<Held>{};
	for (auto step = 0; step < 2'000; ++step) {
		auto const choice = random() % 10;
		if (choice == 0) {
			ledger.forget(slot);
			held = std::nullopt;
			continue;
		}
		// Before a snapshot the pending drafts are replayed: the stored state then covers every token but the last.
		if (choice == 1) {
			auto const pending = ledger.materialize(slot);
			auto const expected = held.has_value() && held->accepted > 0 ? std::optional{held->accepted} : std::nullopt;
			if (pending != expected) {
				return false;
			}
			if (held) {
				held->accepted = 0;
				auto const& kept = ledger.held(slot);
				if (!kept || kept->state.pending != 0 || kept->state.committed != held->position) {
					return false;
				}
			}
			continue;
		}
		auto const depth = std::span{depths}[random() % depths.size()];
		auto const continuing = held.has_value() && choice > 2;
		auto const row = sched::Row{
		    .slot = slot,
		    .request = continuing ? held->request : sched::RequestHandle{.bits = random() % 4},
		    .position = continuing ? held->position : 100 + (random() % 1000),
		    .token = continuing ? held->token : static_cast<decode::Token>(random() % 1000),
		    .rule = rule(),
		    .most = 1 + (random() % (1 + model::drafts(depth))),
		};
		// A fresh row may still match what the slot holds by chance; the model decides by the same facts.
		auto const continues = held.has_value() && held->request == row.request && held->position == row.position && held->token == row.token;
		auto const planned = ledger.plan(row, depth);
		auto const drafts = continues ? std::min<std::uint32_t>(held->drafted, static_cast<std::uint32_t>(model::drafts(depth))) : 0U;
		auto const replays = continues ? held->accepted : 0U;
		if (planned.run.drafts != drafts || planned.run.replays != replays || planned.run.position != row.position || planned.run.token != row.token ||
		    planned.run.slot != slot.index()) {
			return false;
		}
		if (continues && !std::ranges::equal(std::span{planned.drafts}.first(drafts), std::span{held->drafts}.first(drafts))) {
			return false;
		}
		auto outcome = dd::SlotOutcome{};
		outcome.accepted = static_cast<unsigned>(random() % (drafts + 1));
		// Now and then a token ends the turn.
		std::ranges::generate(outcome.committed, [&] {
			auto const pick = random() % 12;
			return pick == 0 ? ends[0] : pick == 1 ? ends[1] : static_cast<unsigned>(random() % 1000);
		});
		std::ranges::generate(outcome.drafts, [&] { return static_cast<unsigned>(random() % 1000); });
		auto const accepted = ledger.settle(row, planned, outcome, depth);
		// The row's limit and its first end-of-turn token cut what it commits; a cut row's drafts continue a sequence it
		// does not have.
		auto count = std::min<std::size_t>(outcome.accepted + 1, row.most);
		for (auto index = std::size_t{0}; index < count; ++index) {
			if (outcome.committed[index] == ends[0] || outcome.committed[index] == ends[1]) {
				count = index + 1;
			}
		}
		auto const cut = count < outcome.accepted + 1;
		if (accepted.count != count || !std::ranges::equal(accepted.view(), std::span{outcome.committed}.first(accepted.count))) {
			return false;
		}
		held = Held{
		    .request = row.request,
		    .position = row.position + count,
		    .token = outcome.committed[count - 1],
		    .drafted = cut ? 0U : static_cast<std::uint32_t>(model::drafts(depth)),
		    .accepted = static_cast<std::uint32_t>(count - 1),
		    .drafts = {},
		};
		std::ranges::copy(std::span{outcome.drafts}.first(held->drafted), held->drafts.begin());
		auto const& kept = ledger.held(slot);
		if (!kept || kept->position != held->position || kept->token != held->token || kept->drafted != held->drafted ||
		    kept->state.pending != held->accepted || kept->state.committed + kept->state.pending != held->position) {
			return false;
		}
	}
	return true;
}

/** A slot never seen starts fresh: no drafts, nothing to replay, its state covering exactly its position. */
[[nodiscard]] auto first_row_is_fresh() -> bool {
	auto const ledger = decode::Ledger{};
	auto const row = sched::Row{.slot = kv::Slot::at(3), .request = {.bits = 9}, .position = 41, .token = 5, .rule = rule(), .most = 5};
	auto const planned = ledger.plan(row, model::DraftDepth::Four);
	return planned.run.drafts == 0 && planned.run.replays == 0 && planned.state.committed == 41 && planned.state.rows == 1;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "a slot never seen starts with no drafts and nothing to replay", .holds = first_row_is_fresh()},
	    Case{.name = "plans and settlements follow the protocol's model (seed 1)", .holds = run(1)},
	    Case{.name = "plans and settlements follow the protocol's model (seed 2)", .holds = run(2)},
	    Case{.name = "plans and settlements follow the protocol's model (seed 3)", .holds = run(3)},
	};
	return engine::testing::report(cases);
}

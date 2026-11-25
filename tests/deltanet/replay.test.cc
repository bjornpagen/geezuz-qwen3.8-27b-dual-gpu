// SPDX-License-Identifier: AGPL-3.0-only
//
// Lazy replay against its oracle: a slot that runs its sequence one token per cycle, with no drafts. Under random
// draft depths and verdicts, with rejected drafts that are tokens the sequence never holds, the replaying slot's stored
// state and window must equal the plain slot's at every committed position, bit for bit, and every row the verdict
// accepted must have produced the plain slot's output for its position.
import std;
import engine.deltanet;
import engine.deltanet_fixtures;
import engine.testing;

namespace {

namespace dn = engine::deltanet;
namespace fx = engine::deltanet_fixtures;

/** The plain slot: every token of `sequence` in its own cycle; returns each position's outputs. */
[[nodiscard]] auto plain_outputs(fx::WeightsStore const& weights, dn::LayerBuffers start, std::span<fx::Token const> sequence)
    -> std::vector<std::vector<dn::Bf16>> {
	auto outputs = std::vector<std::vector<dn::Bf16>>{};
	auto settled = dn::settled({.tokens = 0});
	for (auto const& token : sequence) {
		auto rows = fx::RowsStore{};
		rows.push(token);
		auto const running = dn::run(settled, 0);
		dn::decode(weights.view(), start.view(), rows.view(), dn::cycle_work(running, 0, 0));
		settled = dn::settle(running, 0);
		auto const out = rows.output(0);
		outputs.emplace_back(out.begin(), out.end());
	}
	return outputs;
}

[[nodiscard]] auto replay_matches_plain(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto const weights = fx::random_weights(generator);
	auto const start = fx::random_layer(generator);
	constexpr auto positions = 48;
	auto sequence = std::vector<fx::Token>{};
	for (auto position = 0; position < positions + dn::max_rows; ++position) {
		sequence.push_back(fx::random_token(generator));
	}
	auto const expected = plain_outputs(weights, start, sequence);

	auto lazy = start;
	auto plain = start;
	auto plain_position = std::uint64_t{0};
	auto settled = dn::settled({.tokens = 0});
	auto holds = true;
	auto cycles = 0;
	while (dn::stored(settled).tokens + settled.pending < positions) {
		auto const drafts = std::uniform_int_distribution<std::uint32_t>{0, dn::max_drafts}(generator);
		auto const running = dn::run(settled, drafts);
		auto const first = dn::first_position(running);
		auto const accepted = std::uniform_int_distribution<std::uint32_t>{0, drafts}(generator);
		auto rows = fx::RowsStore{};
		for (auto row = std::uint32_t{0}; row < running.rows; ++row) {
			// Rows past the accepted drafts are rejected: tokens the sequence does not hold there.
			rows.push(row <= accepted ? sequence.at(first + row) : fx::random_token(generator));
		}
		dn::decode(weights.view(), lazy.view(), rows.view(), dn::cycle_work(running, 3, 0));
		for (auto row = std::uint32_t{0}; row <= accepted; ++row) {
			holds = holds && fx::same_bits(rows.output(row), expected.at(first + row));
		}
		settled = dn::settle(running, accepted);
		while (plain_position < settled.committed) {
			auto plain_rows = fx::RowsStore{};
			plain_rows.push(sequence.at(plain_position));
			dn::decode(weights.view(), plain.view(), plain_rows.view(), {.slot = 0, .replays = 0, .rows = 1, .first_row = 0});
			++plain_position;
		}
		holds = holds && fx::same_stored(lazy, plain);
		++cycles;
	}
	return holds && cycles > 0;
}

/** The ledger's arithmetic: each cycle stores its replays and first row, and leaves its accepted drafts pending. */
[[nodiscard]] auto ledger_advances_by_replays_and_first_row() -> bool {
	auto holds = true;
	for (auto pending = std::uint32_t{0}; pending <= dn::max_replays; ++pending) {
		for (auto drafts = std::uint32_t{0}; drafts <= dn::max_drafts; ++drafts) {
			for (auto accepted = std::uint32_t{0}; accepted <= drafts; ++accepted) {
				auto const slot = dn::Settled{.committed = 100, .pending = pending};
				auto const running = dn::run(slot, drafts);
				auto const next = dn::settle(running, accepted);
				holds = holds && running.replays == pending && running.rows == drafts + 1 && dn::first_position(running) == 100 + pending &&
				        next.committed == 100 + pending + 1 && next.pending == accepted &&
				        dn::first_position(dn::run(next, 0)) == dn::first_position(running) + accepted + 1 &&
				        dn::stored(next).tokens == next.committed;
			}
		}
	}
	auto const chunk = dn::prompt_chunk({.tokens = 7}, 5, 2, 9);
	return holds && chunk.after.tokens == 12 && chunk.work.tokens == 5 && chunk.work.slot == 2 && chunk.work.first_row == 9;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the ledger stores each cycle's replays and first row, and holds its accepted drafts",
		                      .holds = ledger_advances_by_replays_and_first_row()},
	    engine::testing::Case{.name = "lazy replay equals the plain sequence, seed 1", .holds = replay_matches_plain(1)},
	    engine::testing::Case{.name = "lazy replay equals the plain sequence, seed 2", .holds = replay_matches_plain(2)},
	};
	return engine::testing::report(cases);
}

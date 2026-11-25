// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.prims;
import engine.testing;

namespace {

namespace prims = engine::device::prims;

/**
 * An mbarrier as the isa defines it: phase k completes when its expected arrivals are in, then the next phase begins
 * with the same count; `test_wait.parity p` holds exactly when the current phase's parity is not p.
 */
struct ModelBarrier {
	unsigned expected;
	unsigned pending;
	unsigned phase;

	void arrive() {
		pending -= 1;
		if (pending == 0) {
			phase += 1;
			pending = expected;
		}
	}

	[[nodiscard]] auto test(unsigned parity) const -> bool {
		return phase % 2 != parity;
	}
};

template<int stages>
[[nodiscard]] auto cursor_counts_stages_and_wraps() -> bool {
	using Cursor = prims::StageCursor<stages>;
	auto cursor = Cursor::start();
	auto holds = true;
	for (auto n = 0U; n < 1000; ++n) {
		holds = holds && cursor.stage == n % stages && cursor.parity == (n / stages) % 2;
		cursor = cursor.next();
	}
	return holds;
}

/**
 * One producer and `consumers` consumers pass `items` items through the ring under a random interleaving, every
 * wait a parity test on the model barrier and every move as the device ring makes it. The consumers must read each
 * item exactly once and in order, the producer must never refill a stage a consumer has yet to read, and the run
 * must never deadlock.
 */
template<int stages>
[[nodiscard]] auto ring_protocol_is_safe_and_live(unsigned consumers, std::uint64_t seed) -> bool {
	using Cursor = prims::StageCursor<stages>;
	constexpr auto items = 20'000U;
	auto full = std::vector<ModelBarrier>(stages, ModelBarrier{.expected = 1, .pending = 1, .phase = 0});
	auto empty = std::vector<ModelBarrier>(stages, ModelBarrier{.expected = consumers, .pending = consumers, .phase = 0});
	auto slot = std::vector<unsigned>(stages, 0);
	auto unread = std::vector<unsigned>(stages, 0);
	auto producer = Cursor::start();
	auto produced = 0U;
	auto readers = std::vector<Cursor>(consumers, Cursor::start());
	auto consumed = std::vector<unsigned>(consumers, 0);
	auto random = engine::testing::generator(seed);
	auto holds = true;
	while (std::ranges::any_of(consumed, [](unsigned count) { return count < items; })) {
		auto ready = std::vector<unsigned>{};
		if (produced < items && empty.at(producer.stage).test(producer.empty_parity())) {
			ready.push_back(consumers);
		}
		for (auto consumer = 0U; consumer < consumers; ++consumer) {
			auto const at = readers.at(consumer);
			if (consumed.at(consumer) < items && full.at(at.stage).test(at.full_parity())) {
				ready.push_back(consumer);
			}
		}
		if (ready.empty()) {
			return false;
		}
		auto const actor = ready.at(static_cast<std::size_t>(random() % ready.size()));
		if (actor == consumers) {
			holds = holds && unread.at(producer.stage) == 0;
			slot.at(producer.stage) = produced;
			unread.at(producer.stage) = consumers;
			full.at(producer.stage).arrive();
			producer = producer.next();
			produced += 1;
		} else {
			auto& at = readers.at(actor);
			holds = holds && slot.at(at.stage) == consumed.at(actor) && unread.at(at.stage) > 0;
			unread.at(at.stage) -= 1;
			empty.at(at.stage).arrive();
			at = at.next();
			consumed.at(actor) += 1;
		}
	}
	return holds && produced == items;
}

/** Waiting on the stage's current parity for the empty barrier, the slip the cursor exists to prevent, deadlocks. */
[[nodiscard]] auto waiting_on_the_wrong_parity_deadlocks() -> bool {
	auto const empty = ModelBarrier{.expected = 1, .pending = 1, .phase = 0};
	auto const cursor = prims::StageCursor<4>::start();
	return !empty.test(cursor.full_parity()) && empty.test(cursor.empty_parity());
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the cursor walks the stages and flips parity at each wrap (2 stages)", .holds = cursor_counts_stages_and_wraps<2>()},
	    engine::testing::Case{.name = "the cursor walks the stages and flips parity at each wrap (5 stages)", .holds = cursor_counts_stages_and_wraps<5>()},
	    engine::testing::Case{.name = "the producer's first wait passes on a fresh empty barrier and a consumer's would not",
	                          .holds = waiting_on_the_wrong_parity_deadlocks()},
	    engine::testing::Case{.name = "a 2-stage ring with 1 consumer is safe and live (seed 21)", .holds = ring_protocol_is_safe_and_live<2>(1, 21)},
	    engine::testing::Case{.name = "a 4-stage ring with 4 consumers is safe and live (seed 22)", .holds = ring_protocol_is_safe_and_live<4>(4, 22)},
	    engine::testing::Case{.name = "a 7-stage ring with 3 consumers is safe and live (seed 23)", .holds = ring_protocol_is_safe_and_live<7>(3, 23)},
	};
	return engine::testing::report(cases);
}

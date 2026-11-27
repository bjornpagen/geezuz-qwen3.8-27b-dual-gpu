// SPDX-License-Identifier: AGPL-3.0-only
//
// The per-phase profile's layout and its table. Every walk's phases must land in distinct slots below the profile's
// count, a cycle of any depth in the deepest cycle's order, and a prompt's end and a logits ticket in the slots of the
// cycle phases they share. The table turns counters into times at the clock the tickets measured and leaves out what
// never ran; a profile no kernel wrote has no table.
import std;
import engine.decode;
import engine.decode_device;
import engine.model;
import engine.schedule;
import engine.testing;

namespace {

namespace decode = engine::decode;
namespace dd = engine::decode_device;
namespace model = engine::model;
namespace schedule = engine::schedule;

using model::DraftDepth;

/** The slots a walk's phases fall in, in walk order. */
struct Slots {
	std::vector<std::size_t> slots;
	std::vector<schedule::Step> steps;
	std::vector<schedule::Pass> passes;

	void phase(schedule::Step step, schedule::Pass pass, std::size_t image_phase, std::size_t /*row_tiles*/) {
		slots.push_back(dd::profile_slot(pass, image_phase));
		steps.push_back(step);
		passes.push_back(pass);
	}

	void task(dd::Point const& /*point*/) {}

	void barrier() {}
};

/** The deepest cycle's walk: its phases' steps and passes in slot order, as the walk and the schedule agree. */
[[nodiscard]] auto deepest_walk() -> Slots {
	auto walked = Slots{};
	dd::walk_cycle(model::max_draft_depth, walked);
	return walked;
}

/** A cycle of depth `D` counts its phases in the deepest cycle's first slots, in order. */
template<DraftDepth D>
[[nodiscard]] auto cycles_fill_the_first_slots() -> bool {
	constexpr auto const& program = schedule::cycle<D, dd::budget>;
	auto walked = Slots{};
	dd::walk_cycle(D, walked);
	if (walked.slots.size() != program.phase_count) {
		return false;
	}
	for (auto index = std::size_t{0}; index < program.phase_count; ++index) {
		auto const& phase = std::span{program.phases}[index];
		if (walked.slots[index] != index || dd::profile_slot(phase.pass, phase.image_phase) != index) {
			return false;
		}
	}
	return true;
}

/** The profile holds exactly one slot per phase of the deepest cycle. */
[[nodiscard]] auto slots_are_the_deepest_cycles_phases() -> bool {
	constexpr auto const& deepest = schedule::cycle<model::max_draft_depth, dd::budget>;
	return dd::profile_phases == deepest.phase_count && dd::profile_phases == schedule::max_cycle_phases;
}

/** A prompt's end and a logits ticket count each phase in the slot of the deepest cycle's phase with the same step. */
[[nodiscard]] auto other_walks_share_the_cycles_slots() -> bool {
	auto const deepest = deepest_walk();
	auto prompt = Slots{};
	dd::walk_prompt_end(model::max_draft_depth, prompt);
	auto logits = Slots{};
	dd::walk_logits(logits);
	auto const lands = [&](Slots const& walked) {
		auto seen = std::set<std::size_t>{};
		for (auto index = std::size_t{0}; index < walked.slots.size(); ++index) {
			auto const slot = walked.slots[index];
			if (slot >= dd::profile_phases || !seen.insert(slot).second) {
				return false;
			}
			auto const& theirs = deepest.steps.at(slot);
			auto const& mine = walked.steps[index];
			if (theirs.op != mine.op || theirs.stream != mine.stream || theirs.layer != mine.layer) {
				return false;
			}
		}
		return !walked.slots.empty();
	};
	return lands(prompt) && lands(logits);
}

[[nodiscard]] auto near(double value, double expected) -> bool {
	return std::abs(value - expected) <= 1e-9 * std::max(1.0, std::abs(expected));
}

/** The mtp phase the synthetic profile visits: the catch-up pass's second phase. */
inline constexpr auto mtp_slot = schedule::verify_phases + 1;

/**
 * Synthetic counters at a measured 1.5 GHz: a verify-pass phase visited twice and an mtp phase visited once, every
 * other slot never.
 */
[[nodiscard]] auto synthetic_profile() -> std::unique_ptr<dd::Profile> {
	auto profile = std::make_unique<dd::Profile>();
	profile->tickets = 4;
	profile->ticket_cycles = 6'000'000'000;
	profile->ticket_nanoseconds = 4'000'000'000;
	auto const blocks = std::uint64_t{dd::sms};
	auto& verify = std::span{profile->phase}[3];
	verify = {.visits = 2,
	          .cycles = 2 * blocks * 3000,
	          .slowest = 4500,
	          .fastest_complement = ~std::uint64_t{1500},
	          .barrier = 2 * blocks * 750,
	          .peer = 2 * blocks * 1500,
	          .masks = 0,
	          .bytes = std::uint64_t{2} * 8'000'000};
	auto& mtp = std::span{profile->phase}[mtp_slot];
	mtp = {.visits = 1, .cycles = blocks * 600, .slowest = 900, .fastest_complement = ~std::uint64_t{300}, .barrier = 0, .peer = 0, .masks = blocks * 150, .bytes = 0};
	return profile;
}

/**
 * The synthetic profile's table: each field its sum over block-visits converted at the measured clock, and the slots
 * never visited left out. It holds no optional: clang-tidy's optional-access dataflow crashes on a function that holds
 * one and reads fields of the schedule's steps, whose header reaches this unit through two modules.
 */
[[nodiscard]] auto reads_as_converted(decode::ProfileTable const& table) -> bool {
	if (table.tickets != 4 || !near(table.sm_gigahertz, 1.5) || table.phases.size() != 2) {
		return false;
	}
	auto const deepest = deepest_walk();
	auto const& first = table.phases[0];
	auto const& second = table.phases[1];
	auto const named = [&](decode::PhaseTime const& time, std::size_t slot) {
		auto const& step = deepest.steps.at(slot);
		auto const& pass = deepest.passes.at(slot);
		return time.step.op == step.op && time.step.layer == step.layer && time.pass.kind == pass.kind && time.pass.draft == pass.draft;
	};
	// 3000 cycles at 1.5 GHz are 2 microseconds; 8 MB in 2 microseconds is 4000 GB/s.
	return named(first, 3) && first.visits == 2 && near(first.block_microseconds, 2.0) && near(first.slowest_microseconds, 3.0) &&
	       near(first.fastest_microseconds, 1.0) && near(first.barrier_microseconds, 0.5) && near(first.peer_microseconds, 1.0) &&
	       near(first.mask_microseconds, 0.0) && near(first.gigabytes_per_second, 4000.0) && named(second, mtp_slot) &&
	       second.pass.kind != schedule::PassKind::Verify && second.visits == 1 && near(second.block_microseconds, 0.4) &&
	       near(second.mask_microseconds, 0.1) && near(second.gigabytes_per_second, 0.0);
}

[[nodiscard]] auto the_table_converts_at_the_measured_clock() -> bool {
	auto const profile = synthetic_profile();
	auto const table = decode::profile_table(*profile);
	return table.has_value() && reads_as_converted(*table);
}

/** A profile no kernel wrote (any build but the profile build) has no table. */
[[nodiscard]] auto an_unwritten_profile_has_no_table() -> bool {
	auto const profile = std::make_unique<dd::Profile>();
	return !decode::profile_table(*profile).has_value();
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "the profile has one slot per phase of the deepest cycle", .holds = slots_are_the_deepest_cycles_phases()},
	    Case{.name = "a depth-one cycle's phases fill the first slots in order", .holds = cycles_fill_the_first_slots<DraftDepth::One>()},
	    Case{.name = "a depth-two cycle's phases fill the first slots in order", .holds = cycles_fill_the_first_slots<DraftDepth::Two>()},
	    Case{.name = "a depth-three cycle's phases fill the first slots in order", .holds = cycles_fill_the_first_slots<DraftDepth::Three>()},
	    Case{.name = "a depth-four cycle's phases fill every slot in order", .holds = cycles_fill_the_first_slots<DraftDepth::Four>()},
	    Case{.name = "a prompt's end and a logits ticket count in the slots of the cycle phases they share",
	         .holds = other_walks_share_the_cycles_slots()},
	    Case{.name = "the table converts counters at the measured clock and leaves out what never ran",
	         .holds = the_table_converts_at_the_measured_clock()},
	    Case{.name = "a profile no kernel wrote has no table", .holds = an_unwritten_profile_has_no_table()},
	};
	return engine::testing::report(cases);
}

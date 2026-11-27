// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's interpretation of the decode kernel's walk: the same `walk_cycle` (and `walk_prompt_end`, `walk_logits`)
// the kernel interprets, here counting what it does. Its phases must be the schedule's `cycle<D, B>` phase for phase;
// every task ends in a grid barrier; the partial and candidate exchanges are the schedule's count; and the exchange
// slots and epochs both cards derive alternate and grow across cycles of any depths, which is what lets two receive
// slots serve every exchange.
import std;
import engine.decode_device;
import engine.model;
import engine.schedule;
import engine.testing;

namespace {

namespace dd = engine::decode_device;
namespace model = engine::model;
namespace schedule = engine::schedule;

using dd::Task;
using model::DraftDepth;
using schedule::Op;

struct Visited {
	schedule::Step step;
	schedule::Pass pass;
	std::size_t image_phase;
	std::size_t row_tiles;
};

/** Records the walk: its phases, tasks and barriers, and checks that a barrier follows every task and nothing else. */
struct Recorder {
	std::vector<Visited> phases;
	std::vector<dd::Point> tasks;
	std::size_t barriers = 0;
	bool barrier_after_each_task = true;

	void phase(schedule::Step step, schedule::Pass pass, std::size_t image_phase, std::size_t row_tiles) {
		phases.push_back({.step = step, .pass = pass, .image_phase = image_phase, .row_tiles = row_tiles});
	}

	void task(dd::Point point) {
		barrier_after_each_task = barrier_after_each_task && barriers == tasks.size();
		tasks.push_back(point);
	}

	void barrier() {
		++barriers;
		barrier_after_each_task = barrier_after_each_task && barriers == tasks.size();
	}
};

[[nodiscard]] auto walk(DraftDepth depth) -> Recorder {
	auto recorder = Recorder{};
	dd::walk_cycle(depth, recorder);
	return recorder;
}

template<DraftDepth D>
[[nodiscard]] auto phases_are_the_schedules() -> bool {
	constexpr auto const& program = schedule::cycle<D, dd::budget>;
	auto const walked = walk(D);
	if (walked.phases.size() != program.phase_count) {
		return false;
	}
	for (auto index = std::size_t{0}; index < program.phase_count; ++index) {
		auto const& planned = std::span{program.phases}[index];
		auto const& visited = walked.phases[index];
		auto const same = visited.step.op == planned.step.op && visited.step.stream == planned.step.stream &&
		                  visited.step.layer == planned.step.layer && visited.pass.kind == planned.pass.kind &&
		                  visited.pass.draft == planned.pass.draft && visited.image_phase == planned.image_phase &&
		                  visited.row_tiles == planned.row_tiles;
		if (!same) {
			return false;
		}
	}
	return true;
}

/**
 * The tasks each op's phase runs, written down independently of the walk: a norm before the phases that read a
 * normalized residual, the ranking's segments, folds and final step after each lm_head, and the mtp input's gather and
 * norm before its fc.
 */
[[nodiscard]] constexpr auto tasks_of(Op op) -> std::size_t {
	switch (op) {
	case Op::Embed:
	case Op::DeltanetCore:
	case Op::DeltanetZ:
	case Op::DeltanetOut:
	case Op::AttentionCore:
	case Op::AttentionGate:
	case Op::AttentionMerge:
	case Op::AttentionOut:
	case Op::MlpOut:
	case Op::Verify:
	case Op::Draft:
		return 1;
	case Op::DeltanetIn:
	case Op::AttentionIn:
	case Op::MlpIn:
		return 2;
	case Op::MtpIn:
		return 3;
	case Op::Head:
		// Norm, lm_head, 243 segments, two folds (to 31, then 4), the final rank.
		return 6;
	case Op::DraftHead:
		// Norm, draft lm_head, 32 segments, one fold (to 4), the final rank.
		return 5;
	}
	std::unreachable();
}

template<DraftDepth D>
[[nodiscard]] auto barriers_are_counted() -> bool {
	constexpr auto const& program = schedule::cycle<D, dd::budget>;
	auto expected = std::size_t{0};
	for (auto index = std::size_t{0}; index < program.phase_count; ++index) {
		expected += tasks_of(std::span{program.phases}[index].step.op);
	}
	auto const walked = walk(D);
	return walked.barrier_after_each_task && walked.barriers == expected && walked.tasks.size() == expected;
}

template<DraftDepth D>
[[nodiscard]] auto exchanges_are_the_schedules() -> bool {
	constexpr auto const& program = schedule::cycle<D, dd::budget>;
	auto const walked = walk(D);
	auto const partials = std::ranges::count_if(walked.tasks, [](dd::Point const& point) {
		return point.task == Task::Gemv && schedule::exchange(point.step.op) == schedule::Exchange::Partials;
	});
	auto const candidates = std::ranges::count_if(walked.tasks, [](dd::Point const& point) {
		return point.task == Task::RankFinal;
	});
	return std::cmp_equal(partials, schedule::exchanges(program, schedule::Exchange::Partials)) &&
	       std::cmp_equal(candidates, schedule::exchanges(program, schedule::Exchange::Candidates));
}

/** Every task reads its own phase: a norm or a rank belongs to the phase it precedes or follows, never another. */
template<DraftDepth D>
[[nodiscard]] auto tasks_stay_in_their_phase() -> bool {
	auto const walked = walk(D);
	return std::ranges::all_of(walked.tasks, [](dd::Point const& point) {
		switch (point.task) {
		case Task::Norm:
			return dd::normalizes(point.step.op);
		case Task::PromptRow:
			return point.step.op == Op::Head;
		case Task::RankSegments:
		case Task::RankGroups:
		case Task::RankFinal:
			return point.step.op == Op::Head || point.step.op == Op::DraftHead;
		case Task::Verify:
			return point.step.op == Op::Verify;
		case Task::Draft:
			return point.step.op == Op::Draft;
		case Task::Logits:
			return point.step.op == Op::Head;
		case Task::MtpGather:
		case Task::MtpNorm:
			return point.step.op == Op::MtpIn;
		case Task::Embed:
			return point.step.op == Op::Embed;
		case Task::Gemv:
		case Task::DeltanetCore:
		case Task::AttentionCore:
		case Task::AttentionMerge:
			return true;
		}
		std::unreachable();
	});
}

/**
 * The receive slots across cycles of random depths, as both cards count them: exchange e uses slot (e - 1) mod 2 with
 * epoch e. Two slots suffice because a card writes exchange e + 2 only after a phase that waited on exchange e + 1,
 * which the peer pushed after consuming exchange e: so between two uses of one slot, exactly one exchange of the
 * other slot intervenes, and every epoch is larger than the slot's last.
 */
[[nodiscard]] auto slots_alternate(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto ordinal = std::uint64_t{0};
	auto last_epoch = std::array<std::uint64_t, 2>{};
	// The slot the last exchange used; the first exchange has none before it, which the ordinal tells.
	auto last_slot = std::size_t{0};
	for (auto cycle = 0; cycle < 50; ++cycle) {
		auto const depth = std::span{schedule::draft_depths}[std::uniform_int_distribution<std::size_t>{0, 3}(random)];
		auto const walked = walk(depth);
		for (auto const& point : walked.tasks) {
			if (point.task != Task::Gemv || schedule::exchange(point.step.op) != schedule::Exchange::Partials) {
				continue;
			}
			++ordinal;
			auto const slot = static_cast<std::size_t>((ordinal - 1) % schedule::exchange_depth);
			if ((ordinal > 1 && last_slot == slot) || last_epoch.at(slot) >= ordinal) {
				return false;
			}
			last_epoch.at(slot) = ordinal;
			last_slot = slot;
		}
	}
	return ordinal > 0;
}

/** Each mtp pass reads the mtp image phases, the verify pass the trunk's, so a pass's walk names its own weights. */
[[nodiscard]] auto passes_read_their_images() -> bool {
	auto const walked = walk(DraftDepth::Four);
	return std::ranges::all_of(walked.phases, [](Visited const& phase) {
		auto const mtp = phase.pass.kind != schedule::PassKind::Verify;
		return mtp == (phase.image_phase >= schedule::verify_phases) && phase.step.stream == (mtp ? model::Stream::Mtp : phase.step.stream);
	});
}

/**
 * A prompt's end is the cycle's suffix from the lm_head on, phase for phase and task for task, except that the
 * lm_head reads prefill's row where the cycle normalizes its residual.
 */
template<DraftDepth D>
[[nodiscard]] auto prompt_end_is_the_cycles_suffix() -> bool {
	auto const cycle = walk(D);
	auto prompt = Recorder{};
	dd::walk_prompt_end(D, prompt);
	auto const head = std::ranges::find_if(cycle.phases, [](Visited const& phase) {
		return phase.step.op == Op::Head;
	});
	auto const skipped = static_cast<std::size_t>(head - cycle.phases.begin());
	auto const first_task = std::ranges::find_if(cycle.tasks, [](dd::Point const& point) {
		return point.step.op == Op::Head;
	});
	auto const tasks_skipped = static_cast<std::size_t>(first_task - cycle.tasks.begin());
	if (prompt.phases.size() + skipped != cycle.phases.size() || prompt.tasks.size() + tasks_skipped != cycle.tasks.size() ||
	    !prompt.barrier_after_each_task || prompt.barriers != prompt.tasks.size()) {
		return false;
	}
	for (auto index = std::size_t{0}; index < prompt.phases.size(); ++index) {
		auto const& mine = prompt.phases[index];
		auto const& theirs = cycle.phases[skipped + index];
		if (mine.step.op != theirs.step.op || mine.step.layer != theirs.step.layer || mine.pass.kind != theirs.pass.kind ||
		    mine.pass.draft != theirs.pass.draft || mine.image_phase != theirs.image_phase || mine.row_tiles != theirs.row_tiles) {
			return false;
		}
	}
	for (auto index = std::size_t{0}; index < prompt.tasks.size(); ++index) {
		auto const& mine = prompt.tasks[index];
		auto const& theirs = cycle.tasks[tasks_skipped + index];
		auto const expected = index == 0 ? Task::PromptRow : theirs.task;
		if (mine.task != expected || mine.step.op != theirs.step.op || mine.image_phase != theirs.image_phase || mine.level != theirs.level) {
			return false;
		}
	}
	return cycle.tasks[tasks_skipped].task == Task::Norm;
}

/**
 * A logits ticket runs the cycle's verify pass phase for phase and task for task (the same tasks over the same rows,
 * so the same arithmetic) up to the lm_head's matmul, then hands the logits over where the cycle ranks them, and stops:
 * no verdict and no mtp pass. Its walk is the cycle's prefix through that matmul and one more task.
 */
template<DraftDepth D>
[[nodiscard]] auto logits_is_the_cycles_prefix() -> bool {
	auto const cycle = walk(D);
	auto logits = Recorder{};
	dd::walk_logits(logits);
	auto const ranking = std::ranges::find_if(cycle.tasks, [](dd::Point const& point) {
		return point.task == Task::RankSegments && point.step.op == Op::Head;
	});
	auto const shared = static_cast<std::size_t>(ranking - cycle.tasks.begin());
	auto const head = std::ranges::find_if(cycle.phases, [](Visited const& phase) {
		return phase.step.op == Op::Head;
	});
	auto const phases = static_cast<std::size_t>(head - cycle.phases.begin()) + 1;
	if (ranking == cycle.tasks.end() || logits.tasks.size() != shared + 1 || logits.phases.size() != phases || !logits.barrier_after_each_task ||
	    logits.barriers != logits.tasks.size()) {
		return false;
	}
	for (auto index = std::size_t{0}; index < phases; ++index) {
		auto const& mine = logits.phases[index];
		auto const& theirs = cycle.phases[index];
		if (mine.step.op != theirs.step.op || mine.step.layer != theirs.step.layer || mine.step.stream != theirs.step.stream ||
		    mine.pass.kind != theirs.pass.kind || mine.image_phase != theirs.image_phase || mine.row_tiles != theirs.row_tiles) {
			return false;
		}
	}
	for (auto index = std::size_t{0}; index < shared; ++index) {
		auto const& mine = logits.tasks[index];
		auto const& theirs = cycle.tasks[index];
		if (mine.task != theirs.task || mine.step.op != theirs.step.op || mine.step.layer != theirs.step.layer || mine.pass.kind != theirs.pass.kind ||
		    mine.image_phase != theirs.image_phase || mine.row_tiles != theirs.row_tiles || mine.level != theirs.level) {
			return false;
		}
	}
	auto const& tail = logits.tasks.back();
	return cycle.tasks[shared - 1].task == Task::Gemv && cycle.tasks[shared - 1].step.op == Op::Head && tail.task == Task::Logits &&
	       tail.step.op == Op::Head && tail.image_phase == cycle.tasks[shared - 1].image_phase;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "the walk's phases are cycle<One, budget>'s", .holds = phases_are_the_schedules<DraftDepth::One>()},
	    Case{.name = "the walk's phases are cycle<Two, budget>'s", .holds = phases_are_the_schedules<DraftDepth::Two>()},
	    Case{.name = "the walk's phases are cycle<Three, budget>'s", .holds = phases_are_the_schedules<DraftDepth::Three>()},
	    Case{.name = "the walk's phases are cycle<Four, budget>'s", .holds = phases_are_the_schedules<DraftDepth::Four>()},
	    Case{.name = "a grid barrier follows every task, as counted per op, depth one", .holds = barriers_are_counted<DraftDepth::One>()},
	    Case{.name = "a grid barrier follows every task, as counted per op, depth two", .holds = barriers_are_counted<DraftDepth::Two>()},
	    Case{.name = "a grid barrier follows every task, as counted per op, depth three", .holds = barriers_are_counted<DraftDepth::Three>()},
	    Case{.name = "a grid barrier follows every task, as counted per op, depth four", .holds = barriers_are_counted<DraftDepth::Four>()},
	    Case{.name = "partial and candidate exchanges are the schedule's, depth one", .holds = exchanges_are_the_schedules<DraftDepth::One>()},
	    Case{.name = "partial and candidate exchanges are the schedule's, depth four", .holds = exchanges_are_the_schedules<DraftDepth::Four>()},
	    Case{.name = "every task stays in its own phase", .holds = tasks_stay_in_their_phase<DraftDepth::Four>()},
	    Case{.name = "receive slots alternate and epochs grow across cycles (seed 1)", .holds = slots_alternate(1)},
	    Case{.name = "receive slots alternate and epochs grow across cycles (seed 2)", .holds = slots_alternate(2)},
	    Case{.name = "each pass reads its own stream's image phases", .holds = passes_read_their_images()},
	    Case{.name = "a prompt's end is the cycle's suffix from the lm_head, depth one", .holds = prompt_end_is_the_cycles_suffix<DraftDepth::One>()},
	    Case{.name = "a prompt's end is the cycle's suffix from the lm_head, depth four", .holds = prompt_end_is_the_cycles_suffix<DraftDepth::Four>()},
	    Case{.name = "a logits ticket is a depth-one cycle's prefix through the lm_head, then the hand-off",
	         .holds = logits_is_the_cycles_prefix<DraftDepth::One>()},
	    Case{.name = "a logits ticket is a depth-four cycle's prefix through the lm_head, then the hand-off",
	         .holds = logits_is_the_cycles_prefix<DraftDepth::Four>()},
	};
	return engine::testing::report(cases);
}

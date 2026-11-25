// SPDX-License-Identifier: AGPL-3.0-only
//
// One decode cycle as the grid-barrier baseline runs it: the schedule's phases in stream order, each cut into the
// tasks every SM runs between two grid barriers. The walk is the schedule's own (`schedule::walk_verify`, then one
// `walk_mtp` per draft, exactly as `schedule::program` builds `cycle<D, B>`), so the kernel holds no second copy of the
// phase order; this header adds only what a barrier separates. The persistent kernel interprets the walk on the cards,
// and the host tests interpret the same walk to count its phases, barriers and exchanges. It uses no standard headers
// so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "decode/wire.h"
#include "model.h"
#include "schedule/program.h"

namespace engine::device::decode {

using schedule::Op;
using schedule::Pass;
using schedule::PassKind;
using schedule::Step;

/** What every SM does between two grid barriers. */
enum class Task : unsigned char {
	/** The verify rows' embeddings into the fp32 residual, with each tile's sum of squares. */
	Embed,
	/** An rms norm of the residual into the bf16 rows the phase's matmuls read. */
	Norm,
	/** A prompt's last row, already final-normed by prefill, as the lm_head's only row. */
	PromptRow,
	/** The phase's weight matmuls; an exchange phase also pushes its partials and adds both cards' into the residual. */
	Gemv,
	DeltanetCore,
	AttentionCore,
	AttentionMerge,
	/** Each row's best 64 candidates of each 512-row segment of the card's vocabulary. */
	RankSegments,
	/** Eight lists of 64 at a time folded into one, `level` times. */
	RankGroups,
	/** Each row's best 64 of the card, kept and pushed to the peer. */
	RankFinal,
	/** Both cards' candidates merged, the drafts judged, the next token sampled. */
	Verify,
	/** The mtp rows' embeddings and hidden inputs, with their sums of squares. */
	MtpGather,
	/** The mtp input norms into the fc's bf16 rows. */
	MtpNorm,
	/** Both cards' draft candidates merged and the next draft proposed. */
	Draft,
};

/** One task of the walk, with the phase it belongs to. */
struct Point {
	Task task;
	Step step;
	Pass pass;
	/** The image phase whose placements the task reads. */
	Count image_phase;
	/** n8 row tiles the phase computes. */
	Count row_tiles;
	/** For `RankGroups`: which fold, from 1. */
	Count level;
};

// ---------------------------------------------------------------------------------------------------------------------
// The ranking's shape: a vocabulary is cut into segments of 512 rows, each row's segment lists folded eight at a time
// until at most eight remain, which the final rank folds into one.

/** Rows of a card's vocabulary one warp ranks at once: 4 KiB of 8-byte keys in shared memory. */
inline constexpr Count segment_rows = 512;
inline constexpr Count list_length = static_cast<Count>(sampling::top_k_bound);
/** Lists one warp folds at once: their 64 keys each fill a segment. */
inline constexpr Count fold_lists = segment_rows / list_length;

static_assert(fold_lists == 8);

/** Rows of the card's vocabulary a ranking phase scores: the whole lm_head, or the draft vocabulary's prefix of it. */
[[nodiscard]] constexpr auto ranked_rows(Op op) -> Count {
	return op == Op::Head ? model::vocab_per_card : model::draft_vocab_per_card;
}

[[nodiscard]] constexpr auto segments(Op op) -> Count {
	return model::divide_up(ranked_rows(op), segment_rows);
}

/** Lists left after `level` folds. */
[[nodiscard]] constexpr auto lists_after(Op op, Count level) -> Count {
	auto lists = segments(op);
	for (auto fold = Count{0}; fold < level; ++fold) {
		lists = model::divide_up(lists, fold_lists);
	}
	return lists;
}

/** Folds before at most eight lists remain. */
[[nodiscard]] constexpr auto folds(Op op) -> Count {
	auto level = Count{0};
	while (lists_after(op, level) > fold_lists) {
		++level;
	}
	return level;
}

static_assert(segments(Op::Head) == 243 && folds(Op::Head) == 2 && lists_after(Op::Head, 2) == 4);
static_assert(segments(Op::DraftHead) == 32 && folds(Op::DraftHead) == 1 && lists_after(Op::DraftHead, 1) == 4);

// ---------------------------------------------------------------------------------------------------------------------
// The walk.

/** Whether a phase starts with an rms norm of its residual into the rows its matmuls read. */
[[nodiscard]] constexpr auto normalizes(Op op) -> bool {
	switch (op) {
	case Op::DeltanetIn:
	case Op::AttentionIn:
	case Op::MlpIn:
	case Op::Head:
	case Op::DraftHead:
		return true;
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
	case Op::MtpIn:
	case Op::Draft:
		return false;
	}
	__builtin_unreachable();
}

/** Where the lm_head's rows come from: the trunk's final norm of each verify row, or a prompt's last row from prefill. */
enum class HeadInput : unsigned char {
	VerifyRows,
	PromptRow,
};

/** Adapts the schedule's walk to `Visit`: each step of it becomes a phase of the cycle and that phase's tasks. */
template<class Visit>
struct Phases {
	Visit* visit;
	Pass pass;
	Count row_tiles;
	Count image_phase;
	HeadInput head;

	constexpr void run(Task task, Step step, Count level) {
		visit->task(Point{.task = task, .step = step, .pass = pass, .image_phase = image_phase, .row_tiles = row_tiles, .level = level});
		visit->barrier();
	}

	constexpr void operator()(Step step) {
		visit->phase(step, pass, image_phase, row_tiles);
		if (step.op == Op::Head && head == HeadInput::PromptRow) {
			run(Task::PromptRow, step, 0);
		} else if (normalizes(step.op)) {
			run(Task::Norm, step, 0);
		}
		switch (step.op) {
		case Op::Embed:
			run(Task::Embed, step, 0);
			break;
		case Op::DeltanetIn:
		case Op::DeltanetZ:
		case Op::DeltanetOut:
		case Op::AttentionIn:
		case Op::AttentionGate:
		case Op::AttentionOut:
		case Op::MlpIn:
		case Op::MlpOut:
			run(Task::Gemv, step, 0);
			break;
		case Op::DeltanetCore:
			run(Task::DeltanetCore, step, 0);
			break;
		case Op::AttentionCore:
			run(Task::AttentionCore, step, 0);
			break;
		case Op::AttentionMerge:
			run(Task::AttentionMerge, step, 0);
			break;
		case Op::Head:
		case Op::DraftHead:
			run(Task::Gemv, step, 0);
			run(Task::RankSegments, step, 0);
			for (auto level = Count{1}; level <= folds(step.op); ++level) {
				run(Task::RankGroups, step, level);
			}
			run(Task::RankFinal, step, 0);
			break;
		case Op::Verify:
			run(Task::Verify, step, 0);
			break;
		case Op::MtpIn:
			run(Task::MtpGather, step, 0);
			run(Task::MtpNorm, step, 0);
			run(Task::Gemv, step, 0);
			break;
		case Op::Draft:
			run(Task::Draft, step, 0);
			break;
		}
		++image_phase;
	}
};

/** The mtp passes of a cycle of `depth`: the catch-up over the accepted rows, then one step per further draft. */
template<class Visit>
constexpr void walk_drafts(model::DraftDepth depth, Visit& visit) {
	for (auto draft = Count{1}; draft <= model::drafts(depth); ++draft) {
		auto const catch_up = draft == 1;
		auto mtp = Phases<Visit>{.visit = &visit,
		                         .pass = {.kind = catch_up ? PassKind::CatchUp : PassKind::Step, .draft = draft},
		                         .row_tiles = catch_up ? row_tiles : schedule::step_row_tiles,
		                         .image_phase = schedule::verify_phases,
		                         .head = HeadInput::VerifyRows};
		schedule::walk_mtp(mtp);
	}
}

/**
 * Walks one cycle of `depth`: the verify pass over every verify row, then one mtp pass per draft, the first (the
 * catch-up) over the accepted rows and each later one over one row per slot. `Visit` takes `phase(step, pass,
 * image_phase, row_tiles)` at each phase's start, `task(point)` for each task and `barrier()` after each.
 */
template<class Visit>
constexpr void walk_cycle(model::DraftDepth depth, Visit& visit) {
	auto verify = Phases<Visit>{.visit = &visit, .pass = {.kind = PassKind::Verify, .draft = 0}, .row_tiles = row_tiles, .image_phase = 0, .head = HeadInput::VerifyRows};
	schedule::walk_verify(verify);
	walk_drafts(depth, visit);
}

/**
 * Walks a prompt's end at `depth`: the verify pass's last two phases (the lm_head and the verdict) over the one row
 * prefill left, then the cycle's mtp passes. It is a suffix of `walk_cycle` but for the lm_head's input.
 */
template<class Visit>
constexpr void walk_prompt_end(model::DraftDepth depth, Visit& visit) {
	auto head = Phases<Visit>{.visit = &visit,
	                          .pass = {.kind = PassKind::Verify, .draft = 0},
	                          .row_tiles = row_tiles,
	                          .image_phase = schedule::verify_phases - 2,
	                          .head = HeadInput::PromptRow};
	head(Step{.op = Op::Head, .stream = model::Stream::Head, .layer = 0});
	head(Step{.op = Op::Verify, .stream = model::Stream::Head, .layer = 0});
	walk_drafts(depth, visit);
}

}

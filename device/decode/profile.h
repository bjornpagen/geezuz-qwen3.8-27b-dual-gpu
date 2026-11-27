// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode kernel's per-phase profile, held once for both ends. A profile build's kernel accumulates these counters
// in its control block on the card and copies them into the host's mapped profile buffer after every ticket; no other
// build's kernel has the code that writes them, so its image is the one it would be without them. Each phase of a walk
// counts into its slot, the place its pass and image phase give it among the deepest cycle's phases, so a cycle of any
// depth, a prompt's end and a logits ticket all count a phase they share in the same place. It uses no standard
// headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "model.h"
#include "schedule/program.h"

namespace engine::device::decode {

using model::Count;

/** Profile slots: one per phase of the deepest cycle, whose phases every walk's phases are among. */
inline constexpr Count profile_phases = schedule::max_cycle_phases;

/** A phase's slot: the verify pass's phases by image phase, then each mtp pass's phases after the pass before. */
[[nodiscard]] constexpr auto profile_slot(schedule::Pass pass, Count image_phase) -> Count {
	if (pass.kind == schedule::PassKind::Verify) {
		return image_phase;
	}
	return schedule::verify_phases + (pass.draft - 1) * schedule::mtp_phases + (image_phase - schedule::verify_phases);
}

/**
 * One phase's counters on one card, summed over every visit since the engine opened. Times are SM cycles, each block
 * reading its own SM's counter, so a block's time is a difference of two of its own readings and never compares
 * readings of two SMs.
 */
struct PhaseCounters {
	/** Tickets that ran the phase. */
	unsigned long long visits;
	/** Cycles blocks spent in the phase, from its entry to the next phase's (or the walk's end), summed over blocks. */
	unsigned long long cycles;
	/** The most cycles one block spent in one visit. */
	unsigned long long slowest;
	/** The complement of the fewest cycles one block spent in one visit: kept as a maximum, so zero is its start. */
	unsigned long long fastest_complement;
	/** Cycles blocks waited at the grid barriers that end the phase's tasks: the phase's warp arrival, summed. */
	unsigned long long barrier;
	/** Cycles spent waiting on the peer card's flags, summed over the waiting warps of every block. */
	unsigned long long peer;
	/** Cycles blocks waited for the host's masks before ranking, summed. */
	unsigned long long masks;
	/**
	 * Bytes the phase streams from the card's memory by its shape, summed over visits: its image phase's weights on this
	 * card, and for a core the live slots' deltanet state or the kv of every position the rows attend to.
	 */
	unsigned long long bytes;
};

struct Profile {
	/** Tickets that ran a walk. */
	unsigned long long tickets;
	/** Block 0's SM cycles and global-timer nanoseconds over those tickets: the SM clock's rate, as it ran. */
	unsigned long long ticket_cycles;
	unsigned long long ticket_nanoseconds;
	unsigned long long unused;
	PhaseCounters phase[profile_phases];
};

/**
 * What each block waited on since its phase began: the wait sites add here, the phase's end moves it into the phase's
 * counters. Device scratch in the control block; the host never reads it.
 */
struct BlockWaits {
	unsigned long long peer;
	unsigned long long masks;
};

static_assert(sizeof(Profile) % 16 == 0, "the profile copies to the host in whole chunks");

}

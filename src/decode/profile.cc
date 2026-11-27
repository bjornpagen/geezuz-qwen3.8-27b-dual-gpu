// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode kernel's per-phase profile as a table: each phase the profiled tickets ran, named by its op, layer and
// pass, with a block's time in it, its waits at the grid barriers, on the peer card and for the masks, and the rate it
// streamed its bytes at. The kernel counts SM cycles; it also counts block 0's cycles and global-timer nanoseconds over
// the same tickets, so the table converts cycles at the clock the SMs actually ran, not a nominal one. Only a profile
// build's kernel writes the counters; any other build's leaves them zero, and there is no table.
export module engine.decode:profile;

import std;
import engine.decode_device;
import engine.model;
import engine.schedule;

namespace engine::decode {

/** One phase of the profiled tickets, on one card. Times are microseconds, per visit. */
export struct PhaseTime {
	schedule::Step step;
	schedule::Pass pass;
	/** Tickets that ran the phase. */
	std::uint64_t visits;
	/** A block's mean time in the phase, and the most and least one block took. */
	double block_microseconds;
	double slowest_microseconds;
	double fastest_microseconds;
	/** A block's mean wait at the grid barriers ending the phase's tasks: the phase's warp arrival. */
	double barrier_microseconds;
	/** A block's mean wait on the peer card's flags, its waiting warps' waits added, and for the host's masks. */
	double peer_microseconds;
	double mask_microseconds;
	/** The bytes the phase streams per visit over a block's mean time in it. */
	double gigabytes_per_second;
};

/** The profile of one card since the engine opened. */
export struct ProfileTable {
	std::uint64_t tickets;
	/** The SM clock over the profiled tickets: cycles per nanosecond. */
	double sm_gigahertz;
	/** Every phase the tickets ran, in the deepest cycle's order. */
	std::vector<PhaseTime> phases;
};

/** The deepest cycle, whose phases are the profile's slots in order. */
inline constexpr auto const& deepest = schedule::cycle<model::max_draft_depth, decode_device::budget>;

static_assert(deepest.phase_count == decode_device::profile_phases, "every slot of the profile is one phase of the deepest cycle");

/**
 * The table of a card's profile, or none when no profiled ticket ran: the kernel was built without the profiler, or
 * has run no walk yet.
 */
export [[nodiscard]] auto profile_table(decode_device::Profile const& profile) -> std::optional<ProfileTable> {
	if (profile.tickets == 0 || profile.ticket_cycles == 0 || profile.ticket_nanoseconds == 0) {
		return std::nullopt;
	}
	auto const gigahertz = static_cast<double>(profile.ticket_cycles) / static_cast<double>(profile.ticket_nanoseconds);
	auto const microseconds = [&](double cycles) {
		return cycles / gigahertz / 1000.0;
	};
	auto table = ProfileTable{.tickets = profile.tickets, .sm_gigahertz = gigahertz, .phases = {}};
	for (auto slot = std::size_t{0}; slot < deepest.phase_count; ++slot) {
		auto const& counters = std::span{profile.phase}[slot];
		if (counters.visits == 0) {
			continue;
		}
		auto const& phase = std::span{deepest.phases}[slot];
		// A block-visit is one block's stay in one visit of the phase: every sum below is over them.
		auto const stays = static_cast<double>(counters.visits) * static_cast<double>(decode_device::sms);
		auto const block = microseconds(static_cast<double>(counters.cycles) / stays);
		auto const bytes_per_visit = static_cast<double>(counters.bytes) / static_cast<double>(counters.visits);
		table.phases.push_back({
		    .step = phase.step,
		    .pass = phase.pass,
		    .visits = counters.visits,
		    .block_microseconds = block,
		    .slowest_microseconds = microseconds(static_cast<double>(counters.slowest)),
		    .fastest_microseconds = microseconds(static_cast<double>(~counters.fastest_complement)),
		    .barrier_microseconds = microseconds(static_cast<double>(counters.barrier) / stays),
		    .peer_microseconds = microseconds(static_cast<double>(counters.peer) / stays),
		    .mask_microseconds = microseconds(static_cast<double>(counters.masks) / stays),
		    .gigabytes_per_second = block > 0.0 ? bytes_per_visit / (block * 1000.0) : 0.0,
		});
	}
	return table;
}

}

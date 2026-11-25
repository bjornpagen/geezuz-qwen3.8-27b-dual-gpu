// SPDX-License-Identifier: AGPL-3.0-only
//
// How the cycle program lands on the machine: each phase's units split over the SMs, the per-SM transaction stream a
// producer walks, the decode kernel's warp roles and shared memory, and the cross-SM and cross-card buffers and flags
// in the workspace. Every geometry is a compile-time fact checked against the sm_86 limits and the model's workspace.
#pragma once

#include "attn/geometry.h"
#include "deltanet/layout.h"
#include "model.h"
#include "prims/budget.h"
#include "sampling/draw.h"
#include "schedule/program.h"

namespace engine::device::schedule {

/** SMs the persistent decode kernel runs one block on each. */
inline constexpr Count sms = static_cast<Count>(prims::sm86.sms);

// ---------------------------------------------------------------------------------------------------------------------
// Units and their split over the SMs.

/** The smallest piece of a phase's work: one tile of every tensor its placement interleaves. */
[[nodiscard]] constexpr auto unit_bytes(Placement const& placement) -> Count {
	return tile_bytes * placement.read.tensor_count;
}

/** A phase's ring placements: one contiguous extent of the image, in units of one size. */
struct Run {
	Count offset;
	Count unit_bytes;
	Count units;
};

[[nodiscard]] constexpr auto run(CardImage const& layout, ImagePhase const& phase) -> Run {
	if (phase.ring.length == 0) {
		return {.offset = 0, .unit_bytes = tile_bytes, .units = 0};
	}
	auto const& first = layout.placements[phase.ring.begin];
	auto units = Count{0};
	for (auto index = phase.ring.begin; index < phase.ring.begin + phase.ring.length; ++index) {
		units += layout.placements[index].bytes / unit_bytes(layout.placements[index]);
	}
	return {.offset = first.offset, .unit_bytes = unit_bytes(first), .units = units};
}

/** Whether every phase's ring placements are one run of the image in one unit size, so a producer only bumps. */
[[nodiscard]] consteval auto streams_are_runs(Count card) -> bool {
	auto const layout = image(card);
	for (auto const& phase : layout.phases) {
		auto expected = Count{0};
		for (auto index = phase.ring.begin; index < phase.ring.begin + phase.ring.length; ++index) {
			auto const& placement = layout.placements[index];
			auto const first = index == phase.ring.begin;
			if ((!first && placement.offset != expected) || unit_bytes(placement) != unit_bytes(layout.placements[phase.ring.begin]) ||
			    placement.bytes % unit_bytes(placement) != 0) {
				return false;
			}
			expected = placement.offset + placement.bytes;
		}
	}
	return true;
}

static_assert(streams_are_runs(0), "every phase streams one contiguous run of whole units");

/**
 * The units SM `sm` takes of `units`: a contiguous run in stream order, the first `units % sms` SMs one more. The split
 * depends on nothing but the unit count, never on the rows or the live slots, so every output element is summed in
 * one order: within an SM in stream order, across SMs in SM order.
 */
[[nodiscard]] constexpr auto share(Count units, Count sm) -> Range {
	auto const base = units / sms;
	auto const extra = units % sms;
	return {.begin = sm * base + (sm < extra ? sm : extra), .length = base + (sm < extra ? 1 : 0)};
}

/** The bytes of the image SM `sm`'s producer streams for `phase`: its pointer bump. */
[[nodiscard]] constexpr auto extent(CardImage const& layout, ImagePhase const& phase, Count sm) -> Range {
	auto const whole = run(layout, phase);
	auto const part = share(whole.units, sm);
	return {.begin = whole.offset + part.begin * whole.unit_bytes, .length = part.length * whole.unit_bytes};
}

/** Units one output tile (128 output rows) spans: its k tiles. */
[[nodiscard]] constexpr auto units_per_output(Placement const& placement) -> Count {
	auto const& entry = model::inventory[placement.read.tensor[0]];
	return model::card_shape(entry, 0).extents[1] / tile_columns(entry.format);
}

/**
 * The SM that finishes the output tile whose units are [first, first + span) of the phase: it holds the tile's last
 * unit, adds the earlier holders' partials in SM order, and runs the epilogue.
 */
[[nodiscard]] constexpr auto finisher(Count units, Count first, Count span) -> Count {
	auto const last = first + span - 1;
	auto const base = units / sms;
	auto const extra = units % sms;
	auto const long_units = extra * (base + 1);
	return last < long_units ? last / (base + 1) : extra + (last - long_units) / base;
}

// ---------------------------------------------------------------------------------------------------------------------
// The decode kernel: warp roles and shared memory.

/**
 * The decode kernel's warps, by role. One producer warp has the issue rate the stream needs (one 512-byte cp.async
 * warp instruction every ~76 clocks), and sm_86 cannot move registers away from it, so there is exactly one. The
 * consumers are two warpgroups of four, so each of the SM's four tensor cores has two math warps. Two exchange warps
 * push this card's partials and receive the peer's, so neither direction waits on the other, and one control warp
 * runs the cycle's per-slot bookkeeping (the attention chunk plan, the host mailbox, the verdict). Twelve warps keep
 * three per scheduler, the warp count the register model rewards.
 */
struct Roles {
	Count producer_warps;
	Count consumer_warpgroups;
	Count exchange_warps;
	Count control_warps;
};

inline constexpr Roles decode_roles = {.producer_warps = 1, .consumer_warpgroups = 2, .exchange_warps = 2, .control_warps = 1};

inline constexpr Count warpgroup_warps = 4;

/** Output rows of one mma: the weights are its m16 operand. */
inline constexpr Count mma_rows = 16;

[[nodiscard]] constexpr auto warps(Roles const& roles) -> Count {
	return roles.producer_warps + roles.consumer_warpgroups * warpgroup_warps + roles.exchange_warps + roles.control_warps;
}

static_assert(warps(decode_roles) % static_cast<Count>(prims::sm86.schedulers_per_sm) == 0,
              "whole warps per scheduler: a partial round loses registers to the busiest scheduler");
static_assert(decode_roles.consumer_warpgroups * warpgroup_warps * mma_rows == tile_rows,
              "the consumer warps split a tile's 128 rows into one m16 slab each, so no cross-warp sum exists");

/** The ring a producer fills: whole tiles, each with a full and an empty mbarrier. */
struct Ring {
	Count slots;
	Count stage_bytes;
};

inline constexpr Count mbarrier_bytes = 8;

/** Shared memory the consumers' epilogues use: norm partials, a tile's candidates, the attention chunk plan. */
inline constexpr Count epilogue_scratch_bytes = Count{8} * 1024;

/** The widest k span of one tile: the activation columns a unit needs staged. */
inline constexpr Count tile_span = tile_columns(model::Format::E4m3);

static_assert(tile_span >= tile_columns(model::Format::Bf16), "the e4m3 tile is the widest");

/** bf16 activations of one tile's k span for every row, double-buffered so the next unit's stage overlaps the mma. */
[[nodiscard]] constexpr auto activation_bytes(RowBudget budget) -> Count {
	return 2 * tile_span * model::rows(budget) * model::element_bytes(model::Format::Bf16);
}

/** A ring rides over dependency hops of a few microseconds: 64 KiB is the least that does, 80 KiB the most worth having. */
inline constexpr Count min_ring_slots = 4;
inline constexpr Count max_ring_slots = 5;

/** The most stages that fit beside the activations and the epilogue scratch, capped at the useful run-ahead. */
[[nodiscard]] constexpr auto ring(RowBudget budget) -> Ring {
	auto const limit = static_cast<Count>(prims::sm86.max_shared_per_block);
	auto const room = limit - activation_bytes(budget) - epilogue_scratch_bytes;
	auto const slots = room / (tile_bytes + 2 * mbarrier_bytes);
	return {.slots = slots < max_ring_slots ? slots : max_ring_slots, .stage_bytes = tile_bytes};
}

/** The decode kernel's shared memory: the ring, its barriers, the activation stages and the epilogue scratch. */
[[nodiscard]] constexpr auto decode_shared_bytes(RowBudget budget) -> Count {
	auto const r = ring(budget);
	return r.slots * (r.stage_bytes + 2 * mbarrier_bytes) + activation_bytes(budget) + epilogue_scratch_bytes;
}

/** The decode kernel's resource budget at one row budget: one resident block per SM. */
[[nodiscard]] constexpr auto decode_budget(RowBudget budget) -> prims::Budget {
	auto const threads = static_cast<int>(warps(decode_roles)) * prims::sm86.warp_threads;
	auto shape = prims::Budget{.threads = threads, .registers_per_thread = 0, .shared_bytes = 0, .blocks_per_sm = 1};
	shape.registers_per_thread = prims::register_cap(shape);
	shape.shared_bytes = static_cast<int>(decode_shared_bytes(budget));
	return shape;
}

[[nodiscard]] consteval auto decode_geometry_fits() -> bool {
	for (auto const budget : model::row_budgets) {
		auto const r = ring(budget);
		if (r.slots < min_ring_slots || prims::check(decode_budget(budget)) != prims::Fit::Fits) {
			return false;
		}
	}
	return true;
}

static_assert(decode_geometry_fits(), "the decode kernel's warps, registers and shared memory fit one sm_86 SM");
static_assert(prims::check(decode_budget(RowBudget::ThreeTiles)) == prims::Fit::Fits);
static_assert(prims::check(decode_budget(RowBudget::FourTiles)) == prims::Fit::Fits);

// ---------------------------------------------------------------------------------------------------------------------
// The workspace: cross-card exchanges, cross-SM fixups and phase counters.

/** An epoch flag: a monotonic 64-bit count both cards advance identically, so it never wraps. */
inline constexpr Count flag_bytes = 8;

/**
 * Receive slots per exchange kind. A card writes exchange e + 2 only after its phase e + 1, which depends on phase
 * e, which consumed exchange e on both cards; so two slots never overwrite one still being read.
 */
inline constexpr Count exchange_depth = 2;

/** Output tiles every partial exchange carries: the hidden state, in 128-row tiles. */
inline constexpr Count exchange_tiles = model::hidden / tile_rows;

static_assert(model::hidden % tile_rows == 0, "the residual is whole output tiles");

/** One output tile's fp32 partial sums for every row: the payload of one release. */
[[nodiscard]] constexpr auto partial_tile_bytes(RowBudget budget) -> Count {
	return model::rows(budget) * tile_rows * sizeof(float);
}

/**
 * A fixup slot: one SM's partial sums of the one output tile it touches but does not finish (only its last tile can
 * be one), for every tensor a unit interleaves.
 */
[[nodiscard]] constexpr auto fixup_slot_bytes(RowBudget budget) -> Count {
	return 2 * partial_tile_bytes(budget);
}

/** Candidates each card keeps per lm_head row and pushes to the other, which merges both lists in rank order. */
inline constexpr Count candidates_per_row = static_cast<Count>(sampling::top_k_bound);

static_assert(sizeof(sampling::Candidate) == 2 * sizeof(float), "a candidate crosses the bridge as its logit and id, unpadded");

/**
 * One receive slot of a candidate exchange: the card's ranked candidates for every row it computes, written whole
 * and released once. The verify pass computes every verify row and the catch-up pass as many, so the row budget
 * bounds both; a draft step computes fewer.
 */
[[nodiscard]] constexpr auto candidate_slot_bytes(RowBudget budget) -> Count {
	return model::rows(budget) * candidates_per_row * sizeof(sampling::Candidate);
}

static_assert(step_row_tiles <= row_tiles(RowBudget::ThreeTiles), "a draft step's rows fit the smallest candidate slot");

/** Where each workspace buffer of the schedule starts, and the bytes they take together. */
struct Workspace {
	Count partials;
	Count partial_flags;
	Count candidates;
	Count candidate_flags;
	Count fixups;
	Count fixup_flags;
	Count counters;
	/** Attention's split-k records: each chunk group's partial output and running max and sum, merged in order. */
	Count attention_records;
	/** Every slot's deltanet replay ring, the inputs lazy rollback replays: slot s at s times one slot's ring. */
	Count replay_rings;
	Count bytes;
};

/** Buffers start on a 256-byte dram chunk. */
inline constexpr Count workspace_alignment = 256;

[[nodiscard]] constexpr auto workspace(RowBudget budget) -> Workspace {
	auto layout = Workspace{};
	layout.partials = 0;
	layout.partial_flags = align_up(layout.partials + exchange_depth * exchange_tiles * partial_tile_bytes(budget), workspace_alignment);
	layout.candidates = align_up(layout.partial_flags + exchange_depth * exchange_tiles * flag_bytes, workspace_alignment);
	layout.candidate_flags = align_up(layout.candidates + exchange_depth * candidate_slot_bytes(budget), workspace_alignment);
	layout.fixups = align_up(layout.candidate_flags + exchange_depth * flag_bytes, workspace_alignment);
	layout.fixup_flags = align_up(layout.fixups + sms * fixup_slot_bytes(budget), workspace_alignment);
	// One completion counter per phase of the deepest cycle: a consumer waits for every SM to arrive at its
	// dependency's counter.
	layout.counters = align_up(layout.fixup_flags + sms * flag_bytes, workspace_alignment);
	layout.attention_records = align_up(layout.counters + max_cycle_phases * flag_bytes, workspace_alignment);
	layout.replay_rings = align_up(layout.attention_records + attn::record_bytes(budget), workspace_alignment);
	layout.bytes = layout.replay_rings + model::slots * deltanet::ring_bytes_per_slot;
	return layout;
}

[[nodiscard]] consteval auto workspaces_fit() -> bool {
	for (auto const budget : model::row_budgets) {
		if (workspace(budget).bytes > model::workspace_bytes) {
			return false;
		}
	}
	return true;
}

static_assert(workspaces_fit(), "the schedule's buffers fit the card's workspace");

}

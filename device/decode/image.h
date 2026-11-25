// SPDX-License-Identifier: AGPL-3.0-only
//
// Where each image phase's data lies in a card's image, as the kernel looks it up at run time: the start of the run
// its producer streams, each ring placement's block scales, and each side read. Computed once at compile time from the
// schedule's image layout (both cards' layouts agree in every offset), in units of the side alignment so the table
// stays small. It uses no standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "model.h"
#include "schedule/geometry.h"
#include "schedule/program.h"

namespace engine::device::decode {

using model::Count;
using schedule::Op;
using schedule::Part;

/** The table's unit: every run starts on a tile and every side read on a dram chunk. */
inline constexpr Count image_unit = schedule::side_alignment;

static_assert(schedule::tile_bytes % image_unit == 0);

inline constexpr Count max_ring_reads = sizeof(schedule::Reads{}.ring) / sizeof(schedule::Read);
inline constexpr Count max_side_reads = sizeof(schedule::Reads{}.side) / sizeof(schedule::Read);

/** One image phase's offsets, in units of `image_unit`. */
struct PhaseOffsets {
	unsigned run;
	unsigned side[max_side_reads];
	/** Per ring placement, per interleaved tensor: the side read holding its block scales. */
	unsigned scales[max_ring_reads][2];
};

struct ImageTable {
	PhaseOffsets phase[schedule::image_phases];
};

[[nodiscard]] consteval auto in_units(Count offset) -> unsigned {
	if (offset % image_unit != 0 || offset / image_unit >= (Count{1} << 32U)) {
		// Not a constant expression: an offset the table cannot hold.
		__builtin_unreachable();
	}
	return static_cast<unsigned>(offset / image_unit);
}

/** The side read of `op` that is `part` of inventory tensor `tensor`; resolved while compiling. */
[[nodiscard]] consteval auto side_index(Op op, Count tensor, Part part) -> Count {
	auto const r = schedule::reads(op);
	for (auto index = Count{0}; index < r.side_count; ++index) {
		if (r.side[index].tensor[0] == tensor && r.side[index].part == part) {
			return index;
		}
	}
	// Not a constant expression: the phase reads no such side data.
	__builtin_unreachable();
}

[[nodiscard]] consteval auto plan_image_table() -> ImageTable {
	auto const layout = schedule::image(0);
	auto table = ImageTable{};
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		auto const& phase = layout.phases[index];
		auto& entry = table.phase[index];
		entry.run = in_units(schedule::run(layout, phase).offset);
		for (auto side = Count{0}; side < phase.side.length; ++side) {
			entry.side[side] = in_units(layout.placements[phase.side.begin + side].offset);
		}
		for (auto ring = Count{0}; ring < phase.ring.length; ++ring) {
			auto const& read = layout.placements[phase.ring.begin + ring].read;
			for (auto which = Count{0}; which < read.tensor_count; ++which) {
				for (auto side = Count{0}; side < phase.side.length; ++side) {
					auto const& candidate = layout.placements[phase.side.begin + side].read;
					if (candidate.part == Part::Scales && candidate.tensor[0] == read.tensor[which] && candidate.select == read.select) {
						entry.scales[ring][which] = in_units(layout.placements[phase.side.begin + side].offset);
					}
				}
			}
		}
	}
	return table;
}

inline constexpr ImageTable image_table = plan_image_table();

[[nodiscard]] consteval auto image_phase_in(Op op, model::Stream stream) -> Count {
	auto const layout = schedule::image(0);
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		if (layout.phases[index].step.op == op && layout.phases[index].step.stream == stream) {
			return index;
		}
	}
	// Not a constant expression: no image phase computes `op` in `stream`.
	__builtin_unreachable();
}

/** The image phase of each deltanet layer's core, by deltanet ordinal: where its side weights are. */
struct CorePhases {
	unsigned phase[model::trunk_layers_of(model::Mixer::LinearAttention)];
};

[[nodiscard]] consteval auto plan_core_phases() -> CorePhases {
	auto const layout = schedule::image(0);
	auto phases = CorePhases{};
	auto ordinal = Count{0};
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		if (layout.phases[index].step.op == Op::DeltanetCore && layout.phases[index].step.stream == model::Stream::Body) {
			phases.phase[ordinal] = static_cast<unsigned>(index);
			++ordinal;
		}
	}
	return phases;
}

inline constexpr CorePhases core_phases = plan_core_phases();

[[nodiscard]] constexpr auto deltanet_core_phase(int layer) -> Count {
	return core_phases.phase[layer];
}

/** Whether every e4m3 ring placement found its scales: an entry left at zero would read the image's first bytes. */
[[nodiscard]] consteval auto scales_found() -> bool {
	auto const layout = schedule::image(0);
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		auto const& phase = layout.phases[index];
		for (auto ring = Count{0}; ring < phase.ring.length; ++ring) {
			auto const& read = layout.placements[phase.ring.begin + ring].read;
			for (auto which = Count{0}; which < read.tensor_count; ++which) {
				auto const e4m3 = model::inventory[read.tensor[which]].format == model::Format::E4m3;
				if (e4m3 && image_table.phase[index].scales[ring][which] == 0) {
					return false;
				}
			}
		}
	}
	return true;
}

static_assert(scales_found(), "every e4m3 tile run has its block scales in the same phase");

}

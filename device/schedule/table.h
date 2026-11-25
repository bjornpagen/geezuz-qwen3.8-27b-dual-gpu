// SPDX-License-Identifier: AGPL-3.0-only
//
// Where each image phase's data lies in a card's image, as kernels and launches look it up at run time: each ring
// placement's first tile and its tensors' block scales, and each side read. Computed once at compile time from
// `image(card)` (both cards' layouts agree in every offset), so decode, prefill and their hosts index one table and
// never search the layout. It uses no standard headers so that it compiles unchanged as device code and inside a
// module purview.
#pragma once

#include "model.h"
#include "schedule/program.h"

namespace engine::device::schedule {

/** The table's unit: every run starts on a tile and every side read on a dram chunk. */
inline constexpr Count image_unit = side_alignment;

static_assert(tile_bytes % image_unit == 0, "a tile boundary is a table unit");

/** An offset into a card's image, held in table units so the table the decode kernel indexes stays small. */
struct ImageOffset {
	unsigned units;

	[[nodiscard]] constexpr auto bytes() const -> Count {
		return Count{units} * image_unit;
	}
};

[[nodiscard]] consteval auto image_offset(Count bytes) -> ImageOffset {
	if (bytes % image_unit != 0 || bytes / image_unit >= (Count{1} << 32U)) {
		// Not a constant expression: an offset the table cannot hold.
		__builtin_unreachable();
	}
	return {.units = static_cast<unsigned>(bytes / image_unit)};
}

/** One image phase's offsets. */
struct PhaseOffsets {
	/** Per ring placement: its first tile. The first placement's starts the run the phase's producers stream. */
	ImageOffset ring[max_ring_reads];
	/** Per ring placement, per interleaved tensor: the side read holding its block scales (e4m3 placements only). */
	ImageOffset scales[max_ring_reads][2];
	ImageOffset side[max_side_reads];
};

struct ImageTable {
	PhaseOffsets phase[image_phases];
};

[[nodiscard]] consteval auto plan_image_table() -> ImageTable {
	auto const layout = image(0);
	auto table = ImageTable{};
	for (auto index = Count{0}; index < image_phases; ++index) {
		auto const& phase = layout.phases[index];
		auto& entry = table.phase[index];
		for (auto side = Count{0}; side < phase.side.length; ++side) {
			entry.side[side] = image_offset(layout.placements[phase.side.begin + side].offset);
		}
		for (auto ring = Count{0}; ring < phase.ring.length; ++ring) {
			auto const& placed = layout.placements[phase.ring.begin + ring];
			entry.ring[ring] = image_offset(placed.offset);
			for (auto which = Count{0}; which < placed.read.tensor_count; ++which) {
				for (auto side = Count{0}; side < phase.side.length; ++side) {
					auto const& candidate = layout.placements[phase.side.begin + side];
					if (candidate.read.part == Part::Scales && candidate.read.tensor[0] == placed.read.tensor[which] &&
					    candidate.read.select == placed.read.select) {
						entry.scales[ring][which] = image_offset(candidate.offset);
					}
				}
			}
		}
	}
	return table;
}

inline constexpr ImageTable image_table = plan_image_table();

/** The side read of `op` that is `part` of inventory tensor `tensor`; resolved while compiling. */
[[nodiscard]] consteval auto side_index(Op op, Count tensor, Part part) -> Count {
	auto const r = reads(op);
	for (auto index = Count{0}; index < r.side_count; ++index) {
		if (r.side[index].tensor[0] == tensor && r.side[index].part == part) {
			return index;
		}
	}
	// Not a constant expression: the phase reads no such side data.
	__builtin_unreachable();
}

/** The first image phase computing `op` in `stream`: the geometry every layer of that op in that stream shares. */
[[nodiscard]] consteval auto first_phase(Op op, Stream stream) -> Count {
	auto const layout = image(0);
	for (auto index = Count{0}; index < image_phases; ++index) {
		if (layout.phases[index].step.op == op && layout.phases[index].step.stream == stream) {
			return index;
		}
	}
	// Not a constant expression: no image phase computes `op` in `stream`.
	__builtin_unreachable();
}

/** Whether every e4m3 ring placement found its scales: an entry left at zero would read the image's first bytes. */
[[nodiscard]] consteval auto scales_found() -> bool {
	auto const layout = image(0);
	for (auto index = Count{0}; index < image_phases; ++index) {
		auto const& phase = layout.phases[index];
		for (auto ring = Count{0}; ring < phase.ring.length; ++ring) {
			auto const& read = layout.placements[phase.ring.begin + ring].read;
			for (auto which = Count{0}; which < read.tensor_count; ++which) {
				if (model::inventory[read.tensor[which]].format == model::Format::E4m3 &&
				    image_table.phase[index].scales[ring][which].units == 0) {
					return false;
				}
			}
		}
	}
	return true;
}

static_assert(scales_found(), "every e4m3 tile run has its block scales in the same phase");

}

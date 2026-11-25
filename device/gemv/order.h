// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight matmuls' geometry and their one summation order, held once for the decode gemv, the prefill gemm and
// the host emulation that checks both. Pure constexpr over the schedule.
//
// A phase's units are split over the SMs by `schedule::share`, independent of the row count. Each output element is
// then summed in exactly one order (the model's `Sum::Weights`):
// - per SM, for an e4m3 matrix: each unit's tile is an mma chain over its 8 k16 steps from +0, folded as
//   partial = fma(chain, block scale, partial) from +0, units in ascending k;
// - per SM, for a bf16 matrix: one mma chain over all the SM's k16 steps of the output tile, from +0;
// - the output tile's SM partials added left to right in SM order.
// A partial is never -0 (an fma with a +0 addend cannot return -0 unless its product is -0 and its addend too), so
// "from +0" and "from the first partial" give the same bits.
//
// The prefill gemm walks k in the same segments, so for the same weights and activation row it can produce the
// decode's bits exactly.
#pragma once

#include "model.h"
#include "schedule/geometry.h"

namespace engine::device::gemv {

using Count = model::Count;
using model::Range;

/** One ring placement of a phase, as the matmul walks it. Units are counted in the phase's whole run. */
struct Matrix {
	/** Units in the phase's run: the domain the SMs split. */
	Count units;
	/** The placement's first unit within the run. */
	Count first;
	/** 128-row output tiles. */
	Count output_tiles;
	/** Units per output tile, in ascending k. */
	Count k_tiles;
	/** Tiles per unit: one, or two when gate_proj and up_proj interleave row block by row block. */
	Count tensors;
	model::Format format;
};

/** Rows of an output tile, and of a weight tile. */
inline constexpr Count tile_rows = schedule::tile_rows;

/** k one unit spans: a whole scale block for e4m3, 64 for a bf16 tile. */
[[nodiscard]] constexpr auto unit_k(Matrix const& matrix) -> Count {
	return schedule::tile_columns(matrix.format);
}

/** The matrix's k extent: the activation row length it reads. */
[[nodiscard]] constexpr auto k_extent(Matrix const& matrix) -> Count {
	return matrix.k_tiles * unit_k(matrix);
}

/** Units of the placement. */
[[nodiscard]] constexpr auto placement_units(Matrix const& matrix) -> Count {
	return matrix.output_tiles * matrix.k_tiles;
}

/** The SM holding unit `unit` of a run of `units`. */
[[nodiscard]] constexpr auto holder(Count units, Count unit) -> Count {
	return schedule::finisher(units, unit, 1);
}

/** The units of output tile `tile`, in run coordinates. */
[[nodiscard]] constexpr auto tile_units(Matrix const& matrix, Count tile) -> Range {
	return {.begin = matrix.first + tile * matrix.k_tiles, .length = matrix.k_tiles};
}

/** The SMs holding units of output tile `tile`: the first holder and how many. */
[[nodiscard]] constexpr auto tile_holders(Matrix const& matrix, Count tile) -> Range {
	auto const units = tile_units(matrix, tile);
	auto const first = holder(matrix.units, units.begin);
	auto const last = holder(matrix.units, units.begin + units.length - 1);
	return {.begin = first, .length = last - first + 1};
}

/** The k tiles of output tile `tile` that SM `sm` sums, as a range of k-tile indices (empty when it holds none). */
[[nodiscard]] constexpr auto segment(Matrix const& matrix, Count tile, Count sm) -> Range {
	auto const units = tile_units(matrix, tile);
	auto const share = schedule::share(matrix.units, sm);
	auto const begin = share.begin > units.begin ? share.begin : units.begin;
	auto const share_end = share.begin + share.length;
	auto const units_end = units.begin + units.length;
	auto const end = share_end < units_end ? share_end : units_end;
	if (end <= begin) {
		return {.begin = 0, .length = 0};
	}
	return {.begin = begin - units.begin, .length = end - begin};
}

/** The placement's units SM `sm` holds, in run coordinates. */
[[nodiscard]] constexpr auto held(Matrix const& matrix, Count sm) -> Range {
	auto const share = schedule::share(matrix.units, sm);
	auto const begin = share.begin > matrix.first ? share.begin : matrix.first;
	auto const share_end = share.begin + share.length;
	auto const placement_end = matrix.first + placement_units(matrix);
	auto const end = share_end < placement_end ? share_end : placement_end;
	return end <= begin ? Range{.begin = begin, .length = 0} : Range{.begin = begin, .length = end - begin};
}

/** How an SM stands to an output tile it holds units of. */
enum class Holding : unsigned char {
	/** It holds every unit: its partial is the tile's sum. */
	Whole,
	/** It holds the last unit but not the first: it adds the earlier holders' partials, then its own. */
	Finishes,
	/** It holds some units but not the last: its partial goes to the finisher through its fixup slot. */
	Contributes,
};

[[nodiscard]] constexpr auto holding(Matrix const& matrix, Count tile, Count sm) -> Holding {
	auto const holders = tile_holders(matrix, tile);
	if (holders.length == 1) {
		return Holding::Whole;
	}
	return sm == holders.begin + holders.length - 1 ? Holding::Finishes : Holding::Contributes;
}

/**
 * The one geometry fact the fixup protocol rests on: an SM contributes to at most one tile, its last, and finishes
 * after others at most one tile, its first; so one fixup slot per SM suffices, and a finisher's stash of its own first
 * partial can wait in its slot until the slot's contribution is written at the end of its share.
 */
[[nodiscard]] constexpr auto fixups_fit(Matrix const& matrix) -> bool {
	for (auto sm = Count{0}; sm < schedule::sms; ++sm) {
		auto const mine = held(matrix, sm);
		if (mine.length == 0) {
			continue;
		}
		auto const first_tile = (mine.begin - matrix.first) / matrix.k_tiles;
		auto const last_tile = (mine.begin + mine.length - 1 - matrix.first) / matrix.k_tiles;
		for (auto tile = first_tile; tile <= last_tile; ++tile) {
			auto const how = holding(matrix, tile, sm);
			if ((how == Holding::Contributes && tile != last_tile) || (how == Holding::Finishes && tile != first_tile)) {
				return false;
			}
		}
	}
	return true;
}

/** The ring placement `ring_index` of image phase `phase` of a card layout. */
[[nodiscard]] constexpr auto matrix_of(schedule::CardImage const& layout, Count phase, Count ring_index) -> Matrix {
	auto const& at = layout.phases[phase];
	auto const whole = schedule::run(layout, at);
	auto first = Count{0};
	for (auto index = at.ring.begin; index < at.ring.begin + ring_index; ++index) {
		auto const& before = layout.placements[index];
		first += before.bytes / schedule::unit_bytes(before);
	}
	auto const& placement = layout.placements[at.ring.begin + ring_index];
	auto const units = placement.bytes / schedule::unit_bytes(placement);
	auto const k_tiles = schedule::units_per_output(placement);
	return {
	    .units = whole.units,
	    .first = first,
	    .output_tiles = units / k_tiles,
	    .k_tiles = k_tiles,
	    .tensors = placement.read.tensor_count,
	    .format = model::inventory[placement.read.tensor[0]].format,
	};
}

/** The ring placement `ring_index` of image phase `phase` (both cards' layouts agree in every offset). */
[[nodiscard]] consteval auto matrix(Count phase, Count ring_index) -> Matrix {
	return matrix_of(schedule::image(0), phase, ring_index);
}

/** The first image phase whose step is `op` (layer 0 of its stream): the geometry every layer of that op shares. */
[[nodiscard]] consteval auto phase_of(schedule::Op op) -> Count {
	auto const layout = schedule::image(0);
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		if (layout.phases[index].step.op == op) {
			return index;
		}
	}
	// Not a constant expression: no image phase computes `op`.
	__builtin_unreachable();
}

/** The trunk's mlp: gate_proj and up_proj interleaved, then down_proj, as one card streams them. */
inline constexpr Matrix gate_up = matrix(phase_of(schedule::Op::MlpIn), 0);
inline constexpr Matrix down = matrix(phase_of(schedule::Op::MlpOut), 0);

static_assert(gate_up.tensors == 2 && gate_up.format == model::Format::E4m3 && k_extent(gate_up) == model::hidden);
static_assert(down.tensors == 1 && down.output_tiles * tile_rows == model::hidden && k_extent(down) == model::ffn_per_card);

/**
 * Whether every ring placement of the image keeps the fixup protocol's one-slot law. Every layer of an op in one stream
 * has the same geometry, so each (op, stream) is checked once.
 */
[[nodiscard]] consteval auto every_matrix_fits() -> bool {
	auto const layout = schedule::image(0);
	for (auto phase = Count{0}; phase < schedule::image_phases; ++phase) {
		auto const& step = layout.phases[phase].step;
		auto seen = false;
		for (auto earlier = Count{0}; earlier < phase; ++earlier) {
			seen = seen || (layout.phases[earlier].step.op == step.op && layout.phases[earlier].step.stream == step.stream);
		}
		if (seen) {
			continue;
		}
		for (auto index = Count{0}; index < layout.phases[phase].ring.length; ++index) {
			if (!fixups_fit(matrix_of(layout, phase, index))) {
				return false;
			}
		}
	}
	return true;
}

static_assert(every_matrix_fits(), "an SM contributes to at most its last tile and finishes after others at most its first");

}

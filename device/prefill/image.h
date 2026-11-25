// SPDX-License-Identifier: AGPL-3.0-only
//
// Where each weight a prefill chunk reads lies in a card's image: per image phase, each ring placement's first tile and
// its tensors' block scales, and each side read; and per decoder layer, the image phase of each of its ops. Computed
// once at compile time from the schedule's image layout (both cards' layouts agree in every offset), so the host turns
// a layer into kernel parameters by indexing, never by searching. It uses no standard headers so that it compiles
// unchanged as device code and inside a module purview.
#pragma once

#include "model.h"
#include "schedule/geometry.h"
#include "schedule/program.h"

namespace engine::device::prefill {

using model::Count;
using schedule::Op;
using schedule::Part;

inline constexpr Count max_ring_reads = sizeof(schedule::Reads{}.ring) / sizeof(schedule::Read);
inline constexpr Count max_side_reads = sizeof(schedule::Reads{}.side) / sizeof(schedule::Read);

/** One image phase's weights, as byte offsets from the image's start. */
struct PhaseOffsets {
	Count ring[max_ring_reads];
	/** Per ring placement, per interleaved tensor: its block scales (e4m3 placements only). */
	Count scales[max_ring_reads][2];
	Count side[max_side_reads];
};

struct ImageTable {
	PhaseOffsets phase[schedule::image_phases];
};

[[nodiscard]] consteval auto plan_image_table() -> ImageTable {
	auto const layout = schedule::image(0);
	auto table = ImageTable{};
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		auto const& phase = layout.phases[index];
		auto& entry = table.phase[index];
		for (auto side = Count{0}; side < phase.side.length; ++side) {
			entry.side[side] = layout.placements[phase.side.begin + side].offset;
		}
		for (auto ring = Count{0}; ring < phase.ring.length; ++ring) {
			auto const& placed = layout.placements[phase.ring.begin + ring];
			entry.ring[ring] = placed.offset;
			for (auto which = Count{0}; which < placed.read.tensor_count; ++which) {
				for (auto side = Count{0}; side < phase.side.length; ++side) {
					auto const& candidate = layout.placements[phase.side.begin + side].read;
					if (candidate.part == Part::Scales && candidate.tensor[0] == placed.read.tensor[which] &&
					    candidate.select == placed.read.select) {
						entry.scales[ring][which] = layout.placements[phase.side.begin + side].offset;
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
	auto const r = schedule::reads(op);
	for (auto index = Count{0}; index < r.side_count; ++index) {
		if (r.side[index].tensor[0] == tensor && r.side[index].part == part) {
			return index;
		}
	}
	// Not a constant expression: the phase reads no such side data.
	__builtin_unreachable();
}

/** The first image phase computing `op` in `stream`: the geometry every layer of that op in that stream shares. */
[[nodiscard]] consteval auto first_phase(Op op, model::Stream stream) -> Count {
	auto const layout = schedule::image(0);
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		if (layout.phases[index].step.op == op && layout.phases[index].step.stream == stream) {
			return index;
		}
	}
	// Not a constant expression: no image phase computes `op` in `stream`.
	__builtin_unreachable();
}

/** The image phases one decoder layer's chunk walks; the ops its mixer lacks stay zero. */
struct LayerPhases {
	model::Mixer mixer;
	Count mixer_in;
	Count deltanet_core;
	Count deltanet_z;
	Count mixer_out;
	Count attention_gate;
	Count mlp_in;
	Count mlp_out;
};

/** The image phases of every trunk layer, then the mtp head's layer, and the mtp fc's phase. */
struct LayerTable {
	LayerPhases trunk[model::trunk_layers];
	LayerPhases mtp;
	Count mtp_in;
};

/** Files one image phase's step under its layer: one pass over the layout fills every layer at once. */
constexpr void file(LayerPhases& layer, Op op, Count index) {
	switch (op) {
	case Op::DeltanetIn:
	case Op::AttentionIn:
		layer.mixer_in = index;
		break;
	case Op::DeltanetCore:
		layer.deltanet_core = index;
		break;
	case Op::DeltanetZ:
		layer.deltanet_z = index;
		break;
	case Op::DeltanetOut:
	case Op::AttentionOut:
		layer.mixer_out = index;
		break;
	case Op::AttentionGate:
		layer.attention_gate = index;
		break;
	case Op::MlpIn:
		layer.mlp_in = index;
		break;
	case Op::MlpOut:
		layer.mlp_out = index;
		break;
	case Op::Embed:
	case Op::AttentionCore:
	case Op::AttentionMerge:
	case Op::Head:
	case Op::Verify:
	case Op::MtpIn:
	case Op::DraftHead:
	case Op::Draft:
		break;
	}
}

[[nodiscard]] consteval auto plan_layer_table() -> LayerTable {
	auto const layout = schedule::image(0);
	auto table = LayerTable{};
	for (auto layer = Count{0}; layer < model::trunk_layers; ++layer) {
		table.trunk[layer].mixer = model::trunk_mixers[layer];
	}
	table.mtp.mixer = model::mtp_mixer;
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		auto const& step = layout.phases[index].step;
		switch (step.stream) {
		case model::Stream::Body:
			file(table.trunk[step.layer], step.op, index);
			break;
		case model::Stream::Mtp:
			file(table.mtp, step.op, index);
			table.mtp_in = step.op == Op::MtpIn ? index : table.mtp_in;
			break;
		case model::Stream::Head:
			break;
		}
	}
	return table;
}

inline constexpr LayerTable layers = plan_layer_table();

/** The final norm's phase: the lm_head's, whose side data holds it. */
inline constexpr Count head_phase = first_phase(Op::Head, model::Stream::Head);

static_assert(model::mtp_layers == 1, "the mtp head is one decoder layer");

/** Whether every e4m3 ring placement found its scales: an entry left at zero would read the image's first bytes. */
[[nodiscard]] consteval auto scales_found() -> bool {
	auto const layout = schedule::image(0);
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		auto const& phase = layout.phases[index];
		for (auto ring = Count{0}; ring < phase.ring.length; ++ring) {
			auto const& read = layout.placements[phase.ring.begin + ring].read;
			for (auto which = Count{0}; which < read.tensor_count; ++which) {
				if (model::inventory[read.tensor[which]].format == model::Format::E4m3 &&
				    image_table.phase[index].scales[ring][which] == 0) {
					return false;
				}
			}
		}
	}
	return true;
}

static_assert(scales_found(), "every e4m3 tile run has its block scales in the same phase");

}

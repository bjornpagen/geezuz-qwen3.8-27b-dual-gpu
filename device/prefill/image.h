// SPDX-License-Identifier: AGPL-3.0-only
//
// Per decoder layer, the image phase of each op a prefill chunk walks: computed once at compile time from the
// schedule's image layout, so the host turns a layer into kernel parameters by indexing the schedule's image table,
// never by searching. It uses no standard headers so that it compiles unchanged as device code and inside a module
// purview.
#pragma once

#include "model.h"
#include "schedule/program.h"
#include "schedule/table.h"

namespace engine::device::prefill {

using model::Count;
using schedule::first_phase;
using schedule::Op;

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

}

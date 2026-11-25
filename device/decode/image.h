// SPDX-License-Identifier: AGPL-3.0-only
//
// The image phase of each deltanet layer's core, by deltanet ordinal, as the rollback replay looks it up at run time:
// computed once at compile time from the schedule's image layout, beside the schedule's image table the kernel reads
// every offset from. It uses no standard headers so that it compiles unchanged as device code and inside a module
// purview.
#pragma once

#include "model.h"
#include "schedule/program.h"
#include "schedule/table.h"

namespace engine::device::decode {

using model::Count;
using schedule::first_phase;
using schedule::image_table;
using schedule::ImageOffset;
using schedule::Op;
using schedule::Part;
using schedule::side_index;

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

}

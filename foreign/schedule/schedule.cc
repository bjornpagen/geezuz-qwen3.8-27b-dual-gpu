// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode cycle's schedule re-exported to host code, so the baker, the launch functions and the tests read the very
// program the kernels compile. Device code cannot import modules, so the schedule is a device header; this unit gives
// its declarations a module surface unchanged.
module;

#include "schedule/geometry.h"

export module engine.schedule;

namespace engine::schedule {

namespace device = engine::device::schedule;

export using device::Blocks;
export using device::CardImage;
export using device::Exchange;
export using device::ImagePhase;
export using device::Instance;
export using device::Op;
export using device::Part;
export using device::Pass;
export using device::PassKind;
export using device::Phase;
export using device::Placement;
export using device::Program;
export using device::Read;
export using device::Ring;
export using device::Roles;
export using device::Run;
export using device::Workspace;
export using device::Reads;
export using device::Select;
export using device::Step;
export using device::Tensors;
export using device::Traffic;

export using device::attention_chunk_positions;
export using device::candidates_per_row;
export using device::decode_roles;
export using device::epilogue_scratch_bytes;
export using device::exchange_depth;
export using device::exchange_tiles;
export using device::flag_bytes;
export using device::max_ring_slots;
export using device::mbarrier_bytes;
export using device::min_ring_slots;
export using device::mma_rows;
export using device::sms;
export using device::tile_span;
export using device::warpgroup_warps;
export using device::workspace_alignment;
export using device::baseline_row_budget;
export using device::dataflow_row_budget;
export using device::deltanet_state_bytes_per_layer_per_slot;
export using device::draft_depths;
export using device::image_phases;
export using device::kv_bytes_per_position_per_layer;
export using device::max_cycle_phases;
export using device::max_dependencies;
export using device::mtp_phases;
export using device::placements;
export using device::side_alignment;
export using device::step_row_tiles;
export using device::tensor_count;
export using device::tensors;
export using device::tile_bytes;
export using device::tile_rows;
export using device::verify_phases;

export using device::activation_bytes;
export using device::blocks;
export using device::candidate_slot_bytes;
export using device::decode_budget;
export using device::decode_shared_bytes;
export using device::extent;
export using device::finisher;
export using device::fixup_slot_bytes;
export using device::partial_tile_bytes;
export using device::ring;
export using device::run;
export using device::share;
export using device::unit_bytes;
export using device::units_per_output;
export using device::warps;
export using device::workspace;
export using device::cycle;
export using device::cycle_bytes;
export using device::exchange;
export using device::exchange_ordinal;
export using device::exchanges;
export using device::holds;
export using device::image;
export using device::phase_bytes;
export using device::program;
export using device::read_bytes;
export using device::reads;
export using device::row_tiles;
export using device::selected;
export using device::shape;
export using device::tile_columns;
export using device::traffic;

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill's workspace, parameter blocks and image table re-exported to host code unchanged, so the launches fill the
// very structs the kernels read and index the very offsets the schedule laid out.
module;

#include "prefill/layout.h"
#include "prefill/image.h"

export module engine.prefill_device:layout;

namespace engine::prefill_device {

namespace device = engine::device::prefill;

export using device::Address;
export using device::CheckParams;
export using device::DecayParams;
export using device::EmbedParams;
export using device::GemmParams;
export using device::ImageTable;
export using device::KeepParams;
export using device::LayerPhases;
export using device::LayerTable;
export using device::MtpParams;
export using device::NormParams;
export using device::PhaseOffsets;
export using device::PushParams;
export using device::ResidualParams;
export using device::Rows;
export using device::Weights;
export using device::Workspace;
export using device::ZeroParams;

export using device::block_rows;
export using device::exchange_floats;
export using device::exchange_items;
export using device::exchange_slots;
export using device::gemm_blocks;
export using device::hidden_tiles;
export using device::image_table;
export using device::head_phase;
export using device::layers;
export using device::max_row_blocks;
export using device::max_rows;
export using device::rows_blocks;
export using device::rows_budget;
export using device::workspace;

export using device::exchange_offset;
export using device::exchange_slot;
export using device::first_phase;
export using device::item_index;
export using device::padded;
export using device::side_index;

}

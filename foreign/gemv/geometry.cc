// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight matmuls' geometry and summation order, the warps' operand offsets, and the standalone kernels' shapes and
// parameters, re-exported to host code from the device headers the kernels compile, so the host emulation, the
// microbenchmark drivers and the tests read the very definitions the kernels index with.
module;

#include "gemv/layout.h"
#include "gemv/order.h"
#include "gemv/selfcheck.h"

export module engine.gemv:geometry;

namespace engine::gemv {

namespace device = engine::device::gemv;

export using device::Count;
export using device::Holding;
export using device::Matrix;
export using device::Range;

export using device::down;
export using device::fixups_fit;
export using device::gate_up;
export using device::held;
export using device::holder;
export using device::holding;
export using device::k_extent;
export using device::matrix;
export using device::phase_of;
export using device::placement_units;
export using device::segment;
export using device::tile_holders;
export using device::tile_rows;
export using device::tile_units;
export using device::unit_k;

namespace layout {
export using device::layout::activation_offset;
export using device::layout::ActivationTile;
export using device::layout::Element;
export using device::layout::element;
export using device::layout::slot_index;
export using device::layout::weight_offset;

namespace prefill {
export using device::layout::prefill::ActivationTile;
export using device::layout::prefill::block_rows;
export using device::layout::prefill::element;
export using device::layout::prefill::first_row;
export using device::layout::prefill::output_groups;
export using device::layout::prefill::row_groups;
export using device::layout::prefill::tile_slab;
export using device::layout::prefill::warp_row_tiles;
export using device::layout::prefill::warp_slabs;
}
}

namespace selfcheck {
export using device::selfcheck::Address;
export using device::selfcheck::blocks;
export using device::selfcheck::Buffers;
export using device::selfcheck::buffers;
export using device::selfcheck::first_epoch;
export using device::selfcheck::GemmBenchParams;
export using device::selfcheck::GemmCheckParams;
export using device::selfcheck::GemvCheckParams;
export using device::selfcheck::GemvStreamParams;
export using device::selfcheck::prefill_check_rows;
export using device::selfcheck::prefill_rows;
export using device::selfcheck::product_width;
export using device::selfcheck::ring_stages;
export using device::selfcheck::residual_rows;
export using device::selfcheck::residual_tiles;
export using device::selfcheck::ResidualCheckParams;
export using device::selfcheck::threads;
}

}

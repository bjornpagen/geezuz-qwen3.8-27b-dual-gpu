// SPDX-License-Identifier: AGPL-3.0-only
//
// What the gemv's self-check and microbenchmark kernels run and how their buffers are laid out, held once for the
// kernels and the host drivers that fill and read them (through `engine.gemv`). Pure constexpr.
//
// Every kernel here is a real gemv or gemm over a real placement's geometry: the trunk's down_proj (one tensor) and
// gate_proj with up_proj (interleaved), on 82 blocks of the decode's warp roles, one block per SM, so each block is the
// SM the static split names. The host bakes the weights, writes the activations and scales, and for a self-check the
// expected outputs its emulation of every warp predicts; the kernel reports each output that differs.
#pragma once

#include "gemv/order.h"
#include "model.h"
#include "prims/budget.h"
#include "schedule/geometry.h"

namespace engine::device::gemv::selfcheck {

/** The persistent kernels' grid: one block per SM. */
inline constexpr unsigned blocks = static_cast<unsigned>(schedule::sms);

/** The decode kernel's warps (producer, two consumer warpgroups, three that idle here), as the schedule fixes them. */
inline constexpr int threads = static_cast<int>(schedule::warps(schedule::decode_roles)) * 32;

/** Ring stages of the gemv kernels: the decode's smallest ring. */
inline constexpr int ring_stages = static_cast<int>(schedule::min_ring_slots);

/** The epoch the self-checks publish their fixups at; the host zeroes the flags before each launch. */
inline constexpr unsigned long long first_epoch = 1;

/** Buffer sizes of a gemv over `matrix` at `row_tiles` n8 tiles of rows. */
struct Buffers {
	/** bf16 activation rows as 16-byte chunks: rows x k_extent. */
	Count activation_chunks;
	/** bf16 scales per tensor: output tiles x k tiles. */
	Count scales;
	/** fp32 values of every SM's fixup slot. */
	Count fixup_floats;
	/** fp32 outputs as [tile][tensor][row][128]. */
	Count out_floats;
	/** bf16 product words of interleaved gate and up: rows x output width / 2. */
	Count product_words;
};

[[nodiscard]] constexpr auto buffers(Matrix const& matrix, int row_tiles) -> Buffers {
	auto const rows = Count{8} * static_cast<Count>(row_tiles);
	auto const tile_floats = matrix.tensors * rows * tile_rows;
	return {
	    .activation_chunks = rows * k_extent(matrix) / 8,
	    .scales = matrix.output_tiles * matrix.k_tiles,
	    .fixup_floats = Count{blocks} * tile_floats,
	    .out_floats = matrix.output_tiles * tile_floats,
	    .product_words = rows * matrix.output_tiles * tile_rows / 2,
	};
}

/** The width of down_proj's input: the product's row length. */
inline constexpr int product_width = static_cast<int>(model::ffn_per_card);

static_assert(gate_up.output_tiles * tile_rows == model::ffn_per_card, "gate and up rows are down_proj's input");

/** U22: one prefill chunk of rows through the gemm. */
inline constexpr unsigned prefill_rows = 2048;

/** The prefill self-checks: a few row blocks, enough to cross every segment boundary of every tile. */
inline constexpr unsigned prefill_check_rows = 64;

// The kernels' one parameter struct each, device addresses as 64-bit integers; the kernels turn them back into typed
// pointers. Buffers are as `buffers` sizes them: tiles are the placement's baked 16 KiB tiles in stream order, activations
// bf16 rows as 16-byte chunks, scales bf16 per tensor (one tensor passes the same address twice), fixups fp32 and
// fixup flags 64-bit, a report `prims::selfcheck::Report`, elapsed clocks one 32-bit word per block.

using Address = unsigned long long;

/** `gemv_check_*`: outputs are fp32 tiles [tile][tensor][row][128] or, for gate and up, bf16 product words. */
struct GemvCheckParams {
	Address tiles;
	Address activations;
	Address gate_scales;
	Address up_scales;
	Address fixups;
	Address fixup_flags;
	Address out;
	Address expected;
	Address report;
};

/** `gemv_stream_r*` (U5): down_proj's placement, fp32 tiles out. */
struct GemvStreamParams {
	Address tiles;
	Address activations;
	Address scales;
	Address fixups;
	Address fixup_flags;
	Address out;
	Address elapsed;
};

/** `gemm_check_*`: outputs are fp32 rows [row][width] or, for gate and up, bf16 product words. */
struct GemmCheckParams {
	Address tiles;
	Address activations;
	Address gate_scales;
	Address up_scales;
	Address out;
	Address expected;
	Address report;
};

/** `gemm_prefill_*` (U22). */
struct GemmBenchParams {
	Address tiles;
	Address activations;
	Address gate_scales;
	Address up_scales;
	Address out;
	Address elapsed;
};

/** Rows the residual self-check updates: the larger row budget. */
inline constexpr int residual_rows = static_cast<int>(model::rows(model::RowBudget::FourTiles));
/** Output tiles of the hidden state: one block of one warp each in the residual self-check. */
inline constexpr unsigned residual_tiles = static_cast<unsigned>(model::hidden / tile_rows);

/**
 * `gemv_check_residual`: both cards' sums as [tile][row][128] fp32 (`own`, `peer`), the residual [row][hidden] fp32,
 * updated in place, and the squares [tile][row] it writes; both compared with the host's.
 */
struct ResidualCheckParams {
	Address own;
	Address peer;
	Address residual;
	Address squares;
	Address expected_residual;
	Address expected_squares;
	Address report;
};

}

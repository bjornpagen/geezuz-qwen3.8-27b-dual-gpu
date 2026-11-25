// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill's data on one card, held once for its kernels and the host that launches them: the chunk's row capacity, the
// prefill workspace (the rows each kernel of a chunk hands the next, the cross-card exchange of the row-parallel
// sums, the carried final-norm row of each slot's last chunk), and every kernel's one parameter block. A chunk is one
// slot's prompt tokens at positions [from, from + count); its rows are [0, count) of every row buffer, padded with
// zero rows to whole 32-row blocks of the gemm. It uses no standard headers so that it compiles unchanged as device
// code and inside a module purview.
#pragma once

#include "attn/geometry.h"
#include "deltanet/layout.h"
#include "gemv/layout.h"
#include "gemv/order.h"
#include "model.h"
#include "prims/budget.h"
#include "schedule/geometry.h"

namespace engine::device::prefill {

using model::Count;

/** Prompt rows one chunk holds at most: the scheduler's chunk, about a second of prefill. */
inline constexpr Count max_rows = 2048;
/** The gemm's row block: a chunk computes whole blocks, its padding rows zero. */
inline constexpr Count block_rows = gemv::layout::prefill::block_rows;
inline constexpr Count max_row_blocks = max_rows / block_rows;
/** 128-feature tiles of the hidden state: the residual's and the norms' unit, and the exchange's output tiles. */
inline constexpr Count hidden_tiles = model::hidden / schedule::tile_rows;

static_assert(max_rows % block_rows == 0);

/** Rows a chunk of `count` tokens computes: whole gemm blocks. */
[[nodiscard]] constexpr auto padded(Count count) -> Count {
	return model::divide_up(count, block_rows) * block_rows;
}

// ---------------------------------------------------------------------------------------------------------------------
// The cross-card exchange. A row-parallel matmul's work item (one 128-output tile by one 32-row block) is written
// whole as this card's own fp32 partial and pushed whole into the peer's receive slot, then released by a per-item
// flag carrying the exchange's epoch. Two receive slots alternate by epoch: a card pushes exchange e + 2 into a slot
// only after its residual for e + 1 saw the peer's pushes of e + 1, which the peer issued after its own residual for
// e had read the slot.

inline constexpr Count exchange_slots = 2;
inline constexpr Count item_floats = block_rows * schedule::tile_rows;
inline constexpr Count item_bytes = item_floats * sizeof(float);
/** A whole exchange: every output tile of the hidden state by every row block, [tile][row block][32][128] fp32. */
inline constexpr Count exchange_floats = hidden_tiles * max_row_blocks * item_floats;
inline constexpr Count exchange_items = hidden_tiles * max_row_blocks;

/** The receive slot of exchange `epoch` (from 1). */
[[nodiscard]] constexpr auto exchange_slot(unsigned long long epoch) -> Count {
	return static_cast<Count>(epoch % exchange_slots);
}

/** The float offset of (tile, row) in an exchange buffer: item (tile, row / 32), row % 32 within it. */
[[nodiscard]] constexpr auto exchange_offset(Count tile, Count row) -> Count {
	return ((tile * max_row_blocks + row / block_rows) * block_rows + row % block_rows) * schedule::tile_rows;
}

[[nodiscard]] constexpr auto item_index(Count tile, Count row_block) -> Count {
	return tile * max_row_blocks + row_block;
}

// ---------------------------------------------------------------------------------------------------------------------
// The workspace.

struct Workspace {
	/** fp32 [rows][hidden]: the trunk's residual stream, then the mtp head's; and its tile squares, [tile][rows]. */
	Count residual;
	Count squares;
	/** bf16 [rows][hidden]: the current norm's output, the matmuls' activation rows. */
	Count normed;
	/** The deltanet's rows: bf16 qkv projections, fp32 a and b, the core's fp32 normed outputs, bf16 gated outputs. */
	Count qkv;
	Count gates;
	Count core;
	Count gated;
	/** Attention's fp32 projection rows and its gated bf16 outputs. */
	Count query;
	Count key;
	Count value;
	Count attention_gate;
	Count attention_output;
	/** The mlp's bf16 silu(gate) * up rows. */
	Count product;
	/** The exchange: this card's partials, the peer's two receive slots, and their item flags (u64 [slot][item]). */
	Count own;
	Count receive;
	Count flags;
	/** bf16 [rows][hidden]: the final norm of every chunk row, the mtp head's hidden input and the prompt end's row. */
	Count final_normed;
	/** bf16 [slots][hidden]: each slot's final norm at its last prefilled position, the next chunk's first mtp entry. */
	Count carried;
	/** The mtp head's inputs: fp32 embeddings, bf16 hidden rows, their tile squares, and the fc's bf16 input rows. */
	Count mtp_embedding;
	Count mtp_hidden;
	Count embedding_squares;
	Count hidden_squares;
	Count mtp_input;
	Count bytes;
};

inline constexpr Count alignment = schedule::workspace_alignment;

[[nodiscard]] constexpr auto align(Count value) -> Count {
	return model::divide_up(value, alignment) * alignment;
}

inline constexpr Count float_bytes = 4;
inline constexpr Count bf16_bytes = 2;

/** The end of what is planned so far: each buffer starts aligned after the last. */
struct Planned {
	Count end;

	constexpr auto take(Count bytes) -> Count {
		auto const at = align(end);
		end = at + bytes;
		return at;
	}
};

[[nodiscard]] consteval auto plan_workspace() -> Workspace {
	auto w = Workspace{};
	auto planned = Planned{.end = 0};
	auto const take = [&planned](Count bytes) {
		return planned.take(bytes);
	};
	constexpr Count rows = max_rows;
	w.residual = take(rows * model::hidden * float_bytes);
	w.squares = take(hidden_tiles * rows * float_bytes);
	w.normed = take(rows * model::hidden * bf16_bytes);
	w.qkv = take(rows * deltanet::qkv_row_bytes);
	w.gates = take(rows * deltanet::gate_row_bytes);
	w.core = take(rows * deltanet::output_row_bytes);
	w.gated = take(rows * deltanet::value_features * bf16_bytes);
	w.query = take(rows * attn::query_row_floats * float_bytes);
	w.key = take(rows * attn::kv_row_floats * float_bytes);
	w.value = take(rows * attn::kv_row_floats * float_bytes);
	w.attention_gate = take(rows * attn::gate_row_floats * float_bytes);
	w.attention_output = take(rows * attn::output_row_values * bf16_bytes);
	w.product = take(rows * model::ffn_per_card * bf16_bytes);
	w.own = take(exchange_floats * float_bytes);
	w.receive = take(exchange_slots * exchange_floats * float_bytes);
	w.flags = take(exchange_slots * exchange_items * schedule::flag_bytes);
	w.final_normed = take(rows * model::hidden * bf16_bytes);
	w.carried = take(model::slots * model::hidden * bf16_bytes);
	w.mtp_embedding = take(rows * model::hidden * float_bytes);
	w.mtp_hidden = take(rows * model::hidden * bf16_bytes);
	w.embedding_squares = take(hidden_tiles * rows * float_bytes);
	w.hidden_squares = take(hidden_tiles * rows * float_bytes);
	w.mtp_input = take(rows * model::hidden * bf16_bytes);
	w.bytes = align(planned.end);
	return w;
}

inline constexpr Workspace workspace = plan_workspace();

// The decode loop's workspace shares the card's workspace budget; prefill takes at most two thirds of it.
static_assert(workspace.bytes <= model::workspace_bytes * 2 / 3, "the prefill workspace leaves the decode loop its share");
static_assert(workspace.bytes % 16 == 0 && workspace.flags % 16 == 0, "the workspace zeroes in whole chunks");

// ---------------------------------------------------------------------------------------------------------------------
// The row-wise kernels: eight warps, four blocks per SM, a warp per (row, 128-feature tile); the gemm kernels run the
// decode's warp roles on one block per SM.

inline constexpr prims::Budget rows_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 4};
inline constexpr unsigned rows_blocks = static_cast<unsigned>(prims::sm86.sms * rows_budget.blocks_per_sm);
inline constexpr unsigned rows_warps = rows_blocks * static_cast<unsigned>(rows_budget.threads / 32);
inline constexpr unsigned gemm_blocks = static_cast<unsigned>(prims::sm86.sms);

static_assert(prims::check(rows_budget) == prims::Fit::Fits);

// ---------------------------------------------------------------------------------------------------------------------
// Parameters. Addresses travel as integers and become pointers only on the device; every shape is a constant here, so
// a launch carries only the chunk's runtime facts (its rows, its slot, its positions, its layer's image offsets).

using Address = unsigned long long;

/** How many rows a launch reads (the chunk's tokens) and computes (whole gemm blocks, padding rows zero). */
struct Rows {
	unsigned count;
	unsigned computed;
};

/** `prefill_zero`: `chunks` 16-byte chunks from `target` become zero. */
struct ZeroParams {
	Address target;
	unsigned long long chunks;
};

/** `prefill_embed`: the chunk's token embeddings widened into the residual, with their tile squares. */
struct EmbedParams {
	/** The host image's embedding table, mapped into the card: bf16 [vocab][hidden]. */
	Address embedding;
	/** The chunk's token ids, u32 [max_rows], in mapped host memory. */
	Address tokens;
	Address workspace;
	Rows rows;
};

/** `prefill_norm`: x * factor * (1 + w), rounded once, from a residual and its squares into bf16 rows. */
struct NormParams {
	Address residual;
	Address squares;
	/** The norm's bf16 [hidden] weight in the image. */
	Address weight;
	Address out;
	Rows rows;
};

/** `prefill_decay`: in_proj_a and in_proj_b (bf16 [value heads][hidden] each, read whole) on the normed rows. */
struct DecayParams {
	Address a;
	Address b;
	Address workspace;
	Rows rows;
};

/** A placement's weights in the image: its tiles in stream order and its tensors' bf16 block scales (e4m3 only). */
struct Weights {
	Address tiles;
	Address scales[2];
};

/** A gemm whose epilogue writes this card's rows: fp32 or bf16 projections, the gated core, the mlp product. */
struct GemmParams {
	Weights weights;
	Address workspace;
	unsigned row_blocks;
	unsigned unused;
};

/** A row-parallel gemm whose epilogue pushes each finished item to the peer and releases it with `epoch`. */
struct PushParams {
	Weights weights;
	Address workspace;
	/** The peer card's prefill workspace, over nvlink: this card only ever stores into it. */
	Address peer_workspace;
	unsigned long long epoch;
	unsigned row_blocks;
	unsigned unused;
};

/**
 * `prefill_residual_add` and `prefill_residual_start`: once the peer's item of exchange `epoch` arrived, the cards'
 * sum (card 0 + card 1) meets the residual, which it adds to or starts, with the new tile squares.
 */
struct ResidualParams {
	Address workspace;
	unsigned long long epoch;
	Rows rows;
};

static_assert(schedule::flag_bytes == sizeof(unsigned long long), "an exchange flag is one 64-bit epoch");

/**
 * `prefill_mtp_gather` and `prefill_mtp_norm`: the mtp head's entries [first, first + rows.count) at those
 * positions. Entry e embeds the chunk's token at row e + 1 - from and reads the final norm at row e - from, or the
 * slot's carried row for the entry before the chunk (first = from - 1).
 */
struct MtpParams {
	Address embedding;
	Address tokens;
	Address workspace;
	/** pre_fc_norm_embedding and pre_fc_norm_hidden, bf16 [hidden] each. */
	Address embedding_norm;
	Address hidden_norm;
	unsigned slot;
	unsigned from;
	unsigned first;
	/** This card's rank: it keeps its half of both of the fc's inputs. */
	unsigned card;
	Rows rows;
};

/**
 * `prefill_check_words`: the bring-up self-check. `words` 32-bit words at `got` (a workspace region, a kv block, a
 * slot's state image) against the host emulation's `expected`; `report` (a `prims::selfcheck::Report`, zeroed) gets
 * every mismatch's word index as its input.
 */
struct CheckParams {
	Address got;
	Address expected;
	Address report;
	unsigned long long words;
};

/** `prefill_keep`: the final norm of the chunk's last row becomes the slot's carried row and the prompt's end row. */
struct KeepParams {
	Address workspace;
	/** Where the prompt end's tail reads the last row, bf16 [hidden]. */
	Address prompt;
	unsigned slot;
	unsigned row;
};

}

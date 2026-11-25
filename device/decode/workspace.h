// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode kernel's workspace on one card: the schedule's exchange buffers and flags first, exactly as
// `schedule::workspace` lays them out (the peer stores into them at those offsets), then the activation rows the phases
// hand each other, attention's records, the ranking lists, the replay rings and block tables, and what a slot keeps
// from one cycle to the next. Every offset is a compile-time fact, and the whole is the size the card's memory plan
// counts. It uses no standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "attn/geometry.h"
#include "decode/walk.h"
#include "decode/wire.h"
#include "deltanet/layout.h"
#include "model.h"
#include "sampling/draw.h"
#include "schedule/geometry.h"

namespace engine::device::decode {

/** What a slot keeps between cycles: the drafts the kernel proposed and the distributions they were drawn from. */
struct Book {
	sampling::Survivors proposal[max_drafts];
	int chosen[max_drafts];
	unsigned tokens[max_drafts];
};

/** Where each of the workspace's buffers starts, in bytes from its base. */
struct Workspace {
	schedule::Workspace exchange;
	/** This card's own fp32 partial of each output tile of the exchange in flight: [tile][rows][128]. */
	Count own_partials;
	/** The trunk's and the mtp head's fp32 residual streams, [rows][hidden], and their tile squares, [tile][rows]. */
	Count residual;
	Count squares;
	Count mtp_residual;
	Count mtp_squares;
	/** bf16 [rows][hidden]: a norm's output, the final norm's (kept for the mtp), and the mtp norm's (kept for the next step). */
	Count normed;
	Count final_normed;
	Count mtp_normed;
	/** The mtp head's inputs: fp32 embeddings, bf16 hidden rows, their tile squares, and the fc's bf16 input rows. */
	Count mtp_embedding;
	Count mtp_hidden;
	Count embedding_squares;
	Count hidden_squares;
	Count mtp_input;
	/** The deltanet's rows: bf16 qkv projections, fp32 a and b, the core's fp32 normed outputs, bf16 gated outputs. */
	Count qkv;
	Count gates;
	Count core;
	Count gated;
	/** Attention's fp32 projection rows and its bf16 gated outputs; its records are the schedule's. */
	Count query;
	Count key;
	Count value;
	Count attention_gate;
	Count attention_output;
	/** The mlp's bf16 silu(gate) * up rows. */
	Count product;
	/** fp32 logits [rows][vocab_per_card]: the lm_head's, or the draft lm_head's in the first columns. */
	Count logits;
	/** The ranking's two list buffers of 8-byte keys, folds alternating between them. */
	Count lists[2];
	/** This card's ranked candidates, [rows][64]. */
	Count candidates;
	/** The command and the descriptor of the ticket being served, copied from the ring. */
	Count command;
	Count descriptor;
	/** Per live slot: this cycle's outcome. */
	Count outcomes;
	/** Per slot: its draft book. */
	Count books;
	/** The slots' kv block tables, u32 [slots][kv_table_blocks]; their replay rings are the schedule's. */
	Count tables;
	/** The final norm of a prompt's last row, bf16 [hidden], where prefill leaves it for the prompt's end. */
	Count prompt;
	Count bytes;
};

inline constexpr Count alignment = schedule::workspace_alignment;

[[nodiscard]] constexpr auto align(Count value) -> Count {
	return model::divide_up(value, alignment) * alignment;
}

inline constexpr Count float_bytes = 4;
inline constexpr Count bf16_bytes = 2;
inline constexpr Count key_bytes = 8;
inline constexpr Count tiles = schedule::exchange_tiles;

/** Where the next buffer goes: each starts on the alignment after the one before. */
struct Cursor {
	Count end;

	constexpr auto take(Count bytes) -> Count {
		auto const at = align(end);
		end = at + bytes;
		return at;
	}
};

[[nodiscard]] consteval auto plan_workspace() -> Workspace {
	auto w = Workspace{};
	w.exchange = schedule::workspace(budget);
	auto cursor = Cursor{.end = w.exchange.bytes};
	auto const take = [&cursor](Count bytes) {
		return cursor.take(bytes);
	};
	w.own_partials = take(tiles * schedule::partial_tile_bytes(budget));
	w.residual = take(max_rows * model::hidden * float_bytes);
	w.squares = take(tiles * max_rows * float_bytes);
	w.mtp_residual = take(max_rows * model::hidden * float_bytes);
	w.mtp_squares = take(tiles * max_rows * float_bytes);
	w.normed = take(max_rows * model::hidden * bf16_bytes);
	w.final_normed = take(max_rows * model::hidden * bf16_bytes);
	w.mtp_normed = take(max_rows * model::hidden * bf16_bytes);
	w.mtp_embedding = take(max_rows * model::hidden * float_bytes);
	w.mtp_hidden = take(max_rows * model::hidden * bf16_bytes);
	w.embedding_squares = take(tiles * max_rows * float_bytes);
	w.hidden_squares = take(tiles * max_rows * float_bytes);
	w.mtp_input = take(max_rows * model::hidden * bf16_bytes);
	w.qkv = take(max_rows * deltanet::qkv_row_bytes);
	w.gates = take(max_rows * deltanet::gate_row_bytes);
	w.core = take(max_rows * deltanet::output_row_bytes);
	w.gated = take(max_rows * deltanet::value_features * bf16_bytes);
	w.query = take(max_rows * attn::query_row_floats * float_bytes);
	w.key = take(max_rows * attn::kv_row_floats * float_bytes);
	w.value = take(max_rows * attn::kv_row_floats * float_bytes);
	w.attention_gate = take(max_rows * attn::gate_row_floats * float_bytes);
	w.attention_output = take(max_rows * attn::output_row_values * bf16_bytes);
	w.product = take(max_rows * model::ffn_per_card * bf16_bytes);
	w.logits = take(max_rows * model::vocab_per_card * float_bytes);
	w.lists[0] = take(max_rows * lists_after(Op::Head, 0) * list_length * key_bytes);
	w.lists[1] = take(max_rows * lists_after(Op::Head, 1) * list_length * key_bytes);
	w.candidates = take(max_rows * list_length * sizeof(sampling::Candidate));
	w.command = take(sizeof(unsigned));
	w.descriptor = take(sizeof(Descriptor));
	w.outcomes = take(model::slots * sizeof(SlotOutcome));
	w.books = take(model::slots * sizeof(Book));
	w.tables = take(model::slots * model::kv_table_blocks * sizeof(unsigned));
	w.prompt = take(model::hidden * bf16_bytes);
	w.bytes = align(cursor.end);
	return w;
}

inline constexpr Workspace workspace = plan_workspace();

/** Attention's records in the schedule's space: every record row's 256 fp32 outputs, then every row's running pair. */
inline constexpr Count record_outputs = workspace.exchange.attention_records;
inline constexpr Count record_running = record_outputs + attn::max_record_rows(budget) * attn::head_dim * float_bytes;

static_assert(record_running + attn::max_record_rows(budget) * sizeof(attn::Running) <= workspace.exchange.attention_records + attn::record_bytes(budget));

/** The card's lm_head row-to-token table, its own allocation of the card's workspace. */
inline constexpr Count row_token_bytes = model::workspace_plan.row_tokens;

// The card's memory plan counts these sizes, so the pool is what is left after exactly them.
static_assert(workspace.bytes == model::workspace_plan.decode, "the memory plan holds the decode workspace's size");
static_assert(sizeof(Control) == model::workspace_plan.control, "the memory plan holds the control block's size");
static_assert(lists_after(Op::Head, 0) >= lists_after(Op::DraftHead, 0) && lists_after(Op::Head, 2) <= lists_after(Op::Head, 0),
              "every fold's lists fit the buffer it writes");
static_assert(sizeof(Descriptor) % 16 == 0, "the descriptor copies in whole chunks");

// ---------------------------------------------------------------------------------------------------------------------
// Ticket counters: what every thread of both cards advances identically, so each names one use of a buffer or flag.

/** Receive slot of the `ordinal`-th exchange of a kind (from 1), and the epoch its flag carries. */
[[nodiscard]] constexpr auto exchange_slot(unsigned long long ordinal) -> Count {
	return static_cast<Count>((ordinal - 1) % schedule::exchange_depth);
}

}

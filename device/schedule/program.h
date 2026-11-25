// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode cycle as data: the ordered phases of one cycle, what each reads from the weight image, what it depends
// on, and where it exchanges with the other card; and the image layout, which is the producers' transaction stream
// written down once. Everything is generated at compile time from the model description. The step-1 baseline walks
// the phases with a grid barrier between them, the dataflow core walks the same phases and waits only on each phase's
// dependencies, and the baker lays the image out from `image(card)`: none of them holds a second copy of the order.
// It uses no standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "model.h"

namespace engine::device::schedule {

using model::Count;
using model::Range;
using model::Stream;

// ---------------------------------------------------------------------------------------------------------------------
// The two compile-time parameters of a cycle: its draft depth and its row budget, both the model description's.

using model::DraftDepth;
using model::RowBudget;

inline constexpr DraftDepth draft_depths[] = {DraftDepth::One, DraftDepth::Two, DraftDepth::Three, DraftDepth::Four};

[[nodiscard]] constexpr auto row_tiles(RowBudget budget) -> Count {
	return model::rows(budget) / model::row_tile;
}

/** The grid-barrier baseline runs four tiles of rows: its stream leaves the tensor cores half idle. */
inline constexpr RowBudget baseline_row_budget = RowBudget::FourTiles;

/** The dataflow core ships the model description's row budget. */
inline constexpr RowBudget dataflow_row_budget = model::row_budget;

/** The shape for `drafts` per slot, one or more; past the deepest shape the cap holds. */
[[nodiscard]] constexpr auto depth_of(Count drafts) -> DraftDepth {
	switch (drafts) {
	case 1:
		return DraftDepth::One;
	case 2:
		return DraftDepth::Two;
	case 3:
		return DraftDepth::Three;
	default:
		return DraftDepth::Four;
	}
}

/**
 * The shape of a cycle with `live` slots (1 to `slots`): one row per slot for its committed token, the rest split
 * evenly as drafts, capped at the deepest shape.
 */
[[nodiscard]] constexpr auto shape(Count live, RowBudget budget) -> DraftDepth {
	return depth_of(model::rows(budget) / live - 1);
}

[[nodiscard]] consteval auto every_slot_count_drafts() -> bool {
	for (auto const budget : model::row_budgets) {
		for (auto live = Count{1}; live <= model::slots; ++live) {
			if (model::rows(budget) / live < 2) {
				return false;
			}
		}
	}
	return true;
}

static_assert(every_slot_count_drafts(), "every live-slot count leaves each slot at least one draft row");

// ---------------------------------------------------------------------------------------------------------------------
// Phases.

/** What one phase of the cycle computes. */
enum class Op : unsigned char {
	/** Gathers the verify rows' token embeddings into the fp32 residual stream. */
	Embed,
	/** in_proj_qkv, in_proj_a and in_proj_b on the normalized residual. */
	DeltanetIn,
	/** The causal convolution and the gated delta rule over each live slot's fp32 state; the per-head output norm. */
	DeltanetCore,
	/** in_proj_z: needed only by out_proj's input, so it streams after the core it does not depend on. */
	DeltanetZ,
	/** out_proj on norm(core) * silu(z), summed across the cards into the residual. */
	DeltanetOut,
	/** The query half of q_proj, k_proj and v_proj; q and k norms, rope and the kv append follow from them. */
	AttentionIn,
	/** Split-k attention over each live slot's kv, in chunks at fixed absolute positions. */
	AttentionCore,
	/** The gate half of q_proj: needed only after attention, so it streams after the core it does not depend on. */
	AttentionGate,
	/** Merges the chunks in index order and applies sigmoid(gate). */
	AttentionMerge,
	/** o_proj, summed across the cards into the residual. */
	AttentionOut,
	/** gate_proj and up_proj, interleaved by row block, into silu(gate) * up. */
	MlpIn,
	/** down_proj, summed across the cards into the residual. */
	MlpOut,
	/** The final norm and the card's lm_head rows: each card's best candidates per row. */
	Head,
	/** Merges both cards' candidates, verifies the drafts and samples: the accepted tokens. */
	Verify,
	/** The mtp head's input: normalized embedding and hidden state through fc, summed across the cards. */
	MtpIn,
	/** The mtp norm and the draft vocabulary's lm_head rows: each card's best draft candidates per row. */
	DraftHead,
	/** Merges both cards' draft candidates: the next draft token per slot. */
	Draft,
};

inline constexpr Op ops[] = {
    Op::Embed,         Op::DeltanetIn,    Op::DeltanetCore,   Op::DeltanetZ,    Op::DeltanetOut, Op::AttentionIn,
    Op::AttentionCore, Op::AttentionGate, Op::AttentionMerge, Op::AttentionOut, Op::MlpIn,       Op::MlpOut,
    Op::Head,          Op::Verify,        Op::MtpIn,          Op::DraftHead,    Op::Draft,
};

/** How a phase's result crosses to the other card. */
enum class Exchange : unsigned char {
	None,
	/** fp32 partial sums of the output, pushed per output tile; both cards add them in card order. */
	Partials,
	/** Per-row candidate lists over the card's vocabulary half. */
	Candidates,
};

[[nodiscard]] constexpr auto exchange(Op op) -> Exchange {
	switch (op) {
	case Op::DeltanetOut:
	case Op::AttentionOut:
	case Op::MlpOut:
	case Op::MtpIn:
		return Exchange::Partials;
	case Op::Head:
	case Op::DraftHead:
		return Exchange::Candidates;
	case Op::Embed:
	case Op::DeltanetIn:
	case Op::DeltanetCore:
	case Op::DeltanetZ:
	case Op::AttentionIn:
	case Op::AttentionCore:
	case Op::AttentionGate:
	case Op::AttentionMerge:
	case Op::MlpIn:
	case Op::Verify:
	case Op::Draft:
		return Exchange::None;
	}
	__builtin_unreachable();
}

/** Per-slot data a phase streams besides the image, sized by the live slots and their lengths at run time. */
enum class Traffic : unsigned char {
	None,
	/** Reads and writes each live slot's fp32 deltanet state of the layer. */
	DeltanetState,
	/** Reads each live slot's bf16 kv of the layer. */
	Kv,
};

[[nodiscard]] constexpr auto traffic(Op op) -> Traffic {
	switch (op) {
	case Op::DeltanetCore:
		return Traffic::DeltanetState;
	case Op::AttentionCore:
		return Traffic::Kv;
	case Op::Embed:
	case Op::DeltanetIn:
	case Op::DeltanetZ:
	case Op::DeltanetOut:
	case Op::AttentionIn:
	case Op::AttentionGate:
	case Op::AttentionMerge:
	case Op::AttentionOut:
	case Op::MlpIn:
	case Op::MlpOut:
	case Op::Head:
	case Op::Verify:
	case Op::MtpIn:
	case Op::DraftHead:
	case Op::Draft:
		return Traffic::None;
	}
	__builtin_unreachable();
}

/** One live slot's fp32 deltanet state of one layer on one card; a core reads it and writes it once per cycle. */
inline constexpr Count deltanet_state_bytes_per_layer_per_slot =
    model::value_heads_per_card * model::key_dim * model::value_dim * sizeof(float);

/** Bytes of bf16 kv one live slot's attention core reads per position of its context, on one card. */
inline constexpr Count kv_bytes_per_position_per_layer =
    model::kv_heads_per_card * 2 * model::head_dim * model::element_bytes(model::Format::Bf16);

// ---------------------------------------------------------------------------------------------------------------------
// What each phase reads from the image.

[[nodiscard]] consteval auto same_name(char const* left, char const* right) -> bool {
	auto index = Count{0};
	while (left[index] != '\0' && left[index] == right[index]) {
		++index;
	}
	return left[index] == right[index];
}

[[nodiscard]] consteval auto tensor_named(char const* name) -> Count {
	for (auto index = Count{0}; index < sizeof(model::inventory) / sizeof(model::inventory[0]); ++index) {
		if (same_name(model::inventory[index].name, name)) {
			return index;
		}
	}
	// Not a constant expression: the schedule names a tensor the model description does not hold.
	__builtin_unreachable();
}

inline constexpr Count tensor_count = sizeof(model::inventory) / sizeof(model::inventory[0]);

/** Inventory indices of the tensors the schedule places, resolved once. */
struct Tensors {
	Count final_norm;
	Count lm_head;
	Count fc;
	Count pre_fc_norm_embedding;
	Count pre_fc_norm_hidden;
	Count mtp_norm;
	Count input_norm;
	Count post_norm;
	Count gate;
	Count up;
	Count down;
	Count qkv;
	Count z;
	Count a;
	Count b;
	Count conv;
	Count a_log;
	Count dt_bias;
	Count deltanet_norm;
	Count out;
	Count q;
	Count k;
	Count v;
	Count q_norm;
	Count k_norm;
	Count o;
};

inline constexpr Tensors tensors = {
    .final_norm = tensor_named("model.language_model.norm.weight"),
    .lm_head = tensor_named("lm_head.weight"),
    .fc = tensor_named("fc.weight"),
    .pre_fc_norm_embedding = tensor_named("pre_fc_norm_embedding.weight"),
    .pre_fc_norm_hidden = tensor_named("pre_fc_norm_hidden.weight"),
    .mtp_norm = tensor_named("norm.weight"),
    .input_norm = tensor_named("input_layernorm.weight"),
    .post_norm = tensor_named("post_attention_layernorm.weight"),
    .gate = tensor_named("mlp.gate_proj.weight"),
    .up = tensor_named("mlp.up_proj.weight"),
    .down = tensor_named("mlp.down_proj.weight"),
    .qkv = tensor_named("linear_attn.in_proj_qkv.weight"),
    .z = tensor_named("linear_attn.in_proj_z.weight"),
    .a = tensor_named("linear_attn.in_proj_a.weight"),
    .b = tensor_named("linear_attn.in_proj_b.weight"),
    .conv = tensor_named("linear_attn.conv1d.weight"),
    .a_log = tensor_named("linear_attn.A_log"),
    .dt_bias = tensor_named("linear_attn.dt_bias"),
    .deltanet_norm = tensor_named("linear_attn.norm.weight"),
    .out = tensor_named("linear_attn.out_proj.weight"),
    .q = tensor_named("self_attn.q_proj.weight"),
    .k = tensor_named("self_attn.k_proj.weight"),
    .v = tensor_named("self_attn.v_proj.weight"),
    .q_norm = tensor_named("self_attn.q_norm.weight"),
    .k_norm = tensor_named("self_attn.k_norm.weight"),
    .o = tensor_named("self_attn.o_proj.weight"),
};

/** How a placement holds a tensor. */
enum class Part : unsigned char {
	/** 16 KiB weight tiles, streamed through the producers' shared-memory ring. */
	Tiles,
	/** An e4m3 tensor's bf16 block scales, one row per 128-row block, read by consumers directly. */
	Scales,
	/** A small tensor read whole by consumers directly. */
	Whole,
};

/** Which 128-row blocks of a card's part of a tensor a placement holds. */
enum class Select : unsigned char {
	All,
	/** Each query head's query rows of q_proj. */
	Query,
	/** Each query head's gate rows of q_proj. */
	Gate,
	/** The lm_head rows of the card's half of the draft vocabulary, which the bake permutes to the front. */
	DraftVocabulary,
	/** The rest of the card's lm_head rows. */
	OtherVocabulary,
};

/** One read: one tensor, or two interleaved row block by row block when one epilogue combines them. */
struct Read {
	Count tensor[2];
	Count tensor_count;
	Part part;
	Select select;
};

[[nodiscard]] consteval auto one(Count tensor, Part part, Select select = Select::All) -> Read {
	return {.tensor = {tensor, tensor}, .tensor_count = 1, .part = part, .select = select};
}

/** What one phase reads from the image: tiles through the ring, in stream order, and side data read directly. */
struct Reads {
	Read ring[3];
	Count ring_count;
	Read side[6];
	Count side_count;
};

[[nodiscard]] consteval auto reads(Op op) -> Reads {
	auto const& t = tensors;
	switch (op) {
	case Op::Embed:
	case Op::AttentionCore:
	case Op::AttentionMerge:
	case Op::Verify:
	case Op::Draft:
		return {.ring = {}, .ring_count = 0, .side = {}, .side_count = 0};
	case Op::DeltanetIn:
		return {.ring = {one(t.qkv, Part::Tiles)},
		        .ring_count = 1,
		        .side = {one(t.input_norm, Part::Whole), one(t.qkv, Part::Scales), one(t.a, Part::Whole), one(t.b, Part::Whole)},
		        .side_count = 4};
	case Op::DeltanetCore:
		return {
		    .ring = {},
		    .ring_count = 0,
		    .side = {one(t.conv, Part::Whole), one(t.a_log, Part::Whole), one(t.dt_bias, Part::Whole), one(t.deltanet_norm, Part::Whole)},
		    .side_count = 4};
	case Op::DeltanetZ:
		return {.ring = {one(t.z, Part::Tiles)}, .ring_count = 1, .side = {one(t.z, Part::Scales)}, .side_count = 1};
	case Op::DeltanetOut:
		return {.ring = {one(t.out, Part::Tiles)}, .ring_count = 1, .side = {one(t.out, Part::Scales)}, .side_count = 1};
	case Op::AttentionIn:
		return {.ring = {one(t.q, Part::Tiles, Select::Query), one(t.k, Part::Tiles), one(t.v, Part::Tiles)},
		        .ring_count = 3,
		        .side = {one(t.input_norm, Part::Whole), one(t.q, Part::Scales, Select::Query), one(t.k, Part::Scales),
		                 one(t.v, Part::Scales), one(t.q_norm, Part::Whole), one(t.k_norm, Part::Whole)},
		        .side_count = 6};
	case Op::AttentionGate:
		return {.ring = {one(t.q, Part::Tiles, Select::Gate)},
		        .ring_count = 1,
		        .side = {one(t.q, Part::Scales, Select::Gate)},
		        .side_count = 1};
	case Op::AttentionOut:
		return {.ring = {one(t.o, Part::Tiles)}, .ring_count = 1, .side = {one(t.o, Part::Scales)}, .side_count = 1};
	case Op::MlpIn:
		return {.ring = {{.tensor = {t.gate, t.up}, .tensor_count = 2, .part = Part::Tiles, .select = Select::All}},
		        .ring_count = 1,
		        .side = {one(t.post_norm, Part::Whole), one(t.gate, Part::Scales), one(t.up, Part::Scales)},
		        .side_count = 3};
	case Op::MlpOut:
		return {.ring = {one(t.down, Part::Tiles)}, .ring_count = 1, .side = {one(t.down, Part::Scales)}, .side_count = 1};
	case Op::Head:
		return {.ring = {one(t.lm_head, Part::Tiles, Select::DraftVocabulary), one(t.lm_head, Part::Tiles, Select::OtherVocabulary)},
		        .ring_count = 2,
		        .side = {one(t.final_norm, Part::Whole)},
		        .side_count = 1};
	case Op::MtpIn:
		return {.ring = {one(t.fc, Part::Tiles)},
		        .ring_count = 1,
		        .side = {one(t.pre_fc_norm_embedding, Part::Whole), one(t.pre_fc_norm_hidden, Part::Whole)},
		        .side_count = 2};
	case Op::DraftHead:
		return {.ring = {one(t.lm_head, Part::Tiles, Select::DraftVocabulary)},
		        .ring_count = 1,
		        .side = {one(t.mtp_norm, Part::Whole)},
		        .side_count = 1};
	}
	__builtin_unreachable();
}

// ---------------------------------------------------------------------------------------------------------------------
// Tiles and block selections.

/** One ring stage: a whole 16 KiB tile, 64 dram chunks, one contiguous run of the image. */
inline constexpr Count tile_bytes = Count{16} * 1024;

/** A tile spans one block-scale row of outputs: 128 rows. */
inline constexpr Count tile_rows = model::scale_block;

/** Side data starts on a 256-byte dram chunk. */
inline constexpr Count side_alignment = 256;

[[nodiscard]] constexpr auto tile_columns(model::Format format) -> Count {
	return tile_bytes / (tile_rows * model::element_bytes(format));
}

static_assert(tile_columns(model::Format::E4m3) == model::scale_block, "an e4m3 tile spans one scale block");

/** A selection of 128-row blocks: `take` blocks starting at `first` in each of `groups` runs `stride` blocks apart. */
struct Blocks {
	Count first;
	Count take;
	Count stride;
	Count groups;
};

[[nodiscard]] constexpr auto selected(Blocks const& blocks) -> Count {
	return blocks.take * blocks.groups;
}

/** Whether `block` is in the selection. */
[[nodiscard]] constexpr auto holds(Blocks const& blocks, Count block) -> bool {
	if (block < blocks.first) {
		return false;
	}
	auto const from = block - blocks.first;
	return from / blocks.stride < blocks.groups && from % blocks.stride < blocks.take;
}

[[nodiscard]] constexpr auto row_blocks(Count tensor, Count card) -> Count {
	return model::divide_up(model::card_shape(model::inventory[tensor], card).extents[0], tile_rows);
}

[[nodiscard]] constexpr auto blocks(Select select, Count tensor, Count card) -> Blocks {
	auto const total = row_blocks(tensor, card);
	constexpr auto head_blocks = model::q_proj_rows_per_head / tile_rows;
	constexpr auto query_blocks = model::head_dim / tile_rows;
	constexpr auto draft_blocks = model::draft_vocab_per_card / tile_rows;
	switch (select) {
	case Select::All:
		return {.first = 0, .take = total, .stride = total, .groups = 1};
	case Select::Query:
		return {.first = 0, .take = query_blocks, .stride = head_blocks, .groups = total / head_blocks};
	case Select::Gate:
		return {.first = query_blocks, .take = head_blocks - query_blocks, .stride = head_blocks, .groups = total / head_blocks};
	case Select::DraftVocabulary:
		return {.first = 0, .take = draft_blocks, .stride = total, .groups = 1};
	case Select::OtherVocabulary:
		return {.first = draft_blocks, .take = total - draft_blocks, .stride = total, .groups = 1};
	}
	__builtin_unreachable();
}

static_assert(model::head_dim % tile_rows == 0 && model::q_proj_rows_per_head % tile_rows == 0, "a head is whole blocks");
static_assert(model::draft_vocab_per_card % tile_rows == 0, "the draft vocabulary is whole lm_head blocks");

/** Bytes of one tensor of `read` on `card`. */
[[nodiscard]] constexpr auto read_bytes(Read const& read, Count tensor, Count card) -> Count {
	auto const& entry = model::inventory[tensor];
	auto const shape = model::card_shape(entry, card);
	auto const chosen = selected(blocks(read.select, tensor, card));
	switch (read.part) {
	case Part::Tiles:
		return chosen * tile_rows * shape.extents[1] * model::element_bytes(entry.format);
	case Part::Scales:
		return chosen * model::divide_up(shape.extents[1], model::scale_block) * model::element_bytes(model::Format::Bf16);
	case Part::Whole:
		return model::card_bytes(entry, card);
	}
	__builtin_unreachable();
}

/** Laws every read obeys: tiles are whole, scales belong to e4m3 tiles, and a whole read takes the whole tensor. */
[[nodiscard]] consteval auto reads_are_sound() -> bool {
	for (auto const op : ops) {
		auto const r = reads(op);
		for (auto index = Count{0}; index < r.ring_count + r.side_count; ++index) {
			auto const& read = index < r.ring_count ? r.ring[index] : r.side[index - r.ring_count];
			auto const ring = index < r.ring_count;
			if (ring != (read.part == Part::Tiles)) {
				return false;
			}
			for (auto which = Count{0}; which < read.tensor_count; ++which) {
				auto const tensor = read.tensor[which];
				auto const& entry = model::inventory[tensor];
				for (auto card = Count{0}; card < model::cards; ++card) {
					auto const shape = model::card_shape(entry, card);
					switch (read.part) {
					case Part::Tiles:
						if (shape.rank != 2 || shape.extents[0] % tile_rows != 0 || shape.extents[1] % tile_columns(entry.format) != 0) {
							return false;
						}
						break;
					case Part::Scales:
						if (entry.format != model::Format::E4m3) {
							return false;
						}
						break;
					case Part::Whole:
						if (entry.format == model::Format::E4m3 || read.select != Select::All) {
							return false;
						}
						break;
					}
				}
			}
		}
	}
	return true;
}

static_assert(reads_are_sound(), "every tiled read is whole tiles, every scale read is e4m3, every whole read is plain");

// ---------------------------------------------------------------------------------------------------------------------
// The canonical phase order: one walk, shared by the image layout and every cycle program.

/** A phase of a pass before it is placed in a cycle: what it computes, and which layer of which stream it is in. */
struct Step {
	Op op;
	Stream stream;
	Count layer;
};

template<class Visit>
constexpr void walk_layer(model::Mixer mixer, Stream stream, Count layer, Visit& visit) {
	switch (mixer) {
	case model::Mixer::LinearAttention:
		visit(Step{.op = Op::DeltanetIn, .stream = stream, .layer = layer});
		visit(Step{.op = Op::DeltanetCore, .stream = stream, .layer = layer});
		visit(Step{.op = Op::DeltanetZ, .stream = stream, .layer = layer});
		visit(Step{.op = Op::DeltanetOut, .stream = stream, .layer = layer});
		break;
	case model::Mixer::FullAttention:
		visit(Step{.op = Op::AttentionIn, .stream = stream, .layer = layer});
		visit(Step{.op = Op::AttentionCore, .stream = stream, .layer = layer});
		visit(Step{.op = Op::AttentionGate, .stream = stream, .layer = layer});
		visit(Step{.op = Op::AttentionMerge, .stream = stream, .layer = layer});
		visit(Step{.op = Op::AttentionOut, .stream = stream, .layer = layer});
		break;
	}
	visit(Step{.op = Op::MlpIn, .stream = stream, .layer = layer});
	visit(Step{.op = Op::MlpOut, .stream = stream, .layer = layer});
}

/** The verify pass: the trunk over every verify row, then the lm_head and the verdict. */
template<class Visit>
constexpr void walk_verify(Visit& visit) {
	visit(Step{.op = Op::Embed, .stream = Stream::Body, .layer = 0});
	for (auto layer = Count{0}; layer < model::trunk_layers; ++layer) {
		walk_layer(model::trunk_mixers[layer], Stream::Body, layer, visit);
	}
	visit(Step{.op = Op::Head, .stream = Stream::Head, .layer = 0});
	visit(Step{.op = Op::Verify, .stream = Stream::Head, .layer = 0});
}

/** One mtp pass: its input, its decoder layer, and one draft. */
template<class Visit>
constexpr void walk_mtp(Visit& visit) {
	visit(Step{.op = Op::MtpIn, .stream = Stream::Mtp, .layer = 0});
	for (auto layer = Count{0}; layer < model::mtp_layers; ++layer) {
		walk_layer(model::mtp_mixer, Stream::Mtp, layer, visit);
	}
	visit(Step{.op = Op::DraftHead, .stream = Stream::Mtp, .layer = 0});
	visit(Step{.op = Op::Draft, .stream = Stream::Mtp, .layer = 0});
}

struct Counter {
	Count steps;

	constexpr void operator()(Step /*step*/) {
		++steps;
	}
};

[[nodiscard]] consteval auto verify_step_count() -> Count {
	auto counter = Counter{.steps = 0};
	walk_verify(counter);
	return counter.steps;
}

[[nodiscard]] consteval auto mtp_step_count() -> Count {
	auto counter = Counter{.steps = 0};
	walk_mtp(counter);
	return counter.steps;
}

inline constexpr Count verify_phases = verify_step_count();
inline constexpr Count mtp_phases = mtp_step_count();

/** The image phases: the verify pass, then one mtp pass, whose placements every mtp pass of a cycle reads again. */
inline constexpr Count image_phases = verify_phases + mtp_phases;

// ---------------------------------------------------------------------------------------------------------------------
// The image layout: every placement once, tiles first in stream order, then the side data.

/** Which copy of a tensor a placement holds: model-wide tensors are in the head stream, mtp-wide ones in the mtp. */
struct Instance {
	Stream stream;
	Count layer;
};

[[nodiscard]] constexpr auto instance(Count tensor, Step const& step) -> Instance {
	switch (model::inventory[tensor].scope) {
	case model::Scope::Model:
		return {.stream = Stream::Head, .layer = 0};
	case model::Scope::Mtp:
		return {.stream = Stream::Mtp, .layer = 0};
	case model::Scope::DecoderLayer:
	case model::Scope::LinearAttentionLayer:
	case model::Scope::FullAttentionLayer:
		return {.stream = step.stream, .layer = step.layer};
	}
	__builtin_unreachable();
}

/** One contiguous extent of a card's image. */
struct Placement {
	Read read;
	Instance instance;
	Count offset;
	Count bytes;
};

/** An image phase's placements: indices into the layout's placement list, each range contiguous. */
struct ImagePhase {
	Step step;
	Range ring;
	Range side;
};

[[nodiscard]] constexpr auto same_placement(Read const& read, Instance const& at, Placement const& placed) -> bool {
	auto same = read.tensor_count == placed.read.tensor_count && read.part == placed.read.part && read.select == placed.read.select &&
	            at.stream == placed.instance.stream && at.layer == placed.instance.layer;
	for (auto which = Count{0}; which < read.tensor_count; ++which) {
		same = same && read.tensor[which] == placed.read.tensor[which];
	}
	return same;
}

[[nodiscard]] constexpr auto align_up(Count value, Count to) -> Count {
	return model::divide_up(value, to) * to;
}

template<Count Placements>
struct Image {
	Placement placements[Placements];
	ImagePhase phases[image_phases];
	/** Bytes of the tile region, which starts the image. */
	Count stream_bytes;
	/** Where the side region starts, and the image's end. */
	Count side_offset;
	Count bytes;
};

/** Collects the image phases in walk order. */
struct Collector {
	ImagePhase* phases;
	Count count;

	constexpr void operator()(Step step) {
		phases[count] = ImagePhase{.step = step, .ring = {.begin = 0, .length = 0}, .side = {.begin = 0, .length = 0}};
		++count;
	}
};

/** Reads of model-wide tensors in all the reads tables: a bound on their distinct placements. */
[[nodiscard]] consteval auto count_model_wide_reads() -> Count {
	auto count = Count{0};
	for (auto const op : ops) {
		auto const r = reads(op);
		for (auto index = Count{0}; index < r.ring_count + r.side_count; ++index) {
			auto const& read = index < r.ring_count ? r.ring[index] : r.side[index - r.ring_count];
			auto const scope = model::inventory[read.tensor[0]].scope;
			count += scope == model::Scope::Model || scope == model::Scope::Mtp ? 1 : 0;
		}
	}
	return count;
}

inline constexpr Count model_wide_reads = count_model_wide_reads();

/** Places every read of every image phase once, ring reads first; returns the placement count. */
template<Count Capacity>
constexpr auto place(Image<Capacity>& image, Count card) -> Count {
	auto collector = Collector{.phases = image.phases, .count = 0};
	walk_verify(collector);
	walk_mtp(collector);
	auto count = Count{0};
	auto offset = Count{0};
	Count shared[model_wide_reads] = {};
	auto shared_count = Count{0};
	for (auto region = 0; region < 2; ++region) {
		auto const ring = region == 0;
		if (!ring) {
			image.stream_bytes = offset;
			offset = align_up(offset, side_alignment);
			image.side_offset = offset;
		}
		for (auto& phase : image.phases) {
			auto const r = reads(phase.step.op);
			auto const reads_here = ring ? r.ring_count : r.side_count;
			auto& range = ring ? phase.ring : phase.side;
			for (auto index = Count{0}; index < reads_here; ++index) {
				auto const& read = ring ? r.ring[index] : r.side[index];
				auto const at = instance(read.tensor[0], phase.step);
				// Only a model-wide tensor can be read by two phases (the draft lm_head is a view of the lm_head); a
				// layer's tensors belong to that layer's phases alone.
				auto const model_wide = model::inventory[read.tensor[0]].scope == model::Scope::Model ||
				                        model::inventory[read.tensor[0]].scope == model::Scope::Mtp;
				auto found = count;
				for (auto earlier = Count{0}; model_wide && earlier < shared_count; ++earlier) {
					if (same_placement(read, at, image.placements[shared[earlier]])) {
						found = shared[earlier];
					}
				}
				if (found == count) {
					if (model_wide) {
						shared[shared_count] = count;
						++shared_count;
					}
					auto bytes = Count{0};
					for (auto which = Count{0}; which < read.tensor_count; ++which) {
						bytes += read_bytes(read, read.tensor[which], card);
					}
					offset = ring ? offset : align_up(offset, side_alignment);
					image.placements[count] = Placement{.read = read, .instance = at, .offset = offset, .bytes = bytes};
					offset += bytes;
					++count;
				}
				if (index == 0) {
					range.begin = found;
				} else if (found != range.begin + index) {
					// Not a constant expression: a phase's placements must be one contiguous run of the image.
					__builtin_unreachable();
				}
				range.length = index + 1;
			}
		}
	}
	image.bytes = offset;
	return count;
}

/** Every read of every image phase placed apart: the most placements an image can need. */
inline constexpr Count placement_bound = image_phases * (sizeof(Reads{}.ring) / sizeof(Read) + sizeof(Reads{}.side) / sizeof(Read));

[[nodiscard]] consteval auto placement_count() -> Count {
	auto probe = Image<placement_bound>{};
	return place(probe, 0);
}

inline constexpr Count placements = placement_count();

using CardImage = Image<placements>;

/** One card's image layout. The splits are balanced, so both cards' layouts agree in every offset. */
[[nodiscard]] consteval auto image(Count card) -> CardImage {
	auto result = CardImage{};
	place(result, card);
	return result;
}

/** Placement bytes of one image phase. */
[[nodiscard]] constexpr auto phase_bytes(CardImage const& layout, ImagePhase const& phase) -> Count {
	auto total = Count{0};
	Range const ranges[] = {phase.ring, phase.side};
	for (auto const& range : ranges) {
		for (auto index = range.begin; index < range.begin + range.length; ++index) {
			total += layout.placements[index].bytes;
		}
	}
	return total;
}

// ---------------------------------------------------------------------------------------------------------------------
// The cycle program.

/** Which pass of the cycle a phase is in. */
enum class PassKind : unsigned char {
	/** The trunk and lm_head over every verify row. */
	Verify,
	/** The first mtp pass, over each slot's accepted rows: it yields draft 1. */
	CatchUp,
	/** A later mtp pass, over one row per slot: it yields the next draft. */
	Step,
};

struct Pass {
	PassKind kind;
	/** The draft the pass yields; zero for the verify pass. */
	Count draft;
};

/** The most dependencies any phase has. */
inline constexpr Count max_dependencies = 3;

/** One phase of a cycle. Its reads are its image phase's placements; its dependencies precede it. */
struct Phase {
	Step step;
	Pass pass;
	/** Index into the image's phases. */
	Count image_phase;
	/** n8 row tiles the phase computes. */
	Count row_tiles;
	Count dependencies[max_dependencies];
	Count dependency_count;
};

inline constexpr Count max_cycle_phases = verify_phases + model::drafts(model::max_draft_depth) * mtp_phases;

/** A cycle's phases in stream order: the order producers stream them and the baseline runs them. */
struct Program {
	DraftDepth depth;
	RowBudget budget;
	Count phase_count;
	Phase phases[max_cycle_phases];
};

/** The most recent phase to write each value a later phase reads. */
struct Recent {
	Count residual;
	Count tokens;
	Count mixer_in;
	Count core;
	Count early;
	Count merge;
	Count mlp_in;
	Count logits;
};

struct Builder {
	Program* program;
	Pass pass;
	Count row_tiles;
	Count image_phase;
	Recent* recent;

	constexpr void operator()(Step step) {
		auto& r = *recent;
		Count deps[max_dependencies] = {};
		auto count = Count{0};
		auto const depend = [&](Count on) {
			deps[count] = on;
			++count;
		};
		auto const here = program->phase_count;
		switch (step.op) {
		case Op::Embed:
			break;
		case Op::DeltanetIn:
		case Op::AttentionIn:
			depend(r.residual);
			r.mixer_in = here;
			break;
		case Op::DeltanetCore:
		case Op::AttentionCore:
			depend(r.mixer_in);
			r.core = here;
			break;
		case Op::DeltanetZ:
		case Op::AttentionGate:
			depend(r.residual);
			r.early = here;
			break;
		case Op::AttentionMerge:
			depend(r.core);
			depend(r.early);
			r.merge = here;
			break;
		case Op::DeltanetOut:
			depend(r.core);
			depend(r.early);
			depend(r.residual);
			break;
		case Op::AttentionOut:
			depend(r.merge);
			depend(r.residual);
			break;
		case Op::MlpIn:
			depend(r.residual);
			r.mlp_in = here;
			break;
		case Op::MlpOut:
			depend(r.mlp_in);
			depend(r.residual);
			break;
		case Op::Head:
		case Op::DraftHead:
			depend(r.residual);
			r.logits = here;
			break;
		case Op::Verify:
		case Op::Draft:
			depend(r.logits);
			r.tokens = here;
			break;
		case Op::MtpIn:
			depend(r.tokens);
			depend(r.residual);
			break;
		}
		switch (step.op) {
		case Op::Embed:
		case Op::DeltanetOut:
		case Op::AttentionOut:
		case Op::MlpOut:
		case Op::MtpIn:
			r.residual = here;
			break;
		case Op::DeltanetIn:
		case Op::DeltanetCore:
		case Op::DeltanetZ:
		case Op::AttentionIn:
		case Op::AttentionCore:
		case Op::AttentionGate:
		case Op::AttentionMerge:
		case Op::MlpIn:
		case Op::Head:
		case Op::Verify:
		case Op::DraftHead:
		case Op::Draft:
			break;
		}
		auto phase = Phase{
		    .step = step, .pass = pass, .image_phase = image_phase, .row_tiles = row_tiles, .dependencies = {}, .dependency_count = count};
		for (auto index = Count{0}; index < count; ++index) {
			phase.dependencies[index] = deps[index];
		}
		program->phases[here] = phase;
		++program->phase_count;
		++image_phase;
	}
};

/** Row tiles of one mtp draft step: one row per slot. */
inline constexpr Count step_row_tiles = model::divide_up(model::slots, model::row_tile);

/** The cycle program of one shape and row budget. */
[[nodiscard]] consteval auto program(DraftDepth depth, RowBudget budget) -> Program {
	auto result = Program{.depth = depth, .budget = budget, .phase_count = 0, .phases = {}};
	auto recent = Recent{};
	auto verify = Builder{.program = &result,
	                      .pass = {.kind = PassKind::Verify, .draft = 0},
	                      .row_tiles = row_tiles(budget),
	                      .image_phase = 0,
	                      .recent = &recent};
	walk_verify(verify);
	for (auto draft = Count{1}; draft <= model::drafts(depth); ++draft) {
		auto const catch_up = draft == 1;
		auto mtp = Builder{.program = &result,
		                   .pass = {.kind = catch_up ? PassKind::CatchUp : PassKind::Step, .draft = draft},
		                   // The catch-up pass runs over the accepted rows, at most every verify row.
		                   .row_tiles = catch_up ? row_tiles(budget) : step_row_tiles,
		                   .image_phase = verify_phases,
		                   .recent = &recent};
		walk_mtp(mtp);
	}
	return result;
}

/** The compile-time stream program of one cycle shape and row budget. */
template<DraftDepth CycleDepth, RowBudget Budget>
inline constexpr Program cycle = program(CycleDepth, Budget);

/** Which of the cycle's exchanges of its kind a phase's exchange is, counting from zero. */
[[nodiscard]] constexpr auto exchange_ordinal(Program const& cycle_program, Count phase) -> Count {
	auto const kind = exchange(cycle_program.phases[phase].step.op);
	auto ordinal = Count{0};
	for (auto index = Count{0}; index < phase; ++index) {
		ordinal += exchange(cycle_program.phases[index].step.op) == kind ? 1 : 0;
	}
	return ordinal;
}

[[nodiscard]] constexpr auto exchanges(Program const& cycle_program, Exchange kind) -> Count {
	auto count = Count{0};
	for (auto index = Count{0}; index < cycle_program.phase_count; ++index) {
		count += exchange(cycle_program.phases[index].step.op) == kind ? 1 : 0;
	}
	return count;
}

/** Image bytes one card streams in one cycle of `cycle_program`. */
[[nodiscard]] constexpr auto cycle_bytes(CardImage const& layout, Program const& cycle_program) -> Count {
	auto total = Count{0};
	for (auto index = Count{0}; index < cycle_program.phase_count; ++index) {
		total += phase_bytes(layout, layout.phases[cycle_program.phases[index].image_phase]);
	}
	return total;
}

// The image holds the main pass and the mtp head once, and the cycle reads the mtp head and the draft vocabulary's
// lm_head rows once per draft.
[[nodiscard]] consteval auto cycles_stream_every_byte() -> bool {
	constexpr auto layout = image(0);
	auto sound = true;
	for (auto const budget : model::row_budgets) {
		for (auto const depth : draft_depths) {
			auto const expected = model::main_pass_bytes_per_card + model::mtp_bytes_per_cycle_per_card(depth) +
			                      model::draft_head_bytes_per_cycle_per_card(depth);
			sound = sound && cycle_bytes(layout, program(depth, budget)) == expected;
		}
	}
	return sound;
}

static_assert(cycles_stream_every_byte(), "a cycle streams the main pass once and the mtp head and draft rows per draft");

static_assert(image(0).stream_bytes % tile_bytes == 0, "the tile region is whole tiles");

}

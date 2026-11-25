// SPDX-License-Identifier: AGPL-3.0-only
//
// Where the deltanet's data lives, held once for the kernels and the host: the fp32 state in the order the lanes hold
// it, a slot's whole deltanet memory (its state image) and how a snapshot cuts it into pool blocks, the replay ring of
// lazy rollback, the work items of a decode cycle and of a prefill chunk, and the parameters each kernel takes. It uses
// no standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "deltanet/math.h"
#include "model.h"
#include "prims/selfcheck.h"

namespace engine::device::deltanet {

using model::Count;

// ---------------------------------------------------------------------------------------------------------------------
// Heads and channels on one card.

inline constexpr int layers = static_cast<int>(model::trunk_layers_of(model::Mixer::LinearAttention));
inline constexpr int key_heads = static_cast<int>(model::key_heads_per_card);
inline constexpr int value_heads = static_cast<int>(model::value_heads_per_card);
/** Value heads reading one key head: the key head's group. */
inline constexpr int group = value_heads / key_heads;
/** in_proj_qkv's outputs on one card: its key heads' queries, then their keys, then its value heads' values. */
inline constexpr int channels = static_cast<int>(model::conv_channels / model::cards);
inline constexpr int query_channels = key_heads * dim;
inline constexpr int value_features = value_heads * dim;

static_assert(group * key_heads == value_heads && group == 3, "value head v reads key head v / 3");
static_assert(channels == 2 * query_channels + value_features, "the card's channels are q | k | v");

/** The channels one key head's group reads: its query, its key, and its value heads' values. */
inline constexpr int group_channels = 2 * dim + group * dim;
/** bf16 elements per 16-byte chunk. */
inline constexpr int bf16_per_chunk = 8;
inline constexpr int group_chunks = group_channels / bf16_per_chunk;

/**
 * The card channel of channel `local` of key head `key_head`'s group, whose channels run query | key | values. A
 * card's channels split into the groups' exactly, which the layout test proves.
 */
[[nodiscard]] constexpr auto card_channel(int key_head, int local) -> int {
	if (local < dim) {
		return key_head * dim + local;
	}
	if (local < 2 * dim) {
		return query_channels + key_head * dim + (local - dim);
	}
	return 2 * query_channels + key_head * group * dim + (local - 2 * dim);
}

// ---------------------------------------------------------------------------------------------------------------------
// The state of one value head: 128 x 128 fp32, element (row i, column j) the model's S[i][j], i over the key and j over
// the value dimension. Columns are independent chains; tree `T` gives each group of T::lanes lanes one column, lane r
// rows [r per_lane, (r + 1) per_lane). The state is stored in the order those lanes load it: a warp's unit is its
// columns_per_warp adjacent columns, 128 rows each, laid out so that the warp's load of chunk k of every lane is 512
// contiguous bytes.

inline constexpr int head_floats = dim * dim;
inline constexpr Count head_state_bytes = Count{head_floats} * sizeof(float);

template<class T>
struct StateLayout {
	static constexpr int unit_columns = T::columns_per_warp;
	static constexpr int unit_floats = unit_columns * dim;
	static constexpr int units = dim / unit_columns;
	static constexpr Count unit_bytes = Count{unit_floats} * sizeof(float);

	/** The lane of a warp holding row `row` of the unit's column `column`. */
	[[nodiscard]] static constexpr auto lane(int row, int column) -> int {
		return T::lanes * (column % unit_columns) + row / T::per_lane;
	}

	/** The float index, from the head's start, of element (row, column). */
	[[nodiscard]] static constexpr auto element(int row, int column) -> int {
		int const within = row % T::per_lane;
		int const chunk = (within / 4) * 32 + lane(row, column);
		return (column / unit_columns) * unit_floats + chunk * 4 + within % 4;
	}

	/** The float index of chunk `chunk` (0 .. T::chunks - 1) of lane `lane_index` in unit `unit`. */
	[[nodiscard]] static constexpr auto lane_chunk(int unit, int chunk, int lane_index) -> int {
		return unit * unit_floats + (chunk * 32 + lane_index) * 4;
	}
};

/** The engine's state layout: four columns per warp unit, a 2 KiB quad. */
using State = StateLayout<Column>;

static_assert(State::unit_bytes == 2048 && State::units == 32);

/**
 * A 128-element fp32 vector (a normalized query or key) as tree `T`'s lanes load it: lane r's chunk k at chunk
 * k lanes + r, so a group's load of chunk k is contiguous and lanes of different groups read the same bytes.
 */
template<class T>
[[nodiscard]] constexpr auto vector_element(int element) -> int {
	int const lane_index = element / T::per_lane;
	int const within = element % T::per_lane;
	return ((within / 4) * T::lanes + lane_index) * 4 + within % 4;
}

// ---------------------------------------------------------------------------------------------------------------------
// A slot's state image: every deltanet layer's state (value heads in order), then every layer's convolution window
// (key head groups in order, each its three rows of its group's channels, oldest first). It is exactly the slot state
// the model description budgets, and a snapshot is this image cut into pool blocks.

inline constexpr int window_rows = static_cast<int>(model::conv_kernel) - 1;
inline constexpr Count window_group_bytes = Count{window_rows} * group_channels * sizeof(Bf16);
inline constexpr Count state_image_bytes = Count{layers} * value_heads * head_state_bytes;
inline constexpr Count window_image_bytes = Count{layers} * key_heads * window_group_bytes;
inline constexpr Count slot_image_bytes = state_image_bytes + window_image_bytes;

static_assert(state_image_bytes == model::deltanet_state_bytes_per_slot, "the image holds the budgeted state");
static_assert(window_image_bytes == model::conv_window_bytes_per_slot, "the image holds the budgeted window");
static_assert(slot_image_bytes == model::slot_state_bytes, "a slot's deltanet memory is its state image, nothing more");

/** Byte offset of value head `head`'s state of deltanet layer `layer` in a slot's image. */
[[nodiscard]] constexpr auto state_offset(int layer, int head) -> Count {
	return (Count(layer) * value_heads + Count(head)) * head_state_bytes;
}

/** Byte offset of key head `key_head`'s window rows of deltanet layer `layer` in a slot's image. */
[[nodiscard]] constexpr auto window_offset(int layer, int key_head) -> Count {
	return state_image_bytes + (Count(layer) * key_heads + Count(key_head)) * window_group_bytes;
}

/** The deltanet ordinal of trunk layer `trunk_layer`: how many deltanet layers precede it. */
[[nodiscard]] constexpr auto deltanet_layer(Count trunk_layer) -> int {
	int ordinal = 0;
	for (Count index = 0; index < trunk_layer; ++index) {
		ordinal += model::trunk_mixers[index] == model::Mixer::LinearAttention ? 1 : 0;
	}
	return ordinal;
}

/** One contiguous run of a slot's image. */
struct Extent {
	Count offset;
	Count bytes;
};

/**
 * The run of the image that part `part` of a snapshot holds: block-sized runs in image order, the last one short.
 * Saving copies each part's run to the start of its block; loading copies it back.
 */
[[nodiscard]] constexpr auto snapshot_part(Count part) -> Extent {
	Count const offset = part * model::kv_block_bytes;
	Count const rest = slot_image_bytes - offset;
	return {.offset = offset, .bytes = rest < model::kv_block_bytes ? rest : model::kv_block_bytes};
}

static_assert(model::snapshot_blocks * model::kv_block_bytes >= slot_image_bytes, "the parts cover the image");
static_assert(snapshot_part(model::snapshot_blocks - 1).bytes > 0, "no part is empty");

// ---------------------------------------------------------------------------------------------------------------------
// Lazy replay. A slot's stored state holds only tokens surely in its sequence: each cycle commits the rows' first token
// (the one sampled last cycle) after replaying the drafts the last cycle accepted. A draft row's update is computed for
// its output but not stored; its raw inputs go to the slot's replay ring, and the next cycle replays the accepted
// prefix of them from the stored state. The ring is the slot's, apart from its state image: snapshots never need it,
// because a snapshot is taken at the stored state's position.

inline constexpr int max_drafts = static_cast<int>(model::drafts(model::max_draft_depth));
/** Rows a slot runs in one cycle: its committed token and its drafts. */
inline constexpr int max_rows = 1 + max_drafts;
inline constexpr int max_replays = max_drafts;
/** Tokens one cycle's core runs through a slot's state: the replays, then the rows. */
inline constexpr int max_events = max_replays + max_rows;

/**
 * One draft's raw inputs for one key head's group: in_proj_qkv's bf16 outputs on the group's channels, then in_proj_a's
 * outputs for its value heads and in_proj_b's, fp32 (`model::Value::DeltanetDecayInput`), padded to whole chunks.
 */
inline constexpr int entry_gate_floats = 2 * group;
inline constexpr int entry_gate_chunks = (entry_gate_floats + 3) / 4;
inline constexpr int entry_chunks = group_chunks + entry_gate_chunks;
inline constexpr Count entry_bytes = Count{entry_chunks} * 16;

static_assert(model::precision(model::Value::DeltanetDecayInput) == model::Precision::Fp32);

/** Byte offset of entry `entry` of key head `key_head`'s ring of deltanet layer `layer`, in a slot's ring. */
[[nodiscard]] constexpr auto ring_offset(int layer, int key_head, int entry) -> Count {
	return ((Count(layer) * key_heads + Count(key_head)) * max_drafts + Count(entry)) * entry_bytes;
}

inline constexpr Count ring_bytes_per_slot = Count{layers} * key_heads * max_drafts * entry_bytes;

// ---------------------------------------------------------------------------------------------------------------------
// The activations the core reads and writes, one row per token (verify row or prompt token), as the projections leave
// them: in_proj_qkv's bf16 outputs in card channel order; in_proj_a's and in_proj_b's fp32 outputs, a then b per value
// head; and the core's normalized, weighted fp32 output (`model::Value::DeltanetNormed`), value heads in order.

inline constexpr Count qkv_row_bytes = Count{channels} * sizeof(Bf16);
inline constexpr Count gate_row_bytes = Count{value_heads} * 2 * sizeof(float);
inline constexpr Count output_row_bytes = Count{value_features} * sizeof(float);

/**
 * A device allocation holding T, as a kernel's parameters carry it: the pointer it is on the card. The host never
 * dereferences one; it makes it from a device address in one place, `engine.deltanet`'s `address`.
 */
template<class T>
struct Address {
	T* value;
};

static_assert(sizeof(Address<float>) == 8, "a device address is 64 bits on both sides");

/** The image's side data the core reads for one deltanet layer: each tensor's part on this card. */
struct Weights {
	/** conv1d.weight: [channels][4] bf16, card channel order. */
	Address<Bf16 const> conv;
	/** A_log and dt_bias: one bf16 per value head. */
	Address<Bf16 const> a_log;
	Address<Bf16 const> dt_bias;
	/** The gated norm's weight: [128] bf16. */
	Address<Bf16 const> norm;
};

/** The activation rows a launch reads and writes. */
struct Rows {
	Address<Bf16 const> qkv;
	/** [rows][value heads][2]: a, then b. */
	Address<float const> gates;
	/** [rows][value features]. */
	Address<float> outputs;
};

// ---------------------------------------------------------------------------------------------------------------------
// Decode: one item per live slot and key head group, every item resident at once on its own SM, the whole core one
// wave. An item streams its group's three heads of state (192 KiB) in and its committed state out.

/** One live slot's share of a decode cycle. */
struct SlotCycle {
	/** Which slot: its state image and its ring. */
	unsigned slot;
	/** Accepted drafts of the last cycle to replay from the ring first, at most `max_replays`. */
	unsigned replays;
	/** This cycle's rows, 1 to `max_rows`: the committed token, then the drafts. */
	unsigned rows;
	/** The slot's first row in the activation rows; its rows are consecutive. */
	unsigned first_row;
};

struct DecodeParams {
	/** The slots' state images, slot s at s * slot_image_bytes. */
	Address<unsigned char> images;
	/** The slots' replay rings, slot s at s * ring_bytes_per_slot. */
	Address<unsigned char> rings;
	Weights weights;
	Rows rows;
	unsigned layer;
	/** Live slots this cycle: the first `live` entries of `slots`. */
	unsigned live;
	SlotCycle slots[model::slots];
};

inline constexpr int decode_items = static_cast<int>(model::slots) * key_heads;

// ---------------------------------------------------------------------------------------------------------------------
// Prefill: the same arithmetic over a chunk of prompt tokens, one block per slot and value head, so a prompt alone
// keeps 24 SMs busy. A value head's block reads its group's window but cannot write it while the group's other blocks
// may still read it, so a second launch, after the first, moves each window to the chunk's end.

/** One slot's prompt chunk. */
struct SlotChunk {
	unsigned slot;
	/** Prompt tokens in the chunk, at least one. */
	unsigned tokens;
	unsigned first_row;
};

struct PrefillParams {
	Address<unsigned char> images;
	Weights weights;
	Rows rows;
	unsigned layer;
	unsigned live;
	SlotChunk slots[model::slots];
};

inline constexpr int prefill_items = static_cast<int>(model::slots) * value_heads;
inline constexpr int window_items = static_cast<int>(model::slots) * key_heads;

/** Prompt tokens a prefill block prepares at once: one stage of its pipeline. */
inline constexpr int prefill_block_tokens = 16;

// ---------------------------------------------------------------------------------------------------------------------
// The chain probe: the delta rule alone over prepared inputs, one value head per 128 columns, for measuring a chain's
// latency per token under tree `T` and the throughput of every chain of a layer. Inputs are per value head:
// keys and queries [tokens][128] fp32 in `vector_element<T>` order, values [tokens][128] fp32, gates [tokens].

inline constexpr int probe_heads = value_heads;

struct ProbeParams {
	Address<float const> keys;
	Address<float const> queries;
	Address<float const> values;
	Address<Gates const> gates;
	/** [probe_heads][head_floats] in `StateLayout<T>` order, updated in place. */
	Address<float> state;
	/** [probe_heads][tokens][128] fp32. */
	Address<float> outputs;
	/** Per block: the cycle count at the first token and after the last. */
	Address<unsigned long long> cycles;
	unsigned tokens;
};

/** The tree self-check: one report, zeroed by the host before the launch. */
struct CheckParams {
	Address<prims::selfcheck::Report> report;
};

/** Butterfly rounds each warp of the tree self-check runs, each over fresh values. */
inline constexpr int check_rounds = 64;
inline constexpr int check_blocks = 64;

/**
 * The bits of lane `lane`'s value in round `round` of warp `warp` of block `block`: a finite float of either sign
 * with a random significand and an exponent within 2^±31, so sums both cancel and round.
 */
[[nodiscard]] constexpr auto check_value(unsigned block, unsigned warp, unsigned round, unsigned lane) -> float {
	unsigned long long mixed = (static_cast<unsigned long long>(block) << 48U) ^ (static_cast<unsigned long long>(warp) << 32U) ^
	                           (static_cast<unsigned long long>(round) << 8U) ^ lane;
	mixed ^= mixed >> 33U;
	mixed *= 0xff51afd7ed558ccdULL;
	mixed ^= mixed >> 33U;
	mixed *= 0xc4ceb9fe1a85ec53ULL;
	mixed ^= mixed >> 33U;
	auto const bits = static_cast<unsigned>(mixed);
	unsigned const exponent = 127U - 31U + ((bits >> 23U) & 63U);
	return __builtin_bit_cast(float, (bits & 0x807fffffU) | (exponent << 23U));
}

}

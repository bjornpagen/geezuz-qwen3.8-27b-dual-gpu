// SPDX-License-Identifier: AGPL-3.0-only
//
// The model description: every dimension of Qwen3.8-27B-FP8, the tensors the engine reads from its checkpoint, how
// each tensor divides across the two cards, the serving geometry, and the per-card byte budgets derived from them.
// It is the single source of these facts. Device code includes it; `engine.model` re-exports it to host code. It uses
// no standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

// The rounding section's fixed algorithms are built on the engine's fixed transcendentals, which use no headers either.
#include "prims/transcendental.h"

namespace engine::model {

using Count = decltype(sizeof(0));

/** A ratio of two counts, kept exact. */
struct Share {
	Count numerator;
	Count denominator;
};

[[nodiscard]] constexpr auto divide_up(Count value, Count by) -> Count {
	return (value + by - 1) / by;
}

inline constexpr Count cards = 2;

/** A card of the tensor-parallel pair, by its rank. */
enum class Card : unsigned char {
	Zero,
	One,
};

static_assert(cards == 2, "one Card case per tensor-parallel rank");

inline constexpr Count slots = 8;

/** Verify rows come in tiles of the mma's n operand: tokens are the n8 side, weights the m16 side. */
inline constexpr Count row_tile = 8;

/**
 * Verify rows per decode cycle, shared by every slot's committed token and drafts. A budget between multiples of the
 * tile costs the next tile, so only whole tiles exist; past four, compute overtakes the weight stream.
 */
enum class RowBudget : unsigned char {
	ThreeTiles,
	FourTiles,
};

[[nodiscard]] constexpr auto rows(RowBudget budget) -> Count {
	switch (budget) {
	case RowBudget::ThreeTiles:
		return 3 * row_tile;
	case RowBudget::FourTiles:
		return 4 * row_tile;
	}
	__builtin_unreachable();
}

inline constexpr RowBudget row_budgets[] = {RowBudget::ThreeTiles, RowBudget::FourTiles};

/** The budget the engine ships with; four tiles is the first alternative measured on the cards. */
inline constexpr RowBudget row_budget = RowBudget::ThreeTiles;

/**
 * Draft tokens a slot carries into one cycle. The producer stream depends only on this, so each depth is its own
 * compile-time stream program.
 */
enum class DraftDepth : unsigned char {
	One,
	Two,
	Three,
	Four,
};

[[nodiscard]] constexpr auto drafts(DraftDepth depth) -> Count {
	switch (depth) {
	case DraftDepth::One:
		return 1;
	case DraftDepth::Two:
		return 2;
	case DraftDepth::Three:
		return 3;
	case DraftDepth::Four:
		return 4;
	}
	__builtin_unreachable();
}

inline constexpr DraftDepth max_draft_depth = DraftDepth::Four;

// GDDR6X on each 3090.
inline constexpr Count card_memory_bytes = Count{24} * 1024 * 1024 * 1024;

// The driver maps device memory in pages, so every allocation takes whole pages. 2 MiB is the granularity the driver
// reports for device memory on current nvidia cards; like the reserve below, it is an assumption until bring-up reads
// it on these cards.
inline constexpr Count allocation_granularity = Count{2} * 1024 * 1024;

/** The device memory an allocation of `bytes` takes: whole pages. */
[[nodiscard]] constexpr auto allocated(Count bytes) -> Count {
	return divide_up(bytes, allocation_granularity) * allocation_granularity;
}

// What the driver keeps of each card before the engine allocates: its context, the jit-compiled module and the
// resident kernel's local memory. 512 MiB is an assumption, not a measurement; bring-up on the cards measures it and
// replaces this figure.
inline constexpr Count driver_reserve_bytes = Count{512} * 1024 * 1024;

/** The most of the block pool deltanet snapshots may hold; kv uses every block they do not. */
inline constexpr Share snapshot_share = {.numerator = 1, .denominator = 4};

inline constexpr Count hidden = 5120;
inline constexpr Count vocab = 248'320;

// lm_head has a row for every id below `vocab`, but the tokenizer has tokens only below this: the rows above are
// padding. No tokenizer emits or decodes a padding id, so none is ever sampled, by the engine or the reference: their
// logits take no part in top-k, argmax or the softmax.
inline constexpr Count sampled_vocabulary = 248'077;
inline constexpr Count ffn = 17'408;
inline constexpr Count max_positions = 262'144;
inline constexpr float rms_norm_eps = 1e-6F;

/** How an RMSNorm applies its stored weight. */
enum class NormScale : unsigned char {
	/** x * (1 + w): the stored weight is centred on zero. */
	OnePlusWeight,
	/** x * w. */
	Weight,
};

/** The nonlinearity a gate applies to its gating input before the product. */
enum class Gate : unsigned char {
	Sigmoid,
	Silu,
};

// Every RMSNorm stores a zero-centred weight (layer, q/k, final and mtp norms read near zero, many negative, in the
// checkpoint), except deltanet's gated output norm, whose weights read near one.
inline constexpr NormScale rms_norm_scale = NormScale::OnePlusWeight;
inline constexpr NormScale deltanet_norm_scale = NormScale::Weight;

/** The mlp computes down(silu(gate(x)) * up(x)). */
inline constexpr Gate mlp_gate = Gate::Silu;

/** The token mixer of a decoder layer. */
enum class Mixer : unsigned char {
	/** Gated deltanet: a fixed-size recurrent state per slot, no kv. */
	LinearAttention,
	/** Softmax attention over the kv cache. */
	FullAttention,
};

inline constexpr Count trunk_layers = 64;

// The checkpoint's `layer_types`, as written: three deltanet layers, then one attention layer.
inline constexpr Mixer trunk_mixers[trunk_layers] = {
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention,
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention,
    Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention,
    Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention,
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention,
    Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention,
    Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention,
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention,
    Mixer::LinearAttention, Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention,
    Mixer::FullAttention,   Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,
    Mixer::LinearAttention, Mixer::LinearAttention, Mixer::LinearAttention, Mixer::FullAttention,
};

[[nodiscard]] consteval auto trunk_layers_of(Mixer mixer) -> Count {
	auto count = Count{0};
	for (auto const each : trunk_mixers) {
		count += each == mixer ? 1 : 0;
	}
	return count;
}

// The mtp head is one decoder layer with full attention and its own kv. It reads the trunk's embedding table
// (`mtp_use_dedicated_embeddings: false`) and drafts through the trunk's lm_head rows.
inline constexpr Count mtp_layers = 1;
inline constexpr Mixer mtp_mixer = Mixer::FullAttention;

// Full attention: grouped-query, with an output gate. Each query head's q_proj rows are the head's query followed by
// its gate; the attention output is multiplied by sigmoid(gate) before o_proj.
inline constexpr Count query_heads = 24;
inline constexpr Count kv_heads = 4;
inline constexpr Count head_dim = 256;
inline constexpr Count q_proj_rows_per_head = 2 * head_dim;
inline constexpr Count query_features = query_heads * head_dim;
inline constexpr Gate attention_output_gate = Gate::Sigmoid;

// Rotary embedding on the first `rotary_dim` features of each query and key head; the rest pass through. Pairs are
// half-split: feature j rotates with feature j + rotary_dim / 2 at frequency j, inverse frequency
// rope_theta^(-2j / rotary_dim). The checkpoint's rope is interleaved mrope: frequency j takes its position from
// stream j mod 3 (temporal, height, width) within the first 3 * section entries of that stream, the temporal stream
// otherwise. Text input gives all three streams the token's position, so the interleave selects among equal values:
// every frequency sees the one position, in natural order j = 0 .. rotary_dim / 2 - 1, exactly plain partial rope.
inline constexpr Share partial_rotary = {.numerator = 1, .denominator = 4};
inline constexpr Count rotary_dim = head_dim * partial_rotary.numerator / partial_rotary.denominator;
inline constexpr float rope_theta = 10'000'000.0F;
inline constexpr Count mrope_sections[3] = {11, 11, 10};

// Gated deltanet. Each value head reads key head (value head / (value_heads / key_heads)): the key and query heads
// are repeated, interleaved, across the value heads. The short causal convolution runs over the concatenated q|k|v
// channels. Its output passes an RMSNorm gated by silu(z).
inline constexpr Count key_heads = 16;
inline constexpr Count value_heads = 48;
inline constexpr Count key_dim = 128;
inline constexpr Count value_dim = 128;
inline constexpr Count conv_kernel = 4;
inline constexpr Count value_features = value_heads * value_dim;
inline constexpr Count conv_channels = 2 * key_heads * key_dim + value_features;
inline constexpr Gate deltanet_output_gate = Gate::Silu;

/** Token ids the mtp head drafts over: the rows of `data/draft_vocab_32k.bin`. */
inline constexpr Count draft_vocab = 32'768;

static_assert(trunk_layers_of(Mixer::LinearAttention) + trunk_layers_of(Mixer::FullAttention) == trunk_layers);
static_assert(draft_vocab <= sampled_vocabulary && sampled_vocabulary <= vocab, "every draft id is a token, every token a row");
static_assert(query_heads % kv_heads == 0, "grouped-query attention");
static_assert(value_heads % key_heads == 0, "every value head reads one key head");
static_assert(rotary_dim * partial_rotary.denominator == head_dim * partial_rotary.numerator, "whole rotary features");
static_assert(mrope_sections[0] + mrope_sections[1] + mrope_sections[2] == rotary_dim / 2, "one section per pair");

// Each card holds a contiguous half of every head kind, so every head it holds finds the kv or key head it reads on
// the same card.
inline constexpr Count query_heads_per_card = query_heads / cards;
inline constexpr Count kv_heads_per_card = kv_heads / cards;
inline constexpr Count key_heads_per_card = key_heads / cards;
inline constexpr Count value_heads_per_card = value_heads / cards;
inline constexpr Count ffn_per_card = ffn / cards;
inline constexpr Count vocab_per_card = vocab / cards;
inline constexpr Count draft_vocab_per_card = draft_vocab / cards;

static_assert(query_heads % cards == 0 && kv_heads % cards == 0, "attention heads divide across the cards");
static_assert(key_heads % cards == 0 && value_heads % cards == 0, "deltanet heads divide across the cards");
static_assert(query_heads_per_card % (query_heads / kv_heads) == 0, "a card's query groups are whole");
static_assert(value_heads_per_card % (value_heads / key_heads) == 0, "a card's value groups are whole");
static_assert(ffn % cards == 0 && vocab % cards == 0 && draft_vocab % cards == 0);

/**
 * Words of 64 bits in one card's mask of a verify row: a bit per row of the card's lm_head, in its row order, padding
 * included. The grammar writes masks this wide, the pipeline carries them, and decode copies them to the cards.
 */
inline constexpr Count card_mask_words = divide_up(vocab_per_card, 64);

/** How a tensor's elements are stored in the checkpoint. */
enum class Format : unsigned char {
	/**
	 * e4m3 weight `<name>` with a bf16 companion `<name>_scale_inv` holding one factor per 128 x 128 block (partial at
	 * the edges): weight = e4m3 value x the block's factor.
	 */
	E4m3,
	Bf16,
};

inline constexpr Count scale_block = 128;
inline constexpr char scale_suffix[] = "_scale_inv";

/** Checkpoint tensors whose name begins with this prefix belong to the vision tower, which the engine drops. */
inline constexpr char excluded_prefix[] = "model.visual.";

/** Where a tensor lives, which decides its name prefix and how many instances exist. */
enum class Scope : unsigned char {
	/** Once: embedding, final norm, lm_head. The name is the full name. */
	Model,
	/** Once in the mtp head, outside its decoder layer: `mtp.` + name. */
	Mtp,
	/** In every decoder layer, the trunk's and the mtp head's. */
	DecoderLayer,
	/** In every decoder layer whose mixer is linear attention. */
	LinearAttentionLayer,
	/** In every decoder layer whose mixer is full attention. */
	FullAttentionLayer,
};

inline constexpr char mtp_prefix[] = "mtp.";

/** A decoder layer's tensors are named prefix + layer index + "." + name. */
inline constexpr char trunk_layer_prefix[] = "model.language_model.layers.";
inline constexpr char mtp_layer_prefix[] = "mtp.layers.";

/**
 * How a tensor divides across the cards. A matrix is [output, input]; axis 0 is the output. A sharded split cuts its
 * axis into sections laid end to end and gives card c the c-th equal part of every section.
 */
enum class Split : unsigned char {
	/** The whole tensor on every card. */
	Replicated,
	/** In pinned host memory, gathered by token row; on no card. */
	Host,
	/** Axis 0 in one section: output features (column-parallel), heads, or vocabulary rows. */
	Rows,
	/** Axis 1 in one section: input features (row-parallel); the cards' partial sums are added. */
	Columns,
	/** Axis 0 as the deltanet channels q | k | v, each halved, so a card keeps its key heads and their value heads. */
	DeltanetChannels,
	/** Axis 1 as two hidden-wide inputs (normalized embedding | normalized hidden state), each halved. */
	HiddenPair,
	/**
	 * Axis 0 is the vocabulary, permuted before it is halved: card c's rows are every second draft id from the c-th,
	 * then every second other id from the c-th, each in ascending order (`engine.vocabulary` holds the map). A card
	 * holds no range of token ids, so its shard has no ranges; only its row count is a range fact.
	 */
	Vocabulary,
};

struct Shape {
	Count rank;
	Count extents[3];
};

[[nodiscard]] constexpr auto vector(Count size) -> Shape {
	return {.rank = 1, .extents = {size, 0, 0}};
}

[[nodiscard]] constexpr auto matrix(Count rows, Count columns) -> Shape {
	return {.rank = 2, .extents = {rows, columns, 0}};
}

/** A depthwise convolution's weight: [channels, 1, kernel]. */
[[nodiscard]] constexpr auto depthwise(Count channels, Count kernel) -> Shape {
	return {.rank = 3, .extents = {channels, 1, kernel}};
}

struct Tensor {
	char const* name;
	Shape shape;
	Scope scope;
	Format format;
	Split split;
};

/** Every checkpoint tensor the engine reads; the checkpoint holds nothing else but the vision tower. */
inline constexpr Tensor inventory[] = {
    // Untied: the embedding and lm_head are separate tensors. The embedding stays in host memory.
    {"model.language_model.embed_tokens.weight", matrix(vocab, hidden), Scope::Model, Format::Bf16, Split::Host},
    {"model.language_model.norm.weight", vector(hidden), Scope::Model, Format::Bf16, Split::Replicated},
    {"lm_head.weight", matrix(vocab, hidden), Scope::Model, Format::Bf16, Split::Vocabulary},

    {"fc.weight", matrix(hidden, 2 * hidden), Scope::Mtp, Format::Bf16, Split::HiddenPair},
    {"pre_fc_norm_embedding.weight", vector(hidden), Scope::Mtp, Format::Bf16, Split::Replicated},
    {"pre_fc_norm_hidden.weight", vector(hidden), Scope::Mtp, Format::Bf16, Split::Replicated},
    {"norm.weight", vector(hidden), Scope::Mtp, Format::Bf16, Split::Replicated},

    {"input_layernorm.weight", vector(hidden), Scope::DecoderLayer, Format::Bf16, Split::Replicated},
    {"post_attention_layernorm.weight", vector(hidden), Scope::DecoderLayer, Format::Bf16, Split::Replicated},
    {"mlp.gate_proj.weight", matrix(ffn, hidden), Scope::DecoderLayer, Format::E4m3, Split::Rows},
    {"mlp.up_proj.weight", matrix(ffn, hidden), Scope::DecoderLayer, Format::E4m3, Split::Rows},
    {"mlp.down_proj.weight", matrix(hidden, ffn), Scope::DecoderLayer, Format::E4m3, Split::Columns},

    {"linear_attn.in_proj_qkv.weight", matrix(conv_channels, hidden), Scope::LinearAttentionLayer, Format::E4m3, Split::DeltanetChannels},
    {"linear_attn.in_proj_z.weight", matrix(value_features, hidden), Scope::LinearAttentionLayer, Format::E4m3, Split::Rows},
    {"linear_attn.in_proj_a.weight", matrix(value_heads, hidden), Scope::LinearAttentionLayer, Format::Bf16, Split::Rows},
    {"linear_attn.in_proj_b.weight", matrix(value_heads, hidden), Scope::LinearAttentionLayer, Format::Bf16, Split::Rows},
    {"linear_attn.conv1d.weight", depthwise(conv_channels, conv_kernel), Scope::LinearAttentionLayer, Format::Bf16,
	 Split::DeltanetChannels},
    {"linear_attn.A_log", vector(value_heads), Scope::LinearAttentionLayer, Format::Bf16, Split::Rows},
    {"linear_attn.dt_bias", vector(value_heads), Scope::LinearAttentionLayer, Format::Bf16, Split::Rows},
    {"linear_attn.norm.weight", vector(value_dim), Scope::LinearAttentionLayer, Format::Bf16, Split::Replicated},
    {"linear_attn.out_proj.weight", matrix(hidden, value_features), Scope::LinearAttentionLayer, Format::E4m3, Split::Columns},

    {"self_attn.q_proj.weight", matrix(query_heads * q_proj_rows_per_head, hidden), Scope::FullAttentionLayer, Format::E4m3, Split::Rows},
    {"self_attn.k_proj.weight", matrix(kv_heads * head_dim, hidden), Scope::FullAttentionLayer, Format::E4m3, Split::Rows},
    {"self_attn.v_proj.weight", matrix(kv_heads * head_dim, hidden), Scope::FullAttentionLayer, Format::E4m3, Split::Rows},
    {"self_attn.q_norm.weight", vector(head_dim), Scope::FullAttentionLayer, Format::Bf16, Split::Replicated},
    {"self_attn.k_norm.weight", vector(head_dim), Scope::FullAttentionLayer, Format::Bf16, Split::Replicated},
    {"self_attn.o_proj.weight", matrix(hidden, query_heads* head_dim), Scope::FullAttentionLayer, Format::E4m3, Split::Columns},
};

/** The trunk layers and mtp layers a scope's tensors appear in; zero for the once-only scopes. */
struct Instances {
	Count trunk;
	Count mtp;
};

/** Whether a decoder layer with `mixer` holds the tensors of `scope`. */
[[nodiscard]] constexpr auto in_layer(Scope scope, Mixer mixer) -> bool {
	switch (scope) {
	case Scope::Model:
	case Scope::Mtp:
		return false;
	case Scope::DecoderLayer:
		return true;
	case Scope::LinearAttentionLayer:
		return mixer == Mixer::LinearAttention;
	case Scope::FullAttentionLayer:
		return mixer == Mixer::FullAttention;
	}
	__builtin_unreachable();
}

[[nodiscard]] consteval auto layer_instances(Scope scope) -> Instances {
	auto trunk = Count{0};
	for (auto const mixer : trunk_mixers) {
		trunk += in_layer(scope, mixer) ? 1 : 0;
	}
	return {.trunk = trunk, .mtp = in_layer(scope, mtp_mixer) ? mtp_layers : 0};
}

struct Range {
	Count begin;
	Count length;
};

/** The ranges of `axis` one card holds, in axis order; every other axis is held whole. */
struct Shard {
	Count axis;
	Count count;
	Range ranges[3];
};

struct Sections {
	Count count;
	Count lengths[3];
};

[[nodiscard]] constexpr auto split_axis(Split split) -> Count {
	switch (split) {
	case Split::Replicated:
	case Split::Host:
	case Split::Rows:
	case Split::DeltanetChannels:
	case Split::Vocabulary:
		return 0;
	case Split::Columns:
	case Split::HiddenPair:
		return 1;
	}
	__builtin_unreachable();
}

[[nodiscard]] constexpr auto sections(Tensor const& tensor) -> Sections {
	auto const whole = tensor.shape.extents[split_axis(tensor.split)];
	switch (tensor.split) {
	case Split::Host:
	case Split::Vocabulary:
		return {.count = 0, .lengths = {0, 0, 0}};
	case Split::Replicated:
	case Split::Rows:
	case Split::Columns:
		return {.count = 1, .lengths = {whole, 0, 0}};
	case Split::DeltanetChannels:
		return {.count = 3, .lengths = {key_heads * key_dim, key_heads * key_dim, value_features}};
	case Split::HiddenPair:
		return {.count = 2, .lengths = {hidden, hidden, 0}};
	}
	__builtin_unreachable();
}

[[nodiscard]] constexpr auto shard(Tensor const& tensor, Count card) -> Shard {
	auto const parts = sections(tensor);
	auto result = Shard{.axis = split_axis(tensor.split), .count = parts.count, .ranges = {}};
	auto offset = Count{0};
	for (auto index = Count{0}; index < parts.count; ++index) {
		auto const whole = parts.lengths[index];
		result.ranges[index] = tensor.split == Split::Replicated ? Range{.begin = offset, .length = whole}
		                                                         : Range{.begin = offset + card * (whole / cards), .length = whole / cards};
		offset += whole;
	}
	return result;
}

[[nodiscard]] constexpr auto held(Shard const& part) -> Count {
	auto total = Count{0};
	for (auto index = Count{0}; index < part.count; ++index) {
		total += part.ranges[index].length;
	}
	return total;
}

[[nodiscard]] constexpr auto elements(Shape const& shape) -> Count {
	auto total = Count{1};
	for (auto axis = Count{0}; axis < shape.rank; ++axis) {
		total *= shape.extents[axis];
	}
	return total;
}

/** The shape of the part of `tensor` one card holds. */
[[nodiscard]] constexpr auto card_shape(Tensor const& tensor, Count card) -> Shape {
	auto const part = shard(tensor, card);
	auto shape = tensor.shape;
	// A vocabulary split holds its rows by the permutation, not by a range: each card an equal part.
	shape.extents[part.axis] = tensor.split == Split::Vocabulary ? tensor.shape.extents[0] / cards : held(part);
	return shape;
}

/** The checkpoint shape of an e4m3 tensor's block-scale companion. */
[[nodiscard]] constexpr auto scale_shape(Shape const& weight) -> Shape {
	return matrix(divide_up(weight.extents[0], scale_block), divide_up(weight.extents[1], scale_block));
}

[[nodiscard]] constexpr auto element_bytes(Format format) -> Count {
	switch (format) {
	case Format::E4m3:
		return 1;
	case Format::Bf16:
		return 2;
	}
	__builtin_unreachable();
}

/** Bytes of `shape` stored in `format`, block scales included. */
[[nodiscard]] constexpr auto stored_bytes(Format format, Shape const& shape) -> Count {
	auto const weights = elements(shape) * element_bytes(format);
	return format == Format::E4m3 ? weights + elements(scale_shape(shape)) * element_bytes(Format::Bf16) : weights;
}

[[nodiscard]] constexpr auto card_bytes(Tensor const& tensor, Count card) -> Count {
	return stored_bytes(tensor.format, card_shape(tensor, card));
}

// Laws every split obeys; the kernels and the baker rely on each.
[[nodiscard]] consteval auto splits_are_sound() -> bool {
	for (auto const& tensor : inventory) {
		auto const parts = sections(tensor);
		auto const axis = split_axis(tensor.split);
		auto total = Count{0};
		for (auto index = Count{0}; index < parts.count; ++index) {
			auto const whole = parts.lengths[index];
			total += whole;
			if (tensor.split != Split::Replicated) {
				// Every card holds an equal part, and an e4m3 part starts and ends on a scale block, so the scale
				// tensor splits exactly with its weight.
				auto const step = tensor.format == Format::E4m3 ? cards * scale_block : cards;
				if (whole % step != 0) {
					return false;
				}
			}
		}
		// The vocabulary permutation gives each card half the draft ids and half the others, in whole lm_head blocks.
		auto const permuted = tensor.split != Split::Vocabulary ||
		                      (tensor.shape.extents[0] == vocab && draft_vocab % (cards * scale_block) == 0 && vocab % cards == 0);
		auto const covers = tensor.split == Split::Host || tensor.split == Split::Vocabulary || total == tensor.shape.extents[axis];
		auto const in_rank = axis < tensor.shape.rank;
		auto const blocked = tensor.format != Format::E4m3 || tensor.shape.rank == 2;
		auto const balanced = card_bytes(tensor, 0) == card_bytes(tensor, cards - 1);
		if (!covers || !permuted || !in_rank || !blocked || !balanced) {
			return false;
		}
	}
	return true;
}

static_assert(splits_are_sound(), "every split covers its axis in equal, scale-block-aligned parts on a real axis");

/** The weight streams of one decode cycle. */
enum class Stream : unsigned char {
	/** The 64 trunk decoder layers. */
	Body,
	/** The final norm and the lm_head's vocabulary rows. */
	Head,
	/** The mtp head: its decoder layer, fc and norms. */
	Mtp,
};

[[nodiscard]] consteval auto copies_in(Tensor const& tensor, Stream stream) -> Count {
	auto const layers = layer_instances(tensor.scope);
	switch (stream) {
	case Stream::Body:
		return layers.trunk;
	case Stream::Head:
		return tensor.scope == Scope::Model ? 1 : 0;
	case Stream::Mtp:
		return layers.mtp + (tensor.scope == Scope::Mtp ? 1 : 0);
	}
	__builtin_unreachable();
}

/** Bytes one card reads for one pass over `stream`; every card reads the same (the splits are balanced). */
[[nodiscard]] consteval auto stream_bytes(Stream stream) -> Count {
	auto total = Count{0};
	for (auto const& tensor : inventory) {
		total += copies_in(tensor, stream) * card_bytes(tensor, 0);
	}
	return total;
}

inline constexpr Count body_bytes_per_card = stream_bytes(Stream::Body);
inline constexpr Count head_bytes_per_card = stream_bytes(Stream::Head);
inline constexpr Count mtp_bytes_per_card = stream_bytes(Stream::Mtp);

/**
 * One draft lm_head step: each card's half of the draft vocabulary. It is a view, not a copy: the baker orders each
 * card's lm_head rows so those rows are a contiguous prefix, so these bytes are streamed but take no memory of their own.
 */
inline constexpr Count draft_head_bytes_per_card = draft_vocab_per_card * hidden * element_bytes(Format::Bf16);

/** Host bytes of the embedding table, gathered by token row each cycle. */
inline constexpr Count embedding_host_bytes = vocab * hidden * element_bytes(Format::Bf16);

inline constexpr Count main_pass_bytes_per_card = body_bytes_per_card + head_bytes_per_card;

// d drafts take d mtp passes: the catch-up pass over the accepted tokens already yields the first draft, then d - 1
// steps follow. Each pass ends in one draft lm_head step.
[[nodiscard]] constexpr auto mtp_bytes_per_cycle_per_card(DraftDepth depth) -> Count {
	return drafts(depth) * mtp_bytes_per_card;
}

[[nodiscard]] constexpr auto draft_head_bytes_per_cycle_per_card(DraftDepth depth) -> Count {
	return drafts(depth) * draft_head_bytes_per_card;
}

/** Weight bytes resident on each card: the trunk, the head and the mtp head; the draft lm_head is a view. */
inline constexpr Count weight_bytes_per_card = body_bytes_per_card + head_bytes_per_card + mtp_bytes_per_card;

/** Parameters of the engine's tensors in each format, whole model; block scales count as bf16 parameters. */
[[nodiscard]] consteval auto parameters(Format format) -> Count {
	auto total = Count{0};
	for (auto const& tensor : inventory) {
		auto const layers = layer_instances(tensor.scope);
		auto const copies = tensor.scope == Scope::Model || tensor.scope == Scope::Mtp ? 1 : layers.trunk + layers.mtp;
		if (tensor.format == format) {
			total += copies * elements(tensor.shape);
		}
		if (tensor.format == Format::E4m3 && format == Format::Bf16) {
			total += copies * elements(scale_shape(tensor.shape));
		}
	}
	return total;
}

/** Whether `value` rounds to `stated` multiples of `unit`: a budget figure checked at the precision it is stated. */
[[nodiscard]] consteval auto rounds_to(Count value, Count unit, Count stated) -> bool {
	return (value + unit / 2) / unit == stated;
}

inline constexpr Count gigabyte = 1'000'000'000;

static_assert(rounds_to(body_bytes_per_card, gigabyte / 100, 1219), "body: 12.19 GB per card per cycle");
static_assert(rounds_to(head_bytes_per_card, gigabyte / 100, 127), "lm_head: 1.27 GB per card per cycle");
static_assert(rounds_to(main_pass_bytes_per_card, gigabyte / 100, 1346), "main pass: 13.46 GB per card per cycle");
static_assert(rounds_to(mtp_bytes_per_card, gigabyte / 100, 24), "mtp head: 0.24 GB per card per pass");
static_assert(rounds_to(mtp_bytes_per_cycle_per_card(DraftDepth::Four), gigabyte / 100, 95), "mtp head: 0.95 GB at d = 4");
static_assert(rounds_to(draft_head_bytes_per_card, gigabyte / 100, 17), "draft lm_head: 0.17 GB per card per step");
static_assert(rounds_to(draft_head_bytes_per_cycle_per_card(DraftDepth::Four), gigabyte / 100, 67), "draft lm_head: 0.67 GB at d = 4");
static_assert(rounds_to(embedding_host_bytes, gigabyte / 100, 254), "embedding table: 2.54 GB in host memory");
static_assert(rounds_to(parameters(Format::E4m3), gigabyte / 100, 2470), "24.70b e4m3 parameters");
static_assert(rounds_to(parameters(Format::Bf16), gigabyte / 100, 262), "2.62b bf16 parameters");

/** Layers that keep kv: the trunk's full-attention layers and the mtp head's. */
inline constexpr Count kv_layers = layer_instances(Scope::FullAttentionLayer).trunk + layer_instances(Scope::FullAttentionLayer).mtp;

// Each position of each kv layer holds a key and a value row per kv head, bf16.
inline constexpr Count kv_bytes_per_token_per_card = kv_layers * kv_heads_per_card * 2 * head_dim * element_bytes(Format::Bf16);
inline constexpr Count trunk_kv_bytes_per_token_per_card =
    layer_instances(Scope::FullAttentionLayer).trunk * kv_heads_per_card * 2 * head_dim * element_bytes(Format::Bf16);

/** Positions per kv block: two of the split-k attention's 32-position tiles. */
inline constexpr Count kv_block_tokens = 64;
inline constexpr Count kv_block_bytes = kv_block_tokens * kv_bytes_per_token_per_card;

/** Entries of one slot's block table: enough blocks for the longest sequence the model takes. */
inline constexpr Count kv_table_blocks = divide_up(max_positions, kv_block_tokens);

static_assert(kv_table_blocks == 4096);

static_assert(trunk_kv_bytes_per_token_per_card == Count{32} * 1024, "trunk kv reads: 32 KiB per position per card");

/** fp32 deltanet state of one slot on one card: a key_dim x value_dim matrix per value head per deltanet layer. */
inline constexpr Count deltanet_state_bytes_per_slot =
    trunk_layers_of(Mixer::LinearAttention) * value_heads_per_card * key_dim * value_dim * sizeof(float);

// The convolution's window: the last conv_kernel - 1 inputs per channel, which are bf16 activations.
inline constexpr Count conv_window_bytes_per_slot =
    trunk_layers_of(Mixer::LinearAttention) * (conv_kernel - 1) * (conv_channels / cards) * element_bytes(Format::Bf16);

inline constexpr Count slot_state_bytes = deltanet_state_bytes_per_slot + conv_window_bytes_per_slot;

/** A deltanet snapshot (recurrent state and convolution window) occupies whole blocks. */
inline constexpr Count snapshot_blocks = divide_up(slot_state_bytes, kv_block_bytes);

static_assert(rounds_to(deltanet_state_bytes_per_slot, gigabyte / 1000, 75), "deltanet state: 75 MB per slot per card");

/**
 * The card's workspace: every allocation besides the weights, the slot state and the pool, each its own whole pages.
 * Decode's and prefill's buffers coexist; neither aliases the other by phase. Prefill could reuse only the decode
 * scratch nothing keeps across a stopped loop (attention's records and decode's rows, ~123 MB, ~55 blocks), never the
 * exchange buffers the peer writes or the flags whose epochs both kernels keep; that reuse would make the two hosts
 * share one allocation, for about one percent more pool.
 *
 * Each figure is a layout's size, derived where the layout is defined: the layouts are built from the schedule,
 * attention, deltanet and sampling headers, which all include this one, so this header cannot compute them. It holds
 * the figures as constants, and each layout asserts that its size is exactly its figure, so neither drifts.
 */
struct WorkspacePlan {
	/** The decode kernel's workspace: the schedule's buffers (exchanges, flags, attention's records, the replay rings, the
	 * rope table), then decode's rows, the slots' books and block tables. */
	Count decode;
	/** The lm_head row-to-token table: a u32 token id per row of the card's half. */
	Count row_tokens;
	/** The decode kernel's control block, which the host and both cards' kernels share. */
	Count control;
	/** Prefill's workspace: a chunk's rows, its cross-card exchange and each slot's carried final-norm row. */
	Count prefill;
};

inline constexpr WorkspacePlan workspace_plan = {
    .decode = 210'562'560,
    .row_tokens = vocab_per_card * sizeof(unsigned),
    .control = 384,
    .prefill = 460'775'424,
};

/** The device memory the workspace takes: each allocation in whole pages. */
inline constexpr Count workspace_bytes = allocated(workspace_plan.decode) + allocated(workspace_plan.row_tokens) +
                                         allocated(workspace_plan.control) + allocated(workspace_plan.prefill);

/**
 * Each card's memory, outermost first, each allocation in whole pages. The block pool is what remains, so it is
 * derived, never chosen. Its blocks hold kv or deltanet snapshots; snapshots may hold at most their share, kv any block
 * they do not hold.
 */
struct CardMemory {
	Count driver_reserve;
	Count weights;
	Count workspace;
	Count slot_state;
	Count pool_blocks;
};

[[nodiscard]] consteval auto plan_card_memory() -> CardMemory {
	auto memory = CardMemory{
	    .driver_reserve = driver_reserve_bytes,
	    .weights = allocated(weight_bytes_per_card),
	    .workspace = workspace_bytes,
	    .slot_state = allocated(slots * slot_state_bytes),
	    .pool_blocks = 0,
	};
	auto const room = card_memory_bytes - memory.driver_reserve - memory.weights - memory.workspace - memory.slot_state;
	// The pool is one allocation of whole blocks, so its last page can hold less than a block: take the most blocks
	// whose pages fit.
	memory.pool_blocks = room / kv_block_bytes;
	while (allocated(memory.pool_blocks * kv_block_bytes) > room) {
		--memory.pool_blocks;
	}
	return memory;
}

inline constexpr CardMemory card_memory = plan_card_memory();

/** Blocks of the one pool kv and snapshots share. */
inline constexpr Count pool_blocks = card_memory.pool_blocks;
inline constexpr Count pool_tokens = pool_blocks * kv_block_tokens;
inline constexpr Count pool_bytes = pool_blocks * kv_block_bytes;

/** The most blocks snapshots may hold at once: their share of the pool, in whole snapshots. */
inline constexpr Count snapshot_cap_blocks =
    pool_blocks * snapshot_share.numerator / snapshot_share.denominator / snapshot_blocks * snapshot_blocks;

/** Snapshots the pool holds at most. */
inline constexpr Count snapshot_capacity = snapshot_cap_blocks / snapshot_blocks;

static_assert(card_memory.driver_reserve + card_memory.weights + card_memory.workspace + card_memory.slot_state + allocated(pool_bytes) <=
                  card_memory_bytes,
              "everything a card holds fits its memory");
static_assert(snapshot_capacity >= slots, "every slot can park its state in the prefix cache at once");
static_assert(pool_blocks >= slots, "every slot holds at least one kv block");

[[nodiscard]] consteval auto row_budgets_fit_the_slots() -> bool {
	for (auto const budget : row_budgets) {
		if (rows(budget) < slots || rows(budget) > slots * (1 + drafts(max_draft_depth))) {
			return false;
		}
	}
	return true;
}

static_assert(row_budgets_fit_the_slots(), "every row budget gives each slot its committed row and no slot more than its deepest draft");

// ---------------------------------------------------------------------------------------------------------------------
// Rounding: where the engine rounds to bf16, what stays fp32, the order of every sum, every tree, constant and fixed
// algorithm, held here once. The kernels' arithmetic is these functions (each device header calls them), and the cpu
// reference calls them through `engine.model`, so a kernel and the reference agree bit for bit and any difference
// between them is a bug, not noise. Where a kernel chose a parallel mapping for its speed (a tree's lanes, attention's
// groups of chunks, a weight product's split over the SMs), the mapping is recorded here and the reference follows it.
//
// Every fp32 operation is ieee, rounded to nearest even, with no contraction. Every multiply-accumulate is one fma, a
// single rounding: where the product is exact (bf16 x bf16 in fp32) an fma and a multiply then add agree, and it is an
// fma there too so that every accumulation reads one way. exp, log, rsqrt, sigmoid, silu and softplus are the engine's
// fixed routines (`prims::math`); cos and sin are `rope::rotation`. The tensor core's own accumulation is not fp32's:
// each `mma.m16n8k16` step is what `engine.mma_model` says, and an "mma chain" is a run of steps whose accumulator
// passes from step to step.

namespace fixed = device::prims::math;

/** How a value is held between the operation that produces it and those that consume it. */
enum class Precision : unsigned char {
	/** Rounded to nearest even once, where it is produced; every consumer reads the same bf16. */
	Bf16,
	Fp32,
};

/**
 * Every value one operation of a decoder layer, the head or the mtp head hands to another, per verify row. A value is
 * bf16 exactly where it is a tensor-core operand or is kept across positions (the kv cache, the convolution window);
 * everything else, the residual stream and the deltanet state above all, is fp32.
 */
enum class Value : unsigned char {
	/** A row of `embed_tokens`, as stored; it widens exactly into the residual stream. */
	Embedding,
	/** The residual stream, from the embedding (or the mtp fc) through every layer's two additions. */
	Residual,
	/** input_layernorm(residual): the mixer's matmul input. */
	MixerInput,
	/** in_proj_qkv's output: the convolution's input, kept in its window for the next positions. */
	DeltanetProjection,
	/** in_proj_z's output: it only gates. */
	DeltanetGate,
	/** in_proj_a's and in_proj_b's outputs: the decay's and the write strength's inputs. */
	DeltanetDecayInput,
	/** `deltanet::convolved`: silu(conv(window)) per channel, q | k | v before the l2 norms. */
	DeltanetConvolved,
	/** The l2-normalized key, and the l2-normalized query divided by `deltanet::query_divisor`. */
	DeltanetKeys,
	/** `deltanet::decay` and `deltanet::write_strength`. */
	DeltanetGates,
	/** The recurrent state, a key_dim x value_dim matrix per value head. */
	DeltanetState,
	/** The state read out by the query. */
	DeltanetCore,
	/** `deltanet::normed`: the gated norm before its gate, which the core phase hands the output phase. */
	DeltanetNormed,
	/** `deltanet::gated`: out_proj's input. */
	DeltanetGated,
	/** A matmul output summed across the cards before it is added to the residual. */
	Partial,
	/** q after q_norm and rope: the scores' mma operand. */
	Query,
	/** k after k_norm and rope, and v: kept in the kv cache. */
	KeyValue,
	/** The gate half of q_proj: it only gates. */
	AttentionGate,
	/** `attention::score`. */
	Score,
	/** `attention::weight`, split exactly into three bf16 parts by `split_weight` for the PV mmas. */
	AttentionWeight,
	/** A group's PV sums, maximum and row sum (the decode core's records), and their merge over groups. */
	AttentionGroup,
	/** `attention::output`: o_proj's input. */
	AttentionGated,
	/** post_attention_layernorm(residual): gate_proj's and up_proj's input. */
	MlpInput,
	/** gate_proj's and up_proj's outputs. */
	MlpGateUp,
	/** `mlp_product`: down_proj's input. */
	MlpProduct,
	/**
	 * The final norm's output: the lm_head's input, and the mtp head's hidden input at the same position (it is what a
	 * serving stack hands its drafter, so the drafter was trained on it).
	 */
	FinalNormed,
	/** Logits of the lm_head and of the draft lm_head. */
	Logit,
	/** pre_fc_norm_embedding(embedding) | pre_fc_norm_hidden(hidden): the mtp fc's input. */
	MtpInput,
	/** mtp.norm(mtp residual): the draft lm_head's input, and the next draft step's hidden input. */
	MtpNormed,
};

[[nodiscard]] constexpr auto precision(Value value) -> Precision {
	switch (value) {
	case Value::Embedding:
	case Value::MixerInput:
	case Value::DeltanetProjection:
	case Value::DeltanetGated:
	case Value::Query:
	case Value::KeyValue:
	case Value::AttentionGated:
	case Value::MlpInput:
	case Value::MlpProduct:
	case Value::FinalNormed:
	case Value::MtpInput:
	case Value::MtpNormed:
		return Precision::Bf16;
	case Value::Residual:
	case Value::DeltanetGate:
	case Value::DeltanetDecayInput:
	case Value::DeltanetConvolved:
	case Value::DeltanetKeys:
	case Value::DeltanetGates:
	case Value::DeltanetState:
	case Value::DeltanetCore:
	case Value::DeltanetNormed:
	case Value::Partial:
	case Value::AttentionGate:
	case Value::Score:
	case Value::AttentionWeight:
	case Value::AttentionGroup:
	case Value::MlpGateUp:
	case Value::Logit:
		return Precision::Fp32;
	}
	__builtin_unreachable();
}

// ---------------------------------------------------------------------------------------------------------------------
// Sums over lanes.

/**
 * A fixed-order sum of `width` terms on `lanes` lanes of a warp: lane l folds terms [l w / lanes, (l + 1) w / lanes)
 * in ascending order by fma from +0 (fma(x, y, partial) for a product, fma(x, x, partial) for a square), then the
 * partials meet in a butterfly at distance lanes / 2, ..., 2, 1, each lane adding its partner's value. fp32 addition
 * is commutative, so every lane of the butterfly ends with the same bits.
 */
struct Lanes {
	Count lanes;
	Count width;
};

[[nodiscard]] constexpr auto per_lane(Lanes tree) -> Count {
	return tree.width / tree.lanes;
}

/** A tile of the hidden state: the residual epilogue's warp per 128-output tile, four outputs a lane. */
inline constexpr Lanes hidden_tile = {.lanes = 32, .width = 128};

/** One attention head: a row per warp, eight features a lane (a lane holds one 16-byte chunk of the bf16 row). */
inline constexpr Lanes attention_head = {.lanes = 32, .width = head_dim};

// Eight lanes of sixteen rather than a whole warp per vector: the delta rule's chains are shorter, and four state
// columns per warp keep a lane's share of the state at sixteen registers. The deltanet kernels chose it for that speed;
// every 128-term deltanet sum (both l2 norms, both of the recurrence's dot products, the gated norm) takes it.
inline constexpr Lanes deltanet_vector = {.lanes = 8, .width = 128};

static_assert(hidden % hidden_tile.width == 0 && key_dim == deltanet_vector.width && value_dim == deltanet_vector.width);

/** One lane's fold of x_i y_i over terms [first, first + count), ascending, by fma from +0. */
template<class X, class Y>
[[nodiscard]] constexpr auto lane_fold(X const& x, Y const& y, Count first, Count count) -> float {
	float partial = 0.0F;
	for (Count term = first; term < first + count; ++term) {
		partial = fixed::fma(x[term], y[term], partial);
	}
	return partial;
}

/** The butterfly over `lanes` partials: the total every lane ends with. */
template<Count lanes, class P>
[[nodiscard]] constexpr auto butterfly(P const& partials) -> float {
	static_assert(lanes >= 1 && lanes <= 32 && (lanes & (lanes - 1)) == 0, "a butterfly is a power of two within a warp");
	float level[lanes] = {};
	for (Count lane = 0; lane < lanes; ++lane) {
		level[lane] = partials[lane];
	}
	for (Count distance = lanes / 2; distance != 0; distance /= 2) {
		float next[lanes] = {};
		for (Count lane = 0; lane < lanes; ++lane) {
			next[lane] = level[lane] + level[lane ^ distance];
		}
		for (Count lane = 0; lane < lanes; ++lane) {
			level[lane] = next[lane];
		}
	}
	return level[0];
}

/** sum x_i y_i over terms [first, first + tree.width) under `tree`. */
template<Lanes tree, class X, class Y>
[[nodiscard]] constexpr auto lane_sum(X const& x, Y const& y, Count first) -> float {
	static_assert(tree.width % tree.lanes == 0);
	float partials[tree.lanes] = {};
	for (Count lane = 0; lane < tree.lanes; ++lane) {
		partials[lane] = lane_fold(x, y, first + lane * per_lane(tree), per_lane(tree));
	}
	return butterfly<tree.lanes>(partials);
}

/**
 * Every sum of many fp32 terms, and its one order. The lane trees and the functions named carry each order; this list
 * says which sum takes which.
 */
enum class Sum : unsigned char {
	/**
	 * Every product with a weight matrix. A phase's units (one weight tile of each tensor the phase interleaves) are
	 * split over the SMs as `schedule::share` says, each output tile's units consecutive in ascending k. Per SM and
	 * output element: for an e4m3 tile, an mma chain over its 8 k16 steps in ascending k from +0, then
	 * partial = fma(chain, block scale, partial) from +0; for bf16 tiles, one mma chain over all the SM's k16 steps in
	 * ascending k. The output tile's SM partials are added left to right in SM order, then the cards' sums card 0 +
	 * card 1 (for an axis-1 split), then the residual + that sum (the mtp fc's sum starts the mtp residual instead).
	 * in_proj_a and in_proj_b, read whole, are one mma chain over the whole hidden in ascending k.
	 */
	Weights,
	/**
	 * An rms norm over the hidden size: `hidden_squares` (each 128-feature tile under `hidden_tile`, the tile sums added
	 * left to right from tile 0), then `rms_factor` and `one_plus_scaled`.
	 */
	HiddenSquares,
	/** q_norm and k_norm: the head's squares under `attention_head`, then `rms_factor` and `one_plus_scaled`. */
	HeadSquares,
	/** The deltanet l2 norms of q and k: squares under `deltanet_vector`, then `deltanet::l2_factor`. */
	KeySquares,
	/** The deltanet gated norm: squares under `deltanet_vector`, then `rms_factor` and `deltanet::normed`. */
	ValueSquares,
	/**
	 * The deltanet recurrence, per value column j of a head: `deltanet::decayed` for every i; memory = sum_i S_ij k_i
	 * under `deltanet_vector`; `deltanet::delta`; `deltanet::updated` for every i; out_j = sum_i S_ij q_i under
	 * `deltanet_vector`.
	 */
	DeltanetProducts,
	/** The causal convolution: `deltanet::convolved`. */
	Convolution,
	/** q . k: one mma chain over the head's 16 k16 steps in ascending feature order from +0; then `attention::score`. */
	Scores,
	/** A chunk's row sum of its scaled weights: one per lane (position within the chunk), the butterfly alone. */
	ChunkWeights,
	/**
	 * A chunk's PV sums: the group's running output times the chunk's rescale (`attention::rescaled`), then one mma
	 * chain from it over the hi parts of the chunk's two k16 steps of positions in ascending order, then the mid parts',
	 * then the lo parts' (six mmas).
	 */
	ChunkValues,
	/** A group's chunks in order, by the online softmax: `attention::advance` and `attention::running_total`. */
	GroupChunks,
	/** A row's groups in order, the first taken as it is and each later one folded in by `attention::merged`. */
	GroupMerge,
};

/** The sum of squares over a hidden-sized row: `Sum::HiddenSquares`. */
template<class X>
[[nodiscard]] constexpr auto hidden_squares(X const& x) -> float {
	float total = lane_sum<hidden_tile>(x, x, 0);
	for (Count tile = 1; tile < hidden / hidden_tile.width; ++tile) {
		total = total + lane_sum<hidden_tile>(x, x, tile * hidden_tile.width);
	}
	return total;
}

// ---------------------------------------------------------------------------------------------------------------------
// Norms.

/** 1 / sqrt(mean square + eps) from a sum of `width` squares; the mean is a division. */
[[nodiscard]] constexpr auto rms_factor(float squares, Count width) -> float {
	float const mean = squares / static_cast<float>(width);
	return fixed::rsqrt(mean + rms_norm_eps);
}

/** A zero-centred norm's output before its rounding: (x * factor) * (1 + w), 1 + w rounded in fp32. */
[[nodiscard]] constexpr auto one_plus_scaled(float x, float factor, float weight) -> float {
	float const scaled = x * factor;
	float const gain = 1.0F + weight;
	return scaled * gain;
}

static_assert(rms_norm_scale == NormScale::OnePlusWeight && deltanet_norm_scale == NormScale::Weight);

/** silu(gate) * up: `Value::MlpProduct` before its rounding. */
[[nodiscard]] constexpr auto mlp_product(float gate, float up) -> float {
	return fixed::silu(gate) * up;
}

static_assert(mlp_gate == Gate::Silu);

// ---------------------------------------------------------------------------------------------------------------------
// Attention.

namespace attention {

/** Positions per chunk at fixed absolute positions: one per lane of the softmax warp, two per kv block. */
inline constexpr Count chunk_positions = kv_block_tokens / 2;

// Chunks are folded in groups at fixed chunk indices: eight chunks (256 positions) per group below position 32768,
// sixty-four (2048 positions) from there. A group is the unit the decode core assigns to an SM and never splits, so a
// row's arithmetic depends on its query, its kv and its position alone. Short groups near the start keep many SMs busy
// on short contexts; long groups far out keep a long context's records few. The attention kernels chose the mapping;
// it fixes the order every row's softmax is summed in.
inline constexpr Count near_group_chunks = 8;
inline constexpr Count near_chunks = 1024;
inline constexpr Count far_group_chunks = 64;
inline constexpr Count near_groups = near_chunks / near_group_chunks;
inline constexpr Count max_chunks = max_positions / chunk_positions;

static_assert(chunk_positions == 32 && kv_block_tokens == 2 * chunk_positions, "a kv block is two chunks of a warp's lanes");
static_assert(near_chunks % near_group_chunks == 0 && (max_chunks - near_chunks) % far_group_chunks == 0, "groups tile the chunks");

/** The group holding chunk `chunk`. */
[[nodiscard]] constexpr auto group_of(Count chunk) -> Count {
	return chunk < near_chunks ? chunk / near_group_chunks : near_groups + (chunk - near_chunks) / far_group_chunks;
}

/** The first chunk of group `group`. */
[[nodiscard]] constexpr auto group_begin(Count group) -> Count {
	return group < near_groups ? group * near_group_chunks : near_chunks + (group - near_groups) * far_group_chunks;
}

/** The groups a row seeing positions [0, limit] folds: every group holding a position it sees. */
[[nodiscard]] constexpr auto groups_seen(Count limit) -> Count {
	return group_of(limit / chunk_positions) + 1;
}

/** 1 / sqrt(head_dim), exact: the scores' scale is a power of two. */
inline constexpr float score_scale = 0x1p-4F;

static_assert(head_dim == 256, "the score scale is 1 / sqrt(head_dim), exactly");

/**
 * The attention weights enter the PV mmas as P * 2^40 = hi + mid + lo exactly (`split_weight`). Without the scale the
 * lo part of a weight below 2^-110 falls under bf16's subnormal step; from 2^38 up every weight splits exactly. The
 * row sum carries the same scale, so the division cancels it.
 */
inline constexpr float weight_scale = 0x1p40F;

/** The scores' mma chain scaled: `Value::Score`. */
[[nodiscard]] constexpr auto score(float dot) -> float {
	return dot * score_scale;
}

/** The larger of two maxima, neither a nan (a group's running maximum starts at -infinity). */
[[nodiscard]] constexpr auto larger(float left, float right) -> float {
	return right > left ? right : left;
}

/** A chunk's step of a group's online softmax: the new running maximum, and the rescale of what ran before. */
struct Advance {
	float maximum;
	float rescale;
};

/**
 * The step for a chunk whose visible scores' largest is `chunk_maximum`, after a running maximum `running`
 * (-infinity at the group's first chunk, whose rescale is then +0 against a running output and total of +0).
 */
[[nodiscard]] constexpr auto advance(float running, float chunk_maximum) -> Advance {
	float const maximum = larger(running, chunk_maximum);
	return {.maximum = maximum, .rescale = fixed::exp(running - maximum)};
}

/** A visible position's scaled softmax weight under the running maximum: `Value::AttentionWeight`. */
[[nodiscard]] constexpr auto weight(float score_value, float maximum) -> float {
	return fixed::exp(score_value - maximum) * weight_scale;
}

/** The group's running row sum after a chunk whose weights sum (`Sum::ChunkWeights`) to `sum`. */
[[nodiscard]] constexpr auto running_total(float total, float rescale, float sum) -> float {
	return fixed::fma(total, rescale, sum);
}

/** A running output element before the chunk's PV mmas accumulate onto it. */
[[nodiscard]] constexpr auto rescaled(float output, float rescale) -> float {
	return output * rescale;
}

/** Which of two merged groups is rescaled: the one whose maximum is not the larger. */
enum class Side : unsigned char {
	Left,
	Right,
};

/** How two adjacent groups merge: the merged maximum, the side rescaled, and its factor. */
struct Merge {
	float maximum;
	Side scaled;
	float factor;
};

// The group with the larger maximum has factor e^0 = 1 exactly, so it is the fma's addend and the other group's
// product its multiply: one rounding per merged value, where scaling both sides would round three times. On a tie the
// right group is the one multiplied, by exactly 1.
[[nodiscard]] constexpr auto merge(float left_maximum, float right_maximum) -> Merge {
	if (right_maximum > left_maximum) {
		return {.maximum = right_maximum, .scaled = Side::Left, .factor = fixed::exp(left_maximum - right_maximum)};
	}
	return {.maximum = left_maximum, .scaled = Side::Right, .factor = fixed::exp(right_maximum - left_maximum)};
}

/** A merged output element or row sum, the earlier group on the left. */
[[nodiscard]] constexpr auto merged(float left, float right, Merge const& how) -> float {
	return how.scaled == Side::Left ? fixed::fma(left, how.factor, right) : fixed::fma(right, how.factor, left);
}

/** The merged output element over the row sum, gated: `Value::AttentionGated` before its rounding. */
[[nodiscard]] constexpr auto output(float value, float total, float gate) -> float {
	float const mean = value / total;
	return mean * fixed::sigmoid(gate);
}

static_assert(attention_output_gate == Gate::Sigmoid);

/** A scaled attention weight's parts as bf16 bit patterns, largest first; their exact sum is the weight. */
struct WeightParts {
	unsigned short high;
	unsigned short middle;
	unsigned short low;
};

inline constexpr unsigned bf16_bits = 0xffff0000U;

/**
 * The split of a scaled weight in [0, 2^40]: each part is the top 16 bits of the fp32 pattern of what the larger parts
 * leave (a truncation, so each remainder is exact and the last part is exactly bf16).
 */
[[nodiscard]] constexpr auto split_weight(float scaled) -> WeightParts {
	unsigned const high = __builtin_bit_cast(unsigned, scaled) & bf16_bits;
	float const rest = scaled - __builtin_bit_cast(float, high);
	unsigned const middle = __builtin_bit_cast(unsigned, rest) & bf16_bits;
	float const low = rest - __builtin_bit_cast(float, middle);
	return {
	    .high = static_cast<unsigned short>(high >> 16U),
	    .middle = static_cast<unsigned short>(middle >> 16U),
	    .low = static_cast<unsigned short>(__builtin_bit_cast(unsigned, low) >> 16U),
	};
}

}

// ---------------------------------------------------------------------------------------------------------------------
// The gated deltanet.

namespace deltanet {

inline constexpr float l2_norm_eps = 1e-6F;

// The checkpoint's reference code for the recurrence (transformers' `torch_recurrent_gated_delta_rule`, the rule the
// engine runs for decode and prefill alike) divides the normalized query by `query.shape[-1] ** 0.5`, a python float
// that the fp32 tensor's division takes as its fp32 rounding, 8 sqrt(2) to nearest. Its chunked prefill form and the
// fla kernels multiply by 128^-1/2 instead; the two differ in the last bit for some queries, and the engine follows
// the recurrence.
inline constexpr float query_divisor = 0x1.6a09e6p+3F;

static_assert(key_dim == 128, "the query divisor is sqrt(key_dim)");

/** One channel's convolution output for the newest of `conv_kernel` inputs: fma from +0, oldest first, then silu. */
template<class T, class X>
[[nodiscard]] constexpr auto convolved(T const& taps, X const& inputs) -> float {
	float sum = 0.0F;
	for (Count tap = 0; tap < conv_kernel; ++tap) {
		sum = fixed::fma(taps[tap], inputs[tap], sum);
	}
	return fixed::silu(sum);
}

/** 1 / sqrt(sum of squares + eps), the l2 norm's factor. */
[[nodiscard]] constexpr auto l2_factor(float squares) -> float {
	return fixed::rsqrt(squares + l2_norm_eps);
}

/** A normalized key element. */
[[nodiscard]] constexpr auto key(float x, float factor) -> float {
	return x * factor;
}

/** A normalized query element. */
[[nodiscard]] constexpr auto query(float x, float factor) -> float {
	float const normalized = x * factor;
	return normalized / query_divisor;
}

/** e^g, g = -e^A_log softplus(a + dt_bias): how much of the state survives the token. */
[[nodiscard]] constexpr auto decay(float a, float a_log, float dt_bias) -> float {
	float const shifted = a + dt_bias;
	float const rate = fixed::exp(a_log);
	float const step = rate * fixed::softplus(shifted);
	return fixed::exp(-step);
}

/** sigmoid(b): how much of the token's error the state takes in. */
[[nodiscard]] constexpr auto write_strength(float b) -> float {
	return fixed::sigmoid(b);
}

[[nodiscard]] constexpr auto decayed(float state, float decay_value) -> float {
	return state * decay_value;
}

/** A column's error on the token's value, scaled by the write strength. */
[[nodiscard]] constexpr auto delta(float value, float memory, float strength) -> float {
	float const error = value - memory;
	return error * strength;
}

[[nodiscard]] constexpr auto updated(float state, float key_element, float column_delta) -> float {
	return fixed::fma(key_element, column_delta, state);
}

/** The gated norm before its gate: (x * factor) * w, `Value::DeltanetNormed`. */
[[nodiscard]] constexpr auto normed(float core, float factor, float weight) -> float {
	float const scaled = core * factor;
	return scaled * weight;
}

/** out_proj's input before its rounding: normed * silu(z), `Value::DeltanetGated`. */
[[nodiscard]] constexpr auto gated(float normed_value, float z) -> float {
	return normed_value * fixed::silu(z);
}

static_assert(deltanet_output_gate == Gate::Silu);

}

/**
 * The mtp head drafts as it was trained and is served: its entry at position i reads the trunk's `FinalNormed` at
 * position i and the embedding of the token at position i + 1, and its rope and kv use position i. A draft step reads
 * the previous entry's `MtpNormed` and the token it drafted, at the next position. (Checked against vLLM's Qwen3.5
 * MTP drafter: the trunk returns its hidden states after the final norm, the proposer shifts the input ids left by
 * one and copies the target positions unchanged, and the MTP model returns hidden states after `mtp.norm`, which the
 * next step reads.)
 */
inline constexpr Count mtp_token_lead = 1;

// ---------------------------------------------------------------------------------------------------------------------
// Rope.

namespace rope {

/**
 * rope_theta^(-2j / rotary_dim) to the nearest fp32, j = 0 .. rotary_dim / 2 - 1. A rotation's angle is
 * fp32(position) * inverse_frequency[j], rounded to fp32, as the checkpoint's serving code computes it.
 */
inline constexpr float inverse_frequency[rotary_dim / 2] = {
    0x1.000000p+0F,  0x1.356656p-1F,  0x1.75f034p-2F,  0x1.c3f06cp-3F,  0x1.111aeep-3F,  0x1.4a12aep-4F,  0x1.8eec7ep-5F,
    0x1.e222ecp-6F,  0x1.235a72p-6F,  0x1.6020a4p-7F,  0x1.a99428p-8F,  0x1.012cfap-8F,  0x1.36d21ap-9F,  0x1.77a7d8p-10F,
    0x1.c603c4p-11F, 0x1.125c04p-11F, 0x1.4b96bep-12F, 0x1.90c182p-13F, 0x1.e459c6p-14F, 0x1.24b0fep-14F, 0x1.61bea2p-15F,
    0x1.ab8884p-16F, 0x1.025b58p-16F, 0x1.383f88p-17F, 0x1.796182p-18F, 0x1.c8198cp-19F, 0x1.139e96p-19F, 0x1.4d1c98p-20F,
    0x1.9298acp-21F, 0x1.e69338p-22F, 0x1.26091cp-22F, 0x1.635e88p-23F,
};

static_assert(rotary_dim == 64 && rope_theta == 10'000'000.0F, "the table is rope_theta's");

/** The rotated feature pairs of a head: feature j turns with feature j + pairs. */
inline constexpr Count pairs = rotary_dim / 2;

/** cos and sin of one feature pair's angle at one position. */
struct Rotation {
	float cosine;
	float sine;
};

// pi / 2 in three parts: the first two have 33 significant bits, so k times either is exact for every k below 2^20.
inline constexpr double half_pi_high = 0x1.921fb544p+0;
inline constexpr double half_pi_middle = 0x1.0b4611a6p-34;
inline constexpr double half_pi_low = 0x1.3198a2e037073p-69;
inline constexpr double two_over_pi = 0x1.45f306dc9c883p-1;
inline constexpr double round_shifter = 0x1.8p52;

/** sin r for |r| <= pi / 4: its Taylor series to r^17, truncated under 2^-60. */
[[nodiscard]] constexpr auto sine_series(double r) -> double {
	double const z = r * r;
	double p = 1.0 / 355687428096000.0;
	p = p * z - 1.0 / 1307674368000.0;
	p = p * z + 1.0 / 6227020800.0;
	p = p * z - 1.0 / 39916800.0;
	p = p * z + 1.0 / 362880.0;
	p = p * z - 1.0 / 5040.0;
	p = p * z + 1.0 / 120.0;
	p = p * z - 1.0 / 6.0;
	return r + r * (z * p);
}

/** cos r for |r| <= pi / 4: its Taylor series to r^18, truncated under 2^-60. */
[[nodiscard]] constexpr auto cosine_series(double r) -> double {
	double const z = r * r;
	double p = -1.0 / 6402373705728000.0;
	p = p * z + 1.0 / 20922789888000.0;
	p = p * z - 1.0 / 87178291200.0;
	p = p * z + 1.0 / 479001600.0;
	p = p * z - 1.0 / 3628800.0;
	p = p * z + 1.0 / 40320.0;
	p = p * z - 1.0 / 720.0;
	p = p * z + 1.0 / 24.0;
	p = p * z - 0.5;
	return 1.0 + z * p;
}

[[nodiscard]] constexpr auto angle(Count position, Count pair) -> float {
	return static_cast<float>(position) * inverse_frequency[pair];
}

/**
 * cos and sin of the pair's angle at `position`, each the fp32 rounding of a double evaluation with an error near
 * 2^-60: a fixed sequence of ieee double operations, so every machine computes the same bits (libm's cos and sin
 * differ between platforms). The angle, at most max_positions, is reduced by pi / 2 in three parts. This is the only
 * sine and cosine the engine has: attention reads it from a table the device fills by evaluating this very function.
 */
[[nodiscard]] constexpr auto rotation(Count position, Count pair) -> Rotation {
	double const x = static_cast<double>(angle(position, pair));
	double const k = (x * two_over_pi + round_shifter) - round_shifter;
	double const r = ((x - k * half_pi_high) - k * half_pi_middle) - k * half_pi_low;
	double const s = sine_series(r);
	double const c = cosine_series(r);
	switch (static_cast<unsigned long long>(k) % 4U) {
	case 0:
		return {.cosine = static_cast<float>(c), .sine = static_cast<float>(s)};
	case 1:
		return {.cosine = static_cast<float>(-s), .sine = static_cast<float>(c)};
	case 2:
		return {.cosine = static_cast<float>(-c), .sine = static_cast<float>(-s)};
	default:
		return {.cosine = static_cast<float>(s), .sine = static_cast<float>(-c)};
	}
}

static_assert(max_positions <= Count{1} << 20U, "k times the first two parts of pi / 2 stays exact");

// Each rotated feature is one fma: x_j cos - x_(j+32) sin and x_(j+32) cos + x_j sin are each a two-term sum whose
// products are inexact, so one product is fused and the other rounded first. Either choice rounds twice; the feature's
// own cosine term is the fused one.

/** Feature j of a rotated pair: x_j cos - x_(j+32) sin. */
[[nodiscard]] constexpr auto first(float low, float high, Rotation turn) -> float {
	return fixed::fma(low, turn.cosine, -(high * turn.sine));
}

/** Feature j + 32 of a rotated pair: x_(j+32) cos + x_j sin. */
[[nodiscard]] constexpr auto second(float low, float high, Rotation turn) -> float {
	return fixed::fma(high, turn.cosine, low * turn.sine);
}

}

}

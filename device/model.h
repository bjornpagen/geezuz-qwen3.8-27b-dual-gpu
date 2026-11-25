// SPDX-License-Identifier: AGPL-3.0-only
//
// The model description: every dimension of Qwen3.8-27B-FP8, the tensors the engine reads from its checkpoint, how
// each tensor divides across the two cards, the serving geometry, and the per-card byte budgets derived from them.
// It is the single source of these facts. Device code includes it; `engine.model` re-exports it to host code. It uses
// no standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

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

/** Activations, partial-sum exchange buffers, rings and scratch per card. */
inline constexpr Count workspace_bytes = 1'000'000'000;

// What the driver keeps of each card before the engine allocates: its context, the jit-compiled module and the
// resident kernel's local memory. 512 MiB is an assumption, not a measurement; bring-up on the cards measures it and
// replaces this figure.
inline constexpr Count driver_reserve_bytes = Count{512} * 1024 * 1024;

/** The most of the block pool deltanet snapshots may hold; kv uses every block they do not. */
inline constexpr Share snapshot_share = {.numerator = 1, .denominator = 4};

inline constexpr Count hidden = 5120;
inline constexpr Count vocab = 248'320;
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
    {"lm_head.weight", matrix(vocab, hidden), Scope::Model, Format::Bf16, Split::Rows},

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
	shape.extents[part.axis] = held(part);
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
		auto const covers = tensor.split == Split::Host || total == tensor.shape.extents[axis];
		auto const in_rank = axis < tensor.shape.rank;
		auto const blocked = tensor.format != Format::E4m3 || tensor.shape.rank == 2;
		auto const balanced = card_bytes(tensor, 0) == card_bytes(tensor, cards - 1);
		if (!covers || !in_rank || !blocked || !balanced) {
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
 * Each card's memory, outermost first. The block pool is what remains, so it is derived, never chosen. Its blocks hold
 * kv or deltanet snapshots; snapshots may hold at most their share, kv any block they do not hold.
 */
struct CardMemory {
	Count driver_reserve;
	Count weights;
	Count workspace;
	Count slot_state;
	Count pool_blocks;
};

[[nodiscard]] consteval auto plan_card_memory() -> CardMemory {
	auto const fixed = driver_reserve_bytes + weight_bytes_per_card + workspace_bytes + slots * slot_state_bytes;
	return {
	    .driver_reserve = driver_reserve_bytes,
	    .weights = weight_bytes_per_card,
	    .workspace = workspace_bytes,
	    .slot_state = slots * slot_state_bytes,
	    .pool_blocks = (card_memory_bytes - fixed) / kv_block_bytes,
	};
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

static_assert(card_memory.driver_reserve + card_memory.weights + card_memory.workspace + card_memory.slot_state + pool_bytes <=
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

}

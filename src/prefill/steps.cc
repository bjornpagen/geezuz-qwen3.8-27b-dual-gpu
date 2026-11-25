// SPDX-License-Identifier: AGPL-3.0-only
//
// A prompt chunk as one card runs it: the fixed sequence of launches, each a typed kernel with its parameters. Per
// chunk the embedding; per trunk layer its norm, its mixer (the deltanet's projections, decay inputs, core, window
// move, gate and output; or attention's projections, gate, kv append and prefill attention, and output), its mlp, and
// after each row-parallel matmul the exchange's residual update; then the final norm, the mtp head's pass over the
// chunk's entries, and the carried row. Both cards run the same sequence with their own memories; each exchange is
// one epoch both share. A sequence is data, so the tests can run it through an emulated card.
export module engine.prefill:steps;

import std;
import engine.attn;
import engine.core;
import engine.deltanet;
import engine.driver;
import engine.model;
import engine.prefill_device;
import engine.schedule;
import :memories;

namespace engine::prefill {

namespace pd = engine::prefill_device;
namespace attn_device = engine::device::attn;

using model::Count;
using schedule::Op;
using schedule::Part;

/** One launch of kernel `K` with its parameters. */
export template<driver::Kernel K>
struct Launch {
	K::Params params;
};

/** A launch of prefill's own image. */
export using PrefillLaunch =
    std::variant<Launch<pd::Zero>, Launch<pd::Embed>, Launch<pd::Norm>, Launch<pd::Decay>, Launch<pd::ResidualAdd>,
	             Launch<pd::ResidualStart>, Launch<pd::MtpGather>, Launch<pd::MtpNorm>, Launch<pd::Keep>, Launch<pd::DeltanetIn>,
	             Launch<pd::DeltanetZ>, Launch<pd::DeltanetOut>, Launch<pd::AttentionQuery>, Launch<pd::AttentionKey>,
	             Launch<pd::AttentionValue>, Launch<pd::AttentionGate>, Launch<pd::AttentionOut>, Launch<pd::MlpIn>, Launch<pd::MlpOut>,
	             Launch<pd::MtpFc>, Launch<pd::MtpAttentionQuery>, Launch<pd::MtpAttentionKey>, Launch<pd::MtpAttentionValue>,
	             Launch<pd::MtpAttentionGate>, Launch<pd::MtpAttentionOut>, Launch<pd::MtpMlpIn>, Launch<pd::MtpMlpOut>>;

/** A launch of attention's image: the prompt rows' kv append, then the prefill attention. */
export using AttentionLaunch = std::variant<Launch<attn::Append>, Launch<attn::Prefill>>;

/** A launch of the deltanet's image: the prefill core, then the window move. */
export using DeltanetLaunch = std::variant<Launch<deltanet::Prefill>, Launch<deltanet::PrefillWindow>>;

/** One step of a chunk, by the image its kernel is in. */
export using Step = std::variant<PrefillLaunch, AttentionLaunch, DeltanetLaunch>;

/**
 * Whether the slot's final norm at the position before the chunk is still in its carried row: it is after the slot's
 * previous chunk of the same request, and gone after a prefix-cache hit or a preemption.
 */
export enum class Carry : std::uint8_t {
	Absent,
	Present,
};

/** One slot's prompt chunk: `count` tokens at positions [from, from + count). */
export struct ChunkFacts {
	std::uint32_t slot;
	std::uint32_t from;
	std::uint32_t count;
	Carry carry;

	[[nodiscard]] constexpr auto operator==(ChunkFacts const&) const -> bool = default;
};

/**
 * The mtp head's entries a chunk writes: entry e reads the final norm at position e and embeds the token at e + 1, so a
 * chunk can write every entry up to its own last position's, which waits for the next token. The entry before the
 * chunk is written from the carried row; without it, that one entry's kv stays zero (a fresh block's), which only the
 * drafter reads, masked: the cost is draft acceptance at one position, never the output.
 */
export struct MtpEntries {
	std::uint32_t first;
	std::uint32_t count;

	[[nodiscard]] constexpr auto operator==(MtpEntries const&) const -> bool = default;
};

export [[nodiscard]] constexpr auto mtp_entries(ChunkFacts const& chunk) -> MtpEntries {
	core::invariant(chunk.count >= 1 && chunk.count <= pd::max_rows);
	core::invariant(chunk.carry == Carry::Absent || chunk.from >= 1);
	auto const first = chunk.carry == Carry::Present ? chunk.from - 1 : chunk.from;
	auto const end = chunk.from + chunk.count - 1;
	return {.first = first, .count = end - first};
}

/** Where one card's chunk runs: its memories, its prefill workspace, the peer's, and the inputs both cards map. */
export struct CardView {
	model::Card card;
	CardMemories memories;
	Address workspace;
	Address peer_workspace;
	Address embedding;
	/** The chunk's token ids, u32 [max_rows], in mapped host memory. */
	Address tokens;

	[[nodiscard]] constexpr auto operator==(CardView const&) const -> bool = default;
};

/** The rows a launch over `count` chunk rows reads and computes. */
[[nodiscard]] constexpr auto rows_of(std::uint32_t count) -> pd::Rows {
	return {.count = count, .computed = core::narrow<unsigned>(pd::padded(count))};
}

/** The full-attention layers before trunk layer `layer`: its kv layer. */
export [[nodiscard]] constexpr auto kv_layer(Count layer) -> unsigned {
	auto ordinal = 0U;
	for (auto index = Count{0}; index < layer; ++index) {
		ordinal += model::trunk_mixers[index] == model::Mixer::FullAttention ? 1U : 0U;
	}
	return ordinal;
}

/** The mtp head's kv layer: after every trunk layer's. */
/** The gemm's row block, as the launches count row blocks. */
inline constexpr unsigned block_rows = static_cast<unsigned>(pd::block_rows);

export inline constexpr unsigned mtp_kv_layer = static_cast<unsigned>(model::kv_layers - 1);

static_assert(model::trunk_layers_of(model::Mixer::FullAttention) == mtp_kv_layer);

/** The attention kernels' rows and the prompt positions they sit at. */
struct Prompt {
	std::uint32_t start;
	pd::Rows rows;
};

/** The weight matmul launches of one mixer kind in one stream. */
template<class Query, class Key, class Value, class Gate, class Out, class In, class Down>
struct AttentionKernels {
	using QueryKernel = Query;
	using KeyKernel = Key;
	using ValueKernel = Value;
	using GateKernel = Gate;
	using OutKernel = Out;
	using MlpInKernel = In;
	using MlpOutKernel = Down;
};

using TrunkKernels =
    AttentionKernels<pd::AttentionQuery, pd::AttentionKey, pd::AttentionValue, pd::AttentionGate, pd::AttentionOut, pd::MlpIn, pd::MlpOut>;
using MtpKernels = AttentionKernels<pd::MtpAttentionQuery, pd::MtpAttentionKey, pd::MtpAttentionValue, pd::MtpAttentionGate,
                                    pd::MtpAttentionOut, pd::MtpMlpIn, pd::MtpMlpOut>;

/**
 * Builds one card's sequence for one chunk, part by part: `begin`, each trunk layer in order, `end`. Every exchange
 * takes the next epoch; both cards' builders, started at the same epoch, hand out the same epochs.
 */
export class Sequence {
public:
	Sequence(ChunkFacts chunk, CardView view, std::uint64_t epoch) : chunk_{chunk}, view_{view}, epoch_{epoch} {
		core::invariant(chunk.slot < model::slots && chunk.count >= 1 && chunk.count <= pd::max_rows);
		core::invariant(Count{chunk.from} + chunk.count <= model::max_positions);
		core::invariant(epoch >= 1);
	}

	/** The slot's deltanet state starts empty at a prompt's first position, then the chunk's embeddings. */
	auto begin() -> void {
		// The pool has no reset command and a slot's image holds its previous request's state, so a prompt's first
		// chunk clears it; every later chunk resumes what the slot's previous chunk or the cache left.
		if (chunk_.from == 0) {
			add<pd::Zero>(
			    {.target = view_.memories.states + Count{chunk_.slot} * model::slot_state_bytes, .chunks = model::slot_state_bytes / 16});
		}
		add<pd::Embed>({.embedding = view_.embedding, .tokens = view_.tokens, .workspace = view_.workspace, .rows = rows()});
	}

	/** Trunk layer `layer`: its mixer and its mlp. */
	auto layer(Count layer) -> void {
		core::invariant(layer < model::trunk_layers);
		auto const& phases = pd::layers.trunk[layer];
		switch (phases.mixer) {
		case model::Mixer::LinearAttention:
			deltanet_mixer(layer, phases);
			break;
		case model::Mixer::FullAttention:
			attention_mixer<TrunkKernels>(phases, kv_layer(layer), Prompt{.start = chunk_.from, .rows = rows()});
			break;
		}
		mlp<TrunkKernels>(phases, rows());
	}

	/** The final norm of every row, the mtp head's pass over the chunk's entries, and the carried row. */
	auto end() -> void {
		norm(side(pd::head_phase, pd::side_index(Op::Head, schedule::tensors.final_norm, Part::Whole)), pd::workspace.final_normed, rows());
		auto const entries = mtp_entries(chunk_);
		if (entries.count > 0) {
			mtp(entries);
		}
		add<pd::Keep>({.workspace = view_.workspace, .prompt = view_.memories.prompt, .slot = chunk_.slot, .row = chunk_.count - 1});
	}

	/** The next exchange's epoch: where the next chunk's sequence starts. */
	[[nodiscard]] auto epoch() const -> std::uint64_t {
		return epoch_;
	}

	[[nodiscard]] auto steps() && -> std::vector<Step> {
		return std::move(steps_);
	}

private:
	template<class K>
	auto add(K::Params params) -> void {
		steps_.emplace_back(PrefillLaunch{Launch<K>{.params = params}});
	}

	[[nodiscard]] auto rows() const -> pd::Rows {
		return rows_of(chunk_.count);
	}

	[[nodiscard]] auto side(Count phase, Count index) const -> Address {
		return view_.memories.image + pd::image_table.phase[phase].side[index];
	}

	/** Ring placement `ring` of `phase`: its tiles and its tensors' scales (one tensor names its scales twice). */
	[[nodiscard]] auto weights(Count phase, Count ring, Count tensors) const -> pd::Weights {
		auto const& entry = pd::image_table.phase[phase];
		auto const image = view_.memories.image;
		return {.tiles = image + entry.ring[ring], .scales = {image + entry.scales[ring][0], image + entry.scales[ring][tensors - 1]}};
	}

	[[nodiscard]] auto gemm(Count phase, Count ring, Count tensors, pd::Rows rows) const -> pd::GemmParams {
		return {
		    .weights = weights(phase, ring, tensors), .workspace = view_.workspace, .row_blocks = rows.computed / block_rows, .unused = 0};
	}

	auto norm(Address weight, Count out, pd::Rows rows) -> void {
		add<pd::Norm>({.residual = view_.workspace + pd::workspace.residual,
		               .squares = view_.workspace + pd::workspace.squares,
		               .weight = weight,
		               .out = view_.workspace + out,
		               .rows = rows});
	}

	/** A row-parallel matmul pushing its partials, then the residual meeting both cards' sums. */
	template<class Push, class Meet>
	auto exchange(Count phase, pd::Rows rows) -> void {
		auto const epoch = epoch_++;
		add<Push>({.weights = weights(phase, 0, 1),
		           .workspace = view_.workspace,
		           .peer_workspace = view_.peer_workspace,
		           .epoch = epoch,
		           .row_blocks = rows.computed / block_rows,
		           .unused = 0});
		add<Meet>({.workspace = view_.workspace, .epoch = epoch, .rows = rows});
	}

	auto deltanet_mixer(Count layer, pd::LayerPhases const& phases) -> void {
		auto const& t = schedule::tensors;
		norm(side(phases.mixer_in, pd::side_index(Op::DeltanetIn, t.input_norm, Part::Whole)), pd::workspace.normed, rows());
		add<pd::DeltanetIn>(gemm(phases.mixer_in, 0, 1, rows()));
		add<pd::Decay>({.a = side(phases.mixer_in, pd::side_index(Op::DeltanetIn, t.a, Part::Whole)),
		                .b = side(phases.mixer_in, pd::side_index(Op::DeltanetIn, t.b, Part::Whole)),
		                .workspace = view_.workspace,
		                .rows = rows()});
		auto const core = [&](Count index) {
			return deltanet::address<deltanet::Bf16 const>(side(phases.deltanet_core, index));
		};
		auto params = deltanet::PrefillParams{
		    .images = deltanet::address<unsigned char>(view_.memories.states),
		    .weights = {.conv = core(pd::side_index(Op::DeltanetCore, t.conv, Part::Whole)),
			            .a_log = core(pd::side_index(Op::DeltanetCore, t.a_log, Part::Whole)),
			            .dt_bias = core(pd::side_index(Op::DeltanetCore, t.dt_bias, Part::Whole)),
			            .norm = core(pd::side_index(Op::DeltanetCore, t.deltanet_norm, Part::Whole))},
		    .rows = {.qkv = deltanet::address<deltanet::Bf16 const>(view_.workspace + pd::workspace.qkv),
			         .gates = deltanet::address<float const>(view_.workspace + pd::workspace.gates),
			         .outputs = deltanet::address<float>(view_.workspace + pd::workspace.core)},
		    .layer = static_cast<unsigned>(deltanet::deltanet_layer(layer)),
		    .live = 1,
		    .slots = {},
		};
		params.slots[0] = {.slot = chunk_.slot, .tokens = chunk_.count, .first_row = 0};
		steps_.emplace_back(DeltanetLaunch{Launch<deltanet::Prefill>{.params = params}});
		steps_.emplace_back(DeltanetLaunch{Launch<deltanet::PrefillWindow>{.params = params}});
		add<pd::DeltanetZ>(gemm(phases.deltanet_z, 0, 1, rows()));
		exchange<pd::DeltanetOut, pd::ResidualAdd>(phases.mixer_out, rows());
	}

	template<class Kernels>
	auto attention_mixer(pd::LayerPhases const& phases, unsigned kv, Prompt prompt) -> void {
		auto const& t = schedule::tensors;
		norm(side(phases.mixer_in, pd::side_index(Op::AttentionIn, t.input_norm, Part::Whole)), pd::workspace.normed, prompt.rows);
		add<typename Kernels::QueryKernel>(gemm(phases.mixer_in, 0, 1, prompt.rows));
		add<typename Kernels::KeyKernel>(gemm(phases.mixer_in, 1, 1, prompt.rows));
		add<typename Kernels::ValueKernel>(gemm(phases.mixer_in, 2, 1, prompt.rows));
		add<typename Kernels::GateKernel>(gemm(phases.attention_gate, 0, 1, prompt.rows));
		auto const params = attn_device::PromptParams{
		    .slot = chunk_.slot,
		    .start = prompt.start,
		    .count = prompt.rows.count,
		    .unused = 0,
		    .kv = {.blocks = view_.memories.pool, .tables = view_.memories.tables, .layer = kv, .unused = 0},
		    .rows = {.query = view_.workspace + pd::workspace.query,
			         .key = view_.workspace + pd::workspace.key,
			         .value = view_.workspace + pd::workspace.value,
			         .gate = view_.workspace + pd::workspace.attention_gate},
		    .preparation = {.query = side(phases.mixer_in, pd::side_index(Op::AttentionIn, t.q_norm, Part::Whole)),
			                .key = side(phases.mixer_in, pd::side_index(Op::AttentionIn, t.k_norm, Part::Whole)),
			                .rotations = view_.memories.rotations},
		    .output = view_.workspace + pd::workspace.attention_output,
		};
		// The append writes every row's kv before the attention reads any, since a row reads the rows before it.
		steps_.emplace_back(AttentionLaunch{Launch<attn::Append>{.params = params}});
		steps_.emplace_back(AttentionLaunch{Launch<attn::Prefill>{.params = params}});
		exchange<typename Kernels::OutKernel, pd::ResidualAdd>(phases.mixer_out, prompt.rows);
	}

	template<class Kernels>
	auto mlp(pd::LayerPhases const& phases, pd::Rows rows) -> void {
		norm(side(phases.mlp_in, pd::side_index(Op::MlpIn, schedule::tensors.post_norm, Part::Whole)), pd::workspace.normed, rows);
		add<typename Kernels::MlpInKernel>(gemm(phases.mlp_in, 0, 2, rows));
		exchange<typename Kernels::MlpOutKernel, pd::ResidualAdd>(phases.mlp_out, rows);
	}

	/** The mtp head over the chunk's entries: its inputs and fc into a fresh residual, then its decoder layer. */
	auto mtp(MtpEntries entries) -> void {
		auto const& t = schedule::tensors;
		auto const rows = rows_of(entries.count);
		auto const params = pd::MtpParams{
		    .embedding = view_.embedding,
		    .tokens = view_.tokens,
		    .workspace = view_.workspace,
		    .embedding_norm = side(pd::layers.mtp_in, pd::side_index(Op::MtpIn, t.pre_fc_norm_embedding, Part::Whole)),
		    .hidden_norm = side(pd::layers.mtp_in, pd::side_index(Op::MtpIn, t.pre_fc_norm_hidden, Part::Whole)),
		    .slot = chunk_.slot,
		    .from = chunk_.from,
		    .first = entries.first,
		    .card = static_cast<unsigned>(std::to_underlying(view_.card)),
		    .rows = rows,
		};
		add<pd::MtpGather>(params);
		add<pd::MtpNorm>(params);
		exchange<pd::MtpFc, pd::ResidualStart>(pd::layers.mtp_in, rows);
		attention_mixer<MtpKernels>(pd::layers.mtp, mtp_kv_layer, Prompt{.start = entries.first, .rows = rows});
		mlp<MtpKernels>(pd::layers.mtp, rows);
	}

	ChunkFacts chunk_;
	CardView view_;
	std::uint64_t epoch_;
	std::vector<Step> steps_;
};

static_assert(model::mtp_mixer == model::Mixer::FullAttention, "the mtp head's mixer is attention");

/** One card's whole sequence for a chunk, and the epoch the next chunk starts at. */
export struct Sequenced {
	std::vector<Step> steps;
	std::uint64_t next_epoch;
};

export [[nodiscard]] auto sequence(ChunkFacts const& chunk, CardView const& view, std::uint64_t epoch) -> Sequenced {
	auto built = Sequence{chunk, view, epoch};
	built.begin();
	for (auto layer = Count{0}; layer < model::trunk_layers; ++layer) {
		built.layer(layer);
	}
	built.end();
	auto const next = built.epoch();
	return {.steps = std::move(built).steps(), .next_epoch = next};
}

}

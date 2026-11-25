// SPDX-License-Identifier: AGPL-3.0-only
//
// The model the reference computes: the checkpoint, the worker threads, and every weight product's plan, resolved
// once at open so the forward pass never looks a name or a phase up.
export module engine.ref:model;

import std;
import engine.core;
import engine.model;
import engine.schedule;
import engine.thread_pool;
import :checkpoint;
import :numeric;
import :plan;

namespace engine::ref {

namespace model = engine::model;
namespace schedule = engine::schedule;

using model::Count;
using schedule::Op;
using schedule::Select;

/** The production shape every generic step is instantiated with: the model description's own dimensions. */
export struct Qwen {
	static constexpr Count hidden = model::hidden;
	static constexpr Count key_heads = model::key_heads;
	static constexpr Count value_heads = model::value_heads;
	static constexpr Count key_dim = model::key_dim;
	static constexpr Count value_dim = model::value_dim;
	static constexpr Count conv_kernel = model::conv_kernel;
	static constexpr Count query_heads = model::query_heads;
	static constexpr Count kv_heads = model::kv_heads;
	static constexpr Count head_dim = model::head_dim;
	static constexpr Count rotary_dim = model::rotary_dim;
};

inline constexpr auto all_rows = std::array{Select::All};

/** The embedding's inventory entry: the one tensor kept in host memory. */
inline constexpr Count embedding_tensor = static_cast<Count>(
    std::ranges::find(model::inventory, model::Split::Host, &model::Tensor::split) - std::ranges::begin(model::inventory));

static_assert(embedding_tensor < schedule::tensor_count && model::inventory[embedding_tensor].shape.extents[0] == model::vocab);

/** A bf16 tensor of the checkpoint, copied out once: norms, the convolution and the deltanet's per-head scalars. */
[[nodiscard]] auto bf16_values(Checkpoint const& checkpoint, Count tensor, schedule::Instance at) -> std::vector<Bf16> {
	auto const bytes = checkpoint.data(tensor, at);
	auto values = std::vector<Bf16>(bytes.size() / 2);
	for (auto index = Count{0}; index < values.size(); ++index) {
		values[index] = bf16_at(bytes, index);
	}
	return values;
}

export struct DeltanetPlans {
	Product qkv;
	Product z;
	Product out;
	std::vector<Bf16> conv;
	std::vector<Bf16> a_log;
	std::vector<Bf16> dt_bias;
	std::vector<Bf16> norm;
};

export struct AttentionPlans {
	Product query;
	Product key;
	Product value;
	Product gate;
	Product out;
	std::vector<Bf16> query_norm;
	std::vector<Bf16> key_norm;
};

/** A decoder layer's mlp, and the two norms every decoder layer has. */
export struct MlpPlans {
	Product gate;
	Product up;
	Product down;
	std::vector<Bf16> input_norm;
	std::vector<Bf16> post_norm;
};

/** One trunk layer: its mixer's products and weights, and its mlp. */
export struct LayerPlans {
	std::variant<DeltanetPlans, AttentionPlans> mixer;
	MlpPlans mlp;
};

export struct MtpPlans {
	Product fc;
	std::vector<Bf16> embedding_norm;
	std::vector<Bf16> hidden_norm;
	std::vector<Bf16> norm;
	AttentionPlans attention;
	MlpPlans mlp;
	Product draft_head;
};

/** The products of one decoder layer's phases, as `stream`'s layer `layer` runs them. */
struct Planner {
	Checkpoint const& checkpoint;
	model::Stream stream;
	Count layer;

	[[nodiscard]] auto at() const -> schedule::Instance {
		return {.stream = stream, .layer = layer};
	}

	[[nodiscard]] auto product(Op op, Count tensor, std::span<Select const> selects) const -> Product {
		return plan(op, stream, layer, tensor, selects);
	}

	[[nodiscard]] auto values(Count tensor) const -> std::vector<Bf16> {
		return bf16_values(checkpoint, tensor, at());
	}

	[[nodiscard]] auto deltanet() const -> DeltanetPlans {
		auto const& t = schedule::tensors;
		return {
		    .qkv = product(Op::DeltanetIn, t.qkv, all_rows),
		    .z = product(Op::DeltanetZ, t.z, all_rows),
		    .out = product(Op::DeltanetOut, t.out, all_rows),
		    .conv = values(t.conv),
		    .a_log = values(t.a_log),
		    .dt_bias = values(t.dt_bias),
		    .norm = values(t.deltanet_norm),
		};
	}

	[[nodiscard]] auto attention() const -> AttentionPlans {
		auto const& t = schedule::tensors;
		return {
		    .query = product(Op::AttentionIn, t.q, std::array{Select::Query}),
		    .key = product(Op::AttentionIn, t.k, all_rows),
		    .value = product(Op::AttentionIn, t.v, all_rows),
		    .gate = product(Op::AttentionGate, t.q, std::array{Select::Gate}),
		    .out = product(Op::AttentionOut, t.o, all_rows),
		    .query_norm = values(t.q_norm),
		    .key_norm = values(t.k_norm),
		};
	}

	[[nodiscard]] auto mlp() const -> MlpPlans {
		auto const& t = schedule::tensors;
		return {
		    .gate = product(Op::MlpIn, t.gate, all_rows),
		    .up = product(Op::MlpIn, t.up, all_rows),
		    .down = product(Op::MlpOut, t.down, all_rows),
		    .input_norm = values(t.input_norm),
		    .post_norm = values(t.post_norm),
		};
	}

	[[nodiscard]] auto mixer(model::Mixer kind) const -> std::variant<DeltanetPlans, AttentionPlans> {
		switch (kind) {
		case model::Mixer::LinearAttention:
			return deltanet();
		case model::Mixer::FullAttention:
			return attention();
		}
		std::unreachable();
	}
};

/** The checkpoint, the threads that compute it, and every plan, built once. */
export class Model {
public:
	[[nodiscard]] static auto open(std::filesystem::path const& directory) -> core::Result<Model, CheckpointError> {
		auto const hardware = std::max(std::thread::hardware_concurrency(), 1U);
		auto pool = std::make_unique<thread_pool::Pool>(hardware - 1);
		auto checkpoint = Checkpoint::open(directory, *pool);
		if (!checkpoint) {
			return std::unexpected(checkpoint.error());
		}
		auto const& t = schedule::tensors;
		auto const head_at = schedule::Instance{.stream = model::Stream::Head, .layer = 0};
		auto const mtp_at = schedule::Instance{.stream = model::Stream::Mtp, .layer = 0};
		auto trunk = std::vector<LayerPlans>{};
		for (auto layer = Count{0}; layer < model::trunk_layers; ++layer) {
			auto const planner = Planner{.checkpoint = *checkpoint, .stream = model::Stream::Body, .layer = layer};
			trunk.push_back({.mixer = planner.mixer(model::trunk_mixers[layer]), .mlp = planner.mlp()});
		}
		auto const mtp_layer = Planner{.checkpoint = *checkpoint, .stream = model::Stream::Mtp, .layer = 0};
		static_assert(model::mtp_mixer == model::Mixer::FullAttention && model::mtp_layers == 1, "the mtp head is one attention layer");
		auto mtp = MtpPlans{
		    .fc = plan(Op::MtpIn, model::Stream::Mtp, 0, t.fc, all_rows),
		    .embedding_norm = bf16_values(*checkpoint, t.pre_fc_norm_embedding, mtp_at),
		    .hidden_norm = bf16_values(*checkpoint, t.pre_fc_norm_hidden, mtp_at),
		    .norm = bf16_values(*checkpoint, t.mtp_norm, mtp_at),
		    .attention = mtp_layer.attention(),
		    .mlp = mtp_layer.mlp(),
		    .draft_head = plan(Op::DraftHead, model::Stream::Mtp, 0, t.lm_head, std::array{Select::DraftVocabulary}),
		};
		auto head = plan(Op::Head, model::Stream::Head, 0, t.lm_head, std::array{Select::DraftVocabulary, Select::OtherVocabulary});
		auto final_norm = bf16_values(*checkpoint, t.final_norm, head_at);
		return Model{std::move(*checkpoint), std::move(pool), std::move(trunk), std::move(mtp), std::move(head), std::move(final_norm)};
	}

	[[nodiscard]] auto checkpoint() const -> Checkpoint const& {
		return checkpoint_;
	}

	[[nodiscard]] auto pool() -> thread_pool::Pool& {
		return *pool_;
	}

	[[nodiscard]] auto trunk() const -> std::span<LayerPlans const> {
		return trunk_;
	}

	[[nodiscard]] auto mtp() const -> MtpPlans const& {
		return mtp_;
	}

	[[nodiscard]] auto head() const -> Product const& {
		return head_;
	}

	[[nodiscard]] auto final_norm() const -> std::span<Bf16 const> {
		return final_norm_;
	}

	/** The embedding row of `token`, as stored. */
	[[nodiscard]] auto embedding(std::uint32_t token) const -> std::vector<Bf16> {
		auto const bytes = checkpoint_.data(embedding_tensor, {.stream = model::Stream::Head, .layer = 0});
		auto row = std::vector<Bf16>(model::hidden);
		for (auto index = Count{0}; index < model::hidden; ++index) {
			row[index] = bf16_at(bytes, (Count{token} * model::hidden) + index);
		}
		return row;
	}

private:
	Model(Checkpoint checkpoint, std::unique_ptr<thread_pool::Pool> pool, std::vector<LayerPlans> trunk, MtpPlans mtp, Product head,
	      std::vector<Bf16> final_norm)
	    : checkpoint_{std::move(checkpoint)}, pool_{std::move(pool)}, trunk_{std::move(trunk)}, mtp_{std::move(mtp)}, head_{std::move(head)},
	      final_norm_{std::move(final_norm)} {}

	Checkpoint checkpoint_;
	std::unique_ptr<thread_pool::Pool> pool_;
	std::vector<LayerPlans> trunk_;
	MtpPlans mtp_;
	Product head_;
	std::vector<Bf16> final_norm_;
};

}

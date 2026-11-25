// SPDX-License-Identifier: AGPL-3.0-only
//
// The model's forward pass over a slot's tokens, and the mtp head's catch-up and draft steps. Each row's arithmetic is
// the same whichever rows share the call, as it is on the cards, so a prompt may be fed in any pieces.
export module engine.ref:forward;

import std;
import engine.core;
import engine.model;
import engine.prims;
import engine.sampling;
import engine.schedule;
import engine.vocabulary;
import :accumulation;
import :attention;
import :deltanet;
import :model;
import :numeric;
import :plan;
import :project;
import :sums;

namespace engine::ref {

namespace model = engine::model;

using model::Count;

export using vocabulary::Token;

/** One slot's state: its tokens, every layer's deltanet state or kv, and the mtp head's kv. */
export struct Sequence {
	std::vector<Token> tokens;
	std::vector<DeltanetState<Qwen>> deltanet;
	std::vector<KvCache<Qwen>> kv;
	KvCache<Qwen> mtp_kv;
	/** The final norm's output at the last position: its mtp entry waits for the token after it. */
	std::optional<std::vector<Bf16>> pending;

	/** An empty slot: no tokens, zero state. */
	[[nodiscard]] static auto empty() -> Sequence {
		return {
		    .tokens = {},
		    .deltanet = std::vector<DeltanetState<Qwen>>(model::trunk_layers_of(model::Mixer::LinearAttention)),
		    .kv = std::vector<KvCache<Qwen>>(model::trunk_layers_of(model::Mixer::FullAttention)),
		    .mtp_kv = {},
		    .pending = std::nullopt,
		};
	}
};

/** The residual + a matmul's sum across the cards, element by element (`model::Sum::Weights`' last addition). */
export auto add_into(Rows<float>& residual, Rows<float> const& sum) -> void {
	for (auto const [into, add] : std::views::zip(residual.values, sum.values)) {
		into = into + add;
	}
}

[[nodiscard]] auto normed(Rows<float> const& residual, std::span<Bf16 const> weight) -> Rows<Bf16> {
	auto out = Rows<Bf16>::zeros(residual.count(), model::hidden);
	auto const scale = std::span<Bf16 const, model::hidden>{weight.first(model::hidden)};
	for (auto row = Count{0}; row < residual.count(); ++row) {
		auto const values = hidden_norm<model::hidden>(std::span<float const, model::hidden>{residual.row(row)}, scale);
		std::ranges::copy(values, out.row(row).begin());
	}
	return out;
}

[[nodiscard]] auto rounded(Rows<float> const& values) -> Rows<Bf16> {
	auto out = Rows<Bf16>::zeros(values.count(), values.width);
	std::ranges::transform(values.values, out.values.begin(), round);
	return out;
}

template<Accumulation A>
[[nodiscard]] auto mix(Model& model, DeltanetPlans const& plans, DeltanetState<Qwen>& state, Rows<Bf16> const& in, Count /*first*/)
    -> Rows<float> {
	auto const& checkpoint = model.checkpoint();
	auto const& t = schedule::tensors;
	auto const at = plans.qkv.at;
	auto const projection = rounded(project<A>(checkpoint, model.pool(), plans.qkv, in));
	auto const gate = project<A>(checkpoint, model.pool(), plans.z, in);
	auto const a = whole_product<A>(checkpoint.data(t.a, at), model::value_heads, in);
	auto const b = whole_product<A>(checkpoint.data(t.b, at), model::value_heads, in);
	auto const weights = DeltanetWeights{.conv = plans.conv, .a_log = plans.a_log, .dt_bias = plans.dt_bias, .norm = plans.norm};
	auto gated = Rows<Bf16>::zeros(in.count(), model::value_features);
	for (auto row = Count{0}; row < in.count(); ++row) {
		auto const out = deltanet_step<Qwen>(state, weights,
		                                     {.projection = projection.row(row), .gate = gate.row(row), .a = a.row(row), .b = b.row(row)});
		std::ranges::copy(out, gated.row(row).begin());
	}
	return project<A>(checkpoint, model.pool(), plans.out, gated);
}

template<Accumulation A>
[[nodiscard]] auto mix(Model& model, AttentionPlans const& plans, KvCache<Qwen>& cache, Rows<Bf16> const& in, Count first) -> Rows<float> {
	auto const& checkpoint = model.checkpoint();
	constexpr auto dim = model::head_dim;
	auto const query = project<A>(checkpoint, model.pool(), plans.query, in);
	auto const key = project<A>(checkpoint, model.pool(), plans.key, in);
	auto const value = project<A>(checkpoint, model.pool(), plans.value, in);
	auto const gate = project<A>(checkpoint, model.pool(), plans.gate, in);
	auto const query_norm = std::span<Bf16 const, dim>{std::span{plans.query_norm}.first(dim)};
	auto const key_norm = std::span<Bf16 const, dim>{std::span{plans.key_norm}.first(dim)};
	for (auto row = Count{0}; row < in.count(); ++row) {
		core::invariant(cache.positions() == first + row);
		for (auto head = Count{0}; head < model::kv_heads; ++head) {
			auto const raw = std::span<float const, dim>{key.row(row).subspan(head * dim, dim)};
			auto const positioned_key = positioned<Qwen>(raw, key_norm, first + row);
			cache.keys.insert(cache.keys.end(), positioned_key.begin(), positioned_key.end());
		}
		for (auto const each : value.row(row)) {
			cache.values.push_back(round(each));
		}
	}
	auto attended = Rows<Bf16>::zeros(in.count(), model::query_features);
	model.pool().share(in.count() * model::query_heads, [&](std::size_t index) {
		auto const row = index / model::query_heads;
		auto const head = index % model::query_heads;
		auto const at = head * model::q_proj_rows_per_head;
		auto const raw = std::span<float const, dim>{query.row(row).subspan(at, dim)};
		auto const positioned_query = positioned<Qwen>(raw, query_norm, first + row);
		auto const gates = std::span<float const, dim>{gate.row(row).subspan(at + dim, dim)};
		auto const out = attend<A, Qwen>(cache, positioned_query, head, first + row, gates);
		std::ranges::copy(out, attended.row(row).subspan(head * dim, dim).begin());
	});
	return project<A>(checkpoint, model.pool(), plans.out, attended);
}

/** down_proj's input from gate_proj's and up_proj's outputs: `model::mlp_product`, rounded once (`Value::MlpProduct`). */
export [[nodiscard]] auto mlp_products(Rows<float> const& gate, Rows<float> const& up) -> Rows<Bf16> {
	core::invariant(gate.width == up.width && gate.values.size() == up.values.size());
	auto product = Rows<Bf16>::zeros(gate.count(), gate.width);
	for (auto const [into, g, u] : std::views::zip(product.values, gate.values, up.values)) {
		into = round(model::mlp_product(g, u));
	}
	return product;
}

template<Accumulation A>
auto feed_forward(Model& model, MlpPlans const& plans, Rows<float>& residual) -> void {
	auto const& checkpoint = model.checkpoint();
	auto const in = normed(residual, plans.post_norm);
	auto const gate = project<A>(checkpoint, model.pool(), plans.gate, in);
	auto const up = project<A>(checkpoint, model.pool(), plans.up, in);
	add_into(residual, project<A>(checkpoint, model.pool(), plans.down, mlp_products(gate, up)));
}

/** One decoder layer over `residual`: its mixer, reading and extending `state`, then its mlp. */
template<Accumulation A, class Mixer, class State>
auto layer(Model& model, Mixer const& mixer, MlpPlans const& mlp, State& state, Rows<float>& residual, Count first) -> void {
	auto const in = normed(residual, mlp.input_norm);
	add_into(residual, mix<A>(model, mixer, state, in, first));
	feed_forward<A>(model, mlp, residual);
}

/** The slot's state a trunk layer's mixer reads and extends: its deltanet state or its kv, by the layer's ordinal. */
[[nodiscard]] auto state_of(Sequence& sequence, DeltanetPlans const& /*mixer*/, Count deltanet, Count /*attention*/) -> DeltanetState<Qwen>& {
	return sequence.deltanet.at(deltanet);
}

[[nodiscard]] auto state_of(Sequence& sequence, AttentionPlans const& /*mixer*/, Count /*deltanet*/, Count attention) -> KvCache<Qwen>& {
	return sequence.kv.at(attention);
}

[[nodiscard]] auto embedded(Model const& model, std::span<Token const> tokens) -> Rows<float> {
	auto out = Rows<float>::zeros(tokens.size(), model::hidden);
	for (auto const [row, token] : std::views::enumerate(tokens)) {
		std::ranges::transform(model.embedding(token), out.row(static_cast<Count>(row)).begin(), widen);
	}
	return out;
}

/**
 * The mtp head over entries at consecutive positions from `first`: entry i reads `hidden` row i (the trunk's final
 * norm output, or the previous draft step's) and the embedding of `tokens[i]`. Returns `model::Value::MtpNormed`.
 */
template<Accumulation A>
[[nodiscard]] auto mtp_forward(Model& model, KvCache<Qwen>& cache, Rows<Bf16> const& hidden, std::span<Token const> tokens, Count first)
    -> Rows<Bf16> {
	auto const& plans = model.mtp();
	auto const embeddings = normed(embedded(model, tokens), plans.embedding_norm);
	auto widened = Rows<float>::zeros(hidden.count(), model::hidden);
	std::ranges::transform(hidden.values, widened.values.begin(), widen);
	auto const states = normed(widened, plans.hidden_norm);
	auto input = Rows<Bf16>::zeros(tokens.size(), 2 * model::hidden);
	for (auto row = Count{0}; row < tokens.size(); ++row) {
		std::ranges::copy(embeddings.row(row), input.row(row).begin());
		std::ranges::copy(states.row(row), input.row(row).subspan(model::hidden).begin());
	}
	auto residual = project<A>(model.checkpoint(), model.pool(), plans.fc, input);
	layer<A>(model, plans.attention, plans.mlp, cache, residual, first);
	return normed(residual, plans.norm);
}

/**
 * Feeds `tokens` to the slot at its next positions: the trunk over every one, the logits of each row over the sampled
 * vocabulary (the padding rows are no token's, so they are not logits), and the mtp catch-up over every entry whose
 * next token is now known.
 */
export template<Accumulation A>
[[nodiscard]] auto extend(Model& model, Sequence& sequence, std::span<Token const> tokens) -> Rows<float> {
	core::invariant(!tokens.empty() && std::ranges::all_of(tokens, [](Token token) {
		return token < model::sampled_vocabulary;
	}));
	auto const first = sequence.tokens.size();
	auto residual = embedded(model, tokens);
	auto deltanet = Count{0};
	auto attention = Count{0};
	for (auto const& plans : model.trunk()) {
		std::visit(
		    [&](auto const& mixer) {
			    layer<A>(model, mixer, plans.mlp, state_of(sequence, mixer, deltanet, attention), residual, first);
		    },
		    plans.mixer);
		deltanet += std::holds_alternative<DeltanetPlans>(plans.mixer) ? 1 : 0;
		attention += std::holds_alternative<AttentionPlans>(plans.mixer) ? 1 : 0;
	}
	auto const final = normed(residual, model.final_norm());
	auto const all = project<A>(model.checkpoint(), model.pool(), model.head(), final);
	auto logits = Rows<float>::zeros(tokens.size(), model::sampled_vocabulary);
	for (auto row = Count{0}; row < tokens.size(); ++row) {
		std::ranges::copy(all.row(row).first(model::sampled_vocabulary), logits.row(row).begin());
	}

	auto hidden = Rows<Bf16>::zeros(0, model::hidden);
	auto next = std::vector<Token>{};
	if (sequence.pending) {
		hidden.values = *sequence.pending;
		next.push_back(tokens[0]);
	}
	for (auto row = Count{0}; row + 1 < tokens.size(); ++row) {
		hidden.values.insert(hidden.values.end(), final.row(row).begin(), final.row(row).end());
		next.push_back(tokens[row + 1]);
	}
	if (!next.empty()) {
		// The first entry is the last position's when one waits, else position zero's.
		auto const entry = sequence.pending ? first - 1 : first;
		std::ignore = mtp_forward<A>(model, sequence.mtp_kv, hidden, next, entry);
	}
	auto const last = final.row(tokens.size() - 1);
	sequence.pending = std::vector<Bf16>{last.begin(), last.end()};
	sequence.tokens.insert(sequence.tokens.end(), tokens.begin(), tokens.end());
	return logits;
}

/** The mtp head's best draft token from its draft lm_head rows, ties to the lower id, as the cards' merge ranks them. */
template<Accumulation A>
[[nodiscard]] auto best_draft(Model& model, Rows<Bf16> const& normed_row) -> Token {
	auto const all = project<A>(model.checkpoint(), model.pool(), model.mtp().draft_head, normed_row);
	auto drafts = std::vector<float>(model::draft_vocab);
	for (auto const [index, token] : std::views::enumerate(vocabulary::draft)) {
		drafts[static_cast<Count>(index)] = all.row(0)[token];
	}
	auto const best = sampling::candidates(drafts).front();
	return vocabulary::draft.at(core::narrow<Count>(best.id));
}

/**
 * `depth` greedy drafts after the slot's tokens and `next`, the token the slot will commit next: the catch-up entry
 * for the last position with `next`, then one draft step per further draft. The slot is unchanged.
 */
export template<Accumulation A>
[[nodiscard]] auto draft(Model& model, Sequence const& sequence, Token next, Count depth) -> std::vector<Token> {
	core::invariant(sequence.pending.has_value() && depth >= 1);
	auto cache = sequence.mtp_kv;
	auto hidden = Rows<Bf16>{.width = model::hidden, .values = *sequence.pending};
	auto token = next;
	auto position = sequence.tokens.size() - 1;
	auto drafts = std::vector<Token>{};
	for (auto step = Count{0}; step < depth; ++step) {
		auto const input = std::array{token};
		auto const normed_row = mtp_forward<A>(model, cache, hidden, input, position);
		token = best_draft<A>(model, normed_row);
		drafts.push_back(token);
		hidden = normed_row;
		++position;
	}
	return drafts;
}

}

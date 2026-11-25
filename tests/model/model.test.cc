// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.model;
import engine.testing;

namespace {

namespace model = engine::model;

using model::Count;
using model::Split;
using model::Tensor;

[[nodiscard]] auto named(std::string_view name) -> Tensor const& {
	auto const found = std::ranges::find_if(model::inventory, [name](Tensor const& tensor) {
		return std::string_view{tensor.name} == name;
	});
	return *found;
}

/** The ranges a card holds of a tensor's split axis, in axis order. */
[[nodiscard]] auto ranges_held(Tensor const& tensor, Count card) -> std::vector<model::Range> {
	auto const part = model::shard(tensor, card);
	auto const held = std::span{part.ranges}.first(part.count);
	return {held.begin(), held.end()};
}

/** How many cards hold each index of the split axis, counted range by range. */
[[nodiscard]] auto coverage(Tensor const& tensor) -> std::vector<Count> {
	auto const axis = model::split_axis(tensor.split);
	auto counts = std::vector<Count>(tensor.shape.extents[axis], 0);
	for (auto card = Count{0}; card < model::cards; ++card) {
		for (auto const& range : ranges_held(tensor, card)) {
			for (auto index = range.begin; index < range.begin + range.length; ++index) {
				counts.at(index) += 1;
			}
		}
	}
	return counts;
}

[[nodiscard]] auto expected_holders(Split split) -> Count {
	switch (split) {
	case Split::Host:
		return 0;
	case Split::Replicated:
		return model::cards;
	case Split::Rows:
	case Split::Columns:
	case Split::DeltanetChannels:
	case Split::HiddenPair:
		return 1;
	}
	std::unreachable();
}

[[nodiscard]] auto shards_reassemble_the_whole_tensor() -> bool {
	return std::ranges::all_of(model::inventory, [](Tensor const& tensor) {
		auto const holders = expected_holders(tensor.split);
		return std::ranges::all_of(coverage(tensor), [holders](Count count) {
			return count == holders;
		});
	});
}

[[nodiscard]] auto card_bytes_conserve_the_checkpoint_bytes() -> bool {
	return std::ranges::all_of(model::inventory, [](Tensor const& tensor) {
		auto total = Count{0};
		for (auto card = Count{0}; card < model::cards; ++card) {
			total += model::card_bytes(tensor, card);
		}
		auto const whole = model::stored_bytes(tensor.format, tensor.shape);
		return total == expected_holders(tensor.split) * whole;
	});
}

/** The heads a card holds of `tensor`, given the rows or columns each head spans on the split axis. */
[[nodiscard]] auto heads_held(Tensor const& tensor, Count card, std::size_t section, Count per_head, Count section_offset)
    -> std::pair<Count, Count> {
	auto const range = ranges_held(tensor, card).at(section);
	return {(range.begin - section_offset) / per_head, (range.begin - section_offset + range.length) / per_head};
}

[[nodiscard]] auto attention_heads_find_their_kv_heads_on_the_card() -> bool {
	auto const group = model::query_heads / model::kv_heads;
	for (auto card = Count{0}; card < model::cards; ++card) {
		auto const queries = heads_held(named("self_attn.q_proj.weight"), card, 0, model::q_proj_rows_per_head, 0);
		auto const keys = heads_held(named("self_attn.k_proj.weight"), card, 0, model::head_dim, 0);
		auto const values = heads_held(named("self_attn.v_proj.weight"), card, 0, model::head_dim, 0);
		auto const outputs = heads_held(named("self_attn.o_proj.weight"), card, 0, model::head_dim, 0);
		auto const counted =
		    queries.second - queries.first == model::query_heads_per_card && keys.second - keys.first == model::kv_heads_per_card;
		auto const reads_local = queries.first / group >= keys.first && (queries.second - 1) / group < keys.second;
		if (!counted || !reads_local || keys != values || queries != outputs) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto deltanet_value_heads_find_their_key_heads_on_the_card() -> bool {
	auto const group = model::value_heads / model::key_heads;
	auto const key_width = model::key_heads * model::key_dim;
	auto const& qkv = named("linear_attn.in_proj_qkv.weight");
	auto const& conv = named("linear_attn.conv1d.weight");
	for (auto card = Count{0}; card < model::cards; ++card) {
		auto const queries = heads_held(qkv, card, 0, model::key_dim, 0);
		auto const keys = heads_held(qkv, card, 1, model::key_dim, key_width);
		auto const values = heads_held(qkv, card, 2, model::value_dim, 2 * key_width);
		auto const counted =
		    keys.second - keys.first == model::key_heads_per_card && values.second - values.first == model::value_heads_per_card;
		auto const reads_local = values.first / group >= keys.first && (values.second - 1) / group < keys.second;
		auto const per_value_head = std::array{
		    heads_held(named("linear_attn.in_proj_z.weight"), card, 0, model::value_dim, 0),
		    heads_held(named("linear_attn.in_proj_a.weight"), card, 0, 1, 0),
		    heads_held(named("linear_attn.in_proj_b.weight"), card, 0, 1, 0),
		    heads_held(named("linear_attn.A_log"), card, 0, 1, 0),
		    heads_held(named("linear_attn.dt_bias"), card, 0, 1, 0),
		    heads_held(named("linear_attn.out_proj.weight"), card, 0, model::value_dim, 0),
		};
		auto const same_values = std::ranges::all_of(per_value_head, [values](auto const& heads) {
			return heads == values;
		});
		auto const conv_matches =
		    std::ranges::equal(ranges_held(conv, card), ranges_held(qkv, card), [](model::Range left, model::Range right) {
			    return left.begin == right.begin && left.length == right.length;
		    });
		if (queries != keys || !counted || !reads_local || !same_values || !conv_matches) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto mlp_down_reads_what_gate_and_up_produce() -> bool {
	for (auto card = Count{0}; card < model::cards; ++card) {
		auto const gate = ranges_held(named("mlp.gate_proj.weight"), card).at(0);
		auto const up = ranges_held(named("mlp.up_proj.weight"), card).at(0);
		auto const down = ranges_held(named("mlp.down_proj.weight"), card).at(0);
		auto const same = gate.begin == up.begin && gate.begin == down.begin && gate.length == down.length;
		if (!same || gate.length != model::ffn_per_card) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto mtp_fc_takes_the_same_features_of_both_inputs() -> bool {
	for (auto card = Count{0}; card < model::cards; ++card) {
		auto const part = ranges_held(named("fc.weight"), card);
		if (part.size() != 2) {
			return false;
		}
		auto const embedding = part[0];
		auto const hidden = part[1];
		if (hidden.begin != embedding.begin + model::hidden || hidden.length != embedding.length) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto lm_head_splits_the_vocabulary() -> bool {
	auto const& head = named("lm_head.weight");
	for (auto card = Count{0}; card < model::cards; ++card) {
		auto const rows = ranges_held(head, card).at(0);
		if (rows.begin != card * model::vocab_per_card || rows.length != model::vocab_per_card) {
			return false;
		}
	}
	return ranges_held(named("model.language_model.embed_tokens.weight"), 0).empty();
}

[[nodiscard]] auto inventory_names_are_unique() -> bool {
	auto names = std::vector<std::pair<model::Scope, std::string_view>>{};
	for (auto const& tensor : model::inventory) {
		names.emplace_back(tensor.scope, tensor.name);
	}
	std::ranges::sort(names);
	return std::ranges::adjacent_find(names) == names.end();
}

// A whole snapshot fits in its blocks, and one block fewer would not hold it.
[[nodiscard]] auto snapshot_blocks_are_the_least_that_hold_a_snapshot() -> bool {
	auto const snapshot = model::deltanet_state_bytes_per_slot + model::conv_window_bytes_per_slot;
	return model::snapshot_blocks * model::kv_block_bytes >= snapshot && (model::snapshot_blocks - 1) * model::kv_block_bytes < snapshot;
}

[[nodiscard]] auto kv_bytes_count_every_attention_layer() -> bool {
	auto layers = Count{model::mtp_mixer == model::Mixer::FullAttention ? model::mtp_layers : 0};
	for (auto const mixer : model::trunk_mixers) {
		layers += mixer == model::Mixer::FullAttention ? 1 : 0;
	}
	auto const per_layer = model::kv_heads_per_card * 2 * model::head_dim * sizeof(std::uint16_t);
	return model::kv_bytes_per_token_per_card == layers * per_layer &&
	       model::kv_pool_bytes == model::kv_pool_tokens * model::kv_bytes_per_token_per_card;
}

// The pool takes every whole block the card has left: one block more would not fit.
[[nodiscard]] auto kv_pool_takes_every_block_left() -> bool {
	auto const memory = model::card_memory;
	auto const fixed = memory.driver_reserve + memory.weights + memory.workspace + memory.slot_state;
	auto const blocks = memory.snapshot_pool_blocks + memory.kv_pool_blocks;
	return fixed + blocks * model::kv_block_bytes <= model::card_memory_bytes &&
	       fixed + (blocks + 1) * model::kv_block_bytes > model::card_memory_bytes;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every tensor's card shards cover its split axis once each, or on every card when replicated",
	        .holds = shards_reassemble_the_whole_tensor(),
	    },
	    engine::testing::Case{
	        .name = "the cards' bytes of a tensor add up to the checkpoint's bytes times its holders",
	        .holds = card_bytes_conserve_the_checkpoint_bytes(),
	    },
	    engine::testing::Case{
	        .name = "a card's query heads read only kv heads on the same card, and o_proj reads its query heads",
	        .holds = attention_heads_find_their_kv_heads_on_the_card(),
	    },
	    engine::testing::Case{
	        .name = "a card's deltanet value heads read only key heads on the same card, in every deltanet tensor",
	        .holds = deltanet_value_heads_find_their_key_heads_on_the_card(),
	    },
	    engine::testing::Case{
	        .name = "a card's down_proj columns are the ffn features its gate_proj and up_proj rows produce",
	        .holds = mlp_down_reads_what_gate_and_up_produce(),
	    },
	    engine::testing::Case{
	        .name = "a card's mtp fc columns are the same features of the embedding and of the hidden state",
	        .holds = mtp_fc_takes_the_same_features_of_both_inputs(),
	    },
	    engine::testing::Case{
	        .name = "lm_head splits the vocabulary in halves and the embedding lives on no card",
	        .holds = lm_head_splits_the_vocabulary(),
	    },
	    engine::testing::Case{
	        .name = "no two inventory entries name the same tensor",
	        .holds = inventory_names_are_unique(),
	    },
	    engine::testing::Case{
	        .name = "a snapshot takes the fewest whole kv blocks that hold its state and convolution window",
	        .holds = snapshot_blocks_are_the_least_that_hold_a_snapshot(),
	    },
	    engine::testing::Case{
	        .name = "kv bytes per position count every full-attention layer, the mtp head's included",
	        .holds = kv_bytes_count_every_attention_layer(),
	    },
	    engine::testing::Case{
	        .name = "the kv and snapshot pools take every whole block the card has left after its fixed contents",
	        .holds = kv_pool_takes_every_block_left(),
	    },
	};
	return engine::testing::report(cases);
}

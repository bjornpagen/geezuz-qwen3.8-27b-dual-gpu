// SPDX-License-Identifier: AGPL-3.0-only
//
// Gated full attention for one row, from the projections to o_proj's input: the head norms and rope, the kv append,
// and the chunked softmax with the exact three-way weight split, generic over the head shape.
export module engine.ref:attention;

import std;
import engine.core;
import engine.model;
import engine.prims;
import engine.schedule;
import :accumulation;
import :numeric;
import :sums;

namespace engine::ref {

namespace math = engine::device::prims::math;
namespace model = engine::model;

using model::Count;

export template<class D>
concept AttentionShape = requires {
	{ D::query_heads } -> std::convertible_to<Count>;
	{ D::kv_heads } -> std::convertible_to<Count>;
	{ D::head_dim } -> std::convertible_to<Count>;
	{ D::rotary_dim } -> std::convertible_to<Count>;
} && D::query_heads % D::kv_heads == 0 && D::head_dim % 32 == 0 && D::rotary_dim % 2 == 0 && D::rotary_dim <= D::head_dim;

/** Positions per chunk: one per lane, at fixed absolute positions. */
export inline constexpr Count chunk_positions = schedule::attention_chunk_positions;

static_assert(chunk_positions == lanes, "a chunk's row sum is one butterfly over its positions");

/** One layer's kv for one slot: per position, every kv head's key then every kv head's value, bf16. */
export template<AttentionShape D>
struct KvCache {
	std::vector<Bf16> keys;
	std::vector<Bf16> values;

	[[nodiscard]] auto positions() const -> Count {
		return keys.size() / (D::kv_heads * D::head_dim);
	}

	[[nodiscard]] auto key(Count position, Count head) const -> std::span<Bf16 const, D::head_dim> {
		return std::span<Bf16 const, D::head_dim>{std::span{keys}.subspan(((position * D::kv_heads) + head) * D::head_dim, D::head_dim)};
	}

	[[nodiscard]] auto value(Count position, Count head, Count feature) const -> Bf16 {
		return values[(((position * D::kv_heads) + head) * D::head_dim) + feature];
	}
};

/** Rope on a head's first rotary_dim features at `position`, fp32: each product rounded before the sum. */
export template<AttentionShape D>
auto rotate(std::array<float, D::head_dim>& head, Count position) -> void {
	constexpr auto half = D::rotary_dim / 2;
	for (auto pair = Count{0}; pair < half; ++pair) {
		auto const turn = model::rope::rotation(position, pair);
		auto const low = head.at(pair);
		auto const high = head.at(pair + half);
		head.at(pair) = (low * turn.cosine) - (high * turn.sine);
		head.at(pair + half) = (high * turn.cosine) + (low * turn.sine);
	}
}

/** A head after its norm and rope, rounded: a query's mma operand, or a key as the cache keeps it. */
export template<AttentionShape D>
[[nodiscard]] auto positioned(std::span<float const, D::head_dim> raw, std::span<Bf16 const, D::head_dim> norm, Count position)
    -> std::array<Bf16, D::head_dim> {
	auto head = head_norm<D::head_dim>(raw, norm);
	rotate<D>(head, position);
	auto out = std::array<Bf16, D::head_dim>{};
	std::ranges::transform(head, out.begin(), round);
	return out;
}

/** One chunk's results for one row: its max score, its PV sums and its row sum, both carrying the weight scale. */
template<AttentionShape D>
struct Chunk {
	float most;
	std::array<float, D::head_dim> values;
	float weights;
};

/**
 * The chunk starting at `first` for a row at `position` (seeing positions up to it), head `head` reading kv head
 * `kv_head`: `model::Sum::Scores`, `ChunkValues` and `ChunkWeights`.
 */
template<Accumulation A, AttentionShape D>
[[nodiscard]] auto chunk(KvCache<D> const& cache, std::span<Bf16 const, D::head_dim> query, Count kv_head, Count first, Count position) -> Chunk<D> {
	auto scores = std::array<float, chunk_positions>{};
	auto seen = std::array<bool, chunk_positions>{};
	auto most = -std::numeric_limits<float>::infinity();
	for (auto lane = Count{0}; lane < chunk_positions; ++lane) {
		auto const at = first + lane;
		seen.at(lane) = at <= position;
		if (!seen.at(lane)) {
			continue;
		}
		auto const key = cache.key(at, kv_head);
		auto score = 0.0F;
		for (auto k = Count{0}; k < D::head_dim; k += 16) {
			score = A::step(score, step_of(key, k), step_of(query, k));
		}
		scores.at(lane) = score * model::attention_score_scale;
		most = std::max(most, scores.at(lane));
	}
	auto scaled = std::array<float, chunk_positions>{};
	auto parts = std::array<std::array<Bf16, chunk_positions>, 3>{};
	for (auto lane = Count{0}; lane < chunk_positions; ++lane) {
		scaled.at(lane) = seen.at(lane) ? math::exp(scores.at(lane) - most) * model::attention_weight_scale : 0.0F;
		auto const split = model::split_weight(scaled.at(lane));
		parts[0].at(lane) = {.bits = split.high};
		parts[1].at(lane) = {.bits = split.middle};
		parts[2].at(lane) = {.bits = split.low};
	}
	auto result = Chunk<D>{.most = most, .values = {}, .weights = butterfly(scaled)};
	auto column = std::array<Bf16, chunk_positions>{};
	for (auto feature = Count{0}; feature < D::head_dim; ++feature) {
		for (auto lane = Count{0}; lane < chunk_positions; ++lane) {
			auto const at = first + lane;
			column.at(lane) = at < cache.positions() ? cache.value(at, kv_head, feature) : Bf16{.bits = 0};
		}
		auto sum = 0.0F;
		for (auto k = Count{0}; k < chunk_positions; k += 16) {
			for (auto const& part : parts) {
				sum = A::step(sum, step_of(part, k), step_of(column, k));
			}
		}
		result.values.at(feature) = sum;
	}
	return result;
}

/**
 * One row's attention output for query head `head`, at `position` over the cache: its chunks merged in order
 * (`model::Sum::ChunkMerge`), divided by the row sum, gated by sigmoid(gate) and rounded: o_proj's input for the head.
 */
export template<Accumulation A, AttentionShape D>
[[nodiscard]] auto attend(KvCache<D> const& cache, std::span<Bf16 const, D::head_dim> query, Count head, Count position,
                          std::span<float const, D::head_dim> gate) -> std::array<Bf16, D::head_dim> {
	core::invariant(position < cache.positions());
	auto const kv_head = head / (D::query_heads / D::kv_heads);
	auto chunks = std::vector<Chunk<D>>{};
	for (auto first = Count{0}; first <= position; first += chunk_positions) {
		chunks.push_back(chunk<A, D>(cache, query, kv_head, first, position));
	}
	auto most = -std::numeric_limits<float>::infinity();
	for (auto const& each : chunks) {
		most = std::max(most, each.most);
	}
	auto values = std::array<float, D::head_dim>{};
	auto weights = 0.0F;
	for (auto const& each : chunks) {
		auto const factor = math::exp(each.most - most);
		for (auto feature = Count{0}; feature < D::head_dim; ++feature) {
			values.at(feature) = std::fma(each.values.at(feature), factor, values.at(feature));
		}
		weights = std::fma(each.weights, factor, weights);
	}
	auto out = std::array<Bf16, D::head_dim>{};
	for (auto feature = Count{0}; feature < D::head_dim; ++feature) {
		out.at(feature) = round((values.at(feature) / weights) * math::sigmoid(gate[feature]));
	}
	return out;
}

}

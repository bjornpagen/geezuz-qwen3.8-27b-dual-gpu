// SPDX-License-Identifier: AGPL-3.0-only
//
// Gated full attention for one row, from the projections to o_proj's input: the head norms and rope, the kv append,
// and the softmax over the model's groups of chunks with the exact three-way weight split, generic over the head shape.
export module engine.ref:attention;

import std;
import engine.core;
import engine.model;
import engine.prims;
import :accumulation;
import :numeric;
import :sums;

namespace engine::ref {

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
export inline constexpr Count chunk_positions = model::attention::chunk_positions;

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

/** Rope on a head's first rotary_dim features at `position`, fp32: each rotated feature one fma (`model::rope`). */
export template<AttentionShape D>
auto rotate(std::array<float, D::head_dim>& head, Count position) -> void {
	constexpr auto half = D::rotary_dim / 2;
	for (auto pair = Count{0}; pair < half; ++pair) {
		auto const turn = model::rope::rotation(position, pair);
		auto const low = head.at(pair);
		auto const high = head.at(pair + half);
		head.at(pair) = model::rope::first(low, high, turn);
		head.at(pair + half) = model::rope::second(low, high, turn);
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

/** A group's results for one row: its PV sums and row sum, both carrying the weight scale, and its running maximum. */
template<AttentionShape D>
struct Group {
	std::array<float, D::head_dim> values;
	float maximum;
	float total;
};

/**
 * Group `group` of a row at `position` (seeing positions up to it), head `head` reading kv head `kv_head`: its chunks
 * in order by the online softmax (`model::Sum::GroupChunks`), each chunk's `Scores`, `ChunkWeights` and `ChunkValues`.
 */
template<Accumulation A, AttentionShape D>
[[nodiscard]] auto group_of_row(KvCache<D> const& cache, std::span<Bf16 const, D::head_dim> query, Count kv_head, Count group,
                                Count position) -> Group<D> {
	auto const begin = model::attention::group_begin(group);
	auto const end = std::min(model::attention::group_begin(group + 1), (position / chunk_positions) + 1);
	auto result = Group<D>{.values = {}, .maximum = -std::numeric_limits<float>::infinity(), .total = 0.0F};
	for (auto chunk = begin; chunk < end; ++chunk) {
		auto const first = chunk * chunk_positions;
		auto scores = std::array<float, chunk_positions>{};
		auto most = -std::numeric_limits<float>::infinity();
		for (auto lane = Count{0}; lane < chunk_positions && first + lane <= position; ++lane) {
			auto const key = cache.key(first + lane, kv_head);
			auto dot = 0.0F;
			for (auto k = Count{0}; k < D::head_dim; k += 16) {
				dot = A::step(dot, step_of(key, k), step_of(query, k));
			}
			scores.at(lane) = model::attention::score(dot);
			most = model::attention::larger(most, scores.at(lane));
		}
		auto const opening = chunk == begin;
		auto const step = model::attention::advance(opening ? -std::numeric_limits<float>::infinity() : result.maximum, most);
		auto weights = std::array<float, chunk_positions>{};
		auto parts = std::array<std::array<Bf16, chunk_positions>, 3>{};
		for (auto lane = Count{0}; lane < chunk_positions; ++lane) {
			weights.at(lane) = first + lane <= position ? model::attention::weight(scores.at(lane), step.maximum) : 0.0F;
			auto const split = model::attention::split_weight(weights.at(lane));
			parts[0].at(lane) = {.bits = split.high};
			parts[1].at(lane) = {.bits = split.middle};
			parts[2].at(lane) = {.bits = split.low};
		}
		result.total = model::attention::running_total(opening ? 0.0F : result.total, step.rescale, butterfly(weights));
		result.maximum = step.maximum;
		auto column = std::array<Bf16, chunk_positions>{};
		for (auto feature = Count{0}; feature < D::head_dim; ++feature) {
			for (auto lane = Count{0}; lane < chunk_positions; ++lane) {
				auto const at = first + lane;
				column.at(lane) = at <= position ? cache.value(at, kv_head, feature) : Bf16{.bits = 0};
			}
			auto sum = model::attention::rescaled(result.values.at(feature), step.rescale);
			for (auto const& part : parts) {
				for (auto k = Count{0}; k < chunk_positions; k += 16) {
					sum = A::step(sum, step_of(part, k), step_of(column, k));
				}
			}
			result.values.at(feature) = sum;
		}
	}
	return result;
}

/**
 * One row's attention output for query head `head`, at `position` over the cache: its groups merged in order
 * (`model::Sum::GroupMerge`), divided by the row sum, gated by sigmoid(gate) and rounded: o_proj's input for the head.
 */
export template<Accumulation A, AttentionShape D>
[[nodiscard]] auto attend(KvCache<D> const& cache, std::span<Bf16 const, D::head_dim> query, Count head, Count position,
                          std::span<float const, D::head_dim> gate) -> std::array<Bf16, D::head_dim> {
	core::invariant(position < cache.positions());
	auto const kv_head = head / (D::query_heads / D::kv_heads);
	auto merged = group_of_row<A, D>(cache, query, kv_head, 0, position);
	for (auto group = Count{1}; group < model::attention::groups_seen(position); ++group) {
		auto const next = group_of_row<A, D>(cache, query, kv_head, group, position);
		auto const how = model::attention::merge(merged.maximum, next.maximum);
		for (auto feature = Count{0}; feature < D::head_dim; ++feature) {
			merged.values.at(feature) = model::attention::merged(merged.values.at(feature), next.values.at(feature), how);
		}
		merged.total = model::attention::merged(merged.total, next.total, how);
		merged.maximum = how.maximum;
	}
	auto out = std::array<Bf16, D::head_dim>{};
	for (auto feature = Count{0}; feature < D::head_dim; ++feature) {
		out.at(feature) = round(model::attention::output(merged.values.at(feature), merged.total, gate[feature]));
	}
	return out;
}

}

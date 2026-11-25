// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention emulated on the host one query row at a time, in exactly the kernels' arithmetic: the warp's lanes for the
// norms (lane L holding features 8L to 8L + 7, its fold, then the butterfly), the rotations the device table holds,
// and every tensor-core sum through `engine.mma_model` in the order the kernels issue their mmas. Every rounding point
// and fixed algorithm is the model description's; the emulation only lays them out as the warps do. A kernel run on the
// same inputs must reproduce these bits.
module;

#include "attn/arithmetic.h"
#include "attn/geometry.h"
#include "attn/layout.h"

export module engine.attn:emulation;

import std;
import engine.core;
import engine.mma_model;

namespace engine::attn {

namespace device = engine::device::attn;
namespace prims = engine::device::prims;
namespace model = engine::model;

inline constexpr std::size_t features = device::head_dim;
inline constexpr std::size_t slice = mma_model::instruction_k;

/** A bf16 row of one head: a rotated query or key, or a value. */
export using Row = std::array<prims::Bf16, features>;

/** A group's (or a merge of groups') unnormalized output, scaled by the weight scale, with its maximum and row sum. */
export struct State {
	std::array<float, features> output;
	float maximum;
	float total;
};

/** The kv one slot's row reads, by absolute position, through at least the end of the row's last chunk. */
export struct KvView {
	std::span<Row const> keys;
	std::span<Row const> values;
};

/**
 * q_norm or k_norm then the rope at `position`, as one warp computes it: each lane's squares folded, the butterfly,
 * the norm, and each rotated pair from the table's rotation; rounded once.
 */
export [[nodiscard]] auto prepare_rotated(std::span<float const, features> projection, std::span<prims::Bf16 const, features> weight,
                                          std::uint32_t position) -> Row {
	auto squares = std::array<float, 32>{};
	for (auto lane = std::size_t{0}; lane < 32; ++lane) {
		float chunk[device::lane_features] = {};
		for (auto index = std::size_t{0}; index < device::features_per_chunk; ++index) {
			chunk[index] = projection[lane * device::features_per_chunk + index];
		}
		squares.at(lane) = device::lane_squares(chunk);
	}
	float const inverse = device::inverse_rms(model::butterfly<32>(squares));
	auto normal = std::array<float, features>{};
	for (auto index = std::size_t{0}; index < features; ++index) {
		normal.at(index) = device::normalized(projection[index], inverse, weight[index]);
	}
	auto const half = model::rope::pairs;
	for (auto pair = std::size_t{0}; pair < half; ++pair) {
		auto const turn = model::rope::rotation(position, pair);
		float const low = normal.at(pair);
		float const high = normal.at(pair + half);
		normal.at(pair) = model::rope::first(low, high, turn);
		normal.at(pair + half) = model::rope::second(low, high, turn);
	}
	auto row = Row{};
	std::ranges::transform(normal, row.begin(), prims::round_to_bf16);
	return row;
}

/** A value row: its projection rounded to bf16 once. */
export [[nodiscard]] auto prepare_value(std::span<float const, features> projection) -> Row {
	auto row = Row{};
	std::ranges::transform(projection, row.begin(), prims::round_to_bf16);
	return row;
}

/** One score as the consumer warp's mma chain makes it: the key row is A, the query B, 16 k16 steps in feature order. */
[[nodiscard]] auto score(Row const& key, Row const& query) -> float {
	return model::attention::score(
	    mma_model::chain<features>(0.0F, std::span<prims::Bf16 const, features>{key}, std::span<prims::Bf16 const, features>{query}));
}

/**
 * Group `group` of the row that sees positions [0, limit]: its chunks in order (`model::Sum::GroupChunks`), each
 * raising the running maximum, rescaling the running output and row sum, and adding its weighted values through the
 * bf16 parts of P (`model::Sum::ChunkValues`).
 */
export [[nodiscard]] auto group_state(Row const& query, KvView kv, std::size_t limit, std::size_t group, device::PvParts parts) -> State {
	auto const row_chunks = limit / device::chunk_positions + 1;
	auto const begin = device::group_begin(group);
	auto const end = std::min(device::group_begin(group + 1), row_chunks);
	core::invariant(begin < end && kv.keys.size() >= end * device::chunk_positions && kv.values.size() >= end * device::chunk_positions);
	auto state = State{.output = {}, .maximum = -prims::math::infinity, .total = 0.0F};
	for (auto chunk = begin; chunk < end; ++chunk) {
		auto const start = chunk * device::chunk_positions;
		auto scores = std::array<float, device::chunk_positions>{};
		auto chunk_maximum = -prims::math::infinity;
		for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
			scores.at(position) = score(kv.keys[start + position], query);
			if (start + position <= limit) {
				chunk_maximum = model::attention::larger(chunk_maximum, scores.at(position));
			}
		}
		auto const first = chunk == begin;
		auto const step = model::attention::advance(first ? -prims::math::infinity : state.maximum, chunk_maximum);
		auto weights = std::array<float, device::chunk_positions>{};
		for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
			weights.at(position) = start + position <= limit ? model::attention::weight(scores.at(position), step.maximum) : 0.0F;
		}
		state.total =
		    model::attention::running_total(first ? 0.0F : state.total, step.rescale, model::butterfly<device::chunk_positions>(weights));
		state.maximum = step.maximum;
		auto split = std::array<std::array<prims::Bf16, device::chunk_positions>, 3>{};
		for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
			auto const pieces = device::split(weights.at(position), parts);
			for (auto part = std::size_t{0}; part < 3; ++part) {
				split.at(part).at(position) = pieces.part[part];
			}
		}
		for (auto feature = std::size_t{0}; feature < features; ++feature) {
			auto column = std::array<prims::Bf16, device::chunk_positions>{};
			for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
				column.at(position) = kv.values[start + position].at(feature);
			}
			auto accumulator = model::attention::rescaled(state.output.at(feature), step.rescale);
			for (auto part = std::size_t{0}; part < static_cast<std::size_t>(device::part_count(parts)); ++part) {
				for (auto step_index = std::size_t{0}; step_index < device::chunk_positions / slice; ++step_index) {
					accumulator = mma_model::step(accumulator, std::span{column}.subspan(step_index * slice).first<slice>(),
					                              std::span{split.at(part)}.subspan(step_index * slice).first<slice>());
				}
			}
			state.output.at(feature) = accumulator;
		}
	}
	return state;
}

/** The merge of two adjacent groups, the earlier on the left (`model::Sum::GroupMerge`). */
export [[nodiscard]] auto fold(State const& left, State const& right) -> State {
	auto const how = model::attention::merge(left.maximum, right.maximum);
	auto merged = State{.output = {}, .maximum = how.maximum, .total = model::attention::merged(left.total, right.total, how)};
	for (auto feature = std::size_t{0}; feature < features; ++feature) {
		merged.output.at(feature) = model::attention::merged(left.output.at(feature), right.output.at(feature), how);
	}
	return merged;
}

/** The whole attention of the row that sees positions [0, limit]: its groups merged left to right. */
export [[nodiscard]] auto attend(Row const& query, KvView kv, std::size_t limit, device::PvParts parts) -> State {
	auto state = group_state(query, kv, limit, 0, parts);
	for (auto group = std::size_t{1}; group < model::attention::groups_seen(limit); ++group) {
		state = fold(state, group_state(query, kv, limit, group, parts));
	}
	return state;
}

/** The gated bf16 output of a merged row. */
export [[nodiscard]] auto finish(State const& state, std::span<float const, features> gate) -> Row {
	auto row = Row{};
	for (auto feature = std::size_t{0}; feature < features; ++feature) {
		row.at(feature) = device::finish(state.output.at(feature), state.total, gate[feature]);
	}
	return row;
}

}

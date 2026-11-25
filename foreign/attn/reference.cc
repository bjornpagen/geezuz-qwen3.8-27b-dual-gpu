// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention written straightforwardly for one query row at a time, in exactly the kernels' rounding: the shared
// arithmetic header for everything outside the tensor core, and an `Accumulation` for every tensor-core sum, taken k16
// slice by k16 slice in the order the kernels issue their mmas. With the model of the sm_86 mma as the accumulation it
// is the bit-exact expectation of the kernels; the tests also run it with a test double to prove what holds for any
// accumulation (the tiling, the assignment and the merge change nothing).
module;

#include "attn/arithmetic.h"
#include "attn/geometry.h"
#include "attn/layout.h"

export module engine.attn:reference;

import std;
import engine.core;

namespace engine::attn {

namespace device = engine::device::attn;
namespace prims = engine::device::prims;

inline constexpr std::size_t features = device::head_dim;
inline constexpr std::size_t slice = 16;

/** A bf16 row of one head: a rotated query or key, or a value. */
export using Row = std::array<prims::Bf16, features>;

/**
 * One tensor-core accumulation step: `step(c, a, b)` is c + a_0 b_0 + ... + a_15 b_15 in the rounding of one
 * `mma.sync.m16n8k16` bf16 x bf16 + fp32, `a` the A operand's row and `b` the B operand's column.
 */
export template<class A>
concept Accumulation = requires(float accumulator, std::span<prims::Bf16 const, slice> a, std::span<prims::Bf16 const, slice> b) {
	{ A::step(accumulator, a, b) } -> std::same_as<float>;
};

/** A group's (or a fold of groups') unnormalized output, scaled by the weight scale, with its maximum and total. */
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

/** q_norm or k_norm then the rope at `position`, as the warp computes it (lane L holding features 8L to 8L + 7). */
export [[nodiscard]] auto prepare_rotated(std::span<float const, features> projection, std::span<prims::Bf16 const, features> weight,
                                          std::uint32_t position) -> Row {
	float squares[32] = {};
	for (auto lane = std::size_t{0}; lane < 32; ++lane) {
		float chunk[device::lane_features] = {};
		for (auto index = std::size_t{0}; index < device::features_per_chunk; ++index) {
			chunk[index] = projection[lane * device::features_per_chunk + index];
		}
		squares[lane] = device::lane_squares(chunk);
	}
	float const inverse = device::inverse_rms(device::butterfly_sum(squares));
	auto normal = std::array<float, features>{};
	for (auto index = std::size_t{0}; index < features; ++index) {
		normal.at(index) = device::normalized(projection[index], inverse, weight[index]);
	}
	auto const half = static_cast<std::size_t>(device::rotary_pairs);
	for (auto pair = std::size_t{0}; pair < half; ++pair) {
		auto const turn = device::sincos(device::angle(position, static_cast<int>(pair)));
		float const first = normal.at(pair);
		float const second = normal.at(pair + half);
		normal.at(pair) = device::rotate_first(first, second, turn);
		normal.at(pair + half) = device::rotate_second(first, second, turn);
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

/**
 * Group `group` of the row that sees positions [0, limit]: its chunks in order, each raising the running maximum,
 * rescaling the running output and total, and adding its weighted values through the three bf16 parts of P.
 */
export template<Accumulation A>
[[nodiscard]] auto group_state(Row const& query, KvView kv, std::size_t limit, std::size_t group, device::PvParts parts) -> State {
	auto const row_chunks = limit / device::chunk_positions + 1;
	auto const begin = device::group_begin(group);
	auto const end = std::min(device::group_begin(group + 1), row_chunks);
	core::invariant(begin < end && kv.keys.size() >= end * device::chunk_positions && kv.values.size() >= end * device::chunk_positions);
	auto state = State{.output = {}, .maximum = -device::math::infinity, .total = 0.0F};
	auto const query_slices = std::span{query};
	for (auto chunk = begin; chunk < end; ++chunk) {
		auto const start = chunk * device::chunk_positions;
		float scores[32] = {};
		auto chunk_maximum = -device::math::infinity;
		for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
			auto const key = std::span{kv.keys[start + position]};
			auto dot = 0.0F;
			for (auto step = std::size_t{0}; step < features / slice; ++step) {
				dot = A::step(dot, key.subspan(step * slice).first<slice>(), query_slices.subspan(step * slice).first<slice>());
			}
			scores[position] = device::score(dot);
			if (start + position <= limit) {
				chunk_maximum = device::larger(chunk_maximum, scores[position]);
			}
		}
		auto const first = chunk == begin;
		auto const step = device::advance(first ? -device::math::infinity : state.maximum, chunk_maximum);
		float weights[32] = {};
		for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
			weights[position] = start + position <= limit ? device::weight(scores[position], step.maximum) : 0.0F;
		}
		state.total = device::running_total(first ? 0.0F : state.total, step.rescale, device::butterfly_sum(weights));
		state.maximum = step.maximum;
		auto split = std::array<std::array<prims::Bf16, device::chunk_positions>, 3>{};
		for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
			auto const pieces = device::split(weights[position], parts);
			for (auto part = std::size_t{0}; part < 3; ++part) {
				split.at(part).at(position) = pieces.part[part];
			}
		}
		for (auto feature = std::size_t{0}; feature < features; ++feature) {
			auto column = std::array<prims::Bf16, device::chunk_positions>{};
			for (auto position = std::size_t{0}; position < device::chunk_positions; ++position) {
				column.at(position) = kv.values[start + position].at(feature);
			}
			auto accumulator = state.output.at(feature) * step.rescale;
			for (auto part = std::size_t{0}; part < static_cast<std::size_t>(device::part_count(parts)); ++part) {
				for (auto step_index = std::size_t{0}; step_index < device::chunk_positions / slice; ++step_index) {
					accumulator = A::step(accumulator, std::span{column}.subspan(step_index * slice).first<slice>(),
					                      std::span{split.at(part)}.subspan(step_index * slice).first<slice>());
				}
			}
			state.output.at(feature) = accumulator;
		}
	}
	return state;
}

/** The merge of two adjacent folds, the earlier on the left. */
export [[nodiscard]] auto fold(State const& left, State const& right) -> State {
	auto const weights = device::factors(left.maximum, right.maximum);
	auto merged = State{.output = {}, .maximum = weights.maximum, .total = device::combine(left.total, right.total, weights)};
	for (auto feature = std::size_t{0}; feature < features; ++feature) {
		merged.output.at(feature) = device::combine(left.output.at(feature), right.output.at(feature), weights);
	}
	return merged;
}

/** The groups a row seeing [0, limit] folds. */
export [[nodiscard]] constexpr auto groups_seen(std::size_t limit) -> std::size_t {
	return device::group_of(limit / device::chunk_positions) + 1;
}

/** The whole attention of the row that sees positions [0, limit]: its groups folded left to right. */
export template<Accumulation A>
[[nodiscard]] auto attend(Row const& query, KvView kv, std::size_t limit, device::PvParts parts) -> State {
	auto state = group_state<A>(query, kv, limit, 0, parts);
	for (auto group = std::size_t{1}; group < groups_seen(limit); ++group) {
		state = fold(state, group_state<A>(query, kv, limit, group, parts));
	}
	return state;
}

/** The gated bf16 output of a folded row. */
export [[nodiscard]] auto finish(State const& state, std::span<float const, features> gate) -> Row {
	auto row = Row{};
	for (auto feature = std::size_t{0}; feature < features; ++feature) {
		row.at(feature) = device::finish(state.output.at(feature), state.total, gate[feature]);
	}
	return row;
}

}

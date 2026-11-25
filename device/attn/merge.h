// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode merge as a device building block: one warp per (row, query head) folds the row's group records in group
// order, starting from the first group itself, then applies the sigmoid output gate and writes the bf16 output that
// o_proj reads. Lane L holds features [8L, 8L + 8), so every record and row is read and written a chunk per lane.
#pragma once

#include "attn/address.h"
#include "attn/arithmetic.h"
#include "attn/geometry.h"
#include "attn/layout.h"
#include "prims/memory.h"
#include "prims/thread.h"

namespace engine::device::attn {

/** The calling warp merges `unit` and writes its gated output. */
__attribute__((device, always_inline)) inline void merge(Records records, Address gate, Address output, Cycle const& cycle, MergeUnit unit) {
	auto const once = prims::policy<prims::Eviction::First>();
	auto const reused = prims::policy<prims::Eviction::Last>();
	unsigned const lane = prims::lane();
	SlotWork const work = cycle.work[unit.work];
	Count const head = unit.query_head / group_heads;
	Count const row = Count{unit.token} * group_heads + unit.query_head % group_heads;
	Count const groups_seen = row_groups(work, unit.token);
	auto const* outputs = at<prims::Chunk const>(records.output);
	auto const* running = at<prims::Chunk const>(records.running);
	float value[lane_features] = {};
	float maximum = 0.0F;
	float total = 0.0F;
	Count const first_record = record_row(cycle, unit.work, head, 0, row);
	for (Count group = 0; group < groups_seen; ++group) {
		Count const record = first_record + group * query_rows(work);
		prims::Chunk const pair = prims::load(running + record, once);
		prims::Chunk const low = prims::load(outputs + record * (head_dim / 4) + Count{2} * lane, once);
		prims::Chunk const high = prims::load(outputs + record * (head_dim / 4) + Count{2} * lane + 1U, once);
		float const group_maximum = __builtin_bit_cast(float, pair.word[0]);
		float const group_total = __builtin_bit_cast(float, pair.word[1]);
		float group_value[lane_features] = {};
#pragma unroll
		for (Count word = 0; word < 4; ++word) {
			group_value[word] = __builtin_bit_cast(float, low.word[word]);
			group_value[4 + word] = __builtin_bit_cast(float, high.word[word]);
		}
		// The fold starts from the first group itself, so a row's result never depends on an empty starting state.
		if (group == 0) {
			maximum = group_maximum;
			total = group_total;
#pragma unroll
			for (unsigned index = 0; index < lane_features; ++index) {
				value[index] = group_value[index];
			}
		} else {
			Factors const weights = factors(maximum, group_maximum);
#pragma unroll
			for (unsigned index = 0; index < lane_features; ++index) {
				value[index] = combine(value[index], group_value[index], weights);
			}
			total = combine(total, group_total, weights);
			maximum = weights.maximum;
		}
	}
	Count const token_row = Count{work.first_row} + unit.token;
	auto const* gates = at<prims::Chunk const>(gate) + (token_row * gate_row_floats + Count{unit.query_head} * head_dim) / 4;
	prims::Chunk const gate_low = prims::load(gates + Count{2} * lane, once);
	prims::Chunk const gate_high = prims::load(gates + Count{2} * lane + 1U, once);
	prims::Chunk packed{};
#pragma unroll
	for (Count word = 0; word < 4; ++word) {
		Count const first = 2 * word;
		float const first_gate = __builtin_bit_cast(float, (first < 4 ? gate_low : gate_high).word[first % 4]);
		float const second_gate = __builtin_bit_cast(float, (first + 1 < 4 ? gate_low : gate_high).word[(first + 1) % 4]);
		packed.word[word] = prims::pack(finish(value[first], total, first_gate), finish(value[first + 1], total, second_gate)).bits;
	}
	auto* target = at<prims::Chunk>(output) + (token_row * output_row_values + Count{unit.query_head} * head_dim) / features_per_chunk;
	prims::store(target + lane, packed, reused);
}

}

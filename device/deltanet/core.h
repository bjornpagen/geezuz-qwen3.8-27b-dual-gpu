// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet core's device building blocks: the tree's butterfly on shuffles, lane vectors in shared memory, and the
// roles of one decode item (a live slot's key head group) and one prefill block (a slot's value head over a prompt
// chunk). Each role is a function of its warps; a kernel composes them under its own warp ranges and shared layout,
// the standalone kernels in deltanet.cu as the persistent decode kernel does. The arithmetic is math.h's, so the
// host's emulation of these roles computes the same bits.
#pragma once

#include "deltanet/geometry.h"
#include "deltanet/layout.h"
#include "deltanet/math.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/reduce.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "target.h"

namespace engine::device::deltanet {

using prims::Chunk;
using prims::Eviction;

/** The device pointer a host-handed address carries. */
template<class T>
[[nodiscard]] __attribute__((device)) inline auto at(Address<T> address) -> T* {
	return address.value;
}

/** The chunk `offset` bytes into the allocation at `base`; every offset layout.h gives a chunk is a multiple of 16. */
template<class Byte>
[[nodiscard]] __attribute__((device)) inline auto chunk_at(Byte* base, Count offset) -> Chunk* {
	/* SAFETY: layout.h's offsets of chunk data are 16-byte multiples inside allocations that start 256-byte aligned. */
	return reinterpret_cast<Chunk*>(const_cast<unsigned char*>(reinterpret_cast<unsigned char const*>(base)) + offset);
}

/**
 * Entry `index` of an array in the kernel's parameters, read at constant offsets only: an index the compiler cannot
 * see would copy the whole parameter block into local memory.
 */
template<class T, Count n>
[[nodiscard]] __attribute__((device)) inline auto pick(T const (&entries)[n], unsigned index) -> T {
	T picked = entries[0];
#pragma unroll
	for (Count candidate = 1; candidate < n; ++candidate) {
		if (index == candidate) {
			picked = entries[candidate];
		}
	}
	return picked;
}

/** A runtime fact the host promised; dev builds trap when it is broken, release builds take it as given. */
__attribute__((device)) inline void expect(bool holds) {
	if constexpr (checks == Checks::On) {
		if (!holds) {
			__builtin_trap();
		}
	}
}

/** The tree's total over this lane's group, on every lane of the group: the butterfly `tree_sum` specifies. */
template<class T>
[[nodiscard]] __attribute__((device)) inline auto group_sum(float partial) -> float {
	float total = partial;
#pragma unroll
	for (int distance = T::lanes / 2; distance != 0; distance /= 2) {
		unsigned const received = prims::exchange(__builtin_bit_cast(unsigned, total), static_cast<unsigned>(distance));
		total = combine(total, __builtin_bit_cast(float, received));
	}
	return total;
}

[[nodiscard]] __attribute__((device)) inline auto element(Chunk chunk, int index) -> Bf16 {
	unsigned const word = chunk.word[index / 2];
	return {.bits = static_cast<unsigned short>(index % 2 == 0 ? word & 0xffffU : word >> 16U)};
}

/** Element `index` of a bf16 array that starts 4-byte aligned, read as its 32-bit word under `policy`. */
template<Eviction eviction>
[[nodiscard]] __attribute__((device)) inline auto load_half(Bf16 const* base, int index, prims::Policy<eviction> policy) -> Bf16 {
	/* SAFETY: bf16 side data starts on a 256-byte boundary, so the word holding any of its elements is whole. */
	unsigned const word = prims::load_word(reinterpret_cast<unsigned const*>(base) + index / 2, policy);
	return {.bits = static_cast<unsigned short>(index % 2 == 0 ? word & 0xffffU : word >> 16U)};
}

[[nodiscard]] __attribute__((device)) inline auto pack(Bf16 const (&values)[bf16_per_chunk]) -> Chunk {
	Chunk packed{};
#pragma unroll
	for (int word = 0; word < 4; ++word) {
		packed.word[word] = static_cast<unsigned>(values[word + word].bits) | (static_cast<unsigned>(values[word + word + 1].bits) << 16U);
	}
	return packed;
}

/** Floats [4 chunk, 4 chunk + 4) of `values` as one chunk. */
template<int n>
[[nodiscard]] __attribute__((device)) inline auto pack(float const (&values)[n], int chunk) -> Chunk {
	Chunk packed{};
#pragma unroll
	for (int word = 0; word < 4; ++word) {
		packed.word[word] = __builtin_bit_cast(unsigned, values[chunk * 4 + word]);
	}
	return packed;
}

/** Tree lane `r`'s elements of a vector stored in `vector_element<T>` order in shared memory. */
template<class T>
__attribute__((device)) inline void load_lane(float const* vector, unsigned r, float (&values)[T::per_lane]) {
#pragma unroll
	for (int chunk = 0; chunk < T::chunks; ++chunk) {
		int const offset = (chunk * T::lanes + static_cast<int>(r)) * 4;
		Chunk const loaded = prims::load_shared(prims::shared_address(vector + offset));
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			values[chunk * 4 + word] = __builtin_bit_cast(float, loaded.word[word]);
		}
	}
}

template<class T>
__attribute__((device)) inline void store_lane(float* vector, unsigned r, float const (&values)[T::per_lane]) {
#pragma unroll
	for (int chunk = 0; chunk < T::chunks; ++chunk) {
		int const offset = (chunk * T::lanes + static_cast<int>(r)) * 4;
		prims::store_shared(prims::shared_address(vector + offset), pack(values, chunk));
	}
}

/** Column lane `r`'s 16 consecutive fp32 elements of a 128-element vector in natural order in shared memory. */
__attribute__((device)) inline void load_natural(float const* vector, unsigned r, float (&values)[Column::per_lane]) {
#pragma unroll
	for (int chunk = 0; chunk < Column::chunks; ++chunk) {
		Chunk const loaded = prims::load_shared(prims::shared_address(vector + (Count(Column::per_lane) * r + Count(4) * Count(chunk))));
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			values[chunk * 4 + word] = __builtin_bit_cast(float, loaded.word[word]);
		}
	}
}

/**
 * Normalizes one query or key in place: the group's 8 lanes read the convolved vector in natural order, and write it
 * normalized in `vector_element<Column>` order over it. Every lane reads before the butterfly and writes after it, so
 * no lane overwrites what another has yet to read. Every lane of the warp runs it, `active` or not, because the
 * butterfly needs the whole warp.
 */
__attribute__((device, always_inline)) inline void normalize_vector(float* vector, Role role, bool active, unsigned r) {
	float values[Column::per_lane] = {};
	if (active) {
		load_natural(vector, r, values);
	}
	float const total = group_sum<Column>(partial_squares(values));
	normalize(values, total, role);
	if (active) {
		store_lane<Column>(vector, r, values);
	}
}

/**
 * Normalizes one head's output row for one token and writes it: the group's lanes read the fp32 core outputs
 * `outputs` (128, shared memory), apply the gated norm's scale and weight, and store 16 fp32 elements each to `target`.
 */
__attribute__((device, always_inline)) inline void normalize_output(float const* outputs, Bf16 const* weight, float* target, bool active,
                                                                    unsigned r, prims::Policy<Eviction::Last> reused) {
	float values[Column::per_lane] = {};
	if (active) {
		load_natural(outputs, r, values);
	}
	float const scale = norm_scale(group_sum<Column>(partial_squares(values)));
	if (!active) {
		return;
	}
#pragma unroll
	for (int half = 0; half < 2; ++half) {
		int const first = Column::per_lane * static_cast<int>(r) + bf16_per_chunk * half;
		Chunk const weights = prims::load(chunk_at(weight, Count(first) * sizeof(Bf16)), reused);
		float result[bf16_per_chunk] = {};
#pragma unroll
		for (int index = 0; index < bf16_per_chunk; ++index) {
			result[index] = normalized(values[bf16_per_chunk * half + index], scale, element(weights, index));
		}
#pragma unroll
		for (int chunk = 0; chunk < 2; ++chunk) {
			prims::store(chunk_at(target, (Count(first) + Count(4 * chunk)) * sizeof(float)), pack(result, chunk), reused);
		}
	}
}

/** The four taps of each of a chunk's eight channels, from conv1d's [channel][4] rows. */
struct Taps {
	Four channel[bf16_per_chunk];
};

[[nodiscard]] __attribute__((device)) inline auto load_taps(Bf16 const* conv, int first_channel, prims::Policy<Eviction::Last> reused)
    -> Taps {
	Taps taps{};
#pragma unroll
	for (int part = 0; part < 4; ++part) {
		Chunk const loaded = prims::load(chunk_at(conv, (Count(first_channel) * 4 + Count(part) * bf16_per_chunk) * sizeof(Bf16)), reused);
#pragma unroll
		for (int index = 0; index < bf16_per_chunk; ++index) {
			int const flat = part * bf16_per_chunk + index;
			taps.channel[flat / 4].element[flat % 4] = element(loaded, index);
		}
	}
	return taps;
}

/** The convolution of eight channels over their four newest inputs, oldest first: two chunks of fp32. */
struct Convolved {
	Chunk half[2];
};

[[nodiscard]] __attribute__((device, always_inline)) inline auto convolve_chunk(Taps const& taps, Chunk oldest, Chunk older, Chunk newer,
                                                                                Chunk newest) -> Convolved {
	float out[bf16_per_chunk] = {};
#pragma unroll
	for (int index = 0; index < bf16_per_chunk; ++index) {
		Four const inputs = {.element = {element(oldest, index), element(older, index), element(newer, index), element(newest, index)}};
		out[index] = convolve(taps.channel[index], inputs);
	}
	return {.half = {pack(out, 0), pack(out, 1)}};
}

/** Stores a convolved chunk's eight fp32 at `target` in shared memory. */
__attribute__((device, always_inline)) inline void store_convolved(float* target, Convolved const& values) {
	prims::store_shared(prims::shared_address(target), values.half[0]);
	prims::store_shared(prims::shared_address(target + 4), values.half[1]);
}

/**
 * Where the convolution of a group's channel chunk `u` lands among a token's vectors: chunks of the query, of the
 * key, then of the values, each 8 consecutive fp32.
 */
[[nodiscard]] __attribute__((device)) inline auto convolved_target(float* queries, float* keys, float* values, int u) -> float* {
	constexpr int vector_chunks = dim / bf16_per_chunk;
	if (u < vector_chunks) {
		return queries + Count(bf16_per_chunk) * Count(u);
	}
	if (u < 2 * vector_chunks) {
		return keys + Count(bf16_per_chunk) * Count(u - vector_chunks);
	}
	return values + Count(bf16_per_chunk) * Count(u - 2 * vector_chunks);
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode: the roles of one item, under geometry.h's warps and shared layout.

using StateRing = prims::StageRing<state_stages, warp_threads, consumer_warps>;

/** The ring over a decode block's stages and barriers. */
[[nodiscard]] __attribute__((device)) inline auto state_ring(DecodeShared& shared) -> StateRing {
	return StateRing::at(shared.ring);
}

/** The shared-window address of chunk `chunk` of stage `stage`. */
[[nodiscard]] __attribute__((device)) inline auto stage_chunk(Stage const (&stages)[state_stages], unsigned stage, int chunk)
    -> prims::SharedAddress {
	return prims::shared_address(&stages[stage].bytes[Count(chunk) * 16]);
}

/** Where one item's data lives this cycle. */
struct Item {
	unsigned char* image;
	unsigned char* ring;
	int layer;
	int key_head;
	int replays;
	int rows;
	int first_row;
};

[[nodiscard]] __attribute__((device)) inline auto decode_item(DecodeParams const& params, unsigned slot_index, int key_head) -> Item {
	SlotCycle const work = pick(params.slots, slot_index);
	expect(work.slot < model::slots && work.replays <= static_cast<unsigned>(max_replays) && work.rows >= 1 &&
	       work.rows <= static_cast<unsigned>(max_rows) && params.layer < static_cast<unsigned>(layers));
	return {.image = at(params.images) + Count(work.slot) * slot_image_bytes,
	        .ring = at(params.rings) + Count(work.slot) * ring_bytes_per_slot,
	        .layer = static_cast<int>(params.layer),
	        .key_head = key_head,
	        .replays = static_cast<int>(work.replays),
	        .rows = static_cast<int>(work.rows),
	        .first_row = static_cast<int>(work.first_row)};
}

/** The producer warp: streams the item's state, stage by stage, as fast as the consumers free stages. */
__attribute__((device, always_inline)) inline void produce_state(Item const& item, StateRing ring, Stage (&stages)[state_stages]) {
	unsigned const lane_index = prims::lane();
	auto const once = prims::policy<Eviction::First>();
	Chunk const* source = chunk_at(item.image, state_offset(item.layer, group * item.key_head));
	auto cursor = StateRing::Cursor::start();
	for (int stage = 0; stage < item_stages; ++stage) {
		ring.await_free(cursor, prims::block_backoff);
#pragma unroll 4
		for (int step = 0; step < stage_chunks / warp_threads; ++step) {
			int const chunk = step * warp_threads + static_cast<int>(lane_index);
			prims::copy_async<prims::Prefetch::Chunk256>(stage_chunk(stages, cursor.stage, chunk), source + (stage * stage_chunks + chunk),
			                                             once);
		}
		ring.filled_on_copies(cursor);
		cursor = cursor.next();
	}
}

/**
 * The preparer warps (`preparer_threads` threads, `index` this thread's rank among them, `Barrier` their named
 * barrier): every event's convolution, gates and normalized query and key into `events`; the committed window; the
 * drafts' raw inputs into the ring. Ends with the preparers' arrival on `prepared`.
 */
template<class Barrier>
__attribute__((device, always_inline)) inline void prepare_events(Item const& item, Weights const& weights, Rows const& rows,
                                                                  Events& events, prims::Mbarrier prepared, unsigned index) {
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	int const events_count = item.replays + item.rows;
	Chunk* const window = chunk_at(item.image, window_offset(item.layer, item.key_head));
	Bf16 const* const qkv = at(rows.qkv);
	/* SAFETY: the gate rows are fp32 words as `Rows` lays them out. */
	auto const* const gate_rows = reinterpret_cast<unsigned const*>(at(rows.gates));

	// a for each of the group's heads, then b for each.
	float gate_inputs[entry_gate_chunks * 4] = {};
	int const event_of_gates = static_cast<int>(index) - group_chunks;
	if (static_cast<int>(index) < group_chunks) {
		int const u = static_cast<int>(index);
		int const first_channel = card_channel(item.key_head, u * bf16_per_chunk);
		Taps const taps = load_taps(at(weights.conv), first_channel, reused);
		// The window, then every event's raw input: replays from the ring, rows from the projections.
		Chunk sequence[window_rows + max_events] = {};
#pragma unroll
		for (int row = 0; row < window_rows; ++row) {
			sequence[row] = prims::load(window + (row * group_chunks + u), once);
		}
#pragma unroll
		for (int event = 0; event < max_events; ++event) {
			if (event < item.replays) {
				sequence[window_rows + event] = prims::load(chunk_at(item.ring, ring_offset(item.layer, item.key_head, event)) + u, once);
			} else if (event < events_count) {
				Count const row = Count(item.first_row + event - item.replays);
				sequence[window_rows + event] =
				    prims::load(chunk_at(qkv, row * qkv_row_bytes + Count(first_channel) * sizeof(Bf16)), reused);
			}
		}
#pragma unroll
		for (int event = 0; event < max_events; ++event) {
			if (event < events_count) {
				Convolved const out = convolve_chunk(taps, sequence[event], sequence[event + 1], sequence[event + 2], sequence[event + 3]);
				store_convolved(convolved_target(events.queries[event], events.keys[event], events.values[event], u), out);
			}
			// The committed token is the first row: the window after it is the stored window.
			if (event == item.replays) {
#pragma unroll
				for (int row = 0; row < window_rows; ++row) {
					prims::store(window + (row * group_chunks + u), sequence[event + 1 + row], once);
				}
			}
			// A draft row's raw input waits in the ring for the cycle that learns whether it was accepted.
			if (event > item.replays && event < events_count) {
				prims::store(chunk_at(item.ring, ring_offset(item.layer, item.key_head, event - item.replays - 1)) + u,
				             sequence[window_rows + event], once);
			}
		}
	} else if (event_of_gates >= 0 && event_of_gates < events_count) {
		int const event = event_of_gates;
		if (event < item.replays) {
#pragma unroll
			for (int chunk = 0; chunk < entry_gate_chunks; ++chunk) {
				Chunk const stored =
				    prims::load(chunk_at(item.ring, ring_offset(item.layer, item.key_head, event)) + group_chunks + chunk, once);
#pragma unroll
				for (int word = 0; word < 4; ++word) {
					gate_inputs[chunk * 4 + word] = __builtin_bit_cast(float, stored.word[word]);
				}
			}
		} else {
			Count const row = Count(item.first_row + event - item.replays);
#pragma unroll
			for (int head = 0; head < group; ++head) {
				Count const pair = (row * value_heads + Count(group * item.key_head + head)) * 2;
				gate_inputs[head] = __builtin_bit_cast(float, prims::load_word(gate_rows + pair, reused));
				gate_inputs[group + head] = __builtin_bit_cast(float, prims::load_word(gate_rows + pair + 1, reused));
			}
		}
#pragma unroll
		for (int head = 0; head < group; ++head) {
			int const value_head = group * item.key_head + head;
			events.gates[event][head] =
			    gates(gate_inputs[head], gate_inputs[group + head], load_half(at(weights.a_log), value_head, reused),
				      load_half(at(weights.dt_bias), value_head, reused));
		}
	}
	// Every replay's ring entry has been read before any draft's entry is written over it, and every convolution is in
	// place before the normalizations read it.
	Barrier::sync();
	if (event_of_gates > item.replays && event_of_gates < events_count) {
#pragma unroll
		for (int chunk = 0; chunk < entry_gate_chunks; ++chunk) {
			prims::store(chunk_at(item.ring, ring_offset(item.layer, item.key_head, event_of_gates - item.replays - 1)) + group_chunks +
			                 chunk,
			             pack(gate_inputs, chunk), once);
		}
	}
	// Two vectors per event, one per group of eight lanes.
	unsigned const r = index % 8;
	int const vector_group = static_cast<int>(index / 8);
	constexpr int groups = preparer_threads / 8;
	for (int base = 0; base < 2 * max_events; base += groups) {
		int const vector = base + vector_group;
		int const event = vector / 2;
		bool const active = vector < 2 * events_count;
		Role const role = vector % 2 == 0 ? Role::Query : Role::Key;
		int const safe_event = active ? event : 0;
		normalize_vector(role == Role::Query ? events.queries[safe_event] : events.keys[safe_event], role, active, r);
	}
	prims::sync_warp();
	if (prims::lane() == 0) {
		prepared.arrive();
	}
}

/**
 * Consumer warp `warp` (of `consumer_warps`): quad `warp` of every stage, through every event. The state after the
 * first row is stored as the slot's committed state; every row's output goes to `outputs`.
 */
__attribute__((device, always_inline)) inline void consume_state(Item const& item, StateRing ring, Stage const (&stages)[state_stages],
                                                                 Events const& events, RowOutputs& outputs, int warp) {
	auto const once = prims::policy<Eviction::First>();
	unsigned const lane_index = prims::lane();
	unsigned const r = lane_index % Column::lanes;
	int const column_in_quad = static_cast<int>(lane_index) / Column::lanes;
	int const events_count = item.replays + item.rows;
	auto cursor = StateRing::Cursor::start();
	for (int stage = 0; stage < item_stages; ++stage) {
		int const quad = stage * quads_per_stage + warp;
		int const head = quad / State::units;
		int const unit = quad % State::units;
		int const column = unit * State::unit_columns + column_in_quad;
		float state[Column::per_lane] = {};
		ring.await_filled(cursor, prims::block_backoff);
#pragma unroll
		for (int chunk = 0; chunk < Column::chunks; ++chunk) {
			Chunk const loaded = prims::load_shared(stage_chunk(
			    stages, cursor.stage, warp * (stage_chunks / quads_per_stage) + chunk * warp_threads + static_cast<int>(lane_index)));
#pragma unroll
			for (int word = 0; word < 4; ++word) {
				state[chunk * 4 + word] = __builtin_bit_cast(float, loaded.word[word]);
			}
		}
		prims::sync_warp();
		if (lane_index == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();

		Chunk* const committed = chunk_at(item.image, state_offset(item.layer, group * item.key_head + head));
#pragma unroll 1
		for (int event = 0; event < events_count; ++event) {
			float key[Column::per_lane] = {};
			load_lane<Column>(events.keys[event], r, key);
			Gates const gate = events.gates[event][head];
			float const value = events.values[event][head * dim + column];
			float const projected = group_sum<Column>(decay_and_project(state, key, gate.decay));
			update(state, key, delta(value, projected, gate.beta));
			if (event >= item.replays) {
				float query[Column::per_lane] = {};
				load_lane<Column>(events.queries[event], r, query);
				float const output = group_sum<Column>(read(state, query));
				if (r == 0) {
					outputs.values[event - item.replays][head * dim + column] = output;
				}
			}
			if (event == item.replays) {
#pragma unroll
				for (int chunk = 0; chunk < Column::chunks; ++chunk) {
					prims::store(committed + State::lane_chunk(unit, chunk, static_cast<int>(lane_index)) / 4, pack(state, chunk), once);
				}
			}
		}
	}
}

/** The consumers, after all of them are done: each row's three heads normalized into the output rows. */
__attribute__((device, always_inline)) inline void normalize_rows(Item const& item, Weights const& weights, Rows const& rows,
                                                                  RowOutputs const& outputs, unsigned index) {
	auto const reused = prims::policy<Eviction::Last>();
	unsigned const r = index % 8;
	int const pair = static_cast<int>(index / 8);
	bool const active = pair < item.rows * group;
	int const row = active ? pair / group : 0;
	int const head = pair % group;
	float* const target = at(rows.outputs) + Count(item.first_row + row) * value_features + Count(group * item.key_head + head) * dim;
	normalize_output(&outputs.values[row][Count(head) * dim], at(weights.norm), target, active, r, reused);
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill.

struct Chunked {
	unsigned char* image;
	int layer;
	int value_head;
	int tokens;
	int first_row;
};

[[nodiscard]] __attribute__((device)) inline auto prefill_item(PrefillParams const& params, unsigned slot_index, int value_head)
    -> Chunked {
	SlotChunk const work = pick(params.slots, slot_index);
	expect(work.slot < model::slots && work.tokens >= 1 && params.layer < static_cast<unsigned>(layers));
	return {.image = at(params.images) + Count(work.slot) * slot_image_bytes,
	        .layer = static_cast<int>(params.layer),
	        .value_head = value_head,
	        .tokens = static_cast<int>(work.tokens),
	        .first_row = static_cast<int>(work.first_row)};
}

}

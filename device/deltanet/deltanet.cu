// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet kernels: the decode core over one cycle's live slots, prefill over prompt chunks with its window
// update, the chain probes the microbenchmarks time, and the tree self-check. The decode core here is the same
// composition of core.h's roles the persistent decode kernel makes; standalone, it is what the microbenchmark times
// and what the host checks bit for bit against its emulation on generated inputs.
#include "deltanet/core.h"
#include "deltanet/geometry.h"
#include "deltanet/layout.h"
#include "deltanet/math.h"
#include "prims/budget.h"
#include "prims/order.h"
#include "prims/selfcheck.h"
#include "prims/sync.h"
#include "prims/thread.h"

namespace engine::device::deltanet {

// ---------------------------------------------------------------------------------------------------------------------
// Decode.

using DecodeProducer = prims::Warps<0, producer_warps>;
using DecodeConsumers = prims::Warps<producer_warps, consumer_warps>;
using DecodePreparers = prims::Warps<producer_warps + consumer_warps, preparer_warps>;
using PreparerBarrier = prims::NamedBarrier<1, DecodePreparers::count>;
using ConsumerBarrier = prims::NamedBarrier<2, DecodeConsumers::count>;

static_assert(prims::tiles<DecodeProducer, DecodeConsumers, DecodePreparers>(decode_budget), "the roles tile the block");

// ---------------------------------------------------------------------------------------------------------------------
// Prefill.

using PrefillPreparers = prims::Warps<0, prefill_preparer_warps>;
using PrefillChains = prims::Warps<prefill_preparer_warps, chain_warps>;
using PrefillBarrier = prims::NamedBarrier<1, PrefillPreparers::count>;

static_assert(prims::tiles<PrefillPreparers, PrefillChains>(prefill_budget), "the roles tile the block");

[[nodiscard]] __attribute__((device)) inline auto block_tokens(Chunked const& chunk, int block) -> int {
	int const rest = chunk.tokens - block * prefill_block_tokens;
	return rest < prefill_block_tokens ? rest : prefill_block_tokens;
}

/**
 * The preparer warps of a prefill block: per token block, wait for the chain to release the stage, normalize the
 * outputs it left there, then fill it with the next block's convolution, gates and normalized vectors.
 */
__attribute__((device, always_inline)) inline void prepare_blocks(Chunked const& chunk, Weights const& weights, Rows const& rows,
                                                                  PrefillShared& shared, unsigned index) {
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	int const key_head = chunk.value_head / group;
	int const u = static_cast<int>(index);
	bool const convolves = u < head_chunks;
	// Threads past the convolution's gate one token each; the rest only normalize.
	int const gate_token = u - head_chunks;
	bool const gates_a_token = gate_token >= 0 && gate_token < prefill_block_tokens;
	int const first_channel = card_channel(key_head, head_chunk(chunk.value_head, convolves ? u : 0) * bf16_per_chunk);
	Taps taps{};
	Chunk oldest{};
	Chunk older{};
	Chunk newer{};
	if (convolves) {
		taps = load_taps(at(weights.conv), first_channel, reused);
		constexpr int newest_row = (window_rows - 1) * group_chunks;
		Chunk const* const window = chunk_at(chunk.image, window_offset(chunk.layer, key_head)) + head_chunk(chunk.value_head, u);
		oldest = prims::load(window, once);
		older = prims::load(window + group_chunks, once);
		newer = prims::load(window + newest_row, once);
	}
	Bf16 const a_log = load_half(at(weights.a_log), chunk.value_head, reused);
	Bf16 const dt_bias = load_half(at(weights.dt_bias), chunk.value_head, reused);
	unsigned const r = index % 8;
	int const vector_group = static_cast<int>(index / 8);
	constexpr int groups = prefill_preparer_threads / 8;
	int const blocks = (chunk.tokens + prefill_block_tokens - 1) / prefill_block_tokens;
	for (int block = 0; block < blocks + 2; ++block) {
		int const stage = block % 2;
		TokenBlock& tokens = shared.blocks[stage];
		if (block >= 2) {
			int const done = block - 2;
			prims::Mbarrier{.address = prims::shared_address(&shared.chained[stage])}.wait(static_cast<unsigned>((done / 2) % 2),
			                                                                               prims::block_backoff);
			int const token = vector_group;
			bool const active = token < block_tokens(chunk, done);
			Count const row = Count(chunk.first_row) + Count(done) * prefill_block_tokens + Count(active ? token : 0);
			float* const target = at(rows.outputs) + row * value_features + Count(chunk.value_head) * dim;
			normalize_output(tokens.outputs[active ? token : 0], at(weights.norm), target, active, r, reused);
		}
		if (block >= blocks) {
			continue;
		}
		int const count = block_tokens(chunk, block);
		Count const first = Count(chunk.first_row) + Count(block) * prefill_block_tokens;
		if (convolves) {
			for (int token = 0; token < count; ++token) {
				Chunk const newest = prims::load(
				    chunk_at(at(rows.qkv), (first + Count(token)) * qkv_row_bytes + Count(first_channel) * sizeof(Bf16)), reused);
				store_convolved(convolved_target(tokens.queries[token], tokens.keys[token], tokens.values[token], u),
				                convolve_chunk(taps, oldest, older, newer, newest));
				oldest = older;
				older = newer;
				newer = newest;
			}
		} else if (gates_a_token && gate_token < count) {
			/* SAFETY: the gate rows are fp32 words as `Rows` lays them out. */
			auto const* const gate_rows = reinterpret_cast<unsigned const*>(at(rows.gates));
			Count const pair = ((first + Count(gate_token)) * value_heads + Count(chunk.value_head)) * 2;
			tokens.gates[gate_token] = gates(__builtin_bit_cast(float, prims::load_word(gate_rows + pair, reused)),
			                                 __builtin_bit_cast(float, prims::load_word(gate_rows + pair + 1, reused)), a_log, dt_bias);
		}
		PrefillBarrier::sync();
		for (int base = 0; base < 2 * prefill_block_tokens; base += groups) {
			int const vector = base + vector_group;
			bool const active = vector < 2 * count;
			int const token = active ? vector / 2 : 0;
			Role const role = vector % 2 == 0 ? Role::Query : Role::Key;
			normalize_vector(role == Role::Query ? tokens.queries[token] : tokens.keys[token], role, active, r);
		}
		prims::sync_warp();
		if (prims::lane() == 0) {
			prims::Mbarrier{.address = prims::shared_address(&shared.prepared[stage])}.arrive();
		}
	}
}

/** Chain warp `warp`: its four quads of the head, held in registers through every token of the chunk. */
__attribute__((device, always_inline)) inline void chain_blocks(Chunked const& chunk, PrefillShared& shared, int warp) {
	auto const once = prims::policy<Eviction::First>();
	unsigned const lane_index = prims::lane();
	unsigned const r = lane_index % Column::lanes;
	int const column_in_quad = static_cast<int>(lane_index) / Column::lanes;
	Chunk* const head = chunk_at(chunk.image, state_offset(chunk.layer, chunk.value_head));
	float state[quads_per_chain_warp][Column::per_lane] = {};
#pragma unroll
	for (int quad = 0; quad < quads_per_chain_warp; ++quad) {
#pragma unroll
		for (int part = 0; part < Column::chunks; ++part) {
			Chunk const loaded =
			    prims::load(head + State::lane_chunk(warp * quads_per_chain_warp + quad, part, static_cast<int>(lane_index)) / 4, once);
#pragma unroll
			for (int word = 0; word < 4; ++word) {
				state[quad][part * 4 + word] = __builtin_bit_cast(float, loaded.word[word]);
			}
		}
	}
	int const blocks = (chunk.tokens + prefill_block_tokens - 1) / prefill_block_tokens;
	for (int block = 0; block < blocks; ++block) {
		int const stage = block % 2;
		TokenBlock& tokens = shared.blocks[stage];
		prims::Mbarrier{.address = prims::shared_address(&shared.prepared[stage])}.wait(static_cast<unsigned>((block / 2) % 2),
		                                                                                prims::block_backoff);
		int const count = block_tokens(chunk, block);
		for (int token = 0; token < count; ++token) {
			float key[Column::per_lane] = {};
			float query[Column::per_lane] = {};
			load_lane<Column>(tokens.keys[token], r, key);
			load_lane<Column>(tokens.queries[token], r, query);
			Gates const gate = tokens.gates[token];
#pragma unroll
			for (int quad = 0; quad < quads_per_chain_warp; ++quad) {
				int const column = (warp * quads_per_chain_warp + quad) * State::unit_columns + column_in_quad;
				float const value = tokens.values[token][column];
				float const projected = group_sum<Column>(decay_and_project(state[quad], key, gate.decay));
				update(state[quad], key, delta(value, projected, gate.beta));
				float const output = group_sum<Column>(read(state[quad], query));
				if (r == 0) {
					tokens.outputs[token][column] = output;
				}
			}
		}
		prims::sync_warp();
		if (lane_index == 0) {
			prims::Mbarrier{.address = prims::shared_address(&shared.chained[stage])}.arrive();
		}
	}
#pragma unroll
	for (int quad = 0; quad < quads_per_chain_warp; ++quad) {
#pragma unroll
		for (int part = 0; part < Column::chunks; ++part) {
			prims::store(head + State::lane_chunk(warp * quads_per_chain_warp + quad, part, static_cast<int>(lane_index)) / 4,
			             pack(state[quad], part), once);
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The chain probe and the tree self-check.

template<class T>
__attribute__((device, always_inline)) inline void probe_chains(ProbeParams const& params) {
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	constexpr int warps = probe_budget.threads / warp_threads;
	constexpr int blocks_per_head = StateLayout<T>::units / warps;
	unsigned const lane_index = prims::lane();
	unsigned const r = lane_index % T::lanes;
	int const block = static_cast<int>(prims::block());
	int const head = block / blocks_per_head;
	int const unit = (block % blocks_per_head) * warps + static_cast<int>(prims::warp_in_block());
	int const column = unit * T::columns_per_warp + static_cast<int>(lane_index) / T::lanes;
	Count const tokens = params.tokens;
	Chunk* const state_head = chunk_at(at(params.state), Count(head) * head_state_bytes);
	float state[T::per_lane] = {};
#pragma unroll
	for (int part = 0; part < T::chunks; ++part) {
		Chunk const loaded = prims::load(state_head + StateLayout<T>::lane_chunk(unit, part, static_cast<int>(lane_index)) / 4, once);
#pragma unroll
		for (int word = 0; word < 4; ++word) {
			state[part * 4 + word] = __builtin_bit_cast(float, loaded.word[word]);
		}
	}
	unsigned long long const started = prims::cycles();
	for (Count token = 0; token < tokens; ++token) {
		Count const row = Count(head) * tokens + token;
		float key[T::per_lane] = {};
		float query[T::per_lane] = {};
#pragma unroll
		for (int part = 0; part < T::chunks; ++part) {
			int const lane_offset = (part * T::lanes + static_cast<int>(r)) * 4;
			Count const offset = (row * dim + Count(lane_offset)) * sizeof(float);
			Chunk const k = prims::load(chunk_at(at(params.keys), offset), reused);
			Chunk const q = prims::load(chunk_at(at(params.queries), offset), reused);
#pragma unroll
			for (int word = 0; word < 4; ++word) {
				key[part * 4 + word] = __builtin_bit_cast(float, k.word[word]);
				query[part * 4 + word] = __builtin_bit_cast(float, q.word[word]);
			}
		}
		/* SAFETY: values and gates are fp32 words in arrays the host laid out as ProbeParams says. */
		auto const* const value_words = reinterpret_cast<unsigned const*>(at(params.values));
		auto const* const gate_words = reinterpret_cast<unsigned const*>(at(params.gates));
		float const value = __builtin_bit_cast(float, prims::load_word(value_words + row * dim + Count(column), reused));
		float const decay = __builtin_bit_cast(float, prims::load_word(gate_words + 2 * row, reused));
		float const beta = __builtin_bit_cast(float, prims::load_word(gate_words + 2 * row + 1, reused));
		float const projected = group_sum<T>(decay_and_project(state, key, decay));
		update(state, key, delta(value, projected, beta));
		float const output = group_sum<T>(read(state, query));
		if (r == 0) {
			/* SAFETY: outputs are [probe_heads][tokens][128] fp32. */
			prims::store_word(reinterpret_cast<unsigned*>(at(params.outputs)) + row * dim + Count(column),
			                  __builtin_bit_cast(unsigned, output), once);
		}
	}
	unsigned long long const finished = prims::cycles();
#pragma unroll
	for (int part = 0; part < T::chunks; ++part) {
		prims::store(state_head + StateLayout<T>::lane_chunk(unit, part, static_cast<int>(lane_index)) / 4, pack(state, part), once);
	}
	if (prims::thread_in_block() == 0) {
		unsigned long long* const cycles = at(params.cycles) + Count(block) * 2;
		prims::store_release<prims::Scope::Gpu>(cycles, started);
		prims::store_release<prims::Scope::Gpu>(cycles + 1, finished);
	}
}

/** Counts one comparison into `report`, keeping the first mismatches verbatim. */
__attribute__((device)) inline void record(prims::selfcheck::Report* report, bool agrees, prims::selfcheck::Mismatch mismatch) {
	prims::count(&report->checked, 1U);
	if (!agrees) {
		unsigned const slot = prims::fetch_add(&report->mismatched, 1U);
		if (slot < prims::selfcheck::kept_mismatches) {
			report->first[slot] = mismatch;
		}
	}
}

/** The butterfly on shuffles against `tree_sum` over the group's values gathered to every lane. */
template<class T>
__attribute__((device)) inline void check_tree(prims::selfcheck::Report* report, unsigned round) {
	unsigned const lane_index = prims::lane();
	float const own = check_value(prims::block(), prims::warp_in_block(), round, lane_index);
	float const got = group_sum<T>(own);
	unsigned const first = lane_index - lane_index % static_cast<unsigned>(T::lanes);
	float gathered[T::lanes] = {};
#pragma unroll
	for (int index = 0; index < T::lanes; ++index) {
		gathered[index] =
		    __builtin_bit_cast(float, prims::broadcast(__builtin_bit_cast(unsigned, own), first + static_cast<unsigned>(index)));
	}
	float const expected = tree_sum(gathered);
	unsigned const got_bits = __builtin_bit_cast(unsigned, got);
	unsigned const expected_bits = __builtin_bit_cast(unsigned, expected);
	record(report, got_bits == expected_bits,
	       {.input = round, .detail = static_cast<unsigned>(T::lanes) * 1000U + lane_index, .got = got_bits, .expected = expected_bits});
}

}

namespace deltanet = engine::device::deltanet;
namespace prims = engine::device::prims;
namespace model = engine::model;

/**
 * The decode core of one cycle at one layer: block b is live slot b / 8's key head group b % 8; blocks past the live
 * slots have nothing to do. Every item streams its group's state, runs its replays and rows, commits, and normalizes.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_deltanet_decode_core = deltanet::decode_budget;

extern "C" __attribute__((global, launch_bounds(deltanet::decode_budget.threads, deltanet::decode_budget.blocks_per_sm))) void
deltanet_decode_core(deltanet::DecodeParams params) {
	using namespace engine::device::deltanet;
	unsigned const slot_index = prims::block() / static_cast<unsigned>(key_heads);
	expect(params.live <= model::slots);
	if (slot_index >= params.live) {
		return;
	}
	Item const item = decode_item(params, slot_index, static_cast<int>(prims::block() % static_cast<unsigned>(key_heads)));
	auto& shared = prims::shared_layout<decode_budget, DecodeShared>();
	StateRing const ring = state_ring(shared);
	prims::Mbarrier const prepared = {.address = prims::shared_address(&shared.prepared)};
	if (prims::thread_in_block() == 0) {
		ring.init();
		prepared.init(DecodePreparers::count);
	}
	prims::sync_block();
	unsigned const warp = prims::warp_in_block();
	if (DecodeProducer::holds(warp)) {
		produce_state(item, ring, shared.stages);
	} else if (DecodeConsumers::holds(warp)) {
		unsigned const rank = DecodeConsumers::rank(warp);
		prepared.wait(0, prims::block_backoff);
		consume_state(item, ring, shared.stages, shared.events, shared.outputs, static_cast<int>(rank));
		ConsumerBarrier::sync();
		normalize_rows(item, params.weights, params.rows, shared.outputs, rank * 32U + prims::lane());
	} else {
		prepare_events<PreparerBarrier>(item, params.weights, params.rows, shared.events, prepared,
		                                prims::thread_in_block() - static_cast<unsigned>(DecodePreparers::first * 32));
	}
}

/**
 * The prefill core of one chunk per live slot at one layer: block b is live slot b / 24's value head b % 24. The
 * group's windows stay as they were; `deltanet_prefill_window` moves them after this launch.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_deltanet_prefill = deltanet::prefill_budget;

extern "C" __attribute__((global, launch_bounds(deltanet::prefill_budget.threads, deltanet::prefill_budget.blocks_per_sm))) void
deltanet_prefill(deltanet::PrefillParams params) {
	using namespace engine::device::deltanet;
	unsigned const slot_index = prims::block() / static_cast<unsigned>(value_heads);
	expect(params.live <= model::slots);
	if (slot_index >= params.live) {
		return;
	}
	Chunked const chunk = prefill_item(params, slot_index, static_cast<int>(prims::block() % static_cast<unsigned>(value_heads)));
	auto& shared = prims::shared_layout<prefill_budget, PrefillShared>();
	if (prims::thread_in_block() == 0) {
		for (int stage = 0; stage < 2; ++stage) {
			prims::Mbarrier{.address = prims::shared_address(&shared.prepared[stage])}.init(PrefillPreparers::count);
			prims::Mbarrier{.address = prims::shared_address(&shared.chained[stage])}.init(PrefillChains::count);
		}
	}
	prims::sync_block();
	unsigned const warp = prims::warp_in_block();
	if (PrefillPreparers::holds(warp)) {
		prepare_blocks(chunk, params.weights, params.rows, shared, prims::thread_in_block());
	} else {
		chain_blocks(chunk, shared, static_cast<int>(PrefillChains::rank(warp)));
	}
}

/**
 * Each live slot's windows moved to the end of its chunk: the last three of the old window and the chunk's raw inputs.
 * Launched after `deltanet_prefill`, which reads the old windows. Block b is live slot b / 8's key head group b % 8.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_deltanet_prefill_window = deltanet::window_budget;

extern "C" __attribute__((global, launch_bounds(deltanet::window_budget.threads, deltanet::window_budget.blocks_per_sm))) void
deltanet_prefill_window(deltanet::PrefillParams params) {
	using namespace engine::device::deltanet;
	unsigned const slot_index = prims::block() / static_cast<unsigned>(key_heads);
	int const u = static_cast<int>(prims::thread_in_block());
	expect(params.live <= model::slots);
	if (slot_index >= params.live || u >= group_chunks) {
		return;
	}
	int const key_head = static_cast<int>(prims::block() % static_cast<unsigned>(key_heads));
	Chunked const chunk = prefill_item(params, slot_index, key_head * group);
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	Chunk* const window = chunk_at(chunk.image, window_offset(chunk.layer, key_head)) + u;
	Chunk old[window_rows] = {};
#pragma unroll
	for (int row = 0; row < window_rows; ++row) {
		int const offset = row * group_chunks;
		old[row] = prims::load(window + offset, once);
	}
	int const first_channel = card_channel(key_head, u * bf16_per_chunk);
	Chunk next[window_rows] = {};
#pragma unroll
	for (int row = 0; row < window_rows; ++row) {
		// The window after the chunk is the last three of the old window followed by the chunk's inputs.
		int const index = chunk.tokens + row;
		if (index >= window_rows) {
			Count const token = Count(chunk.first_row + index - window_rows);
			next[row] = prims::load(chunk_at(at(params.rows.qkv), token * qkv_row_bytes + Count(first_channel) * sizeof(Bf16)), reused);
		} else {
			next[row] = index == 1 ? old[1] : old[2];
		}
	}
#pragma unroll
	for (int row = 0; row < window_rows; ++row) {
		int const offset = row * group_chunks;
		prims::store(window + offset, next[row], once);
	}
}

/** The chain probe under the engine's tree, eight lanes per column. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_deltanet_probe_chain_8 = deltanet::probe_budget;

extern "C" __attribute__((global, launch_bounds(deltanet::probe_budget.threads, deltanet::probe_budget.blocks_per_sm))) void
deltanet_probe_chain_8(deltanet::ProbeParams params) {
	deltanet::probe_chains<deltanet::Column>(params);
}

/** The chain probe with a whole warp per column, the alternative the microbenchmark weighs. */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_deltanet_probe_chain_32 = deltanet::probe_budget;

extern "C" __attribute__((global, launch_bounds(deltanet::probe_budget.threads, deltanet::probe_budget.blocks_per_sm))) void
deltanet_probe_chain_32(deltanet::ProbeParams params) {
	deltanet::probe_chains<deltanet::Tree<32>>(params);
}

/**
 * The tree self-check: every warp runs `check_rounds` rounds of the butterfly at eight and at 32 lanes against
 * `tree_sum`; `report` must come back with no mismatches.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_deltanet_check_trees = deltanet::check_budget;

extern "C" __attribute__((global, launch_bounds(deltanet::check_budget.threads, deltanet::check_budget.blocks_per_sm))) void
deltanet_check_trees(deltanet::CheckParams params) {
	for (unsigned round = 0; round < static_cast<unsigned>(deltanet::check_rounds); ++round) {
		deltanet::check_tree<deltanet::Column>(deltanet::at(params.report), round);
		deltanet::check_tree<deltanet::Tree<32>>(deltanet::at(params.report), round);
	}
}

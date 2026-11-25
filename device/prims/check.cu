// SPDX-License-Identifier: AGPL-3.0-only
//
// The primitives' self-check kernels: each runs primitives on the gpu against an oracle and reports what disagreed.
// They are real kernels with no other job, built to ptx now and run at bring-up by the microbenchmark drivers, which
// read their reports through `engine.prims`. Every kernel takes one parameter struct, and its budget, grid and buffer
// layouts are facts in selfcheck.h. Where the device can decide correctness
// itself (the decode, through the mma, the fragment maps, the ring, the reductions) the report is a mismatch count;
// where the oracle lives on the host (the tensor core's accumulation model, the host's transcendentals) the kernel
// produces what the host compares.
#include "prims/budget.h"
#include "prims/fragment.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/numeric.h"
#include "prims/order.h"
#include "prims/reduce.h"
#include "prims/selfcheck.h"
#include "prims/selfcheck_record.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "prims/transcendental.h"

namespace engine::device::prims::selfcheck {

/** Code `code`, with the two nan codes the baker refuses replaced by zero. */
[[nodiscard]] constexpr auto admitted(unsigned code) -> E4m3 {
	auto const candidate = E4m3{.bits = static_cast<unsigned char>(code & 0xffU)};
	return finite(candidate) ? candidate : E4m3{.bits = 0};
}

[[nodiscard]] constexpr auto power_of_two_bf16(int exponent) -> Bf16 {
	return {.bits = static_cast<unsigned short>(static_cast<unsigned>(exponent + 127) << 7U)};
}

}

namespace check = engine::device::prims::selfcheck;
namespace prims = engine::device::prims;

// The activation exponents each warp of the mma pass multiplies the weights by: across fp32's range, every product
// normal, so the tensor core's sum of one nonzero product is exact whatever its accumulation does.
inline constexpr int activation_exponents[8] = {-100, -60, -20, -9, 0, 9, 40, 100};

/**
 * U7, the decode: every pair of codes in every bf16x2 position decoded by `decode_word` against the exact
 * conversion; then all 256 codes as a 16 x 16 a-fragment through `mma.sync` against activations 2^j, which must
 * reproduce each value times 2^j exactly in fp32.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_decode = check::decode_budget;

extern "C" __attribute__((global, launch_bounds(check::decode_budget.threads, check::decode_budget.blocks_per_sm))) void prims_check_decode(check::ReportParams params) {
	using namespace engine::device::prims;
	auto* const report = check::pointer<check::Report>(params.report);
	unsigned const thread = thread_in_block();
	for (unsigned partner = 0; partner < 256; ++partner) {
		E4m3 const first_code = check::admitted(thread);
		E4m3 const second_code = check::admitted(partner);
		unsigned const word = decode::encode(first_code, second_code, second_code, first_code);
		decode::Pairs const got = decode_word(word);
		decode::Pairs const expected = {.top = pack(to_bf16(first_code), to_bf16(second_code)), .bottom = pack(to_bf16(second_code), to_bf16(first_code))};
		check::record(report, got == expected, {.input = word, .detail = 0, .got = got.top.bits ^ got.bottom.bits, .expected = expected.top.bits ^ expected.bottom.bits});
	}

	unsigned const lane_index = lane();
	int const exponent = activation_exponents[warp_in_block()];
	unsigned words[2] = {};
	for (unsigned word = 0; word < 2; ++word) {
		E4m3 codes[4] = {};
		for (unsigned position = 0; position < 4; ++position) {
			fragment::Coordinate const at = fragment::weight(static_cast<int>(lane_index), static_cast<int>(word), static_cast<int>(position));
			codes[position] = check::admitted(static_cast<unsigned>(at.row * 16 + at.column));
		}
		words[word] = decode::encode(codes[0], codes[1], codes[2], codes[3]);
	}
	decode::Pairs const low_k = decode_word(words[0]);
	decode::Pairs const high_k = decode_word(words[1]);
	AFragment const a = {.reg = {low_k.top.bits, low_k.bottom.bits, high_k.top.bits, high_k.bottom.bits}};
	for (int shift = 0; shift < 2; ++shift) {
		BFragment b{};
		for (int reg = 0; reg < 2; ++reg) {
			Bf16 halves[2] = {};
			for (int half = 0; half < 2; ++half) {
				fragment::Coordinate const at = fragment::b(static_cast<int>(lane_index), reg, half);
				halves[half] = at.row == at.column + 8 * shift ? check::power_of_two_bf16(exponent) : Bf16{.bits = 0};
			}
			b.reg[reg] = pack(halves[0], halves[1]).bits;
		}
		Accumulator d = {.reg = {0.0F, 0.0F, 0.0F, 0.0F}};
		mma(d, a, b);
		for (int reg = 0; reg < 4; ++reg) {
			fragment::Coordinate const at = fragment::c(static_cast<int>(lane_index), reg);
			unsigned const code = static_cast<unsigned>(at.row * 16 + at.column + 8 * shift);
			float const expected = value(check::admitted(code)) * math::power_of_two(exponent / 2) * math::power_of_two(exponent - exponent / 2);
			unsigned const got_bits = __builtin_bit_cast(unsigned, d.reg[reg]);
			unsigned const expected_bits = __builtin_bit_cast(unsigned, expected);
			check::record(report, got_bits == expected_bits, {.input = code, .detail = static_cast<unsigned>(exponent + 1000), .got = got_bits, .expected = expected_bits});
		}
	}
}

/**
 * U6, the tensor core's accumulation: runs `mma.sync` on host-crafted probes, one per warp, and writes each D; the
 * host compares them with the reference's bit-exact model of sm_86's 8-product truncated blocks.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_probe_mma = check::mma_budget;

extern "C" __attribute__((global, launch_bounds(check::mma_budget.threads, check::mma_budget.blocks_per_sm))) void prims_probe_mma(check::MmaParams params) {
	using namespace engine::device::prims;
	auto const* const probes = check::pointer<check::MmaProbe const>(params.probes);
	auto* const results = check::pointer<Accumulator>(params.results);
	unsigned const lane_index = lane();
	unsigned const probe = block() * static_cast<unsigned>(check::mma_budget.threads / 32) + warp_in_block();
	auto const once = policy<Eviction::First>();
	check::MmaProbe const& mine = probes[probe];
	auto const* a_words = reinterpret_cast<Chunk const*>(&mine.a[lane_index]);
	auto const* c_words = reinterpret_cast<Chunk const*>(&mine.c[lane_index]);
	Chunk const a_chunk = load_constant(a_words, once);
	Chunk const c_chunk = load_constant(c_words, once);
	AFragment const a = {.reg = {a_chunk.word[0], a_chunk.word[1], a_chunk.word[2], a_chunk.word[3]}};
	BFragment const b = mine.b[lane_index];
	Accumulator d{};
	for (int reg = 0; reg < 4; ++reg) {
		d.reg[reg] = __builtin_bit_cast(float, c_chunk.word[reg]);
	}
	mma(d, a, b);
	Chunk out{};
	for (int reg = 0; reg < 4; ++reg) {
		out.word[reg] = __builtin_bit_cast(unsigned, d.reg[reg]);
	}
	store(reinterpret_cast<Chunk*>(&results[probe * 32U + lane_index]), out, once);
}

using check::ActivationTile;
inline constexpr prims::fragment::Transpose transposes[2] = {prims::fragment::Transpose::Plain, prims::fragment::Transpose::Transposed};

/**
 * The fragment maps on the hardware: a swizzled 16 x 128 bf16 tile whose every element holds its own (row, k) tag is
 * read back with `ldmatrix.x4` and `.trans`, and every register half must hold the tag `fragment::ldmatrix` names.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_fragments = check::fragment_budget;

extern "C" __attribute__((global, launch_bounds(check::fragment_budget.threads, check::fragment_budget.blocks_per_sm))) void prims_check_fragments(check::ReportParams params) {
	using namespace engine::device::prims;
	auto* const report = check::pointer<check::Report>(params.report);
	auto& layout = shared_layout<check::fragment_budget, check::FragmentLayout>();
	SharedAddress const tile = shared_address(layout.tile);
	unsigned const lane_index = lane();
	// Tag of (row, k): row in the high byte, k in the low.
	for (unsigned index = lane_index; index < ActivationTile::rows * ActivationTile::chunks_per_row; index += 32) {
		unsigned const row = index / ActivationTile::chunks_per_row;
		unsigned const chunk = index % ActivationTile::chunks_per_row;
		Chunk tags{};
		for (unsigned word = 0; word < 4; ++word) {
			unsigned const k = chunk * 8 + 2 * word;
			tags.word[word] = ((row << 8U) | k) | (((row << 8U) | (k + 1)) << 16U);
		}
		store_shared(tile.plus(static_cast<unsigned>(ActivationTile::offset(static_cast<int>(row), static_cast<int>(chunk)))), tags);
	}
	sync_warp();
	for (int chunk = 0; chunk < ActivationTile::chunks_per_row; chunk += 2) {
		// Matrix m covers rows 8 (m % 2) .. + 7 and chunk + m / 2; this lane supplies row lane % 8 of matrix lane / 8.
		int const matrix = static_cast<int>(lane_index) / 8;
		int const row = 8 * (matrix % 2) + static_cast<int>(lane_index) % 8;
		SharedAddress const address = tile.plus(static_cast<unsigned>(ActivationTile::offset(row, chunk + matrix / 2)));
		Matrices4 const plain = ldmatrix_x4<fragment::Transpose::Plain>(address);
		Matrices4 const transposed = ldmatrix_x4<fragment::Transpose::Transposed>(address);
#pragma unroll
		for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
			for (int half = 0; half < 2; ++half) {
				for (fragment::Transpose const transpose : transposes) {
					fragment::Coordinate const at = fragment::ldmatrix(static_cast<int>(lane_index), half, transpose);
					unsigned const expected = (static_cast<unsigned>(8 * (reg % 2) + at.row) << 8U) | static_cast<unsigned>((chunk + reg / 2) * 8 + at.column);
					unsigned const loaded = transpose == fragment::Transpose::Plain ? plain.reg[reg] : transposed.reg[reg];
					unsigned const got = (loaded >> (16U * static_cast<unsigned>(half))) & 0xffffU;
					check::record(report, got == expected, {.input = static_cast<unsigned>(chunk), .detail = static_cast<unsigned>(reg * 4 + half * 2 + (transpose == fragment::Transpose::Plain ? 0 : 1)), .got = got, .expected = expected});
				}
			}
		}
	}
}

using Producer = prims::Warps<0, check::stage_ring_producer_warps>;
using Consumers = prims::Warps<check::stage_ring_producer_warps, check::stage_ring_consumer_warps>;
using StageRingCheck = prims::StageRing<check::stage_ring_stages, static_cast<unsigned>(Producer::threads), static_cast<unsigned>(Consumers::count)>;
static_assert(prims::tiles<Producer, Consumers>(check::stage_ring_budget));
static_assert(sizeof(StageRingCheck::Storage) == sizeof(check::StageRingLayout::barriers));

/**
 * U8, the stage ring: one producer warp streams `stage_ring_tiles_per_block` 4 KiB tiles per block through a
 * 4-stage ring with `cp.async` + `cp.async.mbarrier.arrive.noinc`; four consumer warps wait on the full barriers'
 * parity, check every word against `ring_word`, and free the stage; `elapsed[block]` gets the consumers' clocks.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_stage_ring = check::stage_ring_budget;

extern "C" __attribute__((global, launch_bounds(check::stage_ring_budget.threads, check::stage_ring_budget.blocks_per_sm))) void prims_check_stage_ring(check::StageRingParams params) {
	using namespace engine::device::prims;
	auto const* const source = check::pointer<Chunk const>(params.source);
	auto* const report = check::pointer<check::Report>(params.report);
	auto* const elapsed = check::pointer<unsigned long long>(params.elapsed);
	auto& layout = shared_layout<check::stage_ring_budget, check::StageRingLayout>();
	StageRingCheck const ring = StageRingCheck::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	unsigned const lane_index = lane();
	unsigned const first_chunk = block() * check::stage_ring_tiles_per_block * check::stage_ring_tile_chunks;
	StageRingCheck::Cursor cursor = StageRingCheck::Cursor::start();
	if (Producer::holds(warp)) {
		WeightStream<32> stream = {.next = source + first_chunk + lane_index, .policy = policy<Eviction::First>()};
		for (unsigned tile = 0; tile < check::stage_ring_tiles_per_block; ++tile) {
			ring.await_free(cursor, block_backoff);
			for (unsigned chunk = lane_index; chunk < check::stage_ring_tile_chunks; chunk += 32) {
				stream.issue(shared_address(&layout.stage[cursor.stage][chunk]));
			}
			ring.filled_on_copies(cursor);
			cursor = cursor.next();
		}
		wait_all_copies();
		return;
	}
	unsigned long long const start = cycles();
	unsigned const consumer = Consumers::rank(warp) * 32 + lane_index;
	for (unsigned tile = 0; tile < check::stage_ring_tiles_per_block; ++tile) {
		ring.await_filled(cursor, block_backoff);
		for (unsigned chunk = consumer; chunk < check::stage_ring_tile_chunks; chunk += Consumers::threads) {
			Chunk const got = load_shared(shared_address(&layout.stage[cursor.stage][chunk]));
			unsigned const base = (first_chunk + tile * check::stage_ring_tile_chunks + chunk) * 4;
			for (unsigned word = 0; word < 4; ++word) {
				unsigned const expected = check::ring_word(base + word);
				check::record(report, got.word[word] == expected, {.input = base + word, .detail = tile, .got = got.word[word], .expected = expected});
			}
		}
		sync_warp();
		if (lane_index == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();
	}
	if (consumer == 0) {
		elapsed[block()] = cycles() - start;
	}
}

/**
 * The reductions: per trial each lane draws a float spanning many binades; `warp_sum` must give every lane the
 * butterfly's sum evaluated in its fixed order (recomputed per lane from all 32 inputs), and `redux.sync` must agree
 * with an integer butterfly.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_reduce = check::reduce_budget;

extern "C" __attribute__((global, launch_bounds(check::reduce_budget.threads, check::reduce_budget.blocks_per_sm))) void prims_check_reduce(check::ReportParams params) {
	using namespace engine::device::prims;
	auto* const report = check::pointer<check::Report>(params.report);
	unsigned const lane_index = lane();
	for (unsigned trial = 0; trial < check::reduce_trials; ++trial) {
		unsigned const seed = check::ring_word((warp_in_block() * check::reduce_trials + trial) * 32 + lane_index);
		// Sign, an exponent within 2^-20 .. 2^20, and a random mantissa.
		unsigned const bits_of_input = (seed & 0x807fffffU) | ((107U + (seed >> 23U) % 41U) << 23U);
		float const input = __builtin_bit_cast(float, bits_of_input);
		float level[32] = {};
		for (unsigned other = 0; other < 32; ++other) {
			level[other] = __builtin_bit_cast(float, broadcast(bits_of_input, other));
		}
		for (unsigned distance = 16; distance != 0; distance /= 2) {
			float next[32] = {};
			for (unsigned other = 0; other < 32; ++other) {
				next[other] = level[other] + level[other ^ distance];
			}
			for (unsigned other = 0; other < 32; ++other) {
				level[other] = next[other];
			}
		}
		float const total = warp_sum(input);
		unsigned const got = __builtin_bit_cast(unsigned, total);
		unsigned const expected = __builtin_bit_cast(unsigned, level[lane_index]);
		check::record(report, got == expected && ballot(got == broadcast(got, 0)) == full_warp, {.input = bits_of_input, .detail = trial, .got = got, .expected = expected});

		unsigned integer_total = seed;
		for (unsigned distance = 16; distance != 0; distance /= 2) {
			integer_total += exchange(integer_total, distance);
		}
		unsigned const reduced = warp_sum(seed);
		check::record(report, reduced == integer_total, {.input = seed, .detail = trial, .got = reduced, .expected = integer_total});
		unsigned least = seed;
		for (unsigned distance = 16; distance != 0; distance /= 2) {
			unsigned const partner = exchange(least, distance);
			least = partner < least ? partner : least;
		}
		check::record(report, warp_min(seed) == least, {.input = seed, .detail = trial, .got = warp_min(seed), .expected = least});
		unsigned most = seed;
		unsigned any = seed;
		unsigned all = seed;
		int signed_least = static_cast<int>(seed);
		int signed_most = static_cast<int>(seed);
		float float_most = input;
		for (unsigned distance = 16; distance != 0; distance /= 2) {
			unsigned const partner = exchange(most, distance);
			most = partner > most ? partner : most;
			any |= exchange(any, distance);
			all &= exchange(all, distance);
			int const signed_partner_least = static_cast<int>(exchange(static_cast<unsigned>(signed_least), distance));
			signed_least = signed_partner_least < signed_least ? signed_partner_least : signed_least;
			int const signed_partner_most = static_cast<int>(exchange(static_cast<unsigned>(signed_most), distance));
			signed_most = signed_partner_most > signed_most ? signed_partner_most : signed_most;
			float const float_partner = __builtin_bit_cast(float, exchange(__builtin_bit_cast(unsigned, float_most), distance));
			float_most = float_partner > float_most ? float_partner : float_most;
		}
		bool const integers_agree = warp_max(seed) == most && warp_or(seed) == any && warp_and(seed) == all
		                            && warp_min(static_cast<int>(seed)) == signed_least && warp_max(static_cast<int>(seed)) == signed_most;
		check::record(report, integers_agree, {.input = seed, .detail = trial, .got = warp_max(seed), .expected = most});
		float const float_max = warp_max(input);
		check::record(report, __builtin_bit_cast(unsigned, float_max) == __builtin_bit_cast(unsigned, float_most), {.input = seed, .detail = trial, .got = __builtin_bit_cast(unsigned, float_max), .expected = __builtin_bit_cast(unsigned, float_most)});
	}
}

/**
 * The transcendentals on the device: every fp32 input through each function, folded per block into
 * `digests[function * digest_blocks + block]` (which start zeroed). The host folds its own evaluation of the same
 * header over the same partition; equal digests mean the device computes the host's bits.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_digest_transcendentals = check::digest_budget;

extern "C" __attribute__((global, launch_bounds(check::digest_budget.threads, check::digest_budget.blocks_per_sm))) void prims_digest_transcendentals(check::DigestParams params) {
	using namespace engine::device::prims;
	auto* const digests = check::pointer<unsigned long long>(params.digests);
	unsigned long long const run_start = (static_cast<unsigned long long>(block()) * check::digest_budget.threads + thread_in_block()) * check::digest_run;
	unsigned long long sums[check::functions] = {};
	for (unsigned long long offset = 0; offset < check::digest_run; ++offset) {
		auto const input = static_cast<unsigned>(run_start + offset);
		float const x = __builtin_bit_cast(float, input);
		float const outputs[check::functions] = {math::exp(x), math::log(x), math::rsqrt(x), math::sigmoid(x), math::silu(x), math::softplus(x)};
		for (unsigned function = 0; function < check::functions; ++function) {
			sums[function] += check::digest_term(input, __builtin_bit_cast(unsigned, outputs[function]));
		}
	}
	for (unsigned function = 0; function < check::functions; ++function) {
		accumulate(&digests[function * check::digest_blocks + block()], sums[function]);
	}
}

using OrderProducers = prims::Warps<0, check::order_producer_warps>;
using OrderConsumers = prims::Warps<check::order_producer_warps, check::order_consumer_warps>;
using Handoff = prims::NamedBarrier<1, OrderProducers::count + OrderConsumers::count>;
static_assert(prims::tiles<OrderProducers, OrderConsumers>(check::order_budget));

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_order = check::order_budget;

/**
 * Memory order and the remaining primitives on the hardware: one block per SM (82, co-resident), each passing a value
 * to the next block through a step-1 grid barrier, a release counter, gpu- and system-scope epoch flags, the peer
 * push protocol (aimed at this card's own memory) and a relaxed flag ordered by sequentially consistent fences; within a block, producer warps hand shared data to consumer warps
 * on a named barrier, a double-buffered loop sums staged chunks, and the bf16 pack and widen instructions are held
 * to their host models.
 */
extern "C" __attribute__((global, launch_bounds(check::order_budget.threads, check::order_budget.blocks_per_sm))) void prims_check_order(check::OrderParams params) {
	using namespace engine::device::prims;
	auto* const scratch = check::pointer<check::OrderScratch>(params.scratch);
	auto* const report = check::pointer<check::Report>(params.report);
	auto& layout = shared_layout<check::order_budget, check::OrderLayout>();
	unsigned const me = block();
	unsigned const next = (me + 1) % check::order_blocks;
	unsigned const previous = (me + check::order_blocks - 1) % check::order_blocks;
	unsigned const thread = thread_in_block();
	unsigned const warp = warp_in_block();
	auto const reused = policy<Eviction::Last>();
	auto const once = policy<Eviction::First>();

	// Across SMs: each block's value, published four ways, must reach its neighbour.
	Chunk const mine = {.word = {me, me * 3U, me * 5U, me * 7U}};
	if (thread == 0) {
		store(&scratch->slots[me], mine, reused);
	}
	step1::GridBarrier<check::order_blocks>{.arrivals = &scratch->arrivals, .generation = &scratch->generation}.sync(remote_backoff);
	if (thread == 0) {
		Chunk const seen = load(&scratch->slots[next], reused);
		check::record(report, seen.word[0] == next && seen.word[3] == next * 7U, {.input = me, .detail = 1, .got = seen.word[0], .expected = next});

		release_add(&scratch->counter, 1);
		await_count(&scratch->counter, check::order_blocks, remote_backoff);
		check::record(report, load_acquire(&scratch->counter) == check::order_blocks, {.input = me, .detail = 2, .got = load_acquire(&scratch->counter), .expected = check::order_blocks});

		store(&scratch->slots[me], {.word = {me + 1000U, 0, 0, 0}}, reused);
		EpochWriter<Scope::Gpu>{.flag = &scratch->gpu_flags[me]}.publish(1);
		EpochReader<Scope::Gpu>{.flag = &scratch->gpu_flags[previous]}.await(1, remote_backoff);
		Chunk const relayed = load(&scratch->slots[previous], reused);
		check::record(report, relayed.word[0] == previous + 1000U, {.input = me, .detail = 3, .got = relayed.word[0], .expected = previous + 1000U});

		PeerOutbox const outbox = {.slots = scratch->inbox, .flag = {.flag = &scratch->system_flags[me]}};
		outbox.push(me, mine);
		outbox.publish(1);
		PeerInbox const inbox = {.slots = scratch->inbox, .flag = {.flag = &scratch->system_flags[previous]}};
		inbox.await(1, remote_backoff);
		Chunk const pushed = inbox.read(previous, once);
		check::record(report, pushed.word[2] == previous * 5U, {.input = me, .detail = 4, .got = pushed.word[2], .expected = previous * 5U});

		// The fenced relaxed protocol the flag microbenchmark measures against release and acquire.
		store(&scratch->fenced[me], mine, reused);
		fence_sc<Scope::System>();
		store_relaxed<Scope::System>(&scratch->relaxed_flags[me], 1);
		unsigned sleep = remote_backoff.first_ns;
		while (load_relaxed<Scope::System>(&scratch->relaxed_flags[previous]) < 1) {
			nanosleep(sleep);
			sleep = sleep * 2U < remote_backoff.ceiling_ns ? sleep * 2U : remote_backoff.ceiling_ns;
		}
		fence_sc<Scope::System>();
		Chunk const fenced = load(&scratch->fenced[previous], reused);
		check::record(report, fenced.word[1] == previous * 3U, {.input = me, .detail = 12, .got = fenced.word[1], .expected = previous * 3U});

		prefetch_to_l2(&scratch->slots[me]);
		release_priority(&scratch->scratch[me]);
		discard(&scratch->scratch[me]);
	}

	// Within the block: producers hand chunks to consumers on a named barrier, arriving without waiting.
	if (OrderProducers::holds(warp)) {
		unsigned const slot = (OrderProducers::rank(warp) * 32U + lane()) % OrderProducers::threads;
		store_shared(shared_address(&layout.handed[slot]), {.word = {slot, me, slot ^ me, 7U}});
		Handoff::arrive();
	} else {
		Handoff::sync();
		unsigned const source = (OrderConsumers::rank(warp) * 32U + lane()) % OrderProducers::threads;
		Chunk const handed = load_shared(shared_address(&layout.handed[source]));
		check::record(report, handed.word[0] == source && handed.word[2] == (source ^ me), {.input = thread, .detail = 5, .got = handed.word[2], .expected = source ^ me});
	}

	// Register double buffering: the pipelined sum of staged chunks equals the plain one.
	if (warp == 0) {
		unsigned const lane_index = lane();
		for (unsigned step = 0; step < check::order_steps; ++step) {
			store_shared(shared_address(&layout.steps[step][lane_index]), {.word = {step, lane_index, step * lane_index, 1U}});
		}
		sync_warp();
		unsigned pipelined = 0;
		double_buffered<static_cast<int>(check::order_steps)>(
		    [&](int step) { return load_shared(shared_address(&layout.steps[step][lane_index])); },
		    [&](int, Chunk const& chunk) { pipelined += chunk.word[2] + chunk.word[3]; });
		unsigned plain = 0;
		for (unsigned step = 0; step < check::order_steps; ++step) {
			plain += step * lane_index + 1U;
		}
		check::record(report, pipelined == plain, {.input = lane_index, .detail = 6, .got = pipelined, .expected = plain});
	}

	// Copy groups and a lone mbarrier: warp 1 stores words of its own, copies them back into shared memory under a
	// normal policy, waits on the group, counts itself through a one-phase barrier and frees it.
	if (warp == 1) {
		unsigned const lane_index = lane();
		auto const normal = policy<Eviction::Normal>();
		store_word(&scratch->words[me][lane_index], lane_index * 11U + me, normal);
		sync_warp();
		unsigned const chunk = lane_index % 8;
		auto const* words = reinterpret_cast<Chunk const*>(scratch->words[me]);
		copy_async<Prefetch::None>(shared_address(&layout.copied[lane_index]), &words[chunk], normal);
		commit_copies();
		wait_copies<0>();
		Chunk const copied = load_shared(shared_address(&layout.copied[lane_index]));
		unsigned const expected = 4U * chunk * 11U + me;
		check::record(report, copied.word[0] == expected, {.input = lane_index, .detail = 9, .got = copied.word[0], .expected = expected});
		unsigned const reread = load_word(&scratch->words[me][lane_index], normal);
		check::record(report, reread == lane_index * 11U + me, {.input = lane_index, .detail = 10, .got = reread, .expected = lane_index * 11U + me});
		Mbarrier const lone = {.address = shared_address(&layout.barrier)};
		if (lane_index == 0) {
			lone.init(32);
		}
		sync_warp();
		lone.arrive();
		lone.wait(0, block_backoff);
		check::record(report, lone.test(0) && !lone.test(1), {.input = lane_index, .detail = 11, .got = 0, .expected = 0});
		sync_warp();
		if (lane_index == 0) {
			lone.invalidate();
		}
	}

	// The bf16 pack and widen instructions against their models, over inputs spanning fp32.
	for (unsigned trial = 0; trial < 64; ++trial) {
		unsigned const seed = check::ring_word((me * check::order_budget.threads + thread) * 64U + trial);
		float const first = __builtin_bit_cast(float, seed & 0xbfffffffU);
		float const second = __builtin_bit_cast(float, check::ring_word(seed) & 0xbfffffffU);
		Bf16x2 const packed = pack_bf16x2(first, second);
		Bf16x2 const expected = pack(round_to_bf16(first), round_to_bf16(second));
		check::record(report, packed == expected, {.input = seed, .detail = 7, .got = packed.bits, .expected = expected.bits});
		float const wide_first = widen_first(packed);
		float const wide_second = widen_second(packed);
		bool const widened = __builtin_bit_cast(unsigned, wide_first) == __builtin_bit_cast(unsigned, widen(round_to_bf16(first)))
		                     && __builtin_bit_cast(unsigned, wide_second) == __builtin_bit_cast(unsigned, widen(round_to_bf16(second)));
		check::record(report, widened, {.input = seed, .detail = 8, .got = __builtin_bit_cast(unsigned, wide_first), .expected = expected.bits});
	}
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The primitives' self-check kernels: each runs primitives on the gpu against an oracle and reports what disagreed.
// They are real kernels with no other job, built to ptx now and run at bring-up by the microbenchmark drivers, which
// read their reports through `engine.prims` (the layouts in selfcheck.h). Where the device can decide correctness
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
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "prims/transcendental.h"

namespace engine::device::prims::selfcheck {

/** Counts one comparison into `report`, keeping the first mismatches verbatim. */
__attribute__((device)) inline void record(Report* report, bool agrees, Mismatch mismatch) {
	count(&report->checked, 1U);
	if (!agrees) {
		unsigned const slot = fetch_add(&report->mismatched, 1U);
		if (slot < kept_mismatches) {
			report->first[slot] = mismatch;
		}
	}
}

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

inline constexpr prims::Budget decode_budget = {.threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};
static_assert(prims::check(decode_budget) == prims::Fit::Fits);

// The activation exponents each warp of the mma pass multiplies the weights by: across fp32's range, every product
// normal, so the tensor core's sum of one nonzero product is exact whatever its accumulation does.
inline constexpr int activation_exponents[8] = {-100, -60, -20, -9, 0, 9, 40, 100};

/**
 * U7, the decode: every pair of codes in every bf16x2 position decoded by `decode_word` against the exact
 * conversion; then all 256 codes as a 16 x 16 a-fragment through `mma.sync` against activations 2^j, which must
 * reproduce each value times 2^j exactly in fp32. One block of 256 threads; `report` starts zeroed.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_decode = decode_budget;

extern "C" __attribute__((global, launch_bounds(decode_budget.threads, decode_budget.blocks_per_sm))) void prims_check_decode(check::Report* report) {
	using namespace engine::device::prims;
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

inline constexpr prims::Budget mma_budget = {.threads = 128, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 4};
static_assert(prims::check(mma_budget) == prims::Fit::Fits);
inline constexpr unsigned mma_probe_blocks = 64;

/** One warp's mma probe: every lane's operands as the fragment maps lay them out. */
struct MmaProbe {
	prims::AFragment a[32];
	prims::BFragment b[32];
	prims::Accumulator c[32];
};

/**
 * U6, the tensor core's accumulation: runs `mma.sync` on host-crafted probes, one per warp, and writes each D; the
 * host compares them with the reference's bit-exact model of sm_86's 8-product truncated blocks.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_probe_mma = mma_budget;

extern "C" __attribute__((global, launch_bounds(mma_budget.threads, mma_budget.blocks_per_sm))) void prims_probe_mma(MmaProbe const* probes, prims::Accumulator* results) {
	using namespace engine::device::prims;
	unsigned const lane_index = lane();
	unsigned const probe = block() * (mma_budget.threads / 32U) + warp_in_block();
	auto const once = policy<Eviction::First>();
	MmaProbe const& mine = probes[probe];
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

using ActivationTile = prims::fragment::XorSwizzle<256, 16>;
inline constexpr prims::fragment::Transpose transposes[2] = {prims::fragment::Transpose::Plain, prims::fragment::Transpose::Transposed};
static_assert(prims::fragment::conflict_free<ActivationTile>());

struct FragmentLayout {
	alignas(128) unsigned char tile[ActivationTile::bytes];
};

inline constexpr prims::Budget fragment_budget = {.threads = 32, .registers_per_thread = 64, .shared_bytes = sizeof(FragmentLayout), .blocks_per_sm = 1};
static_assert(prims::check(fragment_budget) == prims::Fit::Fits);

/**
 * The fragment maps on the hardware: a swizzled 16 x 128 bf16 tile whose every element holds its own (row, k) tag is
 * read back with `ldmatrix.x4` and `.trans`, and every register half must hold the tag `fragment::ldmatrix` names.
 * One warp; `report` starts zeroed.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_fragments = fragment_budget;

extern "C" __attribute__((global, launch_bounds(fragment_budget.threads, fragment_budget.blocks_per_sm))) void prims_check_fragments(check::Report* report) {
	using namespace engine::device::prims;
	auto& layout = shared_layout<fragment_budget, FragmentLayout>();
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

inline constexpr int ring_stages = 4;
inline constexpr unsigned ring_tile_chunks = 256;
inline constexpr unsigned ring_tiles_per_block = 64;
inline constexpr unsigned ring_consumer_warps = 4;
using StageRingCheck = prims::StageRing<ring_stages, 32, ring_consumer_warps>;
using Producer = prims::Warps<0, 1>;
using Consumers = prims::Warps<1, ring_consumer_warps>;

struct StageRingLayout {
	StageRingCheck::Storage barriers;
	prims::Chunk stage[ring_stages][ring_tile_chunks];
};

inline constexpr prims::Budget stage_ring_budget = {.threads = 160, .registers_per_thread = 64, .shared_bytes = sizeof(StageRingLayout), .blocks_per_sm = 1};
static_assert(prims::check(stage_ring_budget) == prims::Fit::Fits);
static_assert(prims::tiles<Producer, Consumers>(stage_ring_budget));

/**
 * U8, the ring: one producer warp streams `ring_tiles_per_block` 4 KiB tiles per block through a 4-stage ring with
 * `cp.async` + `cp.async.mbarrier.arrive.noinc`; four consumer warps wait on the full barriers' parity, check every
 * word against `ring_word`, and free the stage. `source` holds 82 blocks' tiles; `elapsed[block]` gets the consumers'
 * elapsed clocks.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_stage_ring = stage_ring_budget;

extern "C" __attribute__((global, launch_bounds(stage_ring_budget.threads, stage_ring_budget.blocks_per_sm))) void prims_check_stage_ring(prims::Chunk const* source, check::Report* report, unsigned long long* elapsed) {
	using namespace engine::device::prims;
	auto& layout = shared_layout<stage_ring_budget, StageRingLayout>();
	StageRingCheck const ring = StageRingCheck::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	unsigned const lane_index = lane();
	unsigned const first_chunk = block() * ring_tiles_per_block * ring_tile_chunks;
	StageRingCheck::Cursor cursor = StageRingCheck::Cursor::start();
	if (Producer::holds(warp)) {
		WeightStream<32> stream = {.next = source + first_chunk + lane_index, .policy = policy<Eviction::First>()};
		for (unsigned tile = 0; tile < ring_tiles_per_block; ++tile) {
			ring.await_free(cursor, block_backoff);
			for (unsigned chunk = lane_index; chunk < ring_tile_chunks; chunk += 32) {
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
	for (unsigned tile = 0; tile < ring_tiles_per_block; ++tile) {
		ring.await_filled(cursor, block_backoff);
		for (unsigned chunk = consumer; chunk < ring_tile_chunks; chunk += Consumers::threads) {
			Chunk const got = load_shared(shared_address(&layout.stage[cursor.stage][chunk]));
			unsigned const base = (first_chunk + tile * ring_tile_chunks + chunk) * 4;
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

inline constexpr prims::Budget reduce_budget = {.threads = 256, .registers_per_thread = 128, .shared_bytes = 0, .blocks_per_sm = 1};
static_assert(prims::check(reduce_budget) == prims::Fit::Fits);
inline constexpr unsigned reduce_trials = 256;

/**
 * The reductions: per trial each lane draws a float spanning many binades; `warp_sum` must give every lane the
 * butterfly's sum evaluated in its fixed order (recomputed per lane from all 32 inputs), and `redux.sync` must agree
 * with an integer butterfly. One block; `report` starts zeroed.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_reduce = reduce_budget;

extern "C" __attribute__((global, launch_bounds(reduce_budget.threads, reduce_budget.blocks_per_sm))) void prims_check_reduce(check::Report* report) {
	using namespace engine::device::prims;
	unsigned const lane_index = lane();
	for (unsigned trial = 0; trial < reduce_trials; ++trial) {
		unsigned const seed = check::ring_word((warp_in_block() * reduce_trials + trial) * 32 + lane_index);
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

static_assert(prims::check(check::digest_budget) == prims::Fit::Fits);

/**
 * The transcendentals on the device: every fp32 input through each function, folded per block into
 * `digests[function * digest_blocks + block]` (which start zeroed). The host folds its own evaluation of the same
 * header over the same partition; equal digests mean the device computes the host's bits.
 */
extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_digest_transcendentals = check::digest_budget;

extern "C" __attribute__((global, launch_bounds(check::digest_budget.threads, check::digest_budget.blocks_per_sm))) void prims_digest_transcendentals(unsigned long long* digests) {
	using namespace engine::device::prims;
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

inline constexpr unsigned order_blocks = static_cast<unsigned>(prims::sm86.sms);
inline constexpr unsigned order_steps = 8;
using OrderProducers = prims::Warps<0, 2>;
using OrderConsumers = prims::Warps<2, 2>;
using Handoff = prims::NamedBarrier<1, 4>;

static_assert(OrderProducers::threads == OrderConsumers::threads, "each consumer thread reads one producer thread's chunk");

struct OrderLayout {
	alignas(8) unsigned long long barrier;
	prims::Chunk copied[32];
	prims::Chunk handed[OrderProducers::threads];
	prims::Chunk steps[order_steps][32];
};

inline constexpr prims::Budget order_budget = {.threads = 128, .registers_per_thread = 128, .shared_bytes = sizeof(OrderLayout), .blocks_per_sm = 1};
static_assert(prims::check(order_budget) == prims::Fit::Fits);
static_assert(prims::tiles<OrderProducers, OrderConsumers>(order_budget));

/** The global memory `prims_check_order` works in, zeroed by the host before the launch. */
struct OrderScratch {
	unsigned arrivals;
	unsigned counter;
	unsigned long long generation;
	unsigned long long gpu_flags[order_blocks];
	unsigned long long system_flags[order_blocks];
	prims::Chunk slots[order_blocks];
	prims::Chunk inbox[order_blocks];
	prims::Line scratch[order_blocks];
	alignas(16) unsigned words[order_blocks][32];
};

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_prims_check_order = order_budget;

/**
 * Memory order and the remaining primitives on the hardware: one block per SM (82, co-resident), each passing a value
 * to the next block through a step-1 grid barrier, a release counter, gpu- and system-scope epoch flags and the peer
 * push protocol (aimed at this card's own memory); within a block, producer warps hand shared data to consumer warps
 * on a named barrier, a double-buffered loop sums staged chunks, and the bf16 pack and widen instructions are held
 * to their host models. `report` and `scratch` start zeroed.
 */
extern "C" __attribute__((global, launch_bounds(order_budget.threads, order_budget.blocks_per_sm))) void prims_check_order(OrderScratch* scratch, check::Report* report) {
	using namespace engine::device::prims;
	auto& layout = shared_layout<order_budget, OrderLayout>();
	unsigned const me = block();
	unsigned const next = (me + 1) % order_blocks;
	unsigned const previous = (me + order_blocks - 1) % order_blocks;
	unsigned const thread = thread_in_block();
	unsigned const warp = warp_in_block();
	auto const reused = policy<Eviction::Last>();
	auto const once = policy<Eviction::First>();

	// Across SMs: each block's value, published four ways, must reach its neighbour.
	Chunk const mine = {.word = {me, me * 3U, me * 5U, me * 7U}};
	if (thread == 0) {
		store(&scratch->slots[me], mine, reused);
	}
	step1::GridBarrier<order_blocks>{.arrivals = &scratch->arrivals, .generation = &scratch->generation}.sync(remote_backoff);
	if (thread == 0) {
		Chunk const seen = load(&scratch->slots[next], reused);
		check::record(report, seen.word[0] == next && seen.word[3] == next * 7U, {.input = me, .detail = 1, .got = seen.word[0], .expected = next});

		release_add(&scratch->counter, 1);
		await_count(&scratch->counter, order_blocks, remote_backoff);
		check::record(report, load_acquire(&scratch->counter) == order_blocks, {.input = me, .detail = 2, .got = load_acquire(&scratch->counter), .expected = order_blocks});

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
		for (unsigned step = 0; step < order_steps; ++step) {
			store_shared(shared_address(&layout.steps[step][lane_index]), {.word = {step, lane_index, step * lane_index, 1U}});
		}
		sync_warp();
		unsigned pipelined = 0;
		double_buffered<static_cast<int>(order_steps)>(
		    [&](int step) { return load_shared(shared_address(&layout.steps[step][lane_index])); },
		    [&](int, Chunk const& chunk) { pipelined += chunk.word[2] + chunk.word[3]; });
		unsigned plain = 0;
		for (unsigned step = 0; step < order_steps; ++step) {
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
		unsigned const seed = check::ring_word((me * order_budget.threads + thread) * 64U + trial);
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

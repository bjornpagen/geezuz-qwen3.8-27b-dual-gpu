// SPDX-License-Identifier: AGPL-3.0-only
//
// The synchronisation microbenchmarks: the SM clock against the card's timer, an mbarrier round trip against a named
// barrier pair, the latency of a `cp.async` completion arriving on an mbarrier, the grain of `nanosleep`, the cross-SM
// hop through release counters, and the warp-specialised ring handoff with no data in it.
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/order.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "ubench/device.h"
#include "ubench/shapes.h"

namespace engine::device::ubench::sync {

__attribute__((device)) inline void clock(ClockParams params) {
	using namespace engine::device::prims;
	if (lane() != 0) {
		return;
	}
	unsigned long long const start_ns = nanoseconds();
	unsigned long long const start = cycles();
	unsigned long long now = start;
	while (now - start < calibration_cycles) {
		now = cycles();
	}
	at<Clocks>(params.clocks)[block()] = {.cycles = now - start, .nanoseconds = nanoseconds() - start_ns};
}

struct PingPongLayout {
	alignas(8) unsigned long long there;
	alignas(8) unsigned long long back;
};

/** Warp 0 arrives on `there` and waits for `back`; warp 1 the mirror image. Warp 0's lane 0 times the round trips. */
template<Wait wait>
__attribute__((device)) inline void mbarrier_pingpong(CyclesParams params) {
	using namespace engine::device::prims;
	constexpr Backoff backoff = backoff_for<wait>(block_backoff);
	auto& layout = shared_layout<pingpong_budget, PingPongLayout>();
	Mbarrier const there = {.address = shared_address(&layout.there)};
	Mbarrier const back = {.address = shared_address(&layout.back)};
	if (thread_in_block() == 0) {
		there.init(1);
		back.init(1);
	}
	sync_block();
	bool const leader = lane() == 0;
	if (warp_in_block() == 0) {
		unsigned long long const start = cycles();
		for (unsigned round = 0; round < static_cast<unsigned>(pingpong_iterations); ++round) {
			if (leader) {
				there.arrive();
			}
			back.wait(round & 1U, backoff);
		}
		if (leader) {
			*at<unsigned long long>(params.cycles) = cycles() - start;
		}
		return;
	}
	for (unsigned round = 0; round < static_cast<unsigned>(pingpong_iterations); ++round) {
		there.wait(round & 1U, backoff);
		sync_warp();
		if (leader) {
			back.arrive();
		}
	}
}

/** The same round trip on two named barriers: each warp signals with `bar.arrive` and waits with `bar.sync`. */
__attribute__((device)) inline void named_pingpong(CyclesParams params) {
	using namespace engine::device::prims;
	using There = NamedBarrier<1, 2>;
	using Back = NamedBarrier<2, 2>;
	if (warp_in_block() == 0) {
		unsigned long long const start = cycles();
		for (int round = 0; round < pingpong_iterations; ++round) {
			There::arrive();
			Back::sync();
		}
		if (lane() == 0) {
			*at<unsigned long long>(params.cycles) = cycles() - start;
		}
		return;
	}
	for (int round = 0; round < pingpong_iterations; ++round) {
		There::sync();
		Back::arrive();
	}
}

struct CopyLayout {
	alignas(8) unsigned long long landed;
	prims::Chunk destination;
};

/** One thread copies 16 l2-resident bytes and waits for the copy's own arrival, again and again. */
__attribute__((device)) inline void copy_completion(CopyParams params) {
	using namespace engine::device::prims;
	if (lane() != 0) {
		return;
	}
	auto& layout = shared_layout<copy_budget, CopyLayout>();
	Mbarrier const landed = {.address = shared_address(&layout.landed)};
	landed.init(1);
	Chunk const* const source = at<Chunk const>(params.source);
	auto const reused = policy<Eviction::Last>();
	constexpr Backoff spin = backoff_for<Wait::Spin>(block_backoff);
	unsigned long long const start = cycles();
	for (unsigned round = 0; round < static_cast<unsigned>(copy_iterations); ++round) {
		copy_async<Prefetch::None>(shared_address(&layout.destination), source + round % 32U, reused);
		landed.arrive_on_copies();
		landed.wait(round & 1U, spin);
	}
	*at<unsigned long long>(params.cycles) = cycles() - start;
}

__attribute__((device)) inline void sleep_grain(SleepParams params) {
	using namespace engine::device::prims;
	if (lane() != 0) {
		return;
	}
	SleepReading* const readings = at<SleepReading>(params.readings);
	for (int kind = 0; kind < sleep_kinds; ++kind) {
		SleepReading reading = {.total = 0, .least = ~0ULL, .most = 0};
		for (int sample = 0; sample < sleep_samples; ++sample) {
			unsigned long long const before = cycles();
			nanosleep(sleeps[kind]);
			unsigned long long const slept = cycles() - before;
			reading.total += slept;
			reading.least = slept < reading.least ? slept : reading.least;
			reading.most = slept > reading.most ? slept : reading.most;
		}
		readings[kind] = reading;
	}
}

/**
 * Block 0 publishes round r with a weak store and a release increment and waits for block 1's echo; block 1 waits,
 * reads, echoes. Each checks the value it reads against the round it waited for.
 */
template<Wait wait>
__attribute__((device)) inline void hop(HopParams params) {
	using namespace engine::device::prims;
	if (lane() != 0) {
		return;
	}
	constexpr Backoff backoff = backoff_for<wait>(remote_backoff);
	HopScratch* const scratch = at<HopScratch>(params.scratch);
	HopReading* const reading = at<HopReading>(params.reading);
	auto const reused = policy<Eviction::Last>();
	unsigned mismatches = 0;
	if (block() == 0) {
		reading->first_sm = sm_id();
		unsigned long long const start = cycles();
		for (unsigned round = 1; round <= static_cast<unsigned>(hop_iterations); ++round) {
			store_word(&scratch->forward_value[0], round, reused);
			release_add(&scratch->forward_count, 1);
			await_count(&scratch->backward_count, round, backoff);
			mismatches += load_word(&scratch->backward_value[0], reused) == round ? 0U : 1U;
		}
		reading->cycles = cycles() - start;
		reading->first_mismatches = mismatches;
		return;
	}
	reading->second_sm = sm_id();
	for (unsigned round = 1; round <= static_cast<unsigned>(hop_iterations); ++round) {
		await_count(&scratch->forward_count, round, backoff);
		unsigned const seen = load_word(&scratch->forward_value[0], reused);
		mismatches += seen == round ? 0U : 1U;
		store_word(&scratch->backward_value[0], seen, reused);
		release_add(&scratch->backward_count, 1);
	}
	reading->second_mismatches = mismatches;
}

/** One producer warp signals each stage of a 4-stage ring full; two consumer warpgroups wait for it and free it. */
__attribute__((device)) inline void stage_handoff(CyclesParams params) {
	using namespace engine::device::prims;
	using Ring = StageRing<4, 1, handoff_consumer_warps>;
	using Producer = Warps<0, 1>;
	using Consumers = Warps<1, handoff_consumer_warps>;
	static_assert(tiles<Producer, Consumers>(handoff_budget));
	auto& storage = shared_layout<handoff_budget, Ring::Storage>();
	Ring const ring = Ring::at(storage);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	Ring::Cursor cursor = Ring::Cursor::start();
	if (Producer::holds(warp_in_block())) {
		if (lane() != 0) {
			return;
		}
		unsigned long long const start = cycles();
		for (int stage = 0; stage < handoff_iterations; ++stage) {
			ring.await_free(cursor, block_backoff);
			ring.full(cursor).arrive();
			cursor = cursor.next();
		}
		*at<unsigned long long>(params.cycles) = cycles() - start;
		return;
	}
	for (int stage = 0; stage < handoff_iterations; ++stage) {
		ring.await_filled(cursor, block_backoff);
		sync_warp();
		if (lane() == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();
	}
}

}

namespace ub = engine::device::ubench;
namespace sync = engine::device::ubench::sync;

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_clock = sync::clock_budget;
extern "C" __attribute__((global, launch_bounds(sync::clock_budget.threads, sync::clock_budget.blocks_per_sm))) void ubench_clock(sync::ClockParams params) {
	sync::clock(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_mbarrier_backoff = sync::pingpong_budget;
extern "C" __attribute__((global, launch_bounds(sync::pingpong_budget.threads, sync::pingpong_budget.blocks_per_sm))) void ubench_mbarrier_backoff(sync::CyclesParams params) {
	sync::mbarrier_pingpong<ub::Wait::Backoff>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_mbarrier_spin = sync::pingpong_budget;
extern "C" __attribute__((global, launch_bounds(sync::pingpong_budget.threads, sync::pingpong_budget.blocks_per_sm))) void ubench_mbarrier_spin(sync::CyclesParams params) {
	sync::mbarrier_pingpong<ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_named_barrier = sync::pingpong_budget;
extern "C" __attribute__((global, launch_bounds(sync::pingpong_budget.threads, sync::pingpong_budget.blocks_per_sm))) void ubench_named_barrier(sync::CyclesParams params) {
	sync::named_pingpong(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_copy_completion = sync::copy_budget;
extern "C" __attribute__((global, launch_bounds(sync::copy_budget.threads, sync::copy_budget.blocks_per_sm))) void ubench_copy_completion(sync::CopyParams params) {
	sync::copy_completion(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_sleep_grain = sync::sleep_budget;
extern "C" __attribute__((global, launch_bounds(sync::sleep_budget.threads, sync::sleep_budget.blocks_per_sm))) void ubench_sleep_grain(sync::SleepParams params) {
	sync::sleep_grain(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_hop_backoff = sync::hop_budget;
extern "C" __attribute__((global, launch_bounds(sync::hop_budget.threads, sync::hop_budget.blocks_per_sm))) void ubench_hop_backoff(sync::HopParams params) {
	sync::hop<ub::Wait::Backoff>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_hop_spin = sync::hop_budget;
extern "C" __attribute__((global, launch_bounds(sync::hop_budget.threads, sync::hop_budget.blocks_per_sm))) void ubench_hop_spin(sync::HopParams params) {
	sync::hop<ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stage_handoff = sync::handoff_budget;
extern "C" __attribute__((global, launch_bounds(sync::handoff_budget.threads, sync::handoff_budget.blocks_per_sm))) void ubench_stage_handoff(sync::CyclesParams params) {
	sync::stage_handoff(params);
}

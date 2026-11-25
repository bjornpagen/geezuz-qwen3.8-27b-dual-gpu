// SPDX-License-Identifier: AGPL-3.0-only
//
// Synchronisation inside a block: named barriers for fixed sub-groups, mbarriers with phase-parity waits, and the
// multi-stage ring they make. sm_86 has no `mbarrier.try_wait` and no transaction counts, so every wait is a
// `test_wait.parity` spin with `nanosleep` backoff and copies are counted in arrivals.
#pragma once

#include "prims/isa.h"
#include "prims/stage_ring.h"
#include "prims/shared.h"

namespace engine::device::prims {

static_assert(available(isa::nanosleep));
static_assert(available(isa::named_barrier) && available(isa::warp_sync));
static_assert(available(isa::mbarrier) && available(isa::mbarrier_parity));

/** Sleeps about `nanoseconds` (the hardware picks within [0, 2t]); a waiting warp's issue slots go to the others. */
__attribute__((device)) inline void nanosleep(unsigned nanoseconds) {
	// Why ptx: `nanosleep` has no ungated builtin at ptx 9.3.
	asm volatile("nanosleep.u32 %0;" : : "r"(nanoseconds) : "memory");
}

/** How a spin backs off: the first sleep, doubled after each failed test up to the ceiling. */
struct Backoff {
	unsigned first_ns;
	unsigned ceiling_ns;
};

/** The default for waits inside a block: a stage is due within a microsecond or so. */
inline constexpr Backoff block_backoff = {.first_ns = 32, .ceiling_ns = 256};

/**
 * A named barrier for a fixed sub-group: barrier `id` (1-15; 0 is the whole block's) among `warps` warps, whose
 * threads all execute `sync`, or `arrive` to signal without waiting.
 */
template<int id_, int warps_>
struct NamedBarrier {
	static constexpr int id = id_;
	static constexpr int threads = warps_ * 32;
	static_assert(id >= 1 && id < 16, "ids 1-15 name sub-group barriers; 0 is the block's");
	static_assert(warps_ >= 1 && threads <= 1024);

	__attribute__((device)) static void sync() {
		// Why ptx: the barrier builtins are gated at ptx 9.3, and only `bar.sync` takes a thread count.
		asm volatile("bar.sync %0, %1;" : : "n"(id), "n"(threads) : "memory");
	}

	__attribute__((device)) static void arrive() {
		asm volatile("bar.arrive %0, %1;" : : "n"(id), "n"(threads) : "memory");
	}
};

/** Reconverges the warp and orders its shared-memory accesses before and after. */
__attribute__((device)) inline void sync_warp() {
	asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
}

/** The whole block's barrier: for setup and teardown only, never in a steady-state loop. */
__attribute__((device)) inline void sync_block() {
	asm volatile("bar.sync 0;" : : : "memory");
}

/** An mbarrier: an 8-byte object in shared memory, addressed by its shared-window address. */
struct Mbarrier {
	SharedAddress address;

	/** Arms the barrier to complete each phase after `arrivals` arrivals; one thread, before any use. */
	__attribute__((device)) void init(unsigned arrivals) const {
		asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(address.value), "r"(arrivals) : "memory");
	}

	/** Releases the barrier's memory for reuse; one thread, after every waiter is done with it. */
	__attribute__((device)) void invalidate() const {
		asm volatile("mbarrier.inval.shared.b64 [%0];" : : "r"(address.value) : "memory");
	}

	/** One arrival, with release semantics at block scope; the returned state is not needed by parity waits. */
	__attribute__((device)) void arrive() const {
		asm volatile("mbarrier.arrive.shared.b64 _, [%0];" : : "r"(address.value) : "memory");
	}

	/**
	 * An arrival made by the completion of every `cp.async` this thread issued before it. Counted against the init
	 * count (`.noinc`), so the thread itself never waits for its copies.
	 */
	__attribute__((device)) void arrive_on_copies() const {
		asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(address.value) : "memory");
	}

	/** Whether the phase with parity `parity` has completed; acquire semantics at block scope when it has. */
	[[nodiscard]] __attribute__((device)) auto test(unsigned parity) const -> bool {
		unsigned done = 0;
		asm volatile(
		    "{\n\t.reg .pred pdone;\n\t"
		    "mbarrier.test_wait.parity.shared.b64 pdone, [%1], %2;\n\t"
		    "selp.u32 %0, 1, 0, pdone;\n\t}"
		    : "=r"(done)
		    : "r"(address.value), "r"(parity)
		    : "memory");
		return done != 0;
	}

	/** Spins until the phase with parity `parity` has completed. */
	__attribute__((device)) void wait(unsigned parity, Backoff backoff) const {
		unsigned sleep = backoff.first_ns;
		while (!test(parity)) {
			nanosleep(sleep);
			sleep = sleep * 2U < backoff.ceiling_ns ? sleep * 2U : backoff.ceiling_ns;
		}
	}
};

/**
 * A ring of `stages` stages in shared memory, each guarded by a full and an empty mbarrier: the full barrier
 * completes after `fill_arrivals` arrivals (one `arrive_on_copies` per producer thread per fill), the empty barrier
 * after `drain_arrivals` (one `arrive` per consumer, thread or warp as the kernel decides).
 */
template<int stages_, unsigned fill_arrivals_, unsigned drain_arrivals_>
struct StageRing {
	static constexpr int stages = stages_;
	static constexpr unsigned fill_arrivals = fill_arrivals_;
	static constexpr unsigned drain_arrivals = drain_arrivals_;
	using Cursor = StageCursor<stages>;

	/** The barriers' storage, a member of the kernel's shared layout. */
	struct Storage {
		alignas(8) unsigned long long full[stages];
		alignas(8) unsigned long long empty[stages];
	};

	SharedAddress full_base;
	SharedAddress empty_base;

	[[nodiscard]] __attribute__((device)) static auto at(Storage& storage) -> StageRing {
		return {.full_base = shared_address(storage.full), .empty_base = shared_address(storage.empty)};
	}

	[[nodiscard]] __attribute__((device)) auto full(Cursor cursor) const -> Mbarrier {
		return {.address = full_base.plus(8U * cursor.stage)};
	}

	[[nodiscard]] __attribute__((device)) auto empty(Cursor cursor) const -> Mbarrier {
		return {.address = empty_base.plus(8U * cursor.stage)};
	}

	/** Arms every barrier; one thread, followed by a block-wide barrier before any other use. */
	__attribute__((device)) void init() const {
		for (unsigned stage = 0; stage < static_cast<unsigned>(stages); ++stage) {
			Mbarrier{.address = full_base.plus(8U * stage)}.init(fill_arrivals);
			Mbarrier{.address = empty_base.plus(8U * stage)}.init(drain_arrivals);
		}
	}

	/** The producer: waits until the stage is free to refill. */
	__attribute__((device)) void await_free(Cursor cursor, Backoff backoff) const {
		empty(cursor).wait(cursor.empty_parity(), backoff);
	}

	/** A consumer: waits until the stage's fill has landed. */
	__attribute__((device)) void await_filled(Cursor cursor, Backoff backoff) const {
		full(cursor).wait(cursor.full_parity(), backoff);
	}

	/** The producer thread's arrival: completes when the copies it issued for this stage land. */
	__attribute__((device)) void filled_on_copies(Cursor cursor) const {
		full(cursor).arrive_on_copies();
	}

	/** A consumer is done reading the stage. */
	__attribute__((device)) void drained(Cursor cursor) const {
		empty(cursor).arrive();
	}
};

}

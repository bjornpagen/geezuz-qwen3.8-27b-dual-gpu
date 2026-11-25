// SPDX-License-Identifier: AGPL-3.0-only
//
// Memory order across SMs and across the two cards. Across SMs of one card: weak stores, then a release (a store or
// a counter increment at gpu scope); the consumer polls with acquire loads and `nanosleep` backoff. Across cards:
// push, never pull. A card stores weakly into its own receive slots in the peer's memory and then releases one
// single-writer epoch flag at system scope; the peer polls its own memory. One release per item, never per element;
// no peer atomics.
#pragma once

#include "prims/isa.h"
#include "prims/memory.h"
#include "prims/sync.h"

namespace engine::device::prims {

static_assert(available(isa::scoped_memory_order));

/** Who must observe the ordering: the SMs of this card, or the whole system (the peer card and the host). */
enum class Scope : unsigned char { Gpu, System };

/** An acquire load of a 64-bit word at `scope`: later loads see everything released before the value was stored. */
template<Scope scope>
[[nodiscard]] __attribute__((device)) inline auto load_acquire(unsigned long long const* word) -> unsigned long long {
	unsigned long long value = 0;
	// Why ptx: scoped acquire and release exist only as ptx qualifiers.
	if constexpr (scope == Scope::Gpu) {
		asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
	} else {
		asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
	}
	return value;
}

/** A release store of a 64-bit word at `scope`: everything this thread stored before is visible to an acquirer. */
template<Scope scope>
__attribute__((device)) inline void store_release(unsigned long long* word, unsigned long long value) {
	if constexpr (scope == Scope::Gpu) {
		asm volatile("st.release.gpu.global.u64 [%0], %1;" : : "l"(word), "l"(value) : "memory");
	} else {
		asm volatile("st.release.sys.global.u64 [%0], %1;" : : "l"(word), "l"(value) : "memory");
	}
}

/** A release increment of a counter on this card: many producers, each publishing its own work. */
__attribute__((device)) inline void release_add(unsigned* counter, unsigned amount) {
	asm volatile("red.release.gpu.global.add.u32 [%0], %1;" : : "l"(counter), "r"(amount) : "memory");
}

/** Adds to a counter on this card and returns its previous value; relaxed: it orders nothing else. */
[[nodiscard]] __attribute__((device)) inline auto fetch_add(unsigned* counter, unsigned amount) -> unsigned {
	unsigned before = 0;
	asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(before) : "l"(counter), "r"(amount) : "memory");
	return before;
}

/** Adds to a 32-bit count on this card; relaxed, and commutative, so the order of the adds cannot show. */
__attribute__((device)) inline void count(unsigned* total, unsigned amount) {
	asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" : : "l"(total), "r"(amount) : "memory");
}

/** Adds to a 64-bit total on this card; relaxed, and commutative, so the order of the adds cannot show. */
__attribute__((device)) inline void accumulate(unsigned long long* total, unsigned long long amount) {
	asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" : : "l"(total), "l"(amount) : "memory");
}

/** An acquire load of a 32-bit counter on this card. */
[[nodiscard]] __attribute__((device)) inline auto load_acquire(unsigned const* counter) -> unsigned {
	unsigned value = 0;
	asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(counter) : "memory");
	return value;
}

/** The default for waits across SMs and cards: a hop costs several hundred nanoseconds to microseconds. */
inline constexpr Backoff remote_backoff = {.first_ns = 64, .ceiling_ns = 1024};

/** Waits until a counter on this card reaches `target`. */
__attribute__((device)) inline void await_count(unsigned const* counter, unsigned target, Backoff backoff) {
	unsigned sleep = backoff.first_ns;
	while (load_acquire(counter) < target) {
		nanosleep(sleep);
		sleep = sleep * 2U < backoff.ceiling_ns ? sleep * 2U : backoff.ceiling_ns;
	}
}

/**
 * The writing end of a single-writer flag at `scope`: epochs only grow (64 bits never wrap), so one release store
 * of the item's epoch publishes everything stored before it.
 */
template<Scope scope>
struct EpochWriter {
	unsigned long long* flag;

	__attribute__((device)) void publish(unsigned long long epoch) const {
		store_release<scope>(flag, epoch);
	}
};

/** The reading end of a single-writer flag: the reader's own memory, polled with acquire loads. */
template<Scope scope>
struct EpochReader {
	unsigned long long const* flag;

	/** Waits until the writer has published `epoch` or later. */
	__attribute__((device)) void await(unsigned long long epoch, Backoff backoff) const {
		unsigned sleep = backoff.first_ns;
		while (load_acquire<scope>(flag) < epoch) {
			nanosleep(sleep);
			sleep = sleep * 2U < backoff.ceiling_ns ? sleep * 2U : backoff.ceiling_ns;
		}
	}
};

/**
 * This card's receive slots in the peer's memory and the peer's flag for them. It can only be stored to: the peer
 * never pulls, and this card never reads across the link.
 */
struct PeerOutbox {
	Chunk* slots;
	EpochWriter<Scope::System> flag;

	/** A weak store of one chunk into slot `index`; ordered for the peer by the next `publish`. */
	__attribute__((device)) void push(unsigned index, Chunk value) const {
		asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
		             :
		             : "l"(slots + index), "r"(value.word[0]), "r"(value.word[1]), "r"(value.word[2]), "r"(value.word[3])
		             : "memory");
	}

	/** Releases every chunk this thread pushed, as item `epoch`. */
	__attribute__((device)) void publish(unsigned long long epoch) const {
		flag.publish(epoch);
	}
};

/** The peer's receive slots in this card's memory, read after the peer's flag says an item arrived. */
struct PeerInbox {
	Chunk const* slots;
	EpochReader<Scope::System> flag;

	__attribute__((device)) void await(unsigned long long epoch, Backoff backoff) const {
		flag.await(epoch, backoff);
	}

	/** Slot `index`, read once: evict-first. */
	[[nodiscard]] __attribute__((device)) auto read(unsigned index, Policy<Eviction::First> policy) const -> Chunk {
		return load(slots + index, policy);
	}
};

// The step-1 baseline keeps upstream's grid barriers so it stays comparable; the dataflow core of step 2 replaces
// them with per-tile counters and deletes this namespace.
namespace step1 {

/**
 * A grid-wide barrier over `blocks` co-resident blocks: a counter of arrivals and a generation the last arriver
 * advances. Every thread of every block calls `sync`.
 */
template<unsigned blocks>
struct GridBarrier {
	unsigned* arrivals;
	unsigned long long* generation;

	__attribute__((device)) void sync(Backoff backoff) const {
		sync_block();
		if (__nvvm_read_ptx_sreg_tid_x() == 0) {
			unsigned long long const seen = load_acquire<Scope::Gpu>(generation);
			unsigned before = 0;
			asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(before) : "l"(arrivals) : "memory");
			if (before == blocks - 1) {
				asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" : : "l"(arrivals), "r"(0U) : "memory");
				store_release<Scope::Gpu>(generation, seen + 1);
			} else {
				unsigned sleep = backoff.first_ns;
				while (load_acquire<Scope::Gpu>(generation) == seen) {
					nanosleep(sleep);
					sleep = sleep * 2U < backoff.ceiling_ns ? sleep * 2U : backoff.ceiling_ns;
				}
			}
		}
		sync_block();
	}
};

}

}

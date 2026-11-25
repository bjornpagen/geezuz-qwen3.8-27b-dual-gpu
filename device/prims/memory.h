// SPDX-License-Identifier: AGPL-3.0-only
//
// Global memory under an explicit l2 policy. Every load, store and async copy of global memory a kernel issues
// through this header carries a `createpolicy` cache policy: single-use streams (weights, kv, deltanet state) evict
// first, what is reused within microseconds (activations, partials, flags) evicts last. There is no policy-free
// global load here to reach for, and the weight stream holds its evict-first policy in its type.
#pragma once

#include "prims/chunk.h"
#include "prims/isa.h"
#include "prims/shared.h"

namespace engine::device::prims {

static_assert(available(isa::cache_policy) && available(isa::l2_prefetch_size) && available(isa::l2_discard));
static_assert(available(isa::cp_async));

/** Which l2 eviction priority a policy gives every access it carries. */
enum class Eviction : unsigned char { First, Last, Normal };

/** An l2 cache policy made by `createpolicy`, typed by its priority. */
template<Eviction eviction_>
struct Policy {
	static constexpr Eviction eviction = eviction_;
	unsigned long long bits;
};

/** A policy giving every access it carries the priority `eviction` (fraction 1). Hoist it out of loops. */
template<Eviction eviction>
[[nodiscard]] __attribute__((device)) inline auto policy() -> Policy<eviction> {
	unsigned long long bits = 0;
	// Why ptx: a cache policy exists only as `createpolicy`'s opaque 64-bit result.
	if constexpr (eviction == Eviction::First) {
		asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(bits));
	} else if constexpr (eviction == Eviction::Last) {
		asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(bits));
	} else {
		asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(bits));
	}
	return {.bits = bits};
}

/** Whether an async copy also asks l2 to fetch the whole 256-byte dram chunk around it. */
enum class Prefetch : unsigned char { None, Chunk256 };

/**
 * Copies 16 bytes from global to shared memory asynchronously, through l2 only (`.cg`), under `policy`. Completion
 * is observed by `Mbarrier::arrive_on_copies` or a copy group.
 */
template<Prefetch prefetch, Eviction eviction>
__attribute__((device)) inline void copy_async(SharedAddress destination, Chunk const* source, Policy<eviction> policy) {
	// Why ptx: `cp.async` and its cache hints have no builtin at ptx 9.3.
	if constexpr (prefetch == Prefetch::Chunk256) {
		asm volatile(
		    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %1;\n\t"
		    "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [gaddr], 16, %2;\n\t}"
		    :
		    : "r"(destination.value), "l"(source), "l"(policy.bits)
		    : "memory");
	} else {
		asm volatile(
		    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %1;\n\t"
		    "cp.async.cg.shared.global.L2::cache_hint [%0], [gaddr], 16, %2;\n\t}"
		    :
		    : "r"(destination.value), "l"(source), "l"(policy.bits)
		    : "memory");
	}
}

/** Closes this thread's current group of async copies. */
__attribute__((device)) inline void commit_copies() {
	asm volatile("cp.async.commit_group;" : : : "memory");
}

/** Waits until at most `pending` of this thread's committed copy groups are still in flight. */
template<int pending>
__attribute__((device)) inline void wait_copies() {
	static_assert(pending >= 0 && pending <= 7);
	asm volatile("cp.async.wait_group %0;" : : "n"(pending) : "memory");
}

/** Waits until every async copy this thread issued has landed, grouped or not; a producer's last act. */
__attribute__((device)) inline void wait_all_copies() {
	asm volatile("cp.async.wait_all;" : : : "memory");
}

/**
 * The weight stream as one producer thread sees it: the baked image is the exact sequence of 16-byte transactions
 * the producers issue, so issuing is a copy and a pointer bump. `producer_threads` threads interleave chunk by
 * chunk; the stream always evicts first and prefetches whole dram chunks.
 */
template<int producer_threads>
struct WeightStream {
	Chunk const* next;
	Policy<Eviction::First> policy;

	/** Copies this thread's next chunk to `destination`, then steps past the other producers' chunks. */
	__attribute__((device)) void issue(SharedAddress destination) {
		copy_async<Prefetch::Chunk256>(destination, next, policy);
		next += producer_threads;
	}
};

/** 16 bytes of data that stays read-only for the kernel's lifetime, kept out of l1 (single use). */
template<Eviction eviction>
[[nodiscard]] __attribute__((device)) inline auto load_constant(Chunk const* source, Policy<eviction> policy) -> Chunk {
	Chunk loaded{};
	// Why ptx: the non-coherent path with l1 bypass and an l2 policy is only spelled in ptx.
	asm volatile(
	    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %4;\n\t"
	    "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [gaddr], %5;\n\t}"
	    : "=r"(loaded.word[0]), "=r"(loaded.word[1]), "=r"(loaded.word[2]), "=r"(loaded.word[3])
	    : "l"(source), "l"(policy.bits)
	    : "memory");
	return loaded;
}

/** 16 bytes another thread of the kernel may have written (ordered by an acquire), under `policy`. */
template<Eviction eviction>
[[nodiscard]] __attribute__((device)) inline auto load(Chunk const* source, Policy<eviction> policy) -> Chunk {
	Chunk loaded{};
	asm volatile(
	    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %4;\n\t"
	    "ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [gaddr], %5;\n\t}"
	    : "=r"(loaded.word[0]), "=r"(loaded.word[1]), "=r"(loaded.word[2]), "=r"(loaded.word[3])
	    : "l"(source), "l"(policy.bits)
	    : "memory");
	return loaded;
}

/** One 32-bit word another thread of the kernel may have written, under `policy`. */
template<Eviction eviction>
[[nodiscard]] __attribute__((device)) inline auto load_word(unsigned const* source, Policy<eviction> policy) -> unsigned {
	unsigned loaded = 0;
	asm volatile(
	    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %1;\n\t"
	    "ld.global.L2::cache_hint.u32 %0, [gaddr], %2;\n\t}"
	    : "=r"(loaded)
	    : "l"(source), "l"(policy.bits)
	    : "memory");
	return loaded;
}

/** Stores 16 bytes under `policy`. */
template<Eviction eviction>
__attribute__((device)) inline void store(Chunk* destination, Chunk value, Policy<eviction> policy) {
	asm volatile(
	    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %0;\n\t"
	    "st.global.L2::cache_hint.v4.u32 [gaddr], {%1, %2, %3, %4}, %5;\n\t}"
	    :
	    : "l"(destination), "r"(value.word[0]), "r"(value.word[1]), "r"(value.word[2]), "r"(value.word[3]), "l"(policy.bits)
	    : "memory");
}

/** Stores one 32-bit word under `policy`. */
template<Eviction eviction>
__attribute__((device)) inline void store_word(unsigned* destination, unsigned value, Policy<eviction> policy) {
	asm volatile(
	    "{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %0;\n\t"
	    "st.global.L2::cache_hint.u32 [gaddr], %1, %2;\n\t}"
	    :
	    : "l"(destination), "r"(value), "l"(policy.bits)
	    : "memory");
}

/** Brings the line holding `address` into l2 at evict-last priority: run-ahead past the shared-memory ring. */
__attribute__((device)) inline void prefetch_to_l2(void const* address) {
	asm volatile("{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %0;\n\tprefetch.global.L2::evict_last [gaddr];\n\t}" : : "l"(address) : "memory");
}

/** Returns a line fetched at evict-last priority to normal priority once it has been used. */
__attribute__((device)) inline void release_priority(Line const* line) {
	asm volatile("{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %0;\n\tapplypriority.global.L2::evict_normal [gaddr], 128;\n\t}" : : "l"(line) : "memory");
}

/** Drops a dead scratch line from l2 without writing it back; its contents become indeterminate. */
__attribute__((device)) inline void discard(Line* line) {
	asm volatile("{\n\t.reg .u64 gaddr;\n\tcvta.to.global.u64 gaddr, %0;\n\tdiscard.global.L2 [gaddr], 128;\n\t}" : : "l"(line) : "memory");
}

}

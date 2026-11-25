// SPDX-License-Identifier: AGPL-3.0-only
//
// The soak's on-device integrity checks: the SHA-256 of a card's whole weight image where it lies in memory, absorbed
// a slice per launch and finished by its own launch, and a pattern written over the memory the engine writes while it
// serves, then read back and counted. The extern "C" wrappers name them for the host.
#include "prims/budget.h"
#include "prims/chunk.h"
#include "prims/memory.h"
#include "prims/reduce.h"
#include "prims/thread.h"
#include "soak/digest.h"
#include "soak/layout.h"

namespace engine::device::soak {

using prims::Chunk;
using prims::Eviction;

/** The object a parameter's device address names. */
template<class T>
[[nodiscard]] __attribute__((device)) inline auto at(Address address) -> T* {
	/* SAFETY: the host passes the device address of an allocation laid out as T, by the typed launch that built it. */
	return __builtin_bit_cast(T*, address);
}

/** A hash state as the three chunks it is moved in. */
struct StateChunks {
	Chunk part[3];
};

static_assert(sizeof(StateChunks) == sizeof(HashState));

// The state lives in host memory both ends map, read once and written once per launch: three chunks over pcie, beside a
// slice that takes a second and a half.
[[nodiscard]] __attribute__((device)) inline auto load_state(Address state) -> HashState {
	auto const reused = prims::policy<Eviction::Last>();
	Chunk const* const chunks = at<Chunk const>(state);
	StateChunks loaded{};
	for (int part = 0; part < 3; ++part) {
		loaded.part[part] = prims::load(chunks + part, reused);
	}
	return __builtin_bit_cast(HashState, loaded);
}

__attribute__((device)) inline void store_state(Address state, HashState const& value) {
	auto const reused = prims::policy<Eviction::Last>();
	Chunk* const chunks = at<Chunk>(state);
	StateChunks const stored = __builtin_bit_cast(StateChunks, value);
	for (int part = 0; part < 3; ++part) {
		prims::store(chunks + part, stored.part[part], reused);
	}
}

/** The weight image's chunks as the hash reads them: once each, evicted first, kept out of l1. */
struct ImageChunks {
	Chunk const* source;
	prims::Policy<Eviction::First> once;

	[[nodiscard]] __attribute__((device)) auto operator()(unsigned long long index) const -> Chunk {
		return prims::load_constant(source + index, once);
	}
};

struct ImageWords {
	unsigned const* source;
	prims::Policy<Eviction::First> once;

	[[nodiscard]] __attribute__((device)) auto operator()(unsigned index) const -> unsigned {
		return prims::load_word(source + index, once);
	}
};

[[gnu::always_inline]] __attribute__((device)) inline void absorb_slice(AbsorbParams params) {
	if (prims::thread_in_block() != 0U) {
		return;
	}
	HashState hash = load_state(params.state);
	absorb(hash, ImageChunks{.source = at<Chunk const>(params.source), .once = prims::policy<Eviction::First>()}, params.blocks);
	store_state(params.state, hash);
}

[[gnu::always_inline]] __attribute__((device)) inline void finish_digest(FinishParams params) {
	if (prims::thread_in_block() != 0U) {
		return;
	}
	HashState hash = load_state(params.state);
	finish(hash, ImageWords{.source = at<unsigned const>(params.tail), .once = prims::policy<Eviction::First>()}, params.tail_words,
	       params.total_bytes);
	store_state(params.state, hash);
}

// Each step a thread covers `pattern_unroll` chunks a grid's width apart, so every warp instruction touches 512
// contiguous bytes and the step's loads are independent of each other.
inline constexpr unsigned long long grid_threads = static_cast<unsigned long long>(pattern_blocks) * pattern_budget.threads;
inline constexpr unsigned long long step_chunks = grid_threads * pattern_unroll;

[[nodiscard]] __attribute__((device)) inline auto first_chunk(unsigned long long step) -> unsigned long long {
	return step * step_chunks + static_cast<unsigned long long>(prims::block()) * pattern_budget.threads + prims::thread_in_block();
}

[[nodiscard]] __attribute__((device)) inline auto expected(unsigned round, unsigned chunk) -> Chunk {
	return Chunk{{pattern_word(round, chunk, 0), pattern_word(round, chunk, 1), pattern_word(round, chunk, 2), pattern_word(round, chunk, 3)}};
}

__attribute__((device)) inline void write_pattern(WriteParams params) {
	Chunk* const region = at<Chunk>(params.region);
	auto const once = prims::policy<Eviction::First>();
	for (unsigned long long step = 0; step * step_chunks < region_chunks; ++step) {
		unsigned long long const first = first_chunk(step);
		for (int part = 0; part < pattern_unroll; ++part) {
			unsigned long long const chunk = first + static_cast<unsigned long long>(part) * grid_threads;
			if (chunk < region_chunks) {
				prims::store(region + chunk, expected(params.round, static_cast<unsigned>(chunk)), once);
			}
		}
	}
}

__attribute__((device)) inline void check_pattern(CheckParams params) {
	Chunk const* const region = at<Chunk const>(params.region);
	auto const once = prims::policy<Eviction::First>();
	unsigned mismatched = 0;
	unsigned first_bad = 0xffff'ffffU;
	for (unsigned long long step = 0; step * step_chunks < region_chunks; ++step) {
		unsigned long long const first = first_chunk(step);
		Chunk read[pattern_unroll];
		for (int part = 0; part < pattern_unroll; ++part) {
			unsigned long long const chunk = first + static_cast<unsigned long long>(part) * grid_threads;
			read[part] = chunk < region_chunks ? prims::load_constant(region + chunk, once) : Chunk{};
		}
		for (int part = 0; part < pattern_unroll; ++part) {
			unsigned long long const chunk = first + static_cast<unsigned long long>(part) * grid_threads;
			if (chunk >= region_chunks) {
				continue;
			}
			Chunk const want = expected(params.round, static_cast<unsigned>(chunk));
			unsigned differing = 0;
			for (int word = 0; word < 4; ++word) {
				differing += read[part].word[word] != want.word[word] ? 1U : 0U;
			}
			mismatched += differing;
			first_bad = differing != 0U && chunk < first_bad ? static_cast<unsigned>(chunk) : first_bad;
		}
	}
	unsigned const total = prims::warp_sum(mismatched);
	unsigned const lowest = prims::warp_min(first_bad);
	if (prims::lane() == 0U) {
		unsigned long long const warp = static_cast<unsigned long long>(prims::block()) * pattern_warps + prims::warp_in_block();
		Tally* const tally = at<Tally>(params.tallies) + warp;
		auto const reused = prims::policy<Eviction::Last>();
		prims::store_word(&tally->mismatched_words, total, reused);
		prims::store_word(&tally->first_chunk, lowest, reused);
	}
}

}

namespace soak = engine::device::soak;

extern "C" __attribute__((device, used)) constexpr soak::Budget engine_budget_soak_absorb = soak::hash_budget;
extern "C" __attribute__((global, launch_bounds(soak::hash_budget.threads, soak::hash_budget.blocks_per_sm))) void soak_absorb(soak::AbsorbParams params) {
	soak::absorb_slice(params);
}

extern "C" __attribute__((device, used)) constexpr soak::Budget engine_budget_soak_finish = soak::hash_budget;
extern "C" __attribute__((global, launch_bounds(soak::hash_budget.threads, soak::hash_budget.blocks_per_sm))) void soak_finish(soak::FinishParams params) {
	soak::finish_digest(params);
}

extern "C" __attribute__((device, used)) constexpr soak::Budget engine_budget_soak_write = soak::pattern_budget;
extern "C" __attribute__((global, launch_bounds(soak::pattern_budget.threads, soak::pattern_budget.blocks_per_sm))) void soak_write(soak::WriteParams params) {
	soak::write_pattern(params);
}

extern "C" __attribute__((device, used)) constexpr soak::Budget engine_budget_soak_check = soak::pattern_budget;
extern "C" __attribute__((global, launch_bounds(soak::pattern_budget.threads, soak::pattern_budget.blocks_per_sm))) void soak_check(soak::CheckParams params) {
	soak::check_pattern(params);
}

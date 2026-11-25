// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight-stream microbenchmarks: the pure read stream over producer warps and in-flight depth, the same stream in
// row-major tile order, the ring under consumer stalls, a decode-like load for the clock under power, activation reuse
// under the stream's l2 priorities, and the evict-last prefetch and probe that measure what l2 carries past the
// stream. Every kernel is one sweep point; the extern "C" wrappers name them for the host.
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "ubench/device.h"
#include "ubench/shapes.h"

namespace engine::device::ubench::stream {

/** Where chunk `index` of a block's run lies, in chunks from the run's start, for each tile order. */
template<TileOrder order>
[[nodiscard]] __attribute__((device)) inline auto placed(unsigned long long index) -> unsigned long long {
	if constexpr (order == TileOrder::Contiguous) {
		return index;
	} else {
		unsigned long long const tile = index / tile_chunks;
		unsigned long long const within = index % tile_chunks;
		unsigned long long const row = within / 8;
		unsigned long long const part = within % 8;
		return tile / slab_tiles * slab_chunks + row * (row_pitch / 16) + tile % slab_tiles * 8 + part;
	}
}

template<class Shape>
__attribute__((device)) inline void read(ReadParams params) {
	using namespace engine::device::prims;
	using Ring = StageRing<Shape::stages, Shape::producer_threads, Shape::consumer_warps>;
	using Producers = Warps<0, Shape::producer_warps>;
	using Consumers = Warps<Shape::producer_warps, Shape::consumer_warps>;
	static_assert(tiles<Producers, Consumers>(Shape::budget));
	struct Layout {
		typename Ring::Storage barriers;
		Chunk stage[Shape::stages][Shape::stage_chunks];
	};
	auto& layout = shared_layout<Shape::budget, Layout>();
	Ring const ring = Ring::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	typename Ring::Cursor cursor = Ring::Cursor::start();
	if (Producers::holds(warp)) {
		Chunk const* const run = at<Chunk const>(params.source) + block() * block_chunks;
		auto const once = policy<Eviction::First>();
		unsigned const thread = Producers::rank(warp) * 32U + lane();
		for (int stage = 0; stage < Shape::stages_per_block; ++stage) {
			ring.await_free(cursor, block_backoff);
			for (unsigned chunk = thread; chunk < static_cast<unsigned>(Shape::stage_chunks); chunk += Shape::producer_threads) {
				unsigned long long const index = static_cast<unsigned long long>(stage) * Shape::stage_chunks + chunk;
				copy_async<Prefetch::Chunk256>(shared_address(&layout.stage[cursor.stage][chunk]), run + placed<Shape::order>(index), once);
			}
			ring.filled_on_copies(cursor);
			cursor = cursor.next();
		}
		wait_all_copies();
		return;
	}
	for (int stage = 0; stage < Shape::stages_per_block; ++stage) {
		ring.await_filled(cursor, block_backoff);
		sync_warp();
		if (lane() == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();
	}
}

/** The producer warp of a one-producer ring of whole tiles, under `eviction`. */
template<class Ring, prims::Eviction eviction, class Stages>
__attribute__((device)) inline void produce_tiles(Ring const& ring, Stages& stages, prims::Chunk const* run) {
	using namespace engine::device::prims;
	auto const priority = policy<eviction>();
	typename Ring::Cursor cursor = Ring::Cursor::start();
	for (int tile = 0; tile < tiles_per_block; ++tile) {
		ring.await_free(cursor, block_backoff);
		for (unsigned chunk = lane(); chunk < static_cast<unsigned>(tile_chunks); chunk += 32) {
			copy_async<Prefetch::Chunk256>(shared_address(&stages[cursor.stage][chunk]), run + static_cast<unsigned long long>(tile) * tile_chunks + chunk,
			                               priority);
		}
		ring.filled_on_copies(cursor);
		cursor = cursor.next();
	}
	wait_all_copies();
}

template<class Shape>
__attribute__((device)) inline void stall(ReadParams params) {
	using namespace engine::device::prims;
	using Ring = StageRing<Shape::stages, 32, Shape::consumer_warps>;
	struct Layout {
		typename Ring::Storage barriers;
		Chunk stage[Shape::stages][tile_chunks];
	};
	auto& layout = shared_layout<Shape::budget, Layout>();
	Ring const ring = Ring::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	if (warp_in_block() == 0) {
		produce_tiles<Ring, Eviction::First>(ring, layout.stage, at<Chunk const>(params.source) + block() * block_chunks);
		return;
	}
	typename Ring::Cursor cursor = Ring::Cursor::start();
	for (int tile = 0; tile < tiles_per_block; ++tile) {
		ring.await_filled(cursor, block_backoff);
		// The stall a dependency hop puts between a stage landing and its consumers moving on.
		if constexpr (Shape::stall_us > 0) {
			if (tile % Shape::stall_period == Shape::stall_period - 1) {
				unsigned long long const landed = nanoseconds();
				while (nanoseconds() - landed < static_cast<unsigned long long>(Shape::stall_us) * 1000ULL) {
					nanosleep(256);
				}
			}
		}
		sync_warp();
		if (lane() == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();
	}
}

/** Bf16 one, packed twice: the decode-like load's activations, constant so only the stream and the math vary. */
inline constexpr unsigned bf16_ones = 0x3F803F80U;

__attribute__((device)) inline void sustain(SustainParams params) {
	using namespace engine::device::prims;
	using Ring = StageRing<Sustain::stages, 32, Sustain::consumer_warps>;
	using Consumers = Warps<1, Sustain::consumer_warps>;
	struct Layout {
		Ring::Storage barriers;
		Chunk stage[Sustain::stages][tile_chunks];
	};
	auto& layout = shared_layout<Sustain::budget, Layout>();
	Ring const ring = Ring::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	if (!Consumers::holds(warp)) {
		produce_tiles<Ring, Eviction::First>(ring, layout.stage, at<Chunk const>(params.source) + block() * block_chunks);
		return;
	}
	unsigned const rank = Consumers::rank(warp);
	unsigned long long const start_cycles = cycles();
	unsigned long long const start_ns = nanoseconds();
	BFragment const b = {.reg = {bf16_ones, bf16_ones}};
	Accumulator accumulators[4] = {};
	Ring::Cursor cursor = Ring::Cursor::start();
	constexpr unsigned chunks_per_lane = tile_chunks / (Sustain::consumer_warps * 32);
	for (int tile = 0; tile < tiles_per_block; ++tile) {
		ring.await_filled(cursor, block_backoff);
		for (unsigned slice = 0; slice < chunks_per_lane; ++slice) {
			unsigned const chunk = (rank * chunks_per_lane + slice) * 32U + lane();
			AFragments2 const a = decode_chunk(load_shared(shared_address(&layout.stage[cursor.stage][chunk])));
			for (auto const& step : a.step) {
				for (auto& accumulator : accumulators) {
					mma(accumulator, step, b);
				}
			}
		}
		sync_warp();
		if (lane() == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();
	}
	if (rank == 0 && lane() == 0) {
		at<Clocks>(params.clocks)[block()] = {.cycles = cycles() - start_cycles, .nanoseconds = nanoseconds() - start_ns};
	}
}

template<class Shape>
__attribute__((device)) inline void reuse(ReuseParams params) {
	using namespace engine::device::prims;
	using Ring = StageRing<Shape::stages, 32, Shape::consumer_warps>;
	using Consumers = Warps<1, Shape::consumer_warps>;
	constexpr Eviction stream_eviction = Shape::priorities == Priorities::Split ? Eviction::First : Eviction::Normal;
	constexpr Eviction activation_eviction = Shape::priorities == Priorities::Split ? Eviction::Last : Eviction::Normal;
	struct Layout {
		typename Ring::Storage barriers;
		Chunk stage[Shape::stages][tile_chunks];
	};
	auto& layout = shared_layout<Shape::budget, Layout>();
	Ring const ring = Ring::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	if (!Consumers::holds(warp)) {
		produce_tiles<Ring, stream_eviction>(ring, layout.stage, at<Chunk const>(params.source) + block() * block_chunks);
		return;
	}
	Chunk const* const activations = at<Chunk const>(params.activations);
	auto const priority = policy<activation_eviction>();
	unsigned const reader = Consumers::rank(warp) * 32U + lane();
	unsigned long long spent = 0;
	unsigned long long sink = 0;
	typename Ring::Cursor cursor = Ring::Cursor::start();
	for (int tile = 0; tile < tiles_per_block; ++tile) {
		ring.await_filled(cursor, block_backoff);
		unsigned const index = (static_cast<unsigned>(tile) * Consumers::threads + reader + block() * 7919U) % Shape::activation_chunks;
		unsigned long long const before = cycles();
		Chunk const value = load(activations + index, priority);
		sink += value.word[0];
		spent += cycles() - before;
		sync_warp();
		if (lane() == 0) {
			ring.drained(cursor);
		}
		cursor = cursor.next();
	}
	if (reader == 0) {
		at<Rate>(params.rates)[block()] = {.cycles = spent, .sink = sink};
	}
}

}

namespace engine::device::ubench::l2 {

__attribute__((device)) inline void prefetch(PrefetchParams params) {
	using namespace engine::device::prims;
	Chunk const* const segment = at<Chunk const>(params.region) + static_cast<unsigned long long>(block()) * segment_chunks;
	for (unsigned line = thread_in_block(); line < static_cast<unsigned>(segment_bytes / 128); line += prefetch_budget.threads) {
		prefetch_to_l2(segment + static_cast<unsigned long long>(line) * 8U);
	}
}

__attribute__((device)) inline void probe(ProbeParams params) {
	using namespace engine::device::prims;
	Chunk const* const segment = at<Chunk const>(params.region) + static_cast<unsigned long long>(block()) * segment_chunks;
	auto const normal = policy<Eviction::Normal>();
	sync_block();
	unsigned long long const start = cycles();
	unsigned sink = 0;
	for (unsigned chunk = thread_in_block(); chunk < static_cast<unsigned>(segment_chunks); chunk += probe_budget.threads) {
		sink ^= load_constant(segment + chunk, normal).word[0];
	}
	// The block waits for every thread's loads, which their use above already waited for.
	sync_block();
	if (thread_in_block() == 0) {
		at<Rate>(params.cycles)[block()] = {.cycles = cycles() - start, .sink = sink};
	}
}

}

namespace ub = engine::device::ubench;
namespace stream = engine::device::ubench::stream;
using stream::TileOrder;

// The read sweep: producer warps 1, 2, 4, 8 by in-flight KiB per SM, where a stage holds a whole issue per producer.
using ReadP1D2 = stream::Read<1, 2, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d2 = ReadP1D2::budget;
extern "C" __attribute__((global, launch_bounds(ReadP1D2::budget.threads, ReadP1D2::budget.blocks_per_sm))) void ubench_read_p1_d2(stream::ReadParams params) {
	stream::read<ReadP1D2>(params);
}

using ReadP1D4 = stream::Read<1, 4, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d4 = ReadP1D4::budget;
extern "C" __attribute__((global, launch_bounds(ReadP1D4::budget.threads, ReadP1D4::budget.blocks_per_sm))) void ubench_read_p1_d4(stream::ReadParams params) {
	stream::read<ReadP1D4>(params);
}

using ReadP1D8 = stream::Read<1, 8, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d8 = ReadP1D8::budget;
extern "C" __attribute__((global, launch_bounds(ReadP1D8::budget.threads, ReadP1D8::budget.blocks_per_sm))) void ubench_read_p1_d8(stream::ReadParams params) {
	stream::read<ReadP1D8>(params);
}

using ReadP1D16 = stream::Read<1, 16, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d16 = ReadP1D16::budget;
extern "C" __attribute__((global, launch_bounds(ReadP1D16::budget.threads, ReadP1D16::budget.blocks_per_sm))) void ubench_read_p1_d16(stream::ReadParams params) {
	stream::read<ReadP1D16>(params);
}

using ReadP1D32 = stream::Read<1, 32, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d32 = ReadP1D32::budget;
extern "C" __attribute__((global, launch_bounds(ReadP1D32::budget.threads, ReadP1D32::budget.blocks_per_sm))) void ubench_read_p1_d32(stream::ReadParams params) {
	stream::read<ReadP1D32>(params);
}

using ReadP1D64 = stream::Read<1, 64, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d64 = ReadP1D64::budget;
extern "C" __attribute__((global, launch_bounds(ReadP1D64::budget.threads, ReadP1D64::budget.blocks_per_sm))) void ubench_read_p1_d64(stream::ReadParams params) {
	stream::read<ReadP1D64>(params);
}

using ReadP2D4 = stream::Read<2, 4, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p2_d4 = ReadP2D4::budget;
extern "C" __attribute__((global, launch_bounds(ReadP2D4::budget.threads, ReadP2D4::budget.blocks_per_sm))) void ubench_read_p2_d4(stream::ReadParams params) {
	stream::read<ReadP2D4>(params);
}

using ReadP2D8 = stream::Read<2, 8, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p2_d8 = ReadP2D8::budget;
extern "C" __attribute__((global, launch_bounds(ReadP2D8::budget.threads, ReadP2D8::budget.blocks_per_sm))) void ubench_read_p2_d8(stream::ReadParams params) {
	stream::read<ReadP2D8>(params);
}

using ReadP2D16 = stream::Read<2, 16, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p2_d16 = ReadP2D16::budget;
extern "C" __attribute__((global, launch_bounds(ReadP2D16::budget.threads, ReadP2D16::budget.blocks_per_sm))) void ubench_read_p2_d16(stream::ReadParams params) {
	stream::read<ReadP2D16>(params);
}

using ReadP2D32 = stream::Read<2, 32, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p2_d32 = ReadP2D32::budget;
extern "C" __attribute__((global, launch_bounds(ReadP2D32::budget.threads, ReadP2D32::budget.blocks_per_sm))) void ubench_read_p2_d32(stream::ReadParams params) {
	stream::read<ReadP2D32>(params);
}

using ReadP2D64 = stream::Read<2, 64, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p2_d64 = ReadP2D64::budget;
extern "C" __attribute__((global, launch_bounds(ReadP2D64::budget.threads, ReadP2D64::budget.blocks_per_sm))) void ubench_read_p2_d64(stream::ReadParams params) {
	stream::read<ReadP2D64>(params);
}

using ReadP4D8 = stream::Read<4, 8, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p4_d8 = ReadP4D8::budget;
extern "C" __attribute__((global, launch_bounds(ReadP4D8::budget.threads, ReadP4D8::budget.blocks_per_sm))) void ubench_read_p4_d8(stream::ReadParams params) {
	stream::read<ReadP4D8>(params);
}

using ReadP4D16 = stream::Read<4, 16, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p4_d16 = ReadP4D16::budget;
extern "C" __attribute__((global, launch_bounds(ReadP4D16::budget.threads, ReadP4D16::budget.blocks_per_sm))) void ubench_read_p4_d16(stream::ReadParams params) {
	stream::read<ReadP4D16>(params);
}

using ReadP4D32 = stream::Read<4, 32, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p4_d32 = ReadP4D32::budget;
extern "C" __attribute__((global, launch_bounds(ReadP4D32::budget.threads, ReadP4D32::budget.blocks_per_sm))) void ubench_read_p4_d32(stream::ReadParams params) {
	stream::read<ReadP4D32>(params);
}

using ReadP4D64 = stream::Read<4, 64, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p4_d64 = ReadP4D64::budget;
extern "C" __attribute__((global, launch_bounds(ReadP4D64::budget.threads, ReadP4D64::budget.blocks_per_sm))) void ubench_read_p4_d64(stream::ReadParams params) {
	stream::read<ReadP4D64>(params);
}

using ReadP8D16 = stream::Read<8, 16, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p8_d16 = ReadP8D16::budget;
extern "C" __attribute__((global, launch_bounds(ReadP8D16::budget.threads, ReadP8D16::budget.blocks_per_sm))) void ubench_read_p8_d16(stream::ReadParams params) {
	stream::read<ReadP8D16>(params);
}

using ReadP8D32 = stream::Read<8, 32, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p8_d32 = ReadP8D32::budget;
extern "C" __attribute__((global, launch_bounds(ReadP8D32::budget.threads, ReadP8D32::budget.blocks_per_sm))) void ubench_read_p8_d32(stream::ReadParams params) {
	stream::read<ReadP8D32>(params);
}

using ReadP8D64 = stream::Read<8, 64, TileOrder::Contiguous>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p8_d64 = ReadP8D64::budget;
extern "C" __attribute__((global, launch_bounds(ReadP8D64::budget.threads, ReadP8D64::budget.blocks_per_sm))) void ubench_read_p8_d64(stream::ReadParams params) {
	stream::read<ReadP8D64>(params);
}

// The canonical stream with its tiles stored as rows of a row-major matrix instead of contiguous runs.
using ReadRowMajor = stream::Read<1, 64, TileOrder::RowMajor>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_read_p1_d64_row_major = ReadRowMajor::budget;
extern "C" __attribute__((global, launch_bounds(ReadRowMajor::budget.threads, ReadRowMajor::budget.blocks_per_sm))) void ubench_read_p1_d64_row_major(stream::ReadParams params) {
	stream::read<ReadRowMajor>(params);
}

// The ring under stalls: ring KiB per SM by the microseconds each sixteenth tile is held.
using StallR32T0 = stream::Stall<32, 0>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r32_t0 = StallR32T0::budget;
extern "C" __attribute__((global, launch_bounds(StallR32T0::budget.threads, StallR32T0::budget.blocks_per_sm))) void ubench_stall_r32_t0(stream::ReadParams params) {
	stream::stall<StallR32T0>(params);
}

using StallR32T4 = stream::Stall<32, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r32_t4 = StallR32T4::budget;
extern "C" __attribute__((global, launch_bounds(StallR32T4::budget.threads, StallR32T4::budget.blocks_per_sm))) void ubench_stall_r32_t4(stream::ReadParams params) {
	stream::stall<StallR32T4>(params);
}

using StallR32T8 = stream::Stall<32, 8>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r32_t8 = StallR32T8::budget;
extern "C" __attribute__((global, launch_bounds(StallR32T8::budget.threads, StallR32T8::budget.blocks_per_sm))) void ubench_stall_r32_t8(stream::ReadParams params) {
	stream::stall<StallR32T8>(params);
}

using StallR32T12 = stream::Stall<32, 12>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r32_t12 = StallR32T12::budget;
extern "C" __attribute__((global, launch_bounds(StallR32T12::budget.threads, StallR32T12::budget.blocks_per_sm))) void ubench_stall_r32_t12(stream::ReadParams params) {
	stream::stall<StallR32T12>(params);
}

using StallR48T0 = stream::Stall<48, 0>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r48_t0 = StallR48T0::budget;
extern "C" __attribute__((global, launch_bounds(StallR48T0::budget.threads, StallR48T0::budget.blocks_per_sm))) void ubench_stall_r48_t0(stream::ReadParams params) {
	stream::stall<StallR48T0>(params);
}

using StallR48T4 = stream::Stall<48, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r48_t4 = StallR48T4::budget;
extern "C" __attribute__((global, launch_bounds(StallR48T4::budget.threads, StallR48T4::budget.blocks_per_sm))) void ubench_stall_r48_t4(stream::ReadParams params) {
	stream::stall<StallR48T4>(params);
}

using StallR48T8 = stream::Stall<48, 8>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r48_t8 = StallR48T8::budget;
extern "C" __attribute__((global, launch_bounds(StallR48T8::budget.threads, StallR48T8::budget.blocks_per_sm))) void ubench_stall_r48_t8(stream::ReadParams params) {
	stream::stall<StallR48T8>(params);
}

using StallR48T12 = stream::Stall<48, 12>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r48_t12 = StallR48T12::budget;
extern "C" __attribute__((global, launch_bounds(StallR48T12::budget.threads, StallR48T12::budget.blocks_per_sm))) void ubench_stall_r48_t12(stream::ReadParams params) {
	stream::stall<StallR48T12>(params);
}

using StallR64T0 = stream::Stall<64, 0>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r64_t0 = StallR64T0::budget;
extern "C" __attribute__((global, launch_bounds(StallR64T0::budget.threads, StallR64T0::budget.blocks_per_sm))) void ubench_stall_r64_t0(stream::ReadParams params) {
	stream::stall<StallR64T0>(params);
}

using StallR64T4 = stream::Stall<64, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r64_t4 = StallR64T4::budget;
extern "C" __attribute__((global, launch_bounds(StallR64T4::budget.threads, StallR64T4::budget.blocks_per_sm))) void ubench_stall_r64_t4(stream::ReadParams params) {
	stream::stall<StallR64T4>(params);
}

using StallR64T8 = stream::Stall<64, 8>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r64_t8 = StallR64T8::budget;
extern "C" __attribute__((global, launch_bounds(StallR64T8::budget.threads, StallR64T8::budget.blocks_per_sm))) void ubench_stall_r64_t8(stream::ReadParams params) {
	stream::stall<StallR64T8>(params);
}

using StallR64T12 = stream::Stall<64, 12>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r64_t12 = StallR64T12::budget;
extern "C" __attribute__((global, launch_bounds(StallR64T12::budget.threads, StallR64T12::budget.blocks_per_sm))) void ubench_stall_r64_t12(stream::ReadParams params) {
	stream::stall<StallR64T12>(params);
}

using StallR80T0 = stream::Stall<80, 0>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r80_t0 = StallR80T0::budget;
extern "C" __attribute__((global, launch_bounds(StallR80T0::budget.threads, StallR80T0::budget.blocks_per_sm))) void ubench_stall_r80_t0(stream::ReadParams params) {
	stream::stall<StallR80T0>(params);
}

using StallR80T4 = stream::Stall<80, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r80_t4 = StallR80T4::budget;
extern "C" __attribute__((global, launch_bounds(StallR80T4::budget.threads, StallR80T4::budget.blocks_per_sm))) void ubench_stall_r80_t4(stream::ReadParams params) {
	stream::stall<StallR80T4>(params);
}

using StallR80T8 = stream::Stall<80, 8>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r80_t8 = StallR80T8::budget;
extern "C" __attribute__((global, launch_bounds(StallR80T8::budget.threads, StallR80T8::budget.blocks_per_sm))) void ubench_stall_r80_t8(stream::ReadParams params) {
	stream::stall<StallR80T8>(params);
}

using StallR80T12 = stream::Stall<80, 12>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_stall_r80_t12 = StallR80T12::budget;
extern "C" __attribute__((global, launch_bounds(StallR80T12::budget.threads, StallR80T12::budget.blocks_per_sm))) void ubench_stall_r80_t12(stream::ReadParams params) {
	stream::stall<StallR80T12>(params);
}

// Activation reuse under the engine's l2 split and under uniform priorities.
using ReuseSplit = stream::Reuse<stream::Priorities::Split>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_reuse_split = ReuseSplit::budget;
extern "C" __attribute__((global, launch_bounds(ReuseSplit::budget.threads, ReuseSplit::budget.blocks_per_sm))) void ubench_reuse_split(stream::ReuseParams params) {
	stream::reuse<ReuseSplit>(params);
}

using ReuseUniform = stream::Reuse<stream::Priorities::Uniform>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_reuse_uniform = ReuseUniform::budget;
extern "C" __attribute__((global, launch_bounds(ReuseUniform::budget.threads, ReuseUniform::budget.blocks_per_sm))) void ubench_reuse_uniform(stream::ReuseParams params) {
	stream::reuse<ReuseUniform>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_sustain = stream::Sustain::budget;
extern "C" __attribute__((global, launch_bounds(stream::Sustain::budget.threads, stream::Sustain::budget.blocks_per_sm))) void ubench_sustain(stream::SustainParams params) {
	stream::sustain(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_l2_prefetch = ub::l2::prefetch_budget;
extern "C" __attribute__((global, launch_bounds(ub::l2::prefetch_budget.threads, ub::l2::prefetch_budget.blocks_per_sm))) void ubench_l2_prefetch(ub::l2::PrefetchParams params) {
	ub::l2::prefetch(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_l2_probe = ub::l2::probe_budget;
extern "C" __attribute__((global, launch_bounds(ub::l2::probe_budget.threads, ub::l2::probe_budget.blocks_per_sm))) void ubench_l2_probe(ub::l2::ProbeParams params) {
	ub::l2::probe(params);
}

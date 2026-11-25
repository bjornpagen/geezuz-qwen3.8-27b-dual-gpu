// SPDX-License-Identifier: AGPL-3.0-only
//
// The standalone kernels' common body: the decode kernel's warp roles running one gemv (or the prefill gemm) and
// nothing else, so the self-checks and the microbenchmarks exercise exactly the building blocks the decode and
// prefill kernels compose.
#pragma once

#include "gemv/gemm.h"
#include "gemv/gemv.h"
#include "gemv/selfcheck.h"
#include "prims/budget.h"
#include "prims/selfcheck.h"
#include "prims/selfcheck_record.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"

namespace engine::device::gemv::kernels {

using Producer = prims::Warps<0, 1>;
using Consumers = prims::Warps<1, consumer_warps>;
using Idle = prims::Warps<1 + consumer_warps, selfcheck::threads / 32 - 1 - consumer_warps>;
using ConsumerBarrier = prims::NamedBarrier<1, consumer_warps>;
using prims::selfcheck::pointer;
using prims::selfcheck::record;

/** The registers each thread of a one-block-per-SM kernel of `selfcheck::threads` may use. */
inline constexpr int registers =
    prims::register_cap({.threads = selfcheck::threads, .registers_per_thread = 1, .shared_bytes = 0, .blocks_per_sm = 1});

template<Matrix matrix, int row_tiles>
using Gemv = Shape<matrix, row_tiles, selfcheck::ring_stages, ConsumerBarrier>;

template<class S>
struct GemvLayout {
	typename S::Ring::Storage barriers;
	alignas(128) unsigned char stages[selfcheck::ring_stages][schedule::tile_bytes];
	alignas(128) unsigned char activations[2][S::Activations::bytes];
};

template<class S>
inline constexpr prims::Budget gemv_budget = {
    .threads = selfcheck::threads,
    .registers_per_thread = registers,
    .shared_bytes = static_cast<int>(sizeof(GemvLayout<S>)),
    .blocks_per_sm = 1,
};

/** Where a standalone gemv reads: the placement's tiles (its run), activations, scales, and the fixup buffers. */
struct GemvInputs {
	prims::Chunk const* tiles;
	prims::Chunk const* activations;
	unsigned short const* gate_scales;
	unsigned short const* up_scales;
	float* fixups;
	unsigned long long* fixup_flags;
};

/** One block's part of one gemv: the producer warp streams, the consumer warps compute and finish, the rest idle. */
template<class S, class Epilogue>
__attribute__((device, always_inline)) inline void run_gemv(GemvInputs const& inputs, Epilogue& epilogue) {
	using namespace engine::device::prims;
	static_assert(tiles<Producer, Consumers, Idle>(gemv_budget<S>));
	auto& layout = shared_layout<gemv_budget<S>, GemvLayout<S>>();
	auto const ring = S::Ring::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	auto cursor = S::Ring::Cursor::start();
	SharedAddress const stages = shared_address(layout.stages);
	if (Producer::holds(warp)) {
		produce<S>(ring, cursor, stages, inputs.tiles, block());
		wait_all_copies();
		return;
	}
	if (!Consumers::holds(warp)) {
		return;
	}
	Operands const operands = {
	    .activations = inputs.activations,
	    .scales = {inputs.gate_scales, inputs.up_scales},
	    .activation_tiles = {shared_address(layout.activations[0]), shared_address(layout.activations[1])},
	    .fixups = inputs.fixups,
	    .fixup_flags = inputs.fixup_flags,
	    .epoch = selfcheck::first_epoch,
	    .sm = block(),
	};
	consume<S>(ring, cursor, stages, operands, Consumers::rank(warp), epilogue);
}

template<Matrix matrix>
struct GemmLayout {
	typename prefill::Ring<matrix>::Storage barriers;
	alignas(128) unsigned char stages[prefill::stages<matrix>][prefill::Stage<matrix>::bytes];
};

template<Matrix matrix>
inline constexpr prims::Budget gemm_budget = {
    .threads = selfcheck::threads,
    .registers_per_thread = registers,
    .shared_bytes = static_cast<int>(sizeof(GemmLayout<matrix>)),
    .blocks_per_sm = 1,
};

/** One block's part of one prefill gemm over `row_blocks` blocks of 32 rows. */
template<Matrix matrix, class Epilogue>
__attribute__((device, always_inline)) inline void run_gemm(prefill::Operands const& operands, unsigned row_blocks, Epilogue& epilogue) {
	using namespace engine::device::prims;
	static_assert(tiles<Producer, Consumers, Idle>(gemm_budget<matrix>));
	auto& layout = shared_layout<gemm_budget<matrix>, GemmLayout<matrix>>();
	auto const ring = prefill::Ring<matrix>::at(layout.barriers);
	if (thread_in_block() == 0) {
		ring.init();
	}
	sync_block();
	unsigned const warp = warp_in_block();
	auto cursor = prefill::Ring<matrix>::Cursor::start();
	SharedAddress const stages = shared_address(layout.stages);
	if (Producer::holds(warp)) {
		prefill::produce<matrix>(ring, cursor, stages, operands, row_blocks, selfcheck::blocks);
		wait_all_copies();
		return;
	}
	if (Consumers::holds(warp)) {
		prefill::consume<matrix>(ring, cursor, stages, operands, row_blocks, selfcheck::blocks, Consumers::rank(warp), epilogue);
	}
}

/** The gemv inputs a self-check or stream kernel's parameters name. */
template<class Params>
[[nodiscard]] __attribute__((device, always_inline)) inline auto gemv_inputs(Params const& params, selfcheck::Address gate_scales,
                                                                             selfcheck::Address up_scales) -> GemvInputs {
	return {
	    .tiles = pointer<prims::Chunk const>(params.tiles),
	    .activations = pointer<prims::Chunk const>(params.activations),
	    .gate_scales = pointer<unsigned short const>(gate_scales),
	    .up_scales = pointer<unsigned short const>(up_scales),
	    .fixups = pointer<float>(params.fixups),
	    .fixup_flags = pointer<unsigned long long>(params.fixup_flags),
	};
}

}

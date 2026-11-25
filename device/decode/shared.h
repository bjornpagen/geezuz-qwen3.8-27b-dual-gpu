// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode kernel's one shared layout and its budget, as plain data both the kernel and the host's launch read: every
// task's shared memory aliased over the same bytes (the weight phases' ring and activation tiles, attention's core,
// the deltanet core, the ranking's keys, the verdict's distributions). A grid barrier separates every two tasks, so no
// two layouts are ever live at once; each task arms its own barriers after the grid barrier that starts it. It uses no
// standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "attn/layout.h"
#include "decode/key.h"
#include "decode/walk.h"
#include "decode/wire.h"
#include "deltanet/geometry.h"
#include "gemv/layout.h"
#include "model.h"
#include "prims/budget.h"
#include "prims/stage_ring.h"
#include "sampling/draw.h"
#include "schedule/geometry.h"

namespace engine::device::decode {

/** One block per SM: the static split of every phase's units names each SM by its block. */
inline constexpr unsigned sms = static_cast<unsigned>(schedule::sms);
inline constexpr unsigned block_warps = static_cast<unsigned>(schedule::warps(schedule::decode_roles));
inline constexpr unsigned grid_warps = sms * block_warps;

inline constexpr int ring_stages = static_cast<int>(schedule::ring(budget).slots);

/** The activation tiles of the widest unit: an e4m3 tile's k span for every row. */
using WidestActivations = gemv::layout::ActivationTile<model::Format::E4m3, static_cast<int>(row_tiles)>;

struct GemvShared {
	prims::StageBarriers<ring_stages> barriers;
	alignas(128) unsigned char stages[ring_stages][schedule::tile_bytes];
	alignas(128) unsigned char activations[2][WidestActivations::bytes];
};

/** Each warp's keys while it ranks one segment or folds one group of lists. */
struct RankShared {
	alignas(16) Key keys[block_warps][segment_rows];
};

/** One slot's verdict: each verify row's merged candidates and its target distribution. */
struct VerifyShared {
	alignas(16) sampling::Candidate candidates[max_committed][list_length];
	int available[max_committed];
	sampling::Survivors targets[max_committed];
};

/** Attention's core for the verify and catch-up passes (a committed row and up to four drafts a slot), and the steps'. */
using WideTile = attn::RowTile<static_cast<unsigned>(max_committed)>;
using NarrowTile = attn::RowTile<1>;

union Shared {
	GemvShared gemv;
	attn::CoreShared<WideTile> wide;
	attn::CoreShared<NarrowTile> narrow;
	deltanet::DecodeShared deltanet;
	RankShared rank;
	VerifyShared verify;
};

/** The decode kernel's budget: the schedule's warps at one block per SM, every register that leaves, the union's bytes. */
[[nodiscard]] consteval auto kernel_budget() -> prims::Budget {
	auto shape = schedule::decode_budget(budget);
	shape.shared_bytes = static_cast<int>(sizeof(Shared));
	return shape;
}

inline constexpr prims::Budget decode_budget = kernel_budget();

static_assert(prims::check(decode_budget) == prims::Fit::Fits, "the decode kernel fits one sm_86 SM");
static_assert(sizeof(Shared) <= static_cast<Count>(prims::sm86.max_shared_per_block), "every task's shared memory fits 99 KiB");
static_assert(sizeof(GemvShared) <= schedule::decode_shared_bytes(budget), "the weight phases fit the schedule's planned shared memory");
static_assert(sizeof(attn::CoreShared<WideTile>) <= sizeof(Shared) && sizeof(deltanet::DecodeShared) <= sizeof(Shared),
              "attention's and the deltanet's cores alias the weight phases' bytes");
static_assert(decode_budget.threads == static_cast<int>(block_warps) * prims::sm86.warp_threads);

}

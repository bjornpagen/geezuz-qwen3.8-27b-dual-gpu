// SPDX-License-Identifier: AGPL-3.0-only
//
// The persistent decode kernel's view of one card: its memories as typed pointers, its warp roles, the grid barrier,
// and the per-cycle view of the rows each pass computes. Device code only.
#pragma once

#include "attn/core.h"
#include "attn/layout.h"
#include "decode/image.h"
#include "decode/key.h"
#include "decode/shared.h"
#include "decode/walk.h"
#include "decode/wire.h"
#include "decode/workspace.h"
#include "deltanet/geometry.h"
#include "gemv/gemv.h"
#include "model.h"
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/order.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "ring/ring.h"
#include "sampling/draw.h"
#include "schedule/geometry.h"
#include "target.h"

namespace engine::device::decode {

using prims::Chunk;
using prims::Eviction;

// ---------------------------------------------------------------------------------------------------------------------
// Warp roles: the schedule's decode roles, which every building block composed here is written against.

using Producer = prims::Warps<0, 1>;
using Consumers = prims::Warps<1, gemv::consumer_warps>;
/** The exchange and control warps: idle in a weight phase but for its side work, the preparers in a core. */
using Side = prims::Warps<1 + gemv::consumer_warps, 3>;
using ConsumerBarrier = prims::NamedBarrier<1, gemv::consumer_warps>;

static_assert(Producer::count + Consumers::count + Side::count == static_cast<int>(block_warps));
static_assert(attn::Producer::first == Producer::first && attn::Consumers::first == Consumers::first && attn::Preparers::first == Side::first &&
                  attn::Preparers::count == Side::count,
              "attention's roles are the decode kernel's");
static_assert(deltanet::producer_warps == Producer::count && deltanet::consumer_warps == Consumers::count &&
                  deltanet::preparer_warps == Side::count,
              "the deltanet core's roles are the decode kernel's");

static_assert(prims::tiles<Producer, Consumers, Side>(decode_budget), "the roles tile the block");

using GemvRing = gemv::Ring<ring_stages>;

static_assert(__is_same(GemvRing::Storage, prims::StageBarriers<ring_stages>), "the weight phases' ring is the shared layout's");

// ---------------------------------------------------------------------------------------------------------------------
// Memory.

/** A device address the host passed as an integer, as the typed pointer it is. */
template<class T>
[[nodiscard]] __attribute__((device, always_inline)) inline auto at(Address address) -> T* {
	/* SAFETY: the host fills every address from an allocation holding values of T at that offset. */
	return __builtin_bit_cast(T*, address);
}

using Ring = ring::Slot<Descriptor, Result>;
using Grid = prims::step1::GridBarrier<sms>;

/** The card's memories, from the kernel's parameters. */
struct Machine {
	Params params;

	template<class T>
	[[nodiscard]] __attribute__((device, always_inline)) auto local(Count offset) const -> T* {
		return at<T>(params.workspace + offset);
	}

	template<class T>
	[[nodiscard]] __attribute__((device, always_inline)) auto peer(Count offset) const -> T* {
		return at<T>(params.peer_workspace + offset);
	}

	[[nodiscard]] __attribute__((device, always_inline)) auto address(Count offset) const -> Address {
		return params.workspace + offset;
	}

	[[nodiscard]] __attribute__((device, always_inline)) auto image(ImageOffset offset) const -> Address {
		return params.image + offset.bytes();
	}

	[[nodiscard]] __attribute__((device, always_inline)) auto descriptor() const -> Descriptor const& {
		return *local<Descriptor const>(workspace.descriptor);
	}

	[[nodiscard]] __attribute__((device, always_inline)) auto outcome(unsigned live) const -> SlotOutcome* {
		return local<SlotOutcome>(workspace.outcomes) + live;
	}

	[[nodiscard]] __attribute__((device, always_inline)) auto book(unsigned slot) const -> Book* {
		return local<Book>(workspace.books) + slot;
	}

	[[nodiscard]] __attribute__((device, always_inline)) auto grid() const -> Grid {
		auto* const control = at<Control>(params.control);
		return {.arrivals = &control->arrivals.value, .generation = &control->generation.word};
	}
};

/** The running counts every thread of both cards advances identically: each names one use of a flag or a buffer. */
struct Epochs {
	unsigned long long ticket;
	/** Weight placements consumed: the fixup flags' epoch. */
	unsigned long long fixups;
	/** Partial exchanges and candidate exchanges made. */
	unsigned long long partials;
	unsigned long long candidates;
};

[[nodiscard]] __attribute__((device, always_inline)) inline auto global_warp() -> unsigned {
	return prims::block() * block_warps + prims::warp_in_block();
}

/** A trap where the host broke a promise the descriptor carries; dev builds only. */
__attribute__((device, always_inline)) inline void expect(bool holds) {
	if constexpr (checks == Checks::On) {
		if (!holds) {
			__builtin_trap();
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The rows of a pass: which slot's rows sit where, at which positions. Attention's cycle description is exactly this.

/** The rows a pass computes, in attention's terms, and how many there are. */
struct PassRows {
	attn::Cycle cycle;
	unsigned rows;
};

/**
 * The verify pass runs each slot's committed token and its drafts; the catch-up pass the accepted ones (and the
 * committed token) at their positions; draft step k one row per slot at the position after the catch-up's last plus
 * k - 2. Each slot's rows follow the last's.
 */
[[nodiscard]] __attribute__((device)) inline auto pass_rows(Machine const& machine, Pass pass) -> PassRows {
	Descriptor const& descriptor = machine.descriptor();
	PassRows result = {.cycle = {.live = descriptor.live, .work = {}}, .rows = 0};
	for (unsigned index = 0; index < descriptor.live; ++index) {
		SlotRun const& run = descriptor.slots[index];
		unsigned rows = 1;
		unsigned position = run.position;
		switch (pass.kind) {
		case PassKind::Verify:
			rows = 1U + run.drafts;
			break;
		case PassKind::CatchUp:
			rows = 1U + machine.outcome(index)->accepted;
			break;
		case PassKind::Step:
			position = run.position + machine.outcome(index)->accepted + static_cast<unsigned>(pass.draft) - 1U;
			break;
		}
		result.cycle.work[index] = {.slot = run.slot, .position = position, .rows = rows, .first_row = result.rows};
		result.rows += rows;
	}
	return result;
}

/** Where row `row` of a pass belongs: its live slot and its place among that slot's rows. */
struct RowPlace {
	unsigned live;
	unsigned offset;
};

[[nodiscard]] __attribute__((device)) inline auto place(attn::Cycle const& cycle, unsigned row) -> RowPlace {
	unsigned live = 0;
	while (live + 1 < cycle.live && cycle.work[live + 1].first_row <= row) {
		++live;
	}
	return {.live = live, .offset = row - cycle.work[live].first_row};
}

/** The token verify row `row` reads: the slot's committed token, then its drafts from its book. */
[[nodiscard]] __attribute__((device)) inline auto verify_token(Machine const& machine, attn::Cycle const& cycle, unsigned row) -> unsigned {
	RowPlace const at_row = place(cycle, row);
	SlotRun const& run = machine.descriptor().slots[at_row.live];
	return at_row.offset == 0 ? run.token : machine.book(run.slot)->tokens[at_row.offset - 1];
}

}

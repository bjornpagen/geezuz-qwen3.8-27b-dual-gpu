// SPDX-License-Identifier: AGPL-3.0-only
//
// The persistent decode kernel as the host sees it: its ptx image and launch declaration, and the device headers it
// shares with the host re-exported unchanged: what crosses the ring, the mask buffer, the logits buffer and the
// profile, the workspace and shared layouts, the ranking's keys, and the cycle's walk. The kernel's budget is the
// very value it `static_assert`s.
module;

#include "decode/key.h"
#include "decode/profile.h"
#include "decode/shared.h"
#include "decode/walk.h"
#include "decode/wire.h"
#include "decode/workspace.h"

export module engine.decode_device;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;

namespace engine::decode_device {

namespace device = engine::device::decode;
namespace prims = engine::device::prims;

export using device::Action;
export using device::Address;
export using device::Book;
export using device::Command;
export using device::Control;
export using device::Descriptor;
export using device::Key;
export using device::LogitsBuffer;
export using device::LogitsHalf;
export using device::MaskFlag;
export using device::MaskHalf;
export using device::Masks;
export using device::Params;
export using device::PhaseCounters;
export using device::Point;
export using device::Profile;
export using device::Result;
export using device::Shared;
export using device::SlotOutcome;
export using device::SlotRun;
export using device::Tail;
export using device::Task;
export using device::Work;
export using device::Workspace;

export using device::budget;
export using device::decode_budget;
export using device::fold_lists;
export using device::list_length;
export using device::logits_halves;
export using device::logits_rows;
export using device::mask_halves;
export using device::mask_words;
export using device::max_commands;
export using device::max_committed;
export using device::max_drafts;
export using device::max_rows;
export using device::no_candidate;
export using device::profile_phases;
export using device::ring_slots;
export using device::row_tiles;
export using device::row_token_bytes;
export using device::segment_rows;
export using device::sms;
export using device::workspace;

export using device::candidate_of;
export using device::folds;
export using device::key_of;
export using device::lists_after;
export using device::logit_of;
export using device::logits_half;
export using device::mask_half;
export using device::normalizes;
export using device::ordered;
export using device::profile_slot;
export using device::ranked_rows;
export using device::segments;
export using device::walk_cycle;
export using device::walk_logits;
export using device::walk_prompt_end;

constexpr char text[] = {
#embed "decode.ptx" suffix(, '\0')
};

/** The decode kernel's ptx image. */
export inline constexpr ptx::Image image = ptx::image(text);

static_assert(image.header == ptx::abi.header, "every image shares the engine's isa and target");

/**
 * The resident decode loop on one card: every block co-resident, one per SM, serving the card's ring until the exit
 * ticket. Launched once per card with that card's memories and the peer's.
 */
export struct Loop {
	static constexpr ptx::Image image = decode_device::image;
	static constexpr std::string_view entry = "decode_loop";
	static constexpr driver::Dim grid = {.x = device::sms, .y = 1, .z = 1};
	static constexpr prims::Budget budget = device::decode_budget;
	static constexpr driver::Residency residency = driver::Residency::CoResident;
	using Params = device::Params;
};

static_assert(driver::Kernel<Loop>);

/** The decode image's one kernel, as a program. */
export using Program = driver::Program<Loop>;

/**
 * The host's release of one half of the mask buffer: every mask word it stored before is the kernel's once the kernel
 * acquires the ticket here (`ld.acquire.sys` on the card).
 */
export auto release_masks(device::MaskFlag& flag, std::uint64_t ticket) -> void {
	std::atomic_ref{flag.ticket}.store(ticket, std::memory_order_release);
}

/** The ticket a half of the mask buffer was last released for, acquired: what the kernel reads before ranking. */
export [[nodiscard]] auto released_masks(device::MaskFlag const& flag) -> std::uint64_t {
	/* SAFETY: an atomic load writes nothing; atomic_ref only takes a mutable object. */
	return std::atomic_ref{const_cast<unsigned long long&>(flag.ticket)}.load(std::memory_order_acquire);
}

}

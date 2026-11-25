// SPDX-License-Identifier: AGPL-3.0-only
//
// What crosses between the host and the persistent decode kernel, held once for both ends: the cycle descriptor the
// host posts through the resident ring, the result the kernel answers with, the mask buffer the host fills while the
// body runs, the control block both cards' kernels meet in, and the kernel's one parameter block. Every field is a
// plain value of fixed width; addresses travel as integers and become pointers only on the device. It uses no
// standard headers so that it compiles unchanged as device code and inside a module purview.
#pragma once

#include "model.h"
#include "sampling/draw.h"
#include "schedule/program.h"

namespace engine::device::decode {

using model::Count;

/** The row budget the grid-barrier baseline runs: every verify and catch-up row of a cycle. */
inline constexpr model::RowBudget budget = schedule::baseline_row_budget;
inline constexpr Count max_rows = model::rows(budget);
inline constexpr Count row_tiles = schedule::row_tiles(budget);
inline constexpr Count max_drafts = model::drafts(model::max_draft_depth);
/** Tokens one slot commits in a cycle: one per accepted draft and the token decided after them. */
inline constexpr Count max_committed = 1 + max_drafts;

// ---------------------------------------------------------------------------------------------------------------------
// The kv pool's commands, as the kernel applies them before a cycle.

/** What one device command does; each is one of the pool's commands or one block of a snapshot. */
enum class Action : unsigned char {
	/** Every byte of block `first` becomes zero. */
	ZeroBlock,
	/** Positions [0, `third`) of block `second` become block `first`'s, the rest zero. */
	CopyPrefix,
	/** Entry `second` of slot `slot`'s block table names block `first`. */
	SetTableEntry,
	/** Part `second` of slot `slot`'s deltanet state image is copied to the start of block `first`. */
	SaveSnapshotPart,
	/** The start of block `first` is copied back into part `second` of slot `slot`'s state image. */
	LoadSnapshotPart,
	/**
	 * Slot `slot`'s stored deltanet state takes the `first` accepted drafts waiting in its replay ring, every layer: what a
	 * snapshot of the slot must hold. It runs alone in its wave and is applied by the deltanet core itself.
	 */
	Replay,
};

/**
 * One command. Commands of one wave touch no common block, slot image or table entry, so the kernel spreads a wave
 * over every SM at once and puts a grid barrier only between waves; waves never decrease along the list.
 */
struct Command {
	Action action;
	unsigned char slot;
	unsigned short wave;
	unsigned first;
	unsigned second;
	unsigned third;
};

static_assert(sizeof(Command) == 16, "a command is one 16-byte chunk");

/** Commands one ticket carries; the host splits a longer list over command-only tickets ahead of the cycle. */
inline constexpr Count max_commands = 512;

/** Tickets in flight at once: a command-only ticket may precede the cycle it serves. */
inline constexpr Count ring_slots = 2;

// ---------------------------------------------------------------------------------------------------------------------
// The cycle descriptor.

/** What a ticket asks of the kernel besides its commands. */
enum class Work : unsigned char {
	/** Apply the commands and nothing else. */
	Apply,
	/** Apply the commands, then run one decode cycle. */
	Cycle,
	/**
	 * Apply the commands, then finish a prompt: the lm_head over the prompt's last row (whose final norm prefill left in
	 * the workspace's prompt row), its masked sample, and the mtp head's catch-up and draft steps from it. The one live
	 * slot's run names the prompt's last position and carries no drafts.
	 */
	PromptEnd,
};

/** One live slot's part of a cycle. */
struct SlotRun {
	/** The slot's index below `model::slots`; live slots come in ascending order. */
	unsigned slot;
	/** Its kv and deltanet state hold positions [0, position); the committed token sits at `position`. */
	unsigned position;
	unsigned token;
	/** Drafts this cycle verifies, from the slot's draft book: at most what the kernel drafted for it last. */
	unsigned char drafts;
	/** Accepted drafts of the slot's last cycle its deltanet replays from the ring first. */
	unsigned char replays;
	unsigned char unused[2];
	sampling::Rule rule;
};

static_assert(sizeof(sampling::Candidate) == 8, "a candidate crosses the cards as one 8-byte word");
static_assert(sizeof(sampling::Rule) == 48 && alignof(sampling::Rule) == 8, "the host fills each slot's rule in this layout");
static_assert(sizeof(SlotRun) == 64);

struct alignas(16) Descriptor {
	Work work;
	/** The cycle's draft depth: the mtp passes it runs, so the drafts the next cycle can verify. */
	model::DraftDepth depth;
	unsigned char live;
	unsigned char unused;
	unsigned commands;
	SlotRun slots[model::slots];
	Command command[max_commands];
};

/** One live slot's outcome. */
struct SlotOutcome {
	/** Drafts accepted; the slot commits accepted + 1 tokens. */
	unsigned accepted;
	/** The accepted drafts, then the token decided after them. */
	unsigned committed[max_committed];
	/** The drafts of the next cycle, as many as the cycle's depth. */
	unsigned drafts[max_drafts];
};

struct Result {
	/** The card's global timer at the cycle's start and end, in nanoseconds. */
	unsigned long long started;
	unsigned long long finished;
	SlotOutcome slots[model::slots];
};

// ---------------------------------------------------------------------------------------------------------------------
// The mask buffer: each verify row's allowed tokens in the card's lm_head row order, filled by the host while the
// cycle's body streams, released by a flag the kernel acquires before it ranks the lm_head's rows.

inline constexpr Count mask_words = model::vocab_per_card / 32;

static_assert(model::vocab_per_card % 32 == 0, "a card's rows are whole mask words");

/** Two halves by ticket parity, so the host may fill the next cycle's masks while this one reads its own. */
inline constexpr Count mask_halves = 2;

struct alignas(128) MaskFlag {
	/** The ticket whose masks the half holds, released after them. */
	unsigned long long ticket;
};

struct MaskHalf {
	unsigned words[max_rows][mask_words];
};

struct Masks {
	MaskFlag flag[mask_halves];
	MaskHalf half[mask_halves];
};

[[nodiscard]] constexpr auto mask_half(unsigned long long ticket) -> Count {
	return static_cast<Count>(ticket % mask_halves);
}

// ---------------------------------------------------------------------------------------------------------------------
// The control block: the words that must read zero before the kernel starts, which the host uploads zeroed.

struct alignas(128) Line {
	unsigned long long word;
};

struct alignas(128) Tally {
	unsigned value;
};

struct Control {
	/** The grid barrier's arrivals and its generation. */
	Tally arrivals;
	Line generation;
	/** Written by the peer card once it has cleared its own flags: the cards meet here before serving tickets. */
	Line peer_ready;
};

// ---------------------------------------------------------------------------------------------------------------------
// The kernel's parameters: one card's memories and the peer's, as device addresses.

using Address = unsigned long long;

struct Params {
	/** This card's baked image, as the schedule lays it out. */
	Address image;
	/** The host image's embedding table, mapped into the card. */
	Address embedding;
	/** The card's lm_head row-to-token table, u32 per row. */
	Address row_tokens;
	Address workspace;
	/** The peer card's workspace, reached over nvlink: this card only ever stores into it. */
	Address peer_workspace;
	Address control;
	Address peer_control;
	/** The kv pool's blocks. */
	Address pool;
	/** The slots' deltanet state images. */
	Address states;
	/** The resident ring's slots and the mask buffer, in mapped host memory. */
	Address ring;
	Address masks;
	/** This card's rank: it chooses the card's half of the mtp fc's two inputs. */
	unsigned card;
	unsigned unused;
};

}

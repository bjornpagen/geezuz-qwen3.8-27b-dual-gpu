// SPDX-License-Identifier: AGPL-3.0-only
export module engine.kv:device;

import std;
import :geometry;

namespace engine::kv {

/**
 * Every position of `block` reads zero, on both cards. The pool zeroes each block it adds to a table, so attention,
 * which masks scores past a slot's length but still multiplies those rows of v by zero, never meets a NaN left by a
 * former snapshot or by memory never written.
 */
export struct ZeroBlock {
	BlockIndex block;
};

/**
 * Positions [0, positions) of `target` become those of `source`, and the rest of `target` reads zero, on both cards:
 * the copy in copy-on-write, which keeps exactly the positions the writing slot holds.
 */
export struct CopyPrefix {
	BlockIndex source;
	BlockIndex target;
	Count positions;
};

/** Entry `entry` of `slot`'s block table on the device names `block`. */
export struct SetTableEntry {
	Slot slot;
	Count entry;
	BlockIndex block;
};

/** `slot`'s deltanet state (recurrent state and convolution window, each card its own half) is written into `blocks`. */
export template<Geometry G>
struct SaveSnapshot {
	Slot slot;
	std::array<BlockIndex, G.snapshot_blocks> blocks;
};

/** `slot`'s deltanet state becomes the snapshot held in `blocks`. */
export template<Geometry G>
struct LoadSnapshot {
	Slot slot;
	std::array<BlockIndex, G.snapshot_blocks> blocks;
};

/**
 * What the pool asks of the device. The pool owns no device memory: it decides which block holds what, and the device
 * applies these in the order given, before any kernel that reads the blocks or tables they touch.
 */
export template<Geometry G>
using Command = std::variant<ZeroBlock, CopyPrefix, SetTableEntry, SaveSnapshot<G>, LoadSnapshot<G>>;

/** A device the pool can drive: it accepts the pool's commands in order. */
export template<class D, Geometry G>
concept Device = requires(D& device, Command<G> const& command) { device.apply(command); };

}

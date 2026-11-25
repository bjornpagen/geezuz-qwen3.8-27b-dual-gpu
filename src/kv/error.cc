// SPDX-License-Identifier: AGPL-3.0-only
export module engine.kv:error;

import std;
import engine.core;
import :geometry;

namespace engine::kv {

/**
 * The pool has fewer free blocks than a request needs. The scheduler evicts cached blocks and retries, or preempts a
 * slot; `needed - free` is the shortfall either must cover.
 */
export struct [[nodiscard]] Exhausted {
	Count needed;
	Count free;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator==(Exhausted const&) const -> bool = default;
};

/** Why a snapshot was not taken. Either way the prefix cache can evict and try again. */
export enum class SnapshotShortage : std::uint8_t {
	/** The pool already holds as many snapshots as their share allows. */
	Capacity,
	/** Fewer blocks are free than one snapshot occupies. */
	Blocks,
};

export struct [[nodiscard]] SnapshotRefused {
	SnapshotShortage shortage;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (shortage) {
		case SnapshotShortage::Capacity:
		case SnapshotShortage::Blocks:
			return core::Retry::Transient;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(SnapshotRefused const&) const -> bool = default;
};

}

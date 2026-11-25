// SPDX-License-Identifier: AGPL-3.0-only
export module engine.hugepage:error;

import std;
import engine.core;

namespace engine::hugepage {

/** Why a region could not be mapped. */
export enum class Failure : std::uint8_t {
	/**
	 * The machine has no hugepages left for it: the reserved pool (set at bring-up) is short, or other processes hold
	 * it. Freed pages make a retry succeed.
	 */
	Exhausted,
	/** The kernel refused the mapping for any other reason (no hugetlb support, a limit on locked memory). */
	Refused,
};

export struct [[nodiscard]] HugepageError {
	Failure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case Failure::Exhausted:
			return core::Retry::Transient;
		case Failure::Refused:
			return core::Retry::Permanent;
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(HugepageError const&) const -> bool = default;
};

/** The page size regions are made of, and the granule their sizes round up to on every platform. */
export inline constexpr std::size_t page_bytes = std::size_t{2} << 20U;

}

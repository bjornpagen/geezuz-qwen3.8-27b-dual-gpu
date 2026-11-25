// SPDX-License-Identifier: AGPL-3.0-only
//
// The sampling arithmetic the decode kernels run, re-exported to host code so the host reference computes the same
// bits. Device code cannot import modules, so the definition is a device header; this unit gives its value types and
// value functions a module surface unchanged, and puts spans with checked lengths in place of its pointer arguments.
module;

#include "sampling/draw.h"

export module engine.sampling_device;

import std;
import engine.core;

namespace engine::sampling_device {

export using device::sampling::Block;
export using device::sampling::Candidate;
export using device::sampling::Filter;
export using device::sampling::Judgement;
export using device::sampling::Key;
export using device::sampling::Nucleus;
export using device::sampling::Philox_key;
export using device::sampling::Purpose;
export using device::sampling::Rule;
export using device::sampling::Settled;
export using device::sampling::Survivors;
export using device::sampling::Verdict;

export using device::sampling::top_k_bound;

export using device::sampling::exp_nonpositive;
export using device::sampling::key;
export using device::sampling::philox;
export using device::sampling::pick;
export using device::sampling::point_mass;
export using device::sampling::probability_of;
export using device::sampling::propose;
export using device::sampling::ranks_before;
export using device::sampling::residual;
export using device::sampling::uniform;

/** Whether `distribution` is well formed: 1 to top_k_bound entries, nonnegative weights, a normal total. */
export [[nodiscard]] constexpr auto well_formed(Survivors const& distribution) -> bool {
	if (distribution.count < 1 || distribution.count > top_k_bound || !(distribution.total >= 0x1p-1022)) {
		return false;
	}
	auto const weights = std::span{distribution.weight}.first(static_cast<std::size_t>(distribution.count));
	return std::ranges::all_of(weights, [](double weight) {
		return weight >= 0.0;
	});
}

/** The device's `survivors` over a borrowed candidate list: 1 to top_k_bound candidates sorted by `ranks_before`. */
export [[nodiscard]] constexpr auto survivors(std::span<Candidate const> sorted, Filter const& filter) -> Survivors {
	core::invariant(!sorted.empty() && sorted.size() <= std::size_t{top_k_bound});
	core::invariant(std::ranges::is_sorted(sorted, ranks_before));
	core::invariant(filter.top_k >= 1 && filter.top_k <= top_k_bound);
	/* SAFETY: the pointer and count describe `sorted`, which outlives the call. */
	return device::sampling::survivors(sorted.data(), core::narrow<int>(sorted.size()), filter);
}

/** The device's `judge`, for a draft index the drafter drew from `draft` (so its weight is positive). */
export [[nodiscard]] constexpr auto judge(Survivors const& target, Survivors const& draft, int index, double u_accept, double u_residual)
    -> Judgement {
	core::invariant(well_formed(target) && well_formed(draft));
	core::invariant(index >= 0 && index < draft.count && draft.weight[index] > 0.0);
	return device::sampling::judge(target, draft, index, u_accept, u_residual);
}

/**
 * The device's `settle` over one slot's cycle: one more target than drafts, and one chosen index per draft into its
 * distribution.
 */
export [[nodiscard]] constexpr auto settle(Rule const& rule, std::uint32_t position, std::span<Survivors const> targets,
                                           std::span<Survivors const> drafts, std::span<int const> chosen) -> Settled {
	core::invariant(targets.size() == drafts.size() + 1 && chosen.size() == drafts.size());
	core::invariant(std::ranges::all_of(targets, well_formed) && std::ranges::all_of(drafts, well_formed));
	for (auto const [draft, index] : std::views::zip(drafts, chosen)) {
		core::invariant(index >= 0 && index < draft.count && draft.weight[index] > 0.0);
	}
	/* SAFETY: each pointer and the depth describe the spans above, which outlive the call. */
	return device::sampling::settle(rule, position, targets.data(), drafts.data(), chosen.data(), core::narrow<int>(drafts.size()));
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sampling:reference;

import std;
import engine.core;
import engine.sampling_device;
import :params;

namespace engine::sampling {

/**
 * The vocabulary's best min(V, top_k_bound) entries in rank order, from every logit: the exact top-k that the two
 * cards' merge of their halves must reproduce. Logits are finite; a NaN or infinite logit is a bug upstream.
 */
export [[nodiscard]] auto candidates(std::span<float const> logits) -> std::vector<sampling_device::Candidate> {
	core::invariant(!logits.empty() && std::in_range<int>(logits.size()));
	core::invariant(std::ranges::all_of(logits, [](float logit) {
		return std::isfinite(logit);
	}));
	auto ranked = std::views::iota(0, core::narrow<int>(logits.size())) | std::views::transform([&](int id) {
		              return sampling_device::Candidate{.logit = logits[core::narrow<std::size_t>(id)], .id = id};
	              }) |
	              std::ranges::to<std::vector>();
	auto const kept = std::min(ranked.size(), std::size_t{sampling_device::top_k_bound});
	std::ranges::partial_sort(ranked, ranked.begin() + core::narrow<std::ptrdiff_t>(kept), sampling_device::ranks_before);
	ranked.resize(kept);
	return ranked;
}

/** The distribution `params` define over `logits`, as the device computes it from the same logits. */
export [[nodiscard]] auto survivors(Params const& params, std::span<float const> logits) -> sampling_device::Survivors {
	return sampling_device::survivors(candidates(logits), rule(params, 0).filter);
}

/** One token and its probability. */
export struct Outcome {
	std::int32_t token;
	double probability;

	[[nodiscard]] constexpr auto operator==(Outcome const&) const -> bool = default;
};

/**
 * The exact distribution `params` define over `logits`, in rank order: the surviving tokens (a prefix of the
 * candidates under `ranks_before`) and their renormalized probabilities, each its weight over the survivors' total.
 * Every other token has probability zero.
 */
export [[nodiscard]] auto distribution(Params const& params, std::span<float const> logits) -> std::vector<Outcome> {
	auto const kept = survivors(params, logits);
	return std::views::iota(0, kept.count) | std::views::transform([&](int index) {
		       return Outcome{.token = kept.id[index], .probability = kept.weight[index] / kept.total};
	       }) |
	       std::ranges::to<std::vector>();
}

/**
 * The token plain decoding samples at `position` from `logits` under `rule`: a cycle with no drafts, whose bonus
 * draw is this token.
 */
export [[nodiscard]] auto next_token(sampling_device::Rule const& rule, std::uint32_t position, std::span<float const> logits)
    -> std::int32_t {
	auto const targets = std::array{sampling_device::survivors(candidates(logits), rule.filter)};
	return sampling_device::settle(rule, position, targets, {}, {}).token;
}

}

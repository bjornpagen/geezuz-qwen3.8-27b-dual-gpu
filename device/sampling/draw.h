// SPDX-License-Identifier: AGPL-3.0-only
//
// The sampling arithmetic, held once for the device and the host reference: which candidates survive a request's
// filters, their weights, how a uniform becomes a token, and how a drafted token is verified. Everything is
// `constexpr` binary64 in which no rounded product feeds an addition (products only scale, divide or compare), so
// the device's fp contraction has nothing to fuse and both compilers produce the same bits from the same fp32 logits
// and the same uniforms. Raw pointers here are transient borrows of caller-owned arrays.
#pragma once

#include "sampling/exp.h"
#include "sampling/philox.h"

namespace engine::device::sampling {

/**
 * The most candidates a request may sample from. Each card keeps the 64 best logits of its half of the vocabulary
 * per lm_head row, and the pair merges 2 x 64 sorted entries into the row's 64: one warp, four per lane. The buffer
 * is 64 x 8 bytes per row, 12 KB per card at 24 rows. It covers the model card's top_k of 20 and llama.cpp's default
 * of 40; a wider request is refused rather than truncated, because truncation would change its distribution.
 */
inline constexpr int top_k_bound = 64;

/** One vocabulary entry: its fp32 logit and token id. */
struct Candidate {
	float logit;
	int id;
};

/** The one total order on candidates: higher logit first, and on equal logits the lower token id first. */
[[nodiscard]] constexpr auto ranks_before(Candidate a, Candidate b) -> bool {
	return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

/** Whether the nucleus (top-p) cut applies: off is the request's top_p of 1, which keeps every candidate. */
enum class Nucleus : unsigned char { Off, On };

/**
 * A request's filters, in llama.cpp's order: top-k, then top-p and min-p on the temperature-1 distribution of the
 * top-k survivors, then temperature. Valid by construction on the host: temperature in (0, 2], top_k in
 * [1, top_k_bound], top_p in [0, 1) when the nucleus is on, min_p in [0, 1]. Greedy is top_k 1.
 */
struct Filter {
	double temperature;
	double top_p;
	double min_p;
	int top_k;
	Nucleus nucleus;
};

/** What one slot's sampling needs on the device: its filters and the fixed part of its random keys. */
struct Rule {
	unsigned long long seed;
	Filter filter;
	unsigned int request;
};

/**
 * A finite distribution over token ids in rank order: `weight[i]` is token `id[i]`'s unnormalized weight for
 * i < count, `total` their sum accumulated in index order, and the probability of `id[i]` is weight[i] / total.
 * count >= 1 and total >= 2^-1022.
 */
struct Survivors {
	double weight[top_k_bound];
	double total;
	int id[top_k_bound];
	int count;
};

[[nodiscard]] constexpr auto logit_gap(Candidate const* sorted, int index) -> double {
	return static_cast<double>(sorted[index].logit) - static_cast<double>(sorted[0].logit);
}

/**
 * The distribution `filter` defines over the `available` best candidates, sorted by `ranks_before` (all of the
 * vocabulary's best `top_k_bound`, or the whole vocabulary when smaller). The survivors are a prefix of that order:
 * - top-k keeps the first top_k;
 * - top-p keeps the shortest prefix whose temperature-1 weights e^(l_i - l_0), summed in order, reach top_p times
 *   their sum over the top-k survivors (ties at the cut fall to the lower id, by the order);
 * - min-p keeps the leading candidates whose temperature-1 weight is at least min_p (p_i >= min_p * p_max);
 * - each survivor then weighs e^((l_i - l_0) / temperature), so the first weighs exactly 1.
 */
[[nodiscard]] constexpr auto survivors(Candidate const* sorted, int available, Filter const& filter) -> Survivors {
	int count = filter.top_k < available ? filter.top_k : available;
	if (filter.nucleus == Nucleus::On) {
		double mass = 0.0;
		for (int index = 0; index < count; ++index) {
			mass += exp_nonpositive(logit_gap(sorted, index));
		}
		double const reach = filter.top_p * mass;
		double cumulative = 0.0;
		for (int index = 0; index < count; ++index) {
			cumulative += exp_nonpositive(logit_gap(sorted, index));
			if (cumulative >= reach) {
				count = index + 1;
				break;
			}
		}
	}
	for (int index = 1; index < count; ++index) {
		if (exp_nonpositive(logit_gap(sorted, index)) < filter.min_p) {
			count = index;
			break;
		}
	}
	Survivors kept{};
	kept.count = count;
	for (int index = 0; index < count; ++index) {
		kept.id[index] = sorted[index].id;
		kept.weight[index] = exp_nonpositive(logit_gap(sorted, index) / filter.temperature);
		kept.total += kept.weight[index];
	}
	return kept;
}

/** The distribution that is certainly `id`: a greedy target or a greedy draft. */
[[nodiscard]] constexpr auto point_mass(int id) -> Survivors {
	Survivors certain{};
	certain.count = 1;
	certain.id[0] = id;
	certain.weight[0] = 1.0;
	certain.total = 1.0;
	return certain;
}

/** The probability of `id` under `distribution`: zero for an id outside its survivors. */
[[nodiscard]] constexpr auto probability_of(Survivors const& distribution, int id) -> double {
	for (int index = 0; index < distribution.count; ++index) {
		if (distribution.id[index] == id) {
			return distribution.weight[index] / distribution.total;
		}
	}
	return 0.0;
}

/**
 * The inverse cdf over the survivors in rank order: uniform u in [0, 1) selects the first index i with
 * u * total < weight[0] + ... + weight[i], the sums accumulated in the order `total` was, and the last survivor
 * otherwise. So token i owns the uniforms in [c_{i-1} / total, c_i / total): a survivor too light to move the running
 * sum owns none, and the last one's interval ends at 1 because u * total < total for every u <= 1 - 2^-53.
 */
[[nodiscard]] constexpr auto pick(Survivors const& distribution, double u) -> int {
	double const target = u * distribution.total;
	int index = 0;
	double cumulative = distribution.weight[0];
	while (index + 1 < distribution.count && !(target < cumulative)) {
		++index;
		cumulative += distribution.weight[index];
	}
	return index;
}

/**
 * The rejection residual norm(max(0, p - q)) over p's survivors, in p's rank order: a token outside q's survivors
 * (q is zero outside the draft vocabulary) keeps all of p, and tokens outside p's survivors have p = 0, so they
 * cannot appear.
 */
[[nodiscard]] constexpr auto residual(Survivors const& target, Survivors const& draft) -> Survivors {
	Survivors rest{};
	rest.count = target.count;
	for (int index = 0; index < target.count; ++index) {
		double const excess = target.weight[index] / target.total - probability_of(draft, target.id[index]);
		rest.id[index] = target.id[index];
		rest.weight[index] = excess > 0.0 ? excess : 0.0;
		rest.total += rest.weight[index];
	}
	return rest;
}

enum class Verdict : unsigned char { Accept, Reject };

/** A verified draft: accepted, or rejected with the token drawn from the residual in its place. */
struct Judgement {
	Verdict verdict;
	int token;
};

/**
 * Speculative verification (Leviathan et al.; Chen et al.): the draft `draft.id[index]`, drawn from q = `draft`,
 * is accepted with probability min(1, p(x) / q(x)) under the target p, tested as u_accept * q(x) < p(x); on
 * rejection the token is drawn from the residual with u_residual. The output is then distributed exactly as p. A
 * greedy target and draft are point masses, and this reduces to accepting exactly when the draft is the argmax.
 */
[[nodiscard]] constexpr auto judge(Survivors const& target, Survivors const& draft, int index, double u_accept, double u_residual)
    -> Judgement {
	int const token = draft.id[index];
	double const proposed = draft.weight[index] / draft.total;
	double const wanted = probability_of(target, token);
	// With u_accept < 1 the rounded product is below q(x) for every normal q(x), so p(x) >= q(x) always accepts.
	if (u_accept * proposed < wanted) {
		return Judgement{.verdict = Verdict::Accept, .token = token};
	}
	Survivors const rest = residual(target, draft);
	// Essential: the residual's true mass is at least q(x) - p(x), so a computed mass under the smallest normal means
	// p and q agree to rounding everywhere and the rejection itself has probability below that; accepting keeps the
	// inverse cdf on a normal total.
	if (!(rest.total >= 0x1p-1022)) {
		return Judgement{.verdict = Verdict::Accept, .token = token};
	}
	return Judgement{.verdict = Verdict::Reject, .token = rest.id[pick(rest, u_residual)]};
}

[[nodiscard]] constexpr auto key(Rule const& rule, unsigned int position, unsigned int step, Purpose purpose) -> Key {
	return Key{.seed = rule.seed, .request = rule.request, .position = position, .step = step, .purpose = purpose};
}

/** The drafter's proposal for the token at `position`, row `step` of the chain: an index into `draft`. */
[[nodiscard]] constexpr auto propose(Rule const& rule, unsigned int position, unsigned int step, Survivors const& draft) -> int {
	return pick(draft, uniform(key(rule, position, step, Purpose::Draft)));
}

/** The outcome of one slot's cycle: how many drafts were accepted, and the token decided after them. */
struct Settled {
	int accepted;
	int token;
};

/**
 * Verifies a chain of `depth` drafts for the tokens at position, position + 1, ...: `targets[d]` is p and
 * `drafts[d]` is q for the token at position + d, and `chosen[d]` indexes the proposal in `drafts[d]`. Drafts are
 * accepted in order until one is rejected, whose residual token ends the cycle; when all are accepted the bonus
 * token is drawn from `targets[depth]`.
 */
[[nodiscard]] constexpr auto settle(Rule const& rule, unsigned int position, Survivors const* targets, Survivors const* drafts,
                                    int const* chosen, int depth) -> Settled {
	for (int step = 0; step < depth; ++step) {
		auto const row = static_cast<unsigned int>(step);
		Judgement const judgement =
		    judge(targets[step], drafts[step], chosen[step], uniform(key(rule, position + row, row, Purpose::Accept)),
			      uniform(key(rule, position + row, row, Purpose::Residual)));
		if (judgement.verdict == Verdict::Reject) {
			return Settled{.accepted = step, .token = judgement.token};
		}
	}
	auto const row = static_cast<unsigned int>(depth);
	Survivors const& last = targets[depth];
	return Settled{.accepted = depth, .token = last.id[pick(last, uniform(key(rule, position + row, row, Purpose::Sample)))]};
}

}

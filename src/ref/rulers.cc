// SPDX-License-Identifier: AGPL-3.0-only
//
// The rulers between two dumps of one token sequence: top-1 agreement, the kl divergence, each side's perplexity, and
// the decode-path ruler's first divergence.
export module engine.ref:rulers;

import std;
import engine.core;
import engine.model;
import :dump;
import :forward;

namespace engine::ref {

namespace model = engine::model;

using model::Count;

/** exp of the mean negative log-likelihood of each row's next token. */
export [[nodiscard]] auto perplexity(Dump const& dump) -> double {
	if (dump.next.empty()) {
		return 1.0;
	}
	auto total = 0.0;
	for (auto const [row, logit] : std::views::zip(dump.rows, dump.next)) {
		total += row.logsumexp - static_cast<double>(logit);
	}
	return std::exp(total / static_cast<double>(dump.next.size()));
}

/** One row's divergence of `q` from `p`, over p's candidates and one bucket for the rest of each distribution. */
struct RowDivergence {
	double kl;
	/** Whether a candidate of p is not among q's, so q's last kept probability stood in for it: an upper bound. */
	bool bounded;
};

[[nodiscard]] auto row_divergence(DumpRow const& p, DumpRow const& q) -> RowDivergence {
	auto const floor = std::exp(static_cast<double>(q.best.back().logit) - q.logsumexp);
	auto kl = 0.0;
	auto p_mass = 0.0;
	auto q_mass = 0.0;
	auto bounded = false;
	for (auto const& each : p.best) {
		auto const log_p = static_cast<double>(each.logit) - p.logsumexp;
		auto const found = std::ranges::find(q.best, each.id, &Ranked::id);
		bounded = bounded || found == q.best.end();
		auto const log_q = found == q.best.end() ? std::log(floor) : static_cast<double>(found->logit) - q.logsumexp;
		auto const probability = std::exp(log_p);
		kl += probability * (log_p - log_q);
		p_mass += probability;
		q_mass += std::exp(log_q);
	}
	// The rest of each distribution as one outcome; its mass on q's side is at least the smallest normal double.
	auto const p_rest = std::max(0.0, 1.0 - p_mass);
	auto const q_rest = std::max(std::numeric_limits<double>::min(), 1.0 - q_mass);
	if (p_rest > 0.0) {
		kl += p_rest * (std::log(p_rest) - std::log(q_rest));
	}
	return {.kl = kl, .bounded = bounded};
}

export struct Comparison {
	Count rows;
	/** Rows whose best token agrees, over all rows. */
	double top1_agreement;
	/** KL(a || b) per row, over a's best candidates and one bucket for the rest. */
	double mean_kl;
	double max_kl;
	Count max_kl_row;
	/** Rows where one of a's candidates is missing from b's list, so its probability under b is bounded, not known. */
	Count bounded_rows;
	double perplexity_a;
	double perplexity_b;
};

export enum class RulerFailure : std::uint8_t {
	/** The two dumps are not of one token sequence. */
	DifferentTokens,
};

export struct [[nodiscard]] RulerError {
	RulerFailure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case RulerFailure::DifferentTokens:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(RulerError const&) const -> bool = default;
};

export [[nodiscard]] auto compare(Dump const& a, Dump const& b) -> core::Result<Comparison, RulerError> {
	if (a.tokens != b.tokens) {
		return std::unexpected(RulerError{.failure = RulerFailure::DifferentTokens});
	}
	auto result = Comparison{
	    .rows = a.rows.size(),
	    .top1_agreement = 0.0,
	    .mean_kl = 0.0,
	    .max_kl = 0.0,
	    .max_kl_row = 0,
	    .bounded_rows = 0,
	    .perplexity_a = perplexity(a),
	    .perplexity_b = perplexity(b),
	};
	auto agree = Count{0};
	auto total = 0.0;
	for (auto const [index, rows] : std::views::enumerate(std::views::zip(a.rows, b.rows))) {
		auto const& [left, right] = rows;
		agree += left.best.front().id == right.best.front().id ? 1 : 0;
		auto const divergence = row_divergence(left, right);
		total += divergence.kl;
		result.bounded_rows += divergence.bounded ? 1 : 0;
		if (divergence.kl > result.max_kl) {
			result.max_kl = divergence.kl;
			result.max_kl_row = static_cast<Count>(index);
		}
	}
	result.top1_agreement = static_cast<double>(agree) / static_cast<double>(a.rows.size());
	result.mean_kl = total / static_cast<double>(a.rows.size());
	return result;
}

/** Where a run's chosen token first differs from the reference's best, with both rows' candidates. */
export struct Divergence {
	Count row;
	/** The token the run chose after row `row`. */
	Token chosen;
	std::vector<Ranked> reference;
	std::vector<Ranked> run;
};

/**
 * The decode-path ruler: for every row whose next token the run chose (not one it was given), whether the reference,
 * teacher-forced over the same tokens, ranks that token first; the first row where it does not.
 */
export [[nodiscard]] auto first_divergence(Dump const& reference, Dump const& run) -> core::Result<std::optional<Divergence>, RulerError> {
	if (reference.tokens != run.tokens) {
		return std::unexpected(RulerError{.failure = RulerFailure::DifferentTokens});
	}
	for (auto row = run.prompt - 1; row + 1 < run.tokens.size(); ++row) {
		auto const chosen = run.tokens[row + 1];
		if (reference.rows[row].best.front().id != chosen) {
			return Divergence{.row = row, .chosen = chosen, .reference = reference.rows[row].best, .run = run.rows[row].best};
		}
	}
	return std::nullopt;
}

}

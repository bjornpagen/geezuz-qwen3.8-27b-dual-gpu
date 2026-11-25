// SPDX-License-Identifier: AGPL-3.0-only
//
// The dump format and the rulers: encode and decode are inverse, any changed byte is caught, and the rulers agree with
// float64 computed from the full distributions the dumps summarize.
import std;
import engine.model;
import engine.ref;
import engine.testing;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;

using model::Count;

/** Logits whose mass sits on at most 40 tokens, the rest far below: the best 64 then hold all but 1e-15 of it. */
[[nodiscard]] auto peaked(std::mt19937_64& generator) -> std::vector<float> {
	auto logits = std::vector<float>(model::sampled_vocabulary, -40.0F);
	auto pick = std::uniform_int_distribution<Count>{0, model::sampled_vocabulary - 1};
	auto value = std::uniform_real_distribution<float>{-4.0F, 8.0F};
	for (auto count = 0; count < 40; ++count) {
		logits[pick(generator)] = value(generator);
	}
	return logits;
}

/** Each logit nudged, as two arithmetics of one model would. */
[[nodiscard]] auto nudged(std::vector<float> logits, std::mt19937_64& generator) -> std::vector<float> {
	auto noise = std::normal_distribution<float>{0.0F, 0.05F};
	for (auto& each : logits) {
		each += noise(generator);
	}
	return logits;
}

struct Pair {
	std::vector<ref::Token> tokens;
	std::vector<std::vector<float>> a;
	std::vector<std::vector<float>> b;
};

[[nodiscard]] auto pair(std::uint64_t seed, Count rows) -> Pair {
	auto generator = engine::testing::generator(seed);
	auto pick = std::uniform_int_distribution<ref::Token>{0, model::sampled_vocabulary - 1};
	auto result = Pair{};
	for (auto row = Count{0}; row < rows; ++row) {
		result.tokens.push_back(pick(generator));
		result.a.push_back(peaked(generator));
		result.b.push_back(nudged(result.a.back(), generator));
	}
	return result;
}

[[nodiscard]] auto log_softmax(std::vector<float> const& logits) -> std::vector<double> {
	auto const most = static_cast<double>(std::ranges::max(logits));
	auto total = 0.0;
	for (auto const each : logits) {
		total += std::exp(static_cast<double>(each) - most);
	}
	auto out = std::vector<double>{};
	for (auto const each : logits) {
		out.push_back(static_cast<double>(each) - most - std::log(total));
	}
	return out;
}

[[nodiscard]] auto full_kl(std::vector<float> const& p, std::vector<float> const& q) -> double {
	auto const lp = log_softmax(p);
	auto const lq = log_softmax(q);
	auto kl = 0.0;
	for (auto const [a, b] : std::views::zip(lp, lq)) {
		kl += std::exp(a) * (a - b);
	}
	return kl;
}

[[nodiscard]] auto rulers_match_float64(std::uint64_t seed) -> bool {
	auto const data = pair(seed, 6);
	auto const a = ref::dump_of(ref::Source::Tensor, 2, data.tokens, data.a);
	auto const b = ref::dump_of(ref::Source::Ieee, 2, data.tokens, data.b);
	auto const compared = ref::compare(a, b);
	if (!compared) {
		return false;
	}
	auto kl = 0.0;
	auto agree = 0.0;
	auto nll = 0.0;
	for (auto row = Count{0}; row < data.tokens.size(); ++row) {
		kl += full_kl(data.a[row], data.b[row]);
		agree += std::ranges::max_element(data.a[row]) - data.a[row].begin() == std::ranges::max_element(data.b[row]) - data.b[row].begin() ? 1.0 : 0.0;
		if (row + 1 < data.tokens.size()) {
			nll -= log_softmax(data.a[row])[data.tokens[row + 1]];
		}
	}
	kl /= static_cast<double>(data.tokens.size());
	agree /= static_cast<double>(data.tokens.size());
	auto const ppl = std::exp(nll / static_cast<double>(data.tokens.size() - 1));
	return std::abs(compared->mean_kl - kl) <= 1e-3 * kl + 1e-9 && compared->top1_agreement == agree &&
	       std::abs(compared->perplexity_a - ppl) <= 1e-9 * ppl;
}

[[nodiscard]] auto self_comparison_is_null(std::uint64_t seed) -> bool {
	auto const data = pair(seed, 4);
	auto const a = ref::dump_of(ref::Source::Tensor, 1, data.tokens, data.a);
	auto const compared = ref::compare(a, a);
	return compared && compared->top1_agreement == 1.0 && std::abs(compared->mean_kl) < 1e-12 && compared->bounded_rows == 0;
}

[[nodiscard]] auto round_trip_and_tamper(std::uint64_t seed) -> bool {
	auto const data = pair(seed, 5);
	auto const dump = ref::dump_of(ref::Source::Engine, 3, data.tokens, data.a);
	auto bytes = ref::encode(dump);
	auto const back = ref::decode(bytes);
	if (!back || *back != dump) {
		return false;
	}
	auto generator = engine::testing::generator(seed + 1);
	auto where = std::uniform_int_distribution<Count>{0, bytes.size() - 1};
	for (auto trial = 0; trial < 64; ++trial) {
		auto tampered = bytes;
		auto const at = where(generator);
		tampered[at] ^= std::byte{0x10};
		if (ref::decode(tampered).has_value()) {
			return false;
		}
	}
	bytes.pop_back();
	return ref::decode(bytes).error().failure == ref::DumpFailure::Truncated;
}

/** The decode ruler finds the first chosen token the reference does not rank first, and only rows the run chose. */
[[nodiscard]] auto decode_ruler_finds_the_first(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto logits = std::vector<std::vector<float>>{};
	for (auto row = 0; row < 6; ++row) {
		logits.push_back(peaked(generator));
	}
	auto tokens = std::vector<ref::Token>{5, 6};
	for (auto row = Count{1}; row < 5; ++row) {
		auto const best = ref::summarize(logits[row]).best.front().id;
		tokens.push_back(row == 3 ? best + 1 : best);
	}
	auto const reference = ref::dump_of(ref::Source::Tensor, 2, tokens, logits);
	auto const found = ref::first_divergence(reference, reference);
	return found && found->has_value() && (*found)->row == 3 && (*found)->chosen == tokens[4];
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "kl, top-1 agreement and perplexity match float64 over full distributions (seed 31)", .holds = rulers_match_float64(31)},
	    engine::testing::Case{.name = "a dump against itself: full agreement, zero kl, nothing bounded (seed 32)", .holds = self_comparison_is_null(32)},
	    engine::testing::Case{.name = "dumps round-trip, and any flipped byte or missing byte is refused (seed 33)", .holds = round_trip_and_tamper(33)},
	    engine::testing::Case{.name = "the decode ruler reports the first chosen token the reference does not rank first (seed 34)", .holds = decode_ruler_finds_the_first(34)},
	};
	return engine::testing::report(cases);
}

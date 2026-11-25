// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.sampling;
import engine.testing;

namespace {

using engine::sampling::Greedy;
using engine::sampling::Outcome;
using engine::sampling::Params;
using engine::sampling::Defaults;
using engine::sampling::Request;
using engine::sampling::Sample;

// The oracle: the filter chain written plainly from its definition, independently of the engine's arithmetic. It
// sorts with a stable sort on logits alone (so equal logits keep ascending ids), uses libm's exp, normalizes before
// cutting as llama.cpp does, and reports where a cut decision sits within rounding of its threshold.
struct Expected {
	std::vector<int> order;
	int count;
	bool ambiguous;
};

struct Filters {
	double temperature;
	int top_k;
	std::optional<double> top_p;
	double min_p;
};

[[nodiscard]] auto filters_of(Params const& params) -> Filters {
	if (std::holds_alternative<Greedy>(params)) {
		return Filters{.temperature = 1.0, .top_k = 1, .top_p = std::nullopt, .min_p = 0.0};
	}
	auto const& sample = std::get<Sample>(params);
	return Filters{.temperature = sample.temperature(), .top_k = sample.top_k(), .top_p = sample.top_p(), .min_p = sample.min_p()};
}

[[nodiscard]] auto oracle(std::span<float const> logits, Filters const& filters) -> Expected {
	auto order = std::views::iota(0, static_cast<int>(logits.size())) | std::ranges::to<std::vector>();
	std::ranges::stable_sort(order, std::greater{}, [&](int id) {
		return logits[static_cast<std::size_t>(id)];
	});
	auto count = std::min(filters.top_k, static_cast<int>(order.size()));
	auto const top = static_cast<double>(logits[static_cast<std::size_t>(order[0])]);
	auto relative = std::vector<double>{};
	for (auto index = 0; index < count; ++index) {
		relative.push_back(std::exp(static_cast<double>(logits[static_cast<std::size_t>(order[static_cast<std::size_t>(index)])]) - top));
	}
	auto const mass = std::ranges::fold_left(relative, 0.0, std::plus{});
	auto ambiguous = false;
	auto const near = [](double value, double threshold) {
		return std::abs(value - threshold) <= 1e-12 * std::max(1.0, threshold);
	};
	if (filters.top_p) {
		auto cumulative = 0.0;
		for (auto index = 0; index < count; ++index) {
			cumulative += relative[static_cast<std::size_t>(index)] / mass;
			ambiguous = ambiguous || near(cumulative, *filters.top_p);
			if (cumulative >= *filters.top_p) {
				count = index + 1;
				break;
			}
		}
	}
	for (auto index = 0; index < count; ++index) {
		ambiguous = ambiguous || (filters.min_p > 0.0 && near(relative[static_cast<std::size_t>(index)], filters.min_p));
	}
	for (auto index = 1; index < count; ++index) {
		if (relative[static_cast<std::size_t>(index)] < filters.min_p) {
			count = index;
			break;
		}
	}
	return Expected{.order = std::move(order), .count = count, .ambiguous = ambiguous};
}

// The probabilities of the first `count` tokens of `order` at the filter's temperature, by libm.
[[nodiscard]] auto oracle_probabilities(std::span<float const> logits, std::span<int const> order, int count, double temperature)
    -> std::vector<double> {
	auto const top = static_cast<double>(logits[static_cast<std::size_t>(order[0])]);
	auto weights = order.first(static_cast<std::size_t>(count)) | std::views::transform([&](int id) {
		               return std::exp((static_cast<double>(logits[static_cast<std::size_t>(id)]) - top) / temperature);
	               }) |
	               std::ranges::to<std::vector>();
	auto const total = std::ranges::fold_left(weights, 0.0, std::plus{});
	for (auto& weight : weights) {
		weight /= total;
	}
	return weights;
}

[[nodiscard]] auto close(double actual, double expected) -> bool {
	return std::abs(actual - expected) <= 1e-12 * expected + 1e-300;
}

// The engine's distribution equals the oracle's: the same survivors in the same order and the same probabilities.
// At a cut within rounding of its threshold, either side of the cut is the definition's answer, so the engine's own
// count is checked to be adjacent and its probabilities to match that count.
[[nodiscard]] auto matches_oracle(Params const& params, std::span<float const> logits) -> bool {
	auto const filters = filters_of(params);
	auto const expected = oracle(logits, filters);
	auto const actual = engine::sampling::distribution(params, logits);
	auto const count = static_cast<int>(actual.size());
	if (count < 1 || (expected.ambiguous ? std::abs(count - expected.count) > 1 : count != expected.count)) {
		return false;
	}
	auto const probabilities = oracle_probabilities(logits, expected.order, count, filters.temperature);
	auto holds = true;
	for (auto index = std::size_t{0}; index < actual.size(); ++index) {
		holds = holds && actual[index].token == expected.order[index] && close(actual[index].probability, probabilities[index]);
	}
	return holds;
}

/** Every filter at once: a request sets temperature and top_p, the model's defaults top-k and min-p. */
struct Setting {
	double temperature;
	std::int64_t top_k;
	double top_p;
	double min_p;
};

[[nodiscard]] auto params_of(Setting const& setting, std::uint64_t seed) -> Params {
	return engine::sampling::parse(Request{.temperature = setting.temperature, .top_p = setting.top_p},
	                               Defaults{.temperature = 1.0, .top_k = setting.top_k, .top_p = 1.0, .min_p = setting.min_p}, seed)
	    .value();
}

[[nodiscard]] auto params_of(Setting const& setting) -> Params {
	return params_of(setting, 11);
}

// Every logit vector over a four-token vocabulary drawn from a grid with many ties, under every grid of filters.
[[nodiscard]] auto exhaustive_small_vocabulary() -> bool {
	constexpr auto grid = std::array{-1.0F, 0.0F, 0.5F, 2.0F};
	auto holds = true;
	for (auto code = std::size_t{0}; code < 256; ++code) {
		auto const logits = std::array{grid[code & 3U], grid[(code >> 2U) & 3U], grid[(code >> 4U) & 3U], grid[(code >> 6U) & 3U]};
		for (auto const temperature : {0.0, 0.5, 1.0, 2.0}) {
			for (auto const top_k : {1, 2, 3, 4}) {
				for (auto const top_p : {0.0, 0.3, 0.5, 0.9, 1.0}) {
					for (auto const min_p : {0.0, 0.2, 0.5, 1.0}) {
						holds = holds &&
						        matches_oracle(
						            params_of(Setting{.temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p}),
						            logits);
					}
				}
			}
		}
	}
	return holds;
}

[[nodiscard]] auto random_logits(std::mt19937_64& generator) -> std::vector<float> {
	auto size = std::uniform_int_distribution<int>{1, 150};
	auto mode = std::uniform_int_distribution<int>{0, 3};
	auto spread = std::uniform_real_distribution<float>{-15.0F, 15.0F};
	auto pick = std::uniform_int_distribution<int>{0, 5};
	constexpr auto grid = std::array{-2.0F, -1.0F, 0.0F, 0.5F, 1.0F, 3.0F};
	auto const shape = mode(generator);
	auto logits = std::vector<float>(static_cast<std::size_t>(size(generator)));
	for (auto& logit : logits) {
		switch (shape) {
		case 0:
			logit = spread(generator);
			break;
		case 1:
			logit = grid[static_cast<std::size_t>(pick(generator))];
			break;
		case 2:
			// Gaps beyond the exponential's range: weights underflow at temperature 1 and below.
			logit = pick(generator) == 0 ? -1.0e4F : spread(generator);
			break;
		default:
			logit = std::nextafter(7.0F, 7.0F + static_cast<float>(pick(generator) - 3));
			break;
		}
	}
	return logits;
}

[[nodiscard]] auto random_request(std::mt19937_64& generator) -> Setting {
	auto unit = std::uniform_real_distribution<double>{0.0, 1.0};
	auto choice = std::uniform_int_distribution<int>{0, 7};
	auto top_k = std::uniform_int_distribution<int>{1, 64};
	auto const temperatures = std::array{0.0, 1e-3, 0.3, 0.7, 1.0, 1.5, 2.0, 2.0 * unit(generator)};
	auto const top_ks = std::array<std::int64_t, 8>{1, 2, 5, 20, 40, 63, 64, top_k(generator)};
	auto const top_ps = std::array{0.0, 0.1, 0.5, 0.8, 0.95, 1.0, 1.0, unit(generator)};
	auto const min_ps = std::array{0.0, 0.0, 0.01, 0.05, 0.2, 0.9, 1.0, unit(generator)};
	return Setting{
	    .temperature = temperatures[static_cast<std::size_t>(choice(generator))],
	    .top_k = top_ks[static_cast<std::size_t>(choice(generator))],
	    .top_p = top_ps[static_cast<std::size_t>(choice(generator))],
	    .min_p = min_ps[static_cast<std::size_t>(choice(generator))],
	};
}

[[nodiscard]] auto random_vocabularies_match_oracle(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto holds = true;
	for (auto round = 0; round < 40'000; ++round) {
		auto const logits = random_logits(generator);
		holds = holds && matches_oracle(params_of(random_request(generator)), logits);
	}
	return holds;
}

// The candidate list is the exact top-k: the prefix of a full stable sort, over vocabularies wider than the bound.
[[nodiscard]] auto candidates_are_the_exact_top_k(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto size = std::uniform_int_distribution<int>{1, 400};
	auto value = std::uniform_int_distribution<int>{-6, 6};
	auto holds = true;
	for (auto round = 0; round < 5'000; ++round) {
		auto logits = std::vector<float>(static_cast<std::size_t>(size(generator)));
		std::ranges::generate(logits, [&] {
			return static_cast<float>(value(generator)) * 0.25F;
		});
		auto const expected = oracle(logits, Filters{.temperature = 1.0, .top_k = 64, .top_p = std::nullopt, .min_p = 0.0});
		auto const actual = engine::sampling::candidates(logits);
		holds = holds && actual.size() == std::min(logits.size(), std::size_t{64});
		for (auto index = std::size_t{0}; index < actual.size(); ++index) {
			holds = holds && actual[index].id == expected.order[index] &&
			        actual[index].logit == logits[static_cast<std::size_t>(actual[index].id)];
		}
	}
	return holds;
}

[[nodiscard]] auto ties_fall_to_the_lower_id() -> bool {
	auto const flat = std::vector<float>(100, 1.5F);
	auto const top3 = engine::sampling::distribution(
	    params_of(Setting{.temperature = 1.0, .top_k = 3, .top_p = 1.0, .min_p = 0.0}), flat);
	auto const nucleus = engine::sampling::distribution(
	    params_of(Setting{.temperature = 1.0, .top_k = 64, .top_p = 0.05, .min_p = 0.0}), flat);
	auto const greedy = engine::sampling::distribution(Params{Greedy{}}, flat);
	// Four equal weights reach exactly half the mass at the second: the cut keeps the shortest prefix that reaches it.
	auto const half = engine::sampling::distribution(
	    params_of(Setting{.temperature = 1.0, .top_k = 4, .top_p = 0.5, .min_p = 0.0}), flat);
	return top3 == std::vector<Outcome>{{.token = 0, .probability = 1.0 / 3.0},
	                                    {.token = 1, .probability = 1.0 / 3.0},
	                                    {.token = 2, .probability = 1.0 / 3.0}} &&
	       nucleus.size() == 4 && nucleus.back().token == 3 && greedy == std::vector<Outcome>{{.token = 0, .probability = 1.0}} &&
	       half == std::vector<Outcome>{{.token = 0, .probability = 0.5}, {.token = 1, .probability = 0.5}};
}

// The inverse cdf gives each survivor the share of the uniform grid its probability says, never a zero weight, and
// the largest uniform lands on the last survivor that moves the running sum.
[[nodiscard]] auto inverse_cdf_shares(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto constexpr grid = 65'536;
	auto holds = true;
	for (auto round = 0; round < 300; ++round) {
		auto const logits = random_logits(generator);
		auto const params = params_of(random_request(generator));
		auto const kept = engine::sampling::survivors(params, logits);
		auto counts = std::vector<int>(static_cast<std::size_t>(kept.count), 0);
		for (auto step = 0; step < grid; ++step) {
			auto const index = engine::sampling::pick(kept, std::ldexp(static_cast<double>(step), -16));
			holds = holds && index >= 0 && index < kept.count && kept.weight[index] > 0.0;
			counts[static_cast<std::size_t>(index)] += 1;
		}
		for (auto index = 0; index < kept.count; ++index) {
			auto const share = static_cast<double>(counts[static_cast<std::size_t>(index)]) / grid;
			holds = holds && std::abs(share - kept.weight[index] / kept.total) <= 1.0 / grid + 1e-12;
		}
		// A survivor so light that adding it leaves the running sum unchanged owns no uniform.
		auto last = 0;
		auto cumulative = 0.0;
		for (auto index = 0; index < kept.count; ++index) {
			auto const next = cumulative + kept.weight[index];
			last = next > cumulative ? index : last;
			cumulative = next;
		}
		holds = holds && engine::sampling::pick(kept, 1.0 - 0x1p-53) == last;
	}
	return holds;
}

// A uniform exactly on a cumulative boundary belongs to the next survivor: survivor i owns [c_{i-1}, c_i) of the mass.
[[nodiscard]] auto boundaries_belong_to_the_next_survivor() -> bool {
	auto const flat = std::vector<float>(8, 0.0F);
	auto const kept = engine::sampling::survivors(
	    params_of(Setting{.temperature = 1.0, .top_k = 4, .top_p = 1.0, .min_p = 0.0}), flat);
	auto holds = kept.count == 4;
	for (auto index = 0; index < 4; ++index) {
		holds = holds && engine::sampling::pick(kept, index / 4.0) == index &&
		        engine::sampling::pick(kept, std::nextafter((index + 1) / 4.0, 0.0)) == index;
	}
	return holds;
}

[[nodiscard]] auto chi_square(std::span<double const> observed, std::span<double const> expected) -> double {
	auto statistic = 0.0;
	for (auto index = std::size_t{0}; index < observed.size(); ++index) {
		statistic += (observed[index] - expected[index]) * (observed[index] - expected[index]) / expected[index];
	}
	return statistic;
}

// Tokens drawn over consecutive positions follow the reference distribution: chi-square with a 1e-6 tail bound.
[[nodiscard]] auto drawn_tokens_follow_the_distribution(std::uint64_t seed) -> bool {
	auto const logits = std::vector<float>{2.0F, 1.5F, 1.5F, 0.0F, -1.0F, 3.0F, 2.5F, -0.5F, 1.0F, 0.25F};
	auto const params =
	    params_of(Setting{.temperature = 0.8, .top_k = 8, .top_p = 0.97, .min_p = 0.02}, seed);
	auto const rule = engine::sampling::rule(params, 0);
	auto const reference = engine::sampling::distribution(params, logits);
	auto constexpr draws = 400'000;
	auto observed = std::vector<double>(reference.size(), 0.0);
	auto holds = true;
	for (auto position = std::uint32_t{0}; position < draws; ++position) {
		auto const token = engine::sampling::next_token(rule, position, logits);
		auto const found = std::ranges::find(reference, token, &Outcome::token);
		holds = holds && found != reference.end();
		if (found != reference.end()) {
			observed[static_cast<std::size_t>(found - reference.begin())] += 1.0;
		}
	}
	auto const expected = reference | std::views::transform([](Outcome const& outcome) {
		                      return outcome.probability * draws;
	                      }) |
	                      std::ranges::to<std::vector>();
	// 1e-6 upper tail of chi-square for up to 7 degrees of freedom.
	return holds && reference.size() >= 4 && chi_square(observed, expected) < 37.0;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every four-token logit vector on a tie-heavy grid, under every filter grid, matches the oracle",
	        .holds = exhaustive_small_vocabulary(),
	    },
	    engine::testing::Case{.name = "random vocabularies and filters match the oracle (seed 21)",
		                      .holds = random_vocabularies_match_oracle(21)},
	    engine::testing::Case{.name = "random vocabularies and filters match the oracle (seed 22)",
		                      .holds = random_vocabularies_match_oracle(22)},
	    engine::testing::Case{.name = "the candidates are the exact top 64 of a full stable sort (seed 23)",
		                      .holds = candidates_are_the_exact_top_k(23)},
	    engine::testing::Case{.name = "ties at the top-k and top-p cuts and at the argmax fall to the lower id",
		                      .holds = ties_fall_to_the_lower_id()},
	    engine::testing::Case{.name = "a uniform on a cumulative boundary belongs to the next survivor",
		                      .holds = boundaries_belong_to_the_next_survivor()},
	    engine::testing::Case{
	        .name = "the inverse cdf gives each survivor its probability's share of a 2^16 uniform grid (seed 24)",
	        .holds = inverse_cdf_shares(24),
	    },
	    engine::testing::Case{
	        .name = "tokens drawn over 400,000 positions pass chi-square against the distribution (seed 25)",
	        .holds = drawn_tokens_follow_the_distribution(25),
	    },
	};
	return engine::testing::report(cases);
}

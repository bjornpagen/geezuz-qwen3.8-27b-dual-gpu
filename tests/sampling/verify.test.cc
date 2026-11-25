// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.sampling;
import engine.testing;

namespace {

using engine::sampling::Candidate;
using engine::sampling::Greedy;
using engine::sampling::Params;
using engine::sampling::Request;
using engine::sampling::Rule;
using engine::sampling::Survivors;
using engine::sampling::Verdict;

// A distribution over ids 0..n-1 with the given integer weights, out of their sum.
[[nodiscard]] auto counted(std::span<int const> weights, std::span<int const> ids) -> Survivors {
	auto distribution = Survivors{};
	distribution.count = static_cast<int>(weights.size());
	for (auto index = 0; index < distribution.count; ++index) {
		distribution.id[index] = ids[static_cast<std::size_t>(index)];
		distribution.weight[index] = weights[static_cast<std::size_t>(index)];
		distribution.total += distribution.weight[index];
	}
	return distribution;
}

// Every way to split `total` units over `parts` tokens.
[[nodiscard]] auto compositions(int total, int parts) -> std::vector<std::vector<int>> {
	if (parts == 1) {
		return {{total}};
	}
	auto all = std::vector<std::vector<int>>{};
	for (auto first = 0; first <= total; ++first) {
		for (auto rest : compositions(total - first, parts - 1)) {
			rest.insert(rest.begin(), first);
			all.push_back(std::move(rest));
		}
	}
	return all;
}

[[nodiscard]] auto weight_of(Survivors const& distribution, int id) -> double {
	for (auto index = 0; index < distribution.count; ++index) {
		if (distribution.id[index] == id) {
			return distribution.weight[index];
		}
	}
	return 0.0;
}

[[nodiscard]] auto units(Survivors const& distribution, int id) -> std::int64_t {
	return static_cast<std::int64_t>(engine::sampling::probability_of(distribution, id) * 8.0);
}

// With p and q in eighths every quantity is exact in binary64, so the output distribution of the rule (accept x with
// probability min(1, p(x)/q(x)), else draw from the engine's residual) is computed in integers and must equal p
// exactly: q(x) min(1, p/q) = min(p, q)(x), the rejection mass is the sum of (q - p)+, and so
// out(y) * C = C min(p, q)(y) + (sum (q - p)+) c(y), with c the residual's weights and C their total, in eighths.
// q is zero outside its own ids: over three ids, token 3 lies outside the draft vocabulary.
[[nodiscard]] auto output_is_exactly_p(Survivors const& target, Survivors const& draft) -> bool {
	auto const rest = engine::sampling::residual(target, draft);
	auto const rejected_mass = std::ranges::fold_left(std::views::iota(0, 4) | std::views::transform([&](int id) {
		                                                  return std::max<std::int64_t>(0, units(draft, id) - units(target, id));
	                                                  }),
	                                                  std::int64_t{0}, std::plus{});
	auto const rest_total = static_cast<std::int64_t>(rest.total * 8.0);
	auto holds = true;
	for (auto id = 0; id < 4; ++id) {
		auto const kept = std::min(units(target, id), units(draft, id));
		if (rest_total == 0) {
			holds = holds && rejected_mass == 0 && kept == units(target, id);
			continue;
		}
		auto const rest_units = static_cast<std::int64_t>(weight_of(rest, id) * 8.0);
		holds = holds && rest_total * kept + rejected_mass * rest_units == rest_total * units(target, id);
	}
	return holds;
}

// The judge realizes that rule: it accepts exactly the uniforms with u q(x) < p(x) (all of them when p(x) >= q(x)),
// and on rejection draws the residual's token with the residual uniform.
[[nodiscard]] auto judge_realizes_the_rule(Survivors const& target, Survivors const& draft, std::span<double const> uniforms) -> bool {
	auto const rest = engine::sampling::residual(target, draft);
	auto holds = true;
	for (auto index = 0; index < draft.count; ++index) {
		if (draft.weight[index] == 0.0) {
			continue;
		}
		auto const token = draft.id[index];
		auto const p = engine::sampling::probability_of(target, token);
		auto const q = draft.weight[index] / draft.total;
		for (auto const u : uniforms) {
			auto const judgement = engine::sampling::judge(target, draft, index, u, u);
			auto const accept = p >= q ? true : u * q < p;
			holds = holds && (judgement.verdict == Verdict::Accept) == accept;
			holds = holds && judgement.token == (accept ? token : rest.id[engine::sampling::pick(rest, u)]);
		}
	}
	return holds;
}

[[nodiscard]] auto exhaustive_eighths() -> bool {
	auto const targets = compositions(8, 4);
	auto const full_drafts = compositions(8, 4);
	auto const narrow_drafts = compositions(8, 3);
	auto uniforms = std::vector<double>{0.0, 1.0 - 0x1p-53};
	for (auto numerator = 1; numerator < 8; ++numerator) {
		for (auto denominator = numerator + 1; denominator <= 8; ++denominator) {
			auto const ratio = static_cast<double>(numerator) / denominator;
			uniforms.insert(uniforms.end(), {std::nextafter(ratio, 0.0), ratio, std::nextafter(ratio, 1.0)});
		}
	}
	constexpr auto ids = std::array{0, 1, 2, 3};
	constexpr auto shuffled = std::array{2, 0, 3, 1};
	auto holds = true;
	for (auto const& p : targets) {
		auto const target = counted(p, shuffled);
		for (auto const& q : full_drafts) {
			auto const draft = counted(q, ids);
			holds = holds && output_is_exactly_p(target, draft) && judge_realizes_the_rule(target, draft, uniforms);
		}
		for (auto const& q : narrow_drafts) {
			auto const draft = counted(q, std::span{ids}.first(3));
			holds = holds && output_is_exactly_p(target, draft) && judge_realizes_the_rule(target, draft, uniforms);
		}
	}
	return holds;
}

[[nodiscard]] auto sorted_candidates(std::span<float const> logits, std::span<int const> ids) -> std::vector<Candidate> {
	auto candidates = std::views::zip(logits, ids) | std::views::transform([](auto const& pair) {
		                  return Candidate{.logit = std::get<0>(pair), .id = std::get<1>(pair)};
	                  }) |
	                  std::ranges::to<std::vector>();
	std::ranges::sort(candidates, engine::sampling::ranks_before);
	candidates.resize(std::min(candidates.size(), std::size_t{engine::sampling::top_k_bound}));
	return candidates;
}

// Over real distributions from logits, with the drafter restricted to two thirds of the vocabulary, the output
// distribution computed from the rule's parts equals p to rounding.
[[nodiscard]] auto real_distributions_recover_p(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto logit = std::uniform_real_distribution<float>{-6.0F, 6.0F};
	auto unit = std::uniform_real_distribution<double>{0.05, 1.0};
	auto top_k = std::uniform_int_distribution<int>{2, 64};
	auto const all_ids = std::views::iota(0, 200) | std::ranges::to<std::vector>();
	auto const draft_ids = all_ids | std::views::filter([](int id) {
		                       return id % 3 != 0;
	                       }) |
	                       std::ranges::to<std::vector>();
	auto holds = true;
	for (auto round = 0; round < 3'000; ++round) {
		auto const params = engine::sampling::parse(Request{.temperature = 2.0 * unit(generator),
		                                                    .top_k = top_k(generator),
		                                                    .top_p = unit(generator),
		                                                    .min_p = 0.0,
		                                                    .seed = std::nullopt},
		                                            engine::sampling::thinking_defaults, 0)
		                        .value();
		auto const filter = engine::sampling::rule(params, 0).filter;
		auto target_logits = std::vector<float>(all_ids.size());
		std::ranges::generate(target_logits, [&] {
			return logit(generator);
		});
		auto draft_logits = std::vector<float>(draft_ids.size());
		std::ranges::generate(draft_logits, [&] {
			return logit(generator);
		});
		auto const target = engine::sampling::survivors(sorted_candidates(target_logits, all_ids), filter);
		auto const draft = engine::sampling::survivors(sorted_candidates(draft_logits, draft_ids), filter);
		auto const rest = engine::sampling::residual(target, draft);
		auto rejected = 0.0;
		auto output = std::vector<double>(all_ids.size(), 0.0);
		for (auto index = 0; index < draft.count; ++index) {
			auto const token = draft.id[index];
			auto const q = draft.weight[index] / draft.total;
			auto const p = engine::sampling::probability_of(target, token);
			output[static_cast<std::size_t>(token)] += std::min(p, q);
			rejected += q - std::min(p, q);
		}
		for (auto index = 0; index < rest.count; ++index) {
			output[static_cast<std::size_t>(rest.id[index])] += rest.total > 0.0 ? rejected * rest.weight[index] / rest.total : 0.0;
		}
		for (auto const id : all_ids) {
			holds = holds && std::abs(output[static_cast<std::size_t>(id)] - engine::sampling::probability_of(target, id)) <= 1e-12;
		}
	}
	return holds;
}

// Greedy targets and drafts are point masses, and verification is exactly argmax match: accept when the draft is the
// target's argmax, otherwise the argmax replaces it.
[[nodiscard]] auto greedy_is_argmax_match(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto logit = std::uniform_int_distribution<int>{-3, 3};
	auto token = std::uniform_int_distribution<int>{0, 7};
	auto unit = std::uniform_real_distribution<double>{0.0, 1.0};
	auto holds = true;
	for (auto round = 0; round < 100'000; ++round) {
		auto logits = std::vector<float>(8);
		std::ranges::generate(logits, [&] {
			return static_cast<float>(logit(generator));
		});
		auto const target = engine::sampling::survivors(Params{Greedy{}}, logits);
		auto const argmax = static_cast<int>(std::ranges::max_element(logits) - logits.begin());
		auto const drafted = token(generator);
		auto const judgement = engine::sampling::judge(target, engine::sampling::point_mass(drafted), 0, unit(generator), unit(generator));
		holds = holds && target.count == 1 && target.id[0] == argmax && (judgement.verdict == Verdict::Accept) == (drafted == argmax) &&
		        judgement.token == argmax;
	}
	return holds;
}

// A toy model: the next token's logits depend on the previous token. The target scores all six tokens; the drafter
// scores only tokens 0..3, its draft vocabulary.
struct Toy {
	std::array<std::array<float, 6>, 6> target;
	std::array<std::array<float, 4>, 6> draft;
};

[[nodiscard]] auto toy(std::uint64_t seed) -> Toy {
	auto generator = engine::testing::generator(seed);
	auto logit = std::uniform_real_distribution<float>{-2.0F, 2.0F};
	auto model = Toy{};
	for (auto& row : model.target) {
		std::ranges::generate(row, [&] {
			return logit(generator);
		});
	}
	for (auto& row : model.draft) {
		std::ranges::generate(row, [&] {
			return logit(generator);
		});
	}
	return model;
}

constexpr auto toy_ids = std::array{0, 1, 2, 3, 4, 5};

[[nodiscard]] auto toy_target(Toy const& model, Rule const& rule, int previous) -> Survivors {
	return engine::sampling::survivors(sorted_candidates(model.target[static_cast<std::size_t>(previous)], toy_ids), rule.filter);
}

[[nodiscard]] auto toy_draft(Toy const& model, Rule const& rule, int previous) -> Survivors {
	return engine::sampling::survivors(sorted_candidates(model.draft[static_cast<std::size_t>(previous)], std::span{toy_ids}.first(4)),
	                                   rule.filter);
}

// Speculative decoding of the toy model from `context` at `position` until `length` tokens are out, three drafts per
// cycle, every draw keyed as the device keys it.
[[nodiscard]] auto speculate(Toy const& model, Rule const& rule, int context, std::uint32_t position, std::size_t length)
    -> std::vector<int> {
	constexpr auto depth = 3U;
	auto out = std::vector<int>{};
	while (out.size() < length) {
		auto targets = std::vector{toy_target(model, rule, context)};
		auto drafts = std::vector<Survivors>{};
		auto chosen = std::vector<int>{};
		auto drafted = std::vector<int>{};
		auto previous = context;
		for (auto step = 0U; step < depth; ++step) {
			drafts.push_back(toy_draft(model, rule, previous));
			chosen.push_back(engine::sampling::propose(rule, position + step, step, drafts.back()));
			previous = drafts.back().id[chosen.back()];
			drafted.push_back(previous);
			targets.push_back(toy_target(model, rule, previous));
		}
		auto const settled = engine::sampling::settle(rule, position, targets, drafts, chosen);
		out.insert(out.end(), drafted.begin(), drafted.begin() + settled.accepted);
		out.push_back(settled.token);
		context = settled.token;
		position += static_cast<std::uint32_t>(settled.accepted + 1);
	}
	out.resize(length);
	return out;
}

// The Wilson-Hilferty approximation of chi-square's 1e-6 upper tail.
[[nodiscard]] auto chi_square_bound(int degrees) -> double {
	auto const scale = 2.0 / (9.0 * degrees);
	return degrees * std::pow(1.0 - scale + 4.75 * std::sqrt(scale), 3.0);
}

// Speculative sampling's first two tokens are distributed as the target's own: chi-square over the 36 pairs against
// the exact p(t1) p(t2 | t1), across 200,000 seeds.
[[nodiscard]] auto speculative_output_is_distributed_as_p(std::uint64_t model_seed) -> bool {
	auto const model = toy(model_seed);
	auto const params = engine::sampling::parse(Request{.temperature = 0.9, .top_k = 5, .top_p = 0.95, .min_p = 0.01, .seed = std::nullopt},
	                                            engine::sampling::thinking_defaults, 0)
	                        .value();
	auto constexpr context = 2;
	auto constexpr trials = 200'000;
	auto observed = std::vector<double>(36, 0.0);
	for (auto trial = 0; trial < trials; ++trial) {
		auto rule = engine::sampling::rule(params, 0);
		rule.seed = static_cast<std::uint64_t>(trial) * 0x9E3779B97F4A7C15ULL + model_seed;
		auto const tokens = speculate(model, rule, context, 100, 2);
		observed[static_cast<std::size_t>(tokens[0]) * 6 + static_cast<std::size_t>(tokens[1])] += 1.0;
	}
	auto const rule = engine::sampling::rule(params, 0);
	auto const first = toy_target(model, rule, context);
	auto statistic = 0.0;
	auto degrees = -1;
	auto holds = true;
	for (auto pair = 0; pair < 36; ++pair) {
		auto const expected = trials * engine::sampling::probability_of(first, pair / 6) *
		                      engine::sampling::probability_of(toy_target(model, rule, pair / 6), pair % 6);
		auto const seen = observed[static_cast<std::size_t>(pair)];
		if (expected == 0.0) {
			holds = holds && seen == 0.0;
			continue;
		}
		statistic += (seen - expected) * (seen - expected) / expected;
		++degrees;
	}
	return holds && degrees >= 8 && statistic < chi_square_bound(degrees);
}

// Greedy speculation emits exactly greedy decoding, whatever the drafter proposes.
[[nodiscard]] auto greedy_speculation_is_greedy_decoding(std::uint64_t model_seed) -> bool {
	auto const model = toy(model_seed);
	auto const rule = engine::sampling::rule(Params{Greedy{}}, 0);
	auto holds = true;
	for (auto context = 0; context < 6; ++context) {
		auto const speculated = speculate(model, rule, context, 0, 64);
		auto previous = context;
		for (auto const token : speculated) {
			auto const& row = model.target[static_cast<std::size_t>(previous)];
			auto const argmax = static_cast<int>(std::ranges::max_element(row) - row.begin());
			holds = holds && token == argmax;
			previous = token;
		}
	}
	return holds;
}

// Target and draft equal up to a few ulps: the judge stays inside p's survivors and the draft, whatever the uniforms.
[[nodiscard]] auto rounding_level_differences_stay_in_support(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto logit = std::uniform_real_distribution<float>{-4.0F, 4.0F};
	auto nudge = std::uniform_int_distribution<int>{-3, 3};
	auto unit = std::uniform_real_distribution<double>{0.0, 1.0};
	auto const ids = std::views::iota(0, 40) | std::ranges::to<std::vector>();
	auto holds = true;
	for (auto round = 0; round < 20'000; ++round) {
		auto logits = std::vector<float>(ids.size());
		std::ranges::generate(logits, [&] {
			return logit(generator);
		});
		auto const filter = engine::sampling::rule(engine::sampling::parse({}, engine::sampling::thinking_defaults, 0).value(), 0).filter;
		auto const target = engine::sampling::survivors(sorted_candidates(logits, ids), filter);
		auto draft = target;
		for (auto index = 0; index < draft.count; ++index) {
			for (auto step = nudge(generator); step != 0; step += step > 0 ? -1 : 1) {
				draft.weight[index] = std::nextafter(draft.weight[index], step > 0 ? 2.0 : 0.0);
			}
		}
		auto const index = engine::sampling::pick(draft, unit(generator));
		auto const judgement = engine::sampling::judge(target, draft, index, unit(generator), unit(generator));
		holds = holds && engine::sampling::probability_of(target, judgement.token) > 0.0;
		auto const own = engine::sampling::pick(target, unit(generator));
		auto const same = engine::sampling::judge(target, target, own, unit(generator), unit(generator));
		holds = holds && same.verdict == Verdict::Accept && same.token == target.id[own];
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "for every p and q in eighths over four tokens (q also over three), the rule's output is exactly p",
	        .holds = exhaustive_eighths(),
	    },
	    engine::testing::Case{
	        .name = "real distributions with the draft vocabulary a strict subset recover p (seed 31)",
	        .holds = real_distributions_recover_p(31),
	    },
	    engine::testing::Case{.name = "greedy verification is exactly argmax match (seed 32)", .holds = greedy_is_argmax_match(32)},
	    engine::testing::Case{
	        .name = "speculative sampling's first two tokens pass chi-square against the target's (model seed 33)",
	        .holds = speculative_output_is_distributed_as_p(33),
	    },
	    engine::testing::Case{
	        .name = "speculative sampling's first two tokens pass chi-square against the target's (model seed 34)",
	        .holds = speculative_output_is_distributed_as_p(34),
	    },
	    engine::testing::Case{
	        .name = "greedy speculation emits exactly greedy decoding (model seed 35)",
	        .holds = greedy_speculation_is_greedy_decoding(35),
	    },
	    engine::testing::Case{
	        .name = "p and q equal to rounding never leave p's support, and p = q always accepts (seed 36)",
	        .holds = rounding_level_differences_stay_in_support(36),
	    },
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.sampling;
import engine.testing;

namespace {

using engine::sampling::Defaults;
using engine::sampling::Greedy;
using engine::sampling::ParamError;
using engine::sampling::ParamErrorKind;
using engine::sampling::Params;
using engine::sampling::Request;
using engine::sampling::Sample;

static_assert(engine::sampling::parse({}, engine::sampling::thinking_defaults, 0).has_value());
static_assert(engine::sampling::parse({}, engine::sampling::instruct_defaults, 0).has_value());

constexpr auto fresh = std::uint64_t{0x5eed'0000'0000'0001};

// The specification of each field, written independently of `parse`.
[[nodiscard]] auto temperature_ok(double value) -> bool {
	return !std::isnan(value) && value >= 0.0 && value <= 2.0;
}

[[nodiscard]] auto unit_ok(double value) -> bool {
	return !std::isnan(value) && value >= 0.0 && value <= 1.0;
}

[[nodiscard]] auto expected_error(double temperature, std::int64_t top_k, double top_p, double min_p) -> std::optional<ParamErrorKind> {
	if (!temperature_ok(temperature)) {
		return ParamErrorKind::TemperatureOutOfRange;
	}
	if (top_k < 1) {
		return ParamErrorKind::TopKBelowOne;
	}
	if (top_k > 64) {
		return ParamErrorKind::TopKAboveBound;
	}
	if (!unit_ok(top_p)) {
		return ParamErrorKind::TopPOutOfRange;
	}
	if (!unit_ok(min_p)) {
		return ParamErrorKind::MinPOutOfRange;
	}
	return std::nullopt;
}

[[nodiscard]] auto parse_all(double temperature, std::int64_t top_k, double top_p, double min_p, std::optional<std::int64_t> seed)
    -> engine::core::Result<Params, ParamError> {
	return engine::sampling::parse(Request{.temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p, .seed = seed},
	                               engine::sampling::thinking_defaults, fresh);
}

[[nodiscard]] auto sample_of(engine::core::Result<Params, ParamError> const& parsed) -> std::optional<Sample> {
	if (!parsed || !std::holds_alternative<Sample>(*parsed)) {
		return std::nullopt;
	}
	return std::get<Sample>(*parsed);
}

// Accepted exactly when the specification accepts, refused with the first failing field's kind otherwise, and an
// accepted request is greedy exactly when its distribution is a point mass by construction.
[[nodiscard]] auto agrees_with_specification(double temperature, std::int64_t top_k, double top_p, double min_p) -> bool {
	auto const parsed = parse_all(temperature, top_k, top_p, min_p, std::nullopt);
	auto const error = expected_error(temperature, top_k, top_p, min_p);
	if (error) {
		return !parsed && parsed.error().kind == *error;
	}
	if (!parsed) {
		return false;
	}
	auto const point_mass = temperature == 0.0 || top_k == 1 || top_p == 0.0;
	if (point_mass) {
		return std::holds_alternative<Greedy>(*parsed);
	}
	auto const sample = sample_of(parsed);
	return sample.has_value() && sample->temperature() == temperature && sample->top_k() == top_k && sample->min_p() == min_p &&
	       sample->top_p() == (top_p == 1.0 ? std::optional<double>{} : std::optional{top_p}) && sample->seed() == fresh;
}

[[nodiscard]] auto edge_doubles(double low, double high) -> std::vector<double> {
	auto constexpr infinity = std::numeric_limits<double>::infinity();
	return {
	    -infinity,
	    std::numeric_limits<double>::lowest(),
	    -1.0,
	    std::nextafter(low, -infinity),
	    low,
	    std::nextafter(low, infinity),
	    std::numeric_limits<double>::denorm_min(),
	    0.5,
	    std::nextafter(high, -infinity),
	    high,
	    std::nextafter(high, infinity),
	    high + 1.0,
	    std::numeric_limits<double>::max(),
	    infinity,
	    std::numeric_limits<double>::quiet_NaN(),
	};
}

[[nodiscard]] auto every_edge_combination_agrees() -> bool {
	auto const temperatures = edge_doubles(0.0, 2.0);
	auto const units = edge_doubles(0.0, 1.0);
	auto const top_ks = std::array<std::int64_t, 9>{std::numeric_limits<std::int64_t>::min(), -1, 0, 1, 2, 63, 64, 65,
	                                                std::numeric_limits<std::int64_t>::max()};
	auto holds = true;
	for (auto const temperature : temperatures) {
		for (auto const top_k : top_ks) {
			for (auto const top_p : units) {
				for (auto const min_p : units) {
					holds = holds && agrees_with_specification(temperature, top_k, top_p, min_p);
				}
			}
		}
	}
	return holds;
}

[[nodiscard]] auto every_small_top_k_agrees() -> bool {
	auto holds = true;
	for (auto top_k = std::int64_t{-1000}; top_k <= 1000; ++top_k) {
		holds = holds && agrees_with_specification(0.7, top_k, 0.9, 0.05);
	}
	return holds;
}

[[nodiscard]] auto random_requests_agree(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto wide = std::uniform_real_distribution<double>{-0.5, 2.5};
	auto choice = std::uniform_int_distribution<int>{0, 3};
	auto top_k = std::uniform_int_distribution<std::int64_t>{-5, 70};
	auto const snap = [&](double value) {
		// Boundaries are where specifications and code part ways, so they are drawn often.
		switch (choice(generator)) {
		case 0:
			return 0.0;
		case 1:
			return 1.0;
		case 2:
			return 2.0;
		default:
			return value;
		}
	};
	auto holds = true;
	for (auto round = 0; round < 200'000; ++round) {
		holds = holds && agrees_with_specification(snap(wide(generator)), top_k(generator), snap(wide(generator)), snap(wide(generator)));
	}
	return holds;
}

[[nodiscard]] auto absent_fields_take_the_defaults() -> bool {
	auto constexpr defaults = Defaults{.temperature = 0.3, .top_k = 7, .top_p = 0.6, .min_p = 0.1};
	auto const parsed = engine::sampling::parse({}, defaults, fresh);
	auto const sample = sample_of(parsed);
	auto const mixed = sample_of(engine::sampling::parse(
	    Request{.temperature = 1.5, .top_k = 3, .top_p = std::nullopt, .min_p = std::nullopt, .seed = std::nullopt}, defaults, fresh));
	return sample.has_value() && sample->temperature() == 0.3 && sample->top_k() == 7 && sample->top_p() == std::optional{0.6} &&
	       sample->min_p() == 0.1 && mixed.has_value() && mixed->temperature() == 1.5 && mixed->top_k() == 3 &&
	       mixed->top_p() == std::optional{0.6} && mixed->min_p() == 0.1;
}

[[nodiscard]] auto a_named_seed_is_its_bit_pattern() -> bool {
	auto holds = true;
	for (auto const named :
	     std::array<std::int64_t, 6>{0, 1, -1, 42, std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()}) {
		auto const parsed = parse_all(1.0, 20, 0.95, 0.0, named);
		auto const sample = sample_of(parsed);
		holds = holds && sample.has_value() && sample->seed() == std::bit_cast<std::uint64_t>(named);
	}
	return holds;
}

[[nodiscard]] auto every_error_is_permanent() -> bool {
	return std::ranges::all_of(std::array{ParamErrorKind::TemperatureOutOfRange, ParamErrorKind::TopKBelowOne,
	                                      ParamErrorKind::TopKAboveBound, ParamErrorKind::TopPOutOfRange, ParamErrorKind::MinPOutOfRange},
	                           [](ParamErrorKind kind) {
		                           return ParamError{.kind = kind}.retry() == engine::core::Retry::Permanent;
	                           });
}

// The device form keeps the parameters it was made from, and greedy keeps exactly one candidate.
[[nodiscard]] auto rules_carry_the_parameters() -> bool {
	auto const sampled = parse_all(0.6, 33, 0.9, 0.25, 99);
	auto const nucleus_off = parse_all(0.6, 33, 1.0, 0.0, 99);
	if (!sampled || !nucleus_off) {
		return false;
	}
	auto const rule = engine::sampling::rule(*sampled, 3);
	auto const off = engine::sampling::rule(*nucleus_off, 0);
	auto const greedy = engine::sampling::rule(Params{Greedy{}}, 5);
	return rule.seed == 99 && rule.request == 3 && rule.filter.temperature == 0.6 && rule.filter.top_k == 33 && rule.filter.top_p == 0.9 &&
	       rule.filter.min_p == 0.25 && rule.filter.nucleus == engine::sampling::Nucleus::On &&
	       off.filter.nucleus == engine::sampling::Nucleus::Off && greedy.filter.top_k == 1 && greedy.request == 5;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every combination of boundary values is accepted or refused as specified, with the first failing field named",
	        .holds = every_edge_combination_agrees(),
	    },
	    engine::testing::Case{.name = "every top_k in [-1000, 1000] is accepted exactly within [1, 64]",
		                      .holds = every_small_top_k_agrees()},
	    engine::testing::Case{.name = "random requests agree with the specification (seed 1)", .holds = random_requests_agree(1)},
	    engine::testing::Case{.name = "random requests agree with the specification (seed 2)", .holds = random_requests_agree(2)},
	    engine::testing::Case{.name = "absent fields take the given defaults, present ones win",
		                      .holds = absent_fields_take_the_defaults()},
	    engine::testing::Case{.name = "a named seed is its 64-bit two's complement pattern", .holds = a_named_seed_is_its_bit_pattern()},
	    engine::testing::Case{.name = "every refused parameter is a permanent error", .holds = every_error_is_permanent()},
	    engine::testing::Case{.name = "the device rule carries the validated parameters", .holds = rules_carry_the_parameters()},
	};
	return engine::testing::report(cases);
}

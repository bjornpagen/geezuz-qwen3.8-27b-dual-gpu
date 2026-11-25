// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.sampling;
import engine.testing;

namespace {

using engine::sampling::exp_nonpositive;

[[nodiscard]] auto ulps_apart(double a, double b) -> std::uint64_t {
	auto const x = std::bit_cast<std::int64_t>(a);
	auto const y = std::bit_cast<std::int64_t>(b);
	return x > y ? static_cast<std::uint64_t>(x - y) : static_cast<std::uint64_t>(y - x);
}

// The oracle is the platform's libm, correctly rounded or nearly so; within one ulp of it is within two of e^x.
[[nodiscard]] auto within_one_ulp(double x) -> bool {
	return ulps_apart(exp_nonpositive(x), std::exp(x)) <= 1;
}

[[nodiscard]] auto dense_sweep_within_one_ulp() -> bool {
	auto holds = true;
	for (auto step = 0; step <= 2'000'000; ++step) {
		holds = holds && within_one_ulp(-708.0 * step / 2'000'000.0);
	}
	return holds;
}

// Every binade from 2^-1074 up, where the reduction and the series meet their small-argument cases.
[[nodiscard]] auto every_binade_within_one_ulp(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto mantissa = std::uniform_real_distribution<double>{1.0, 2.0};
	auto holds = true;
	for (auto exponent = -1074; exponent <= 9; ++exponent) {
		for (auto draw = 0; draw < 2'000; ++draw) {
			auto const x = -std::ldexp(mantissa(generator), exponent);
			holds = holds && (x < -708.0 || within_one_ulp(x));
		}
	}
	return holds;
}

// Weights must not reorder candidates: the exponential never decreases as its argument grows.
[[nodiscard]] auto monotone(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto argument = std::uniform_real_distribution<double>{-708.0, 0.0};
	auto holds = true;
	for (auto draw = 0; draw < 1'000'000; ++draw) {
		auto const x = argument(generator);
		auto const above = std::nextafter(x, 0.0);
		holds = holds && exp_nonpositive(x) <= exp_nonpositive(above);
	}
	return holds;
}

[[nodiscard]] auto range_ends() -> bool {
	auto const holds =
	    exp_nonpositive(0.0) == 1.0 && exp_nonpositive(-0.0) == 1.0 && exp_nonpositive(-std::numeric_limits<double>::infinity()) == 0.0 &&
	    exp_nonpositive(std::numeric_limits<double>::lowest()) == 0.0 && exp_nonpositive(std::nextafter(-708.0, -1000.0)) == 0.0 &&
	    exp_nonpositive(-708.0) >= std::numeric_limits<double>::min() && exp_nonpositive(-std::numeric_limits<double>::denorm_min()) == 1.0;
	return holds;
}

constexpr auto probes = std::array{0.0, -1e-300, -1e-17, -0.1, -0.34657359027997264, -0.5, -1.0, -2.5, -10.0, -87.3, -300.25, -707.9};

constexpr auto constant_values = [] {
	auto values = std::array<double, probes.size()>{};
	for (auto index = std::size_t{0}; index < probes.size(); ++index) {
		values[index] = exp_nonpositive(probes[index]);
	}
	return values;
}();

// The device evaluates at run time, the header's checks at compile time: both must be the same function.
[[nodiscard]] auto run_time_matches_compile_time() -> bool {
	auto const arguments = std::vector(probes.begin(), probes.end());
	auto holds = true;
	for (auto index = std::size_t{0}; index < arguments.size(); ++index) {
		holds = holds &&
		        std::bit_cast<std::uint64_t>(exp_nonpositive(arguments[index])) == std::bit_cast<std::uint64_t>(constant_values[index]);
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "within one ulp of libm on a 2,000,001-point sweep of [-708, 0]",
		                      .holds = dense_sweep_within_one_ulp()},
	    engine::testing::Case{.name = "within one ulp of libm in every binade down to 2^-1074 (seed 7)",
		                      .holds = every_binade_within_one_ulp(7)},
	    engine::testing::Case{.name = "never decreases as its argument grows (seed 8)", .holds = monotone(8)},
	    engine::testing::Case{.name = "exactly 1 at zero, exactly 0 below -708, normal at -708", .holds = range_ends()},
	    engine::testing::Case{.name = "run-time evaluation equals constant evaluation bit for bit",
		                      .holds = run_time_matches_compile_time()},
	};
	return engine::testing::report(cases);
}

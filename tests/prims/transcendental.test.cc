// SPDX-License-Identifier: AGPL-3.0-only
//
// Every fp32 input through each transcendental, against a double-precision reference: the stated ulp bound holds
// everywhere, nan goes to nan, and infinities and zeros go to the limits. The 2^32 inputs are split into shards, one
// ctest case each, so the sweep runs in parallel within each case's deadline.
import std;
import engine.prims;
import engine.testing;

namespace {

namespace math = engine::device::prims::math;

inline constexpr std::uint64_t shards = 8;

[[nodiscard]] auto reference_silu(double x) -> double {
	return std::isinf(x) && x < 0.0 ? -0.0 : x / (1.0 + std::exp(-x));
}

[[nodiscard]] auto reference_softplus(double x) -> double {
	return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

/** The fp32 ulp at the exact value `exact`: the subnormal step below 2^-126. */
[[nodiscard]] auto ulp_at(double exact) -> double {
	auto const magnitude = std::abs(exact);
	return magnitude < 0x1p-126 ? 0x1p-149 : std::ldexp(1.0, std::ilogb(magnitude) - 23);
}

struct Sweep {
	bool within_bound;
	bool nan_to_nan;
	bool limits_exact;
	double worst_ulp;
	std::uint32_t worst_input;
};

template<class Evaluate, class Reference>
[[nodiscard]] auto sweep(Evaluate evaluate, Reference reference, double bound_ulp, std::uint64_t shard) -> Sweep {
	auto result = Sweep{.within_bound = true, .nan_to_nan = true, .limits_exact = true, .worst_ulp = 0.0, .worst_input = 0};
	auto const first = shard * ((std::uint64_t{1} << 32U) / shards);
	auto const last = first + (std::uint64_t{1} << 32U) / shards;
	for (auto bits = first; bits < last; ++bits) {
		auto const input = static_cast<std::uint32_t>(bits);
		auto const x = std::bit_cast<float>(input);
		auto const got = evaluate(x);
		auto const exact = reference(static_cast<double>(x));
		if (std::isnan(exact)) {
			result.nan_to_nan = result.nan_to_nan && std::isnan(got);
			continue;
		}
		auto const nearest = static_cast<float>(exact);
		if (std::isinf(nearest) || std::isinf(got) || exact == 0.0) {
			result.limits_exact = result.limits_exact && got == nearest && std::signbit(got) == std::signbit(nearest);
			continue;
		}
		auto const error = std::abs(static_cast<double>(got) - exact) / ulp_at(exact);
		if (error > result.worst_ulp) {
			result.worst_ulp = error;
			result.worst_input = input;
		}
	}
	result.within_bound = result.worst_ulp <= bound_ulp;
	return result;
}

/** The sweep of one function's shard, against the bound the header states for it; nothing for an unknown name. */
[[nodiscard]] auto run(std::string_view name, std::uint64_t shard) -> std::optional<Sweep> {
	if (name == "exp") {
		return sweep([](float x) { return math::exp(x); }, [](double x) { return std::exp(x); }, 1.0, shard);
	}
	if (name == "log") {
		return sweep([](float x) { return math::log(x); }, [](double x) { return std::log(x); }, 1.0, shard);
	}
	if (name == "rsqrt") {
		return sweep([](float x) { return math::rsqrt(x); }, [](double x) { return 1.0 / std::sqrt(x); }, 0.51, shard);
	}
	if (name == "sigmoid") {
		return sweep([](float x) { return math::sigmoid(x); }, [](double x) { return 1.0 / (1.0 + std::exp(-x)); }, 2.5, shard);
	}
	if (name == "silu") {
		return sweep([](float x) { return math::silu(x); }, reference_silu, 2.5, shard);
	}
	if (name == "softplus") {
		return sweep([](float x) { return math::softplus(x); }, reference_softplus, 2.0, shard);
	}
	return std::nullopt;
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	auto const usage = [] {
		std::println("usage: transcendental <exp|log|rsqrt|sigmoid|silu|softplus> <shard 0-{}>", shards - 1);
		return 2;
	};
	if (arguments.size() != 3) {
		return usage();
	}
	auto const name = std::string_view{arguments[1]};
	auto const shard_text = std::string_view{arguments[2]};
	constexpr auto digits = std::string_view{"01234567"};
	static_assert(digits.size() == shards);
	auto const shard = digits.find(shard_text);
	if (shard_text.size() != 1 || shard == std::string_view::npos) {
		return usage();
	}
	auto const swept = run(name, std::uint64_t{shard});
	if (!swept) {
		return usage();
	}
	auto const& result = *swept;
	std::println("{} shard {}: worst {:.4f} ulp at {:a}", name, shard, result.worst_ulp, std::bit_cast<float>(result.worst_input));
	auto const cases = std::array{
	    engine::testing::Case{.name = "within the stated ulp bound on every input of the shard", .holds = result.within_bound},
	    engine::testing::Case{.name = "nan wherever the exact function is undefined", .holds = result.nan_to_nan},
	    engine::testing::Case{.name = "infinities and zeros exactly where the exact value rounds to them", .holds = result.limits_exact},
	};
	return engine::testing::report(cases);
}

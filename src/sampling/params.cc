// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sampling:params;

import std;
import engine.core;
import engine.sampling_device;

namespace engine::sampling {

/** The highest temperature a request may ask for: the top of the openai api's documented range. */
export inline constexpr double temperature_maximum = 2.0;

/** Why a request's sampling parameters are refused; each kind is one rejected condition. */
export enum class ParamErrorKind : std::uint8_t {
	/** temperature is outside [0, temperature_maximum] (0 is greedy). */
	TemperatureOutOfRange,
	/** top_k is 0 or negative: unbounded top-k is not offered, since the device samples from its best candidates. */
	TopKBelowOne,
	/** top_k is above top_k_bound, the most candidates the device keeps per row. */
	TopKAboveBound,
	/** top_p is outside [0, 1]. */
	TopPOutOfRange,
	/** min_p is outside [0, 1]. */
	MinPOutOfRange,
};

/** A refused request: the caller answers it with a 400 naming the kind. */
export struct [[nodiscard]] ParamError {
	ParamErrorKind kind;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (kind) {
		case ParamErrorKind::TemperatureOutOfRange:
		case ParamErrorKind::TopKBelowOne:
		case ParamErrorKind::TopKAboveBound:
		case ParamErrorKind::TopPOutOfRange:
		case ParamErrorKind::MinPOutOfRange:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(ParamError const&) const -> bool = default;
};

/** The sampling fields of a request as they arrived; an absent field takes the server's default. */
export struct Request {
	std::optional<double> temperature;
	std::optional<std::int64_t> top_k;
	std::optional<double> top_p;
	std::optional<double> min_p;
	std::optional<std::int64_t> seed;
};

/** What an absent request field becomes. */
export struct Defaults {
	double temperature;
	std::int64_t top_k;
	double top_p;
	double min_p;
};

/** The sampling a thinking turn gets when the request names none: the model's recommended thinking sampling. */
export inline constexpr Defaults thinking_defaults{.temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0};

/** The sampling a non-thinking turn gets when the request names none: the model's recommended instruct sampling. */
export inline constexpr Defaults instruct_defaults{.temperature = 0.7, .top_k = 20, .top_p = 0.8, .min_p = 0.0};

/** Always the most likely token, the lowest id among equals. */
export struct Greedy {
	[[nodiscard]] constexpr auto operator==(Greedy const&) const -> bool = default;
};

export class Sample;

/**
 * A request's validated sampling. Every parameter set whose distribution is a point mass on the argmax (temperature
 * 0, top_k 1, top_p 0) is `Greedy`, so a `Sample` always has a temperature in (0, temperature_maximum], top_k in
 * [2, top_k_bound], a nucleus in (0, 1) or none, and min_p in [0, 1].
 */
export using Params = std::variant<Greedy, Sample>;

/**
 * Validates a request's sampling at the boundary, once: every field present or defaulted is checked in the order
 * temperature, top_k, top_p, min_p, and the first refused one names the error. A request's seed is its 64-bit two's
 * complement pattern; `fresh_seed` is the caller's entropy for a request that names none.
 */
export [[nodiscard]] constexpr auto parse(Request const& request, Defaults const& defaults, std::uint64_t fresh_seed)
    -> core::Result<Params, ParamError>;

/** Temperature, top-k, nucleus, min-p and seed of a sampled request, valid by construction (only `parse` makes one). */
export class Sample {
public:
	[[nodiscard]] constexpr auto temperature() const -> double {
		return temperature_;
	}

	[[nodiscard]] constexpr auto top_k() const -> int {
		return top_k_;
	}

	/** The nucleus mass, or none when the request's top_p is 1. */
	[[nodiscard]] constexpr auto top_p() const -> std::optional<double> {
		return top_p_;
	}

	[[nodiscard]] constexpr auto min_p() const -> double {
		return min_p_;
	}

	[[nodiscard]] constexpr auto seed() const -> std::uint64_t {
		return seed_;
	}

	[[nodiscard]] constexpr auto operator==(Sample const&) const -> bool = default;

private:
	friend constexpr auto parse(Request const& request, Defaults const& defaults, std::uint64_t fresh_seed)
	    -> core::Result<Params, ParamError>;

	constexpr Sample(double temperature, int top_k, std::optional<double> top_p, double min_p, std::uint64_t seed)
	    : temperature_{temperature}, top_k_{top_k}, top_p_{top_p}, min_p_{min_p}, seed_{seed} {}

	double temperature_;
	int top_k_;
	std::optional<double> top_p_;
	double min_p_;
	std::uint64_t seed_;
};

[[nodiscard]] constexpr auto within(double value, double low, double high) -> bool {
	return value >= low && value <= high;
}

constexpr auto parse(Request const& request, Defaults const& defaults, std::uint64_t fresh_seed) -> core::Result<Params, ParamError> {
	auto const temperature = request.temperature.value_or(defaults.temperature);
	auto const top_k = request.top_k.value_or(defaults.top_k);
	auto const top_p = request.top_p.value_or(defaults.top_p);
	auto const min_p = request.min_p.value_or(defaults.min_p);
	if (!within(temperature, 0.0, temperature_maximum)) {
		return std::unexpected(ParamError{.kind = ParamErrorKind::TemperatureOutOfRange});
	}
	if (top_k < 1) {
		return std::unexpected(ParamError{.kind = ParamErrorKind::TopKBelowOne});
	}
	if (top_k > sampling_device::top_k_bound) {
		return std::unexpected(ParamError{.kind = ParamErrorKind::TopKAboveBound});
	}
	if (!within(top_p, 0.0, 1.0)) {
		return std::unexpected(ParamError{.kind = ParamErrorKind::TopPOutOfRange});
	}
	if (!within(min_p, 0.0, 1.0)) {
		return std::unexpected(ParamError{.kind = ParamErrorKind::MinPOutOfRange});
	}
	if (temperature == 0.0 || top_k == 1 || top_p == 0.0) {
		return Params{Greedy{}};
	}
	auto const seed = request.seed
	                      .transform([](std::int64_t named) {
		                      return std::bit_cast<std::uint64_t>(named);
	                      })
	                      .value_or(fresh_seed);
	auto const nucleus = top_p == 1.0 ? std::optional<double>{} : std::optional{top_p};
	return Params{Sample{temperature, core::narrow<int>(top_k), nucleus, min_p, seed}};
}

[[nodiscard]] constexpr auto filter_of(Greedy) -> sampling_device::Filter {
	return {.temperature = 1.0, .top_p = 0.0, .min_p = 0.0, .top_k = 1, .nucleus = sampling_device::Nucleus::Off};
}

[[nodiscard]] constexpr auto filter_of(Sample const& sample) -> sampling_device::Filter {
	return {
	    .temperature = sample.temperature(),
	    .top_p = sample.top_p().value_or(0.0),
	    .min_p = sample.min_p(),
	    .top_k = sample.top_k(),
	    .nucleus = sample.top_p() ? sampling_device::Nucleus::On : sampling_device::Nucleus::Off,
	};
}

[[nodiscard]] constexpr auto seed_of(Greedy) -> std::uint64_t {
	return 0;
}

[[nodiscard]] constexpr auto seed_of(Sample const& sample) -> std::uint64_t {
	return sample.seed();
}

/**
 * The device form of a slot's sampling. `request` separates independent streams under one seed (the choice index;
 * 0 for a single choice). Greedy is the filter that keeps one candidate, so its draws never matter.
 */
export [[nodiscard]] constexpr auto rule(Params const& params, std::uint32_t request) -> sampling_device::Rule {
	return std::visit(
	    [&](auto const& chosen) {
		    return sampling_device::Rule{.seed = seed_of(chosen), .filter = filter_of(chosen), .request = request};
	    },
	    params);
}

}

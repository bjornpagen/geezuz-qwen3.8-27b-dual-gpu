// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.testing;

namespace {

using engine::core::OutOfRange;
using engine::core::Retry;

struct Unclassified {
	int code;

	[[nodiscard]] constexpr auto operator==(Unclassified const&) const -> bool = default;
};

static_assert(engine::core::Classified<OutOfRange>);
static_assert(!engine::core::Classified<Unclassified>);
static_assert(!engine::core::Classified<int>);

template<std::integral To>
[[nodiscard]] auto try_narrow_matches_range_from_every_int16() -> bool {
	auto holds = true;
	for (auto wide = std::int32_t{std::numeric_limits<std::int16_t>::min()}; wide <= std::numeric_limits<std::int16_t>::max(); ++wide) {
		auto const value = static_cast<std::int16_t>(wide);
		auto const fits = wide >= std::int32_t{std::numeric_limits<To>::min()} && wide <= std::int32_t{std::numeric_limits<To>::max()};
		auto const result = engine::core::try_narrow<To>(value);
		holds = holds && result.has_value() == fits;
		holds = holds && (!fits || std::int32_t{*result} == wide);
		holds = holds && (fits || result.error() == OutOfRange{});
	}
	return holds;
}

[[nodiscard]] auto narrow_round_trips_every_fitting_value() -> bool {
	auto holds = true;
	for (auto wide = std::int32_t{-128}; wide <= 127; ++wide) {
		holds = holds && std::int32_t{engine::core::narrow<std::int8_t>(wide)} == wide;
	}
	return holds;
}

[[nodiscard]] auto try_narrow_bounds_of_unsigned_64() -> bool {
	auto constexpr top = std::numeric_limits<std::uint64_t>::max();
	return !engine::core::try_narrow<std::int64_t>(top).has_value() &&
	       !engine::core::try_narrow<std::uint64_t>(std::int64_t{-1}).has_value() &&
	       engine::core::try_narrow<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ==
	           std::uint64_t{std::numeric_limits<std::int64_t>::max()};
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "try_narrow to int8 succeeds exactly on values in range, preserving them",
	        .holds = try_narrow_matches_range_from_every_int16<std::int8_t>(),
	    },
	    engine::testing::Case{
	        .name = "try_narrow to uint8 succeeds exactly on values in range, preserving them",
	        .holds = try_narrow_matches_range_from_every_int16<std::uint8_t>(),
	    },
	    engine::testing::Case{
	        .name = "try_narrow respects the signedness boundaries of 64-bit integers",
	        .holds = try_narrow_bounds_of_unsigned_64(),
	    },
	    engine::testing::Case{
	        .name = "narrow preserves every value that fits",
	        .holds = narrow_round_trips_every_fitting_value(),
	    },
	    engine::testing::Case{
	        .name = "an out-of-range conversion is a permanent failure",
	        .holds = OutOfRange{}.retry() == Retry::Permanent,
	    },
	};
	return engine::testing::report(cases);
}

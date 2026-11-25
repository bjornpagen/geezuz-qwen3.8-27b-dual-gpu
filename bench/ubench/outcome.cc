// SPDX-License-Identifier: AGPL-3.0-only
//
// What a microbenchmark yields: readings with their units, a verdict where the benchmark gates something, and the
// ways a run can fail to yield anything.
export module engine.ubench:outcome;

import std;
import engine.core;
import engine.device;
import engine.model;

namespace engine::ubench {

/** The unit a reading is in. */
export enum class Unit : std::uint8_t {
	GigabytesPerSecond,
	Megahertz,
	Nanoseconds,
	Microseconds,
	Milliseconds,
	Cycles,
	FlopPerClockPerSm,
	PerClockPerSm,
	PerMicrosecond,
	Teraflops,
	Mebibytes,
	Watts,
	Ratio,
	Count,
};

[[nodiscard]] constexpr auto symbol(Unit unit) -> std::string_view {
	switch (unit) {
	case Unit::GigabytesPerSecond:
		return "GB/s";
	case Unit::Megahertz:
		return "MHz";
	case Unit::Nanoseconds:
		return "ns";
	case Unit::Microseconds:
		return "us";
	case Unit::Milliseconds:
		return "ms";
	case Unit::Cycles:
		return "cycles";
	case Unit::FlopPerClockPerSm:
		return "flop/clk/SM";
	case Unit::PerClockPerSm:
		return "/clk/SM";
	case Unit::PerMicrosecond:
		return "/us";
	case Unit::Teraflops:
		return "TFLOPS";
	case Unit::Mebibytes:
		return "MiB";
	case Unit::Watts:
		return "W";
	case Unit::Ratio:
		return "x";
	case Unit::Count:
		return "";
	}
	std::unreachable();
}

/** One measured quantity: what it is (for the operator), its value and its unit. */
export struct Reading {
	std::string label;
	double value;
	Unit unit;

	[[nodiscard]] auto operator==(Reading const&) const -> bool = default;
};

[[nodiscard]] auto render(Reading const& reading) -> std::string {
	return std::format("{}: {:.4g} {}", reading.label, reading.value, symbol(reading.unit));
}

/**
 * What a benchmark's readings decide: most only measure (the decision is the operator's, against the expectation);
 * a gate or a self-check holds or fails.
 */
export enum class Verdict : std::uint8_t { Measured, Holds, Fails };

/** A benchmark's result: every reading, its verdict, and what the readings say beyond their numbers. */
export struct Outcome {
	std::vector<Reading> readings;
	Verdict verdict;
	std::vector<std::string> notes;
};

/** A kernel did not finish before its deadline: the card is hung or a protocol deadlocked. */
export struct Hung {
	std::string_view kernel;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Hung const&) const -> bool = default;
};

/**
 * A kernel finished but its results cannot be a measurement (zero elapsed time, nothing checked, two blocks on one SM
 * where the benchmark needs two): it did not run as built, which on the test double is every kernel.
 */
export struct NoReading {
	std::string_view kernel;
	std::string_view why;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(NoReading const&) const -> bool = default;
};

/** Every way running a benchmark fails to produce an outcome. */
export struct [[nodiscard]] UbenchError {
	std::variant<device::DeviceError, Hung, NoReading> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(UbenchError const&) const -> bool = default;
};

export template<class T>
using Result = core::Result<T, UbenchError>;

/** Lifts a device result into a benchmark result. */
export template<class T>
[[nodiscard]] auto lift(core::Result<T, device::DeviceError> result) -> Result<T> {
	if (!result) {
		return std::unexpected(UbenchError{.reason = std::move(result.error())});
	}
	if constexpr (std::is_void_v<T>) {
		return {};
	} else {
		return std::move(*result);
	}
}

export [[nodiscard]] auto no_reading(std::string_view kernel, std::string_view why) -> std::unexpected<UbenchError> {
	return std::unexpected(UbenchError{.reason = NoReading{.kernel = kernel, .why = why}});
}

/** The label prefix every per-card reading starts with. */
export [[nodiscard]] auto on(model::Card card) -> std::string {
	return std::format("card {}", std::to_underlying(card));
}

/** The operator's account of one device failure, one overload per kind. */
struct DescribeDevice {
	[[nodiscard]] auto operator()(device::DriverUnavailable const& reason) const -> std::string {
		return std::format("driver unavailable: {} {}", reason.name, reason.stage == device::LoadStage::Open ? "would not open" : "lacks an entry point");
	}

	[[nodiscard]] auto operator()(device::DriverTooOld const& reason) const -> std::string {
		return std::format("driver too old: CUDA {} found, {} required", reason.found, reason.required);
	}

	[[nodiscard]] auto operator()(device::CardCount const& reason) const -> std::string {
		return std::format("{} cards found, two required", reason.found);
	}

	[[nodiscard]] auto operator()(device::WrongCard const& reason) const -> std::string {
		return std::format("card {} is not an RTX 3090 (fact {} reads {})", reason.ordinal, std::to_underlying(reason.fact), reason.found);
	}

	[[nodiscard]] auto operator()(device::PeerUnreachable const& reason) const -> std::string {
		return std::format("card {} cannot map card {}'s memory", reason.from, reason.to);
	}

	[[nodiscard]] auto operator()(device::NotNvlink const& reason) const -> std::string {
		return std::format("card {} link {} does not join the peer (fault {})", reason.ordinal, reason.link, std::to_underlying(reason.fault));
	}

	[[nodiscard]] auto operator()(device::Failure const& reason) const -> std::string {
		return std::format("driver call {} failed with status {}", reason.symbol, std::to_underlying(reason.status));
	}

	[[nodiscard]] auto operator()(device::ManagementFailure const& reason) const -> std::string {
		return std::format("nvml call {} failed with status {}", reason.symbol, std::to_underlying(reason.status));
	}

	[[nodiscard]] auto operator()(device::ModuleRejected const& reason) const -> std::string {
		return std::format("the jit refused an image: {}", reason.log);
	}

	[[nodiscard]] auto operator()(device::KernelMissing const& reason) const -> std::string {
		return std::format("an image lacks kernel {}", reason.entry);
	}

	[[nodiscard]] auto operator()(device::KernelSignature const& reason) const -> std::string {
		return std::format("kernel {} does not take one {}-byte parameter struct", reason.entry, reason.expected);
	}

	[[nodiscard]] auto operator()(device::HugepageError const& reason) const -> std::string {
		return std::format("pinned host memory unavailable (failure {})", std::to_underlying(reason.failure));
	}
};

struct Describe {
	[[nodiscard]] auto operator()(device::DeviceError const& failure) const -> std::string {
		return std::visit(DescribeDevice{}, failure.reason);
	}

	[[nodiscard]] auto operator()(Hung const& hung) const -> std::string {
		return std::format("{} did not finish before its deadline", hung.kernel);
	}

	[[nodiscard]] auto operator()(NoReading const& none) const -> std::string {
		return std::format("{} gave no reading: {}", none.kernel, none.why);
	}
};

export [[nodiscard]] auto describe(UbenchError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

}

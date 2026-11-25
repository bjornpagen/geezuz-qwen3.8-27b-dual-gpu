// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.hugepage;
import engine.testing;

namespace {

namespace device = engine::device;

using device::DeviceError;

template<class T, std::size_t N>
[[nodiscard]] auto pick(std::mt19937_64& random, std::array<T, N> const& choices) -> T {
	return choices[std::uniform_int_distribution<std::size_t>{0, N - 1}(random)];
}

constexpr auto symbols = std::array<std::string_view, 3>{"cuMemAlloc_v2", "cuLaunchKernel", "nvmlDeviceGetFieldValues"};
constexpr auto statuses =
    std::array{device::Status::OutOfMemory, device::Status::LaunchFailed, device::Status::Unknown, device::Status{12345}};
constexpr auto management_statuses =
    std::array{device::ManagementStatus::NotSupported, device::ManagementStatus::GpuIsLost, device::ManagementStatus{4242}};

[[nodiscard]] auto optional_size(std::mt19937_64& random) -> std::optional<std::size_t> {
	auto const size = std::uniform_int_distribution<std::size_t>{0, 64}(random);
	return size == 0 ? std::nullopt : std::optional{size};
}

/** A device error of any kind, with facts drawn small so that equal draws occur and distinct ones are near. */
[[nodiscard]] auto any_error(std::mt19937_64& random) -> DeviceError {
	auto const small = [&] {
		return std::uniform_int_distribution{0, 5}(random);
	};
	switch (std::uniform_int_distribution{0, 11}(random)) {
	case 0:
		return {.reason = device::DriverUnavailable{
		            .library = pick(random, std::array{device::Library::Cuda, device::Library::Management}),
		            .stage = pick(random, std::array{device::LoadStage::Open, device::LoadStage::EntryPoint}),
		            .name = pick(random, symbols),
		        }};
	case 1:
		return {.reason = device::DriverTooOld{.found = 12'000 + 10 * small(), .required = 13'030}};
	case 2:
		return {.reason = device::CardCount{.found = small()}};
	case 3:
		return {.reason = device::WrongCard{
		            .ordinal = small(),
		            .fact = pick(random,
					             std::array{device::CardFact::ComputeCapability, device::CardFact::SmCount, device::CardFact::MemorySize}),
		            .found = 80 + small(),
		        }};
	case 4:
		return {.reason = device::PeerUnreachable{.from = small(), .to = small()}};
	case 5:
		return {.reason = device::NotNvlink{
		            .ordinal = small(),
		            .link = static_cast<std::uint32_t>(small()),
		            .fault = pick(random, std::array{device::LinkFault::Missing, device::LinkFault::Down, device::LinkFault::Elsewhere}),
		        }};
	case 6:
		return {.reason = device::Failure{.symbol = pick(random, symbols), .status = pick(random, statuses)}};
	case 7:
		return {.reason = device::ManagementFailure{.symbol = pick(random, symbols), .status = pick(random, management_statuses)}};
	case 8:
		return {.reason = device::ModuleRejected{.status = pick(random, statuses), .log = std::format("ptxas error {}", small())}};
	case 9:
		return {.reason = device::KernelMissing{.entry = pick(random, symbols)}};
	case 10:
		return {.reason = device::KernelSignature{
		            .entry = pick(random, symbols),
		            .expected = static_cast<std::size_t>(small()) * 8,
		            .first = optional_size(random),
		            .second = optional_size(random),
		        }};
	default:
		return {.reason = engine::hugepage::HugepageError{
		            .failure = pick(random, std::array{engine::hugepage::Failure::Exhausted, engine::hugepage::Failure::Refused}),
		        }};
	}
}

/**
 * The description is specific: two errors with the same description are the same error, so the text names every fact
 * that tells failures apart. Searched over many errors of every kind with near-identical facts.
 */
[[nodiscard]] auto equal_descriptions_mean_equal_errors(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto seen = std::map<std::string, DeviceError>{};
	auto kinds = std::set<std::size_t>{};
	auto holds = true;
	for (auto draw = 0; draw < 20'000; ++draw) {
		auto const error = any_error(random);
		auto const text = device::describe(error);
		kinds.insert(error.reason.index());
		auto const [known, fresh] = seen.try_emplace(text, error);
		holds = holds && !text.empty() && (fresh || known->second == error);
	}
	return holds && kinds.size() == std::variant_size_v<decltype(DeviceError::reason)>;
}

/** A failed call is described by its symbol and the vendor's name for the status it returned. */
[[nodiscard]] auto failed_calls_name_their_symbol_and_status() -> bool {
	auto const cuda = device::describe({.reason = device::Failure{.symbol = "cuMemAlloc_v2", .status = device::Status::OutOfMemory}});
	auto const nvml = device::describe(
	    {.reason = device::ManagementFailure{.symbol = "nvmlInit_v2", .status = device::ManagementStatus::DriverNotLoaded}});
	return cuda.contains("cuMemAlloc_v2") && cuda.contains("CUDA_ERROR_OUT_OF_MEMORY") && nvml.contains("nvmlInit_v2") &&
	       nvml.contains("NVML_ERROR_DRIVER_NOT_LOADED");
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "random device errors of every kind: equal descriptions only for equal errors (seed 61)",
	        .holds = equal_descriptions_mean_equal_errors(61),
	    },
	    engine::testing::Case{
	        .name = "a failed vendor call is described by its symbol and the vendor's name for its status",
	        .holds = failed_calls_name_their_symbol_and_status(),
	    },
	};
	return engine::testing::report(cases);
}

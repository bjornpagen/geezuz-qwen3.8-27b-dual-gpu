// SPDX-License-Identifier: AGPL-3.0-only
export module engine.device:describe;

import std;
import engine.core;
import engine.driver;
import engine.hugepage;
import engine.prims;
import engine.ptx;

namespace engine::device {

/** cuda.h's name for a status; a status the engine does not name renders as its number. */
[[nodiscard]] auto status_name(driver::Status status) -> std::string {
	using driver::Status;
	switch (status) {
	case Status::Success:
		return "CUDA_SUCCESS";
	case Status::InvalidValue:
		return "CUDA_ERROR_INVALID_VALUE";
	case Status::OutOfMemory:
		return "CUDA_ERROR_OUT_OF_MEMORY";
	case Status::NotInitialized:
		return "CUDA_ERROR_NOT_INITIALIZED";
	case Status::InsufficientDriver:
		return "CUDA_ERROR_INSUFFICIENT_DRIVER";
	case Status::NoDevice:
		return "CUDA_ERROR_NO_DEVICE";
	case Status::InvalidDevice:
		return "CUDA_ERROR_INVALID_DEVICE";
	case Status::InvalidImage:
		return "CUDA_ERROR_INVALID_IMAGE";
	case Status::InvalidContext:
		return "CUDA_ERROR_INVALID_CONTEXT";
	case Status::PeerAccessUnsupported:
		return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
	case Status::InvalidPtx:
		return "CUDA_ERROR_INVALID_PTX";
	case Status::UnsupportedPtxVersion:
		return "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
	case Status::InvalidHandle:
		return "CUDA_ERROR_INVALID_HANDLE";
	case Status::NotFound:
		return "CUDA_ERROR_NOT_FOUND";
	case Status::NotReady:
		return "CUDA_ERROR_NOT_READY";
	case Status::LaunchOutOfResources:
		return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
	case Status::PeerAccessAlreadyEnabled:
		return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
	case Status::HostMemoryAlreadyRegistered:
		return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
	case Status::HostMemoryNotRegistered:
		return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
	case Status::ContextIsDestroyed:
		return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
	case Status::LaunchFailed:
		return "CUDA_ERROR_LAUNCH_FAILED";
	case Status::CooperativeLaunchTooLarge:
		return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
	case Status::Unknown:
		return "CUDA_ERROR_UNKNOWN";
	}
	// The driver may return a status the engine has no name for.
	return std::format("CUDA status {}", std::to_underlying(status));
}

/** nvml.h's name for a status; a status the engine does not name renders as its number. */
[[nodiscard]] auto status_name(driver::ManagementStatus status) -> std::string {
	using driver::ManagementStatus;
	switch (status) {
	case ManagementStatus::Success:
		return "NVML_SUCCESS";
	case ManagementStatus::Uninitialized:
		return "NVML_ERROR_UNINITIALIZED";
	case ManagementStatus::InvalidArgument:
		return "NVML_ERROR_INVALID_ARGUMENT";
	case ManagementStatus::NotSupported:
		return "NVML_ERROR_NOT_SUPPORTED";
	case ManagementStatus::NoPermission:
		return "NVML_ERROR_NO_PERMISSION";
	case ManagementStatus::NotFound:
		return "NVML_ERROR_NOT_FOUND";
	case ManagementStatus::DriverNotLoaded:
		return "NVML_ERROR_DRIVER_NOT_LOADED";
	case ManagementStatus::GpuIsLost:
		return "NVML_ERROR_GPU_IS_LOST";
	case ManagementStatus::ArgumentVersionMismatch:
		return "NVML_ERROR_ARGUMENT_VERSION_MISMATCH";
	case ManagementStatus::Unknown:
		return "NVML_ERROR_UNKNOWN";
	}
	// NVML may return a status the engine has no name for.
	return std::format("NVML status {}", std::to_underlying(status));
}

[[nodiscard]] auto library_name(driver::Library library) -> std::string_view {
	switch (library) {
	case driver::Library::Cuda:
		return "the CUDA driver";
	case driver::Library::Management:
		return "NVML";
	}
	std::unreachable();
}

[[nodiscard]] auto parameter_name(std::optional<std::size_t> bytes) -> std::string {
	return bytes
	    .transform([](std::size_t size) {
		    return std::format("{} bytes", size);
	    })
	    .value_or("none");
}

/** The operator's account of each kind of device failure, one overload per kind. */
struct Describe {
	[[nodiscard]] auto operator()(driver::DriverUnavailable const& reason) const -> std::string {
		switch (reason.stage) {
		case driver::LoadStage::Open:
			return std::format("{} is unavailable: {} could not be opened", library_name(reason.library), reason.name);
		case driver::LoadStage::EntryPoint:
			return std::format("{} is unavailable: it lacks the entry point {}", library_name(reason.library), reason.name);
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator()(driver::DriverTooOld const& reason) const -> std::string {
		return std::format("the driver is too old: it supports CUDA {}.{}, and jitting the engine's ptx needs CUDA {}.{}",
		                   reason.found / 1000, reason.found % 1000 / 10, reason.required / 1000, reason.required % 1000 / 10);
	}

	[[nodiscard]] auto operator()(driver::CardCount const& reason) const -> std::string {
		return std::format("the driver sees {} cards; the engine runs on exactly two", reason.found);
	}

	[[nodiscard]] auto operator()(driver::WrongCard const& reason) const -> std::string {
		switch (reason.fact) {
		case driver::CardFact::ComputeCapability:
			return std::format("card {} is not an RTX 3090: compute capability {}.{}, not {}.{}", reason.ordinal, reason.found / 10,
			                   reason.found % 10, ptx::abi.header.sm / 10, ptx::abi.header.sm % 10);
		case driver::CardFact::SmCount:
			return std::format("card {} is not an RTX 3090: {} SMs, not {}", reason.ordinal, reason.found, prims::sm86.sms);
		case driver::CardFact::MemorySize:
			return std::format("card {} is not an RTX 3090 with the planned memory: it reports {} bytes ({:.2f} GiB)", reason.ordinal,
			                   reason.found, static_cast<double>(reason.found) / static_cast<double>(std::uint64_t{1} << 30U));
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator()(driver::PeerUnreachable const& reason) const -> std::string {
		return std::format("the driver cannot map card {}'s memory into card {}: no peer access between the cards", reason.to, reason.from);
	}

	[[nodiscard]] auto operator()(driver::NotNvlink const& reason) const -> std::string {
		switch (reason.fault) {
		case driver::LinkFault::Missing:
			return std::format("card {} has no nvlink link {}: the peer path would not be nvlink", reason.ordinal, reason.link);
		case driver::LinkFault::Down:
			return std::format("card {}'s nvlink link {} is down: check the bridge's seating and persistence mode", reason.ordinal,
			                   reason.link);
		case driver::LinkFault::Elsewhere:
			return std::format("card {}'s nvlink link {} leads to a device other than the other card", reason.ordinal, reason.link);
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator()(driver::Failure const& reason) const -> std::string {
		return std::format("the CUDA driver call {} failed with {}", reason.symbol, status_name(reason.status));
	}

	[[nodiscard]] auto operator()(driver::ManagementFailure const& reason) const -> std::string {
		return std::format("the NVML call {} failed with {}", reason.symbol, status_name(reason.status));
	}

	[[nodiscard]] auto operator()(driver::ModuleRejected const& reason) const -> std::string {
		return std::format("the driver's jit refused a ptx image with {}: {}", status_name(reason.status), reason.log);
	}

	[[nodiscard]] auto operator()(driver::KernelMissing const& reason) const -> std::string {
		return std::format("the loaded image has no kernel entry {}", reason.entry);
	}

	[[nodiscard]] auto operator()(driver::KernelSignature const& reason) const -> std::string {
		return std::format(
		    "kernel {} is declared with one {}-byte parameter struct, but its entry's first parameter is {} and its second {}",
		    reason.entry, reason.expected, parameter_name(reason.first), parameter_name(reason.second));
	}

	[[nodiscard]] auto operator()(hugepage::HugepageError const& reason) const -> std::string {
		switch (reason.failure) {
		case hugepage::Failure::Exhausted:
			return "no hugepages are left for pinned host memory: reserve more at bring-up (vm.nr_hugepages, on numa node 0)";
		case hugepage::Failure::Refused:
			return "the kernel refused a hugepage mapping for pinned host memory (no hugetlb support, or a locked-memory limit)";
		}
		std::unreachable();
	}
};

/** One sentence for an operator naming what failed and the facts it failed on: the one description of a device error. */
export [[nodiscard]] auto describe(driver::DeviceError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

}

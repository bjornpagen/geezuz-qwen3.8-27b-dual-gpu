// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:error;

import std;
import engine.core;
import :abi;

namespace engine::driver {

/** Which vendor library a load step concerns. */
export enum class Library : std::uint8_t {
	Cuda,
	Management,
};

/** How far loading a library got before it failed. */
export enum class LoadStage : std::uint8_t {
	Open,
	EntryPoint,
};

/**
 * A vendor library is not there, or lacks an entry point the engine declares. `name` is the soname that would not
 * open or the symbol that would not resolve. On macOS this is the outcome of every start: no `libcuda.so.1` exists.
 */
export struct DriverUnavailable {
	Library library;
	LoadStage stage;
	std::string_view name;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(DriverUnavailable const&) const -> bool = default;
};

/** The driver's CUDA version (major * 1000 + minor * 10) is older than the jit of the engine's ptx isa needs. */
export struct DriverTooOld {
	int found;
	int required;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(DriverTooOld const&) const -> bool = default;
};

/** The machine does not have exactly two cards. */
export struct CardCount {
	int found;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(CardCount const&) const -> bool = default;
};

/** The fact of a card that is not the engine's card. */
export enum class CardFact : std::uint8_t {
	ComputeCapability,
	SmCount,
	MemorySize,
};

/** A card is not an RTX 3090: `found` is the value of `fact` it reported (compute capability as major * 10 + minor). */
export struct WrongCard {
	CUdevice ordinal;
	CardFact fact;
	std::int64_t found;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(WrongCard const&) const -> bool = default;
};

/** The driver cannot map card `to`'s memory into card `from`'s address space. */
export struct PeerUnreachable {
	CUdevice from;
	CUdevice to;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(PeerUnreachable const&) const -> bool = default;
};

/** How an nvlink link fails to carry the peer path. */
export enum class LinkFault : std::uint8_t {
	Missing,
	Down,
	Elsewhere,
};

/** A card's link `link` does not join it to the other card, so peer traffic would not run over nvlink. */
export struct NotNvlink {
	CUdevice ordinal;
	std::uint32_t link;
	LinkFault fault;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(NotNvlink const&) const -> bool = default;
};

/**
 * A CUDA driver call returned an error. Out of memory is transient: device memory another process holds can be
 * released. Every other status is a property of the machine, the image or the program, retried in vain.
 */
export struct Failure {
	std::string_view symbol;
	Status status;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return status == Status::OutOfMemory ? core::Retry::Transient : core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Failure const&) const -> bool = default;
};

/** An NVML call returned an error. */
export struct ManagementFailure {
	std::string_view symbol;
	ManagementStatus status;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(ManagementFailure const&) const -> bool = default;
};

/** The driver's jit refused a ptx image; `log` is the jit's own account of why. */
export struct ModuleRejected {
	Status status;
	std::string log;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto operator==(ModuleRejected const&) const -> bool = default;
};

/** A loaded image has no kernel entry of this name. */
export struct KernelMissing {
	std::string_view entry;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(KernelMissing const&) const -> bool = default;
};

/**
 * A kernel entry's parameters are not the one by-value parameter struct of `expected` bytes its host declaration
 * launches it with: `first` and `second` are the byte sizes of its first two parameters, absent where it has none.
 */
export struct KernelSignature {
	std::string_view entry;
	std::size_t expected;
	std::optional<std::size_t> first;
	std::optional<std::size_t> second;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(KernelSignature const&) const -> bool = default;
};

/** Every way bringing up or using the two cards can fail. */
export struct [[nodiscard]] DeviceError {
	std::variant<DriverUnavailable, DriverTooOld, CardCount, WrongCard, PeerUnreachable, NotNvlink, Failure, ManagementFailure,
	             ModuleRejected, KernelMissing, KernelSignature>
	    reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(DeviceError const&) const -> bool = default;
};

/** Calls a CUDA entry point and turns a non-success status into the failure that names it. */
template<Symbol Name, class... Parameters, class... Arguments>
[[nodiscard]] auto call(Entry<Name, Status(Parameters...)> const& entry, Arguments... arguments) -> core::Result<void, DeviceError> {
	auto const status = entry(arguments...);
	if (status != Status::Success) {
		return std::unexpected(DeviceError{.reason = Failure{.symbol = entry.symbol, .status = status}});
	}
	return {};
}

/** Calls an NVML entry point and turns a non-success status into the failure that names it. */
template<Symbol Name, class... Parameters, class... Arguments>
[[nodiscard]] auto call(Entry<Name, ManagementStatus(Parameters...)> const& entry, Arguments... arguments)
    -> core::Result<void, DeviceError> {
	auto const status = entry(arguments...);
	if (status != ManagementStatus::Success) {
		return std::unexpected(DeviceError{.reason = ManagementFailure{.symbol = entry.symbol, .status = status}});
	}
	return {};
}

}

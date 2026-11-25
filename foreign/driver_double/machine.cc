// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver_double:machine;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;

namespace engine::driver_double {

/** Where one nvlink link of a modelled card leads. */
export enum class LinkModel : std::uint8_t {
	ToPeer,
	Elsewhere,
	Down,
	Missing,
};

/** One modelled card. The defaults are an RTX 3090 bridged to its peer by all four links, running CUDA in P2. */
export struct CardModel {
	int compute_major = ptx::abi.header.sm / 10;
	int compute_minor = ptx::abi.header.sm % 10;
	int sm_count = device::prims::sm86.sms;
	std::uint64_t memory_bytes = 25'447'170'048;
	std::array<LinkModel, 4> links = {LinkModel::ToPeer, LinkModel::ToPeer, LinkModel::ToPeer, LinkModel::ToPeer};
	bool reaches_peer = true;
	std::uint32_t memory_clock = 9501;
	int performance_state = 2;
};

/** The modelled jit accepts every image and finds the entries the ptx text declares. */
export struct JitAccepts {};

/** The modelled jit refuses every image with this status and log. */
export struct JitRejects {
	driver::Status status;
	std::string log;
};

/**
 * The machine the fake driver models. `failing_call` names the one vendor call, counted from zero across CUDA and
 * NVML in call order, that fails with the library's "unknown" status: a search over it reaches every failure path.
 * `pending_queries` is how many times a recorded event answers "not ready" before its work counts as done.
 */
export struct Machine {
	int driver_version = 13030;
	driver::Status init = driver::Status::Success;
	std::vector<CardModel> cards = {CardModel{}, CardModel{}};
	std::optional<std::size_t> failing_call = std::nullopt;
	std::variant<JitAccepts, JitRejects> jit = JitAccepts{};
	std::size_t pending_queries = 0;
};

/** The kinds of driver resource the fake accounts for. */
export enum class Kind : std::uint8_t {
	Context,
	DeviceMemory,
	HostMemory,
	Stream,
	Event,
	Module,
	ManagementSession,
};

/** A misuse of the driver the real one would punish, sometimes silently; a correct engine causes none. */
export enum class Violation : std::uint8_t {
	UnknownHandle,
	DoubleRelease,
	ReleasedAfterContext,
	OutlivedContext,
	NoCurrentContext,
	DestroyedCurrentContext,
	WrongContext,
	Unpinned,
	OutOfBounds,
};

/** One launch as the fake driver saw it: the kernel, its geometry and the exact parameter bytes it was given. */
export struct Launch {
	driver::CUdevice ordinal;
	std::string entry;
	driver::Dim grid;
	driver::Dim block;
	std::uint32_t shared_bytes;
	driver::Residency residency;
	std::vector<std::byte> parameters;
};

/** The pci address the fake gives card `ordinal`, and the one a link that leads elsewhere reports. */
export [[nodiscard]] constexpr auto pci_of(int ordinal) -> driver::PciAddress {
	return {.domain = 0, .bus = 0x01 + 0x40 * static_cast<std::uint32_t>(ordinal), .device = 0};
}

export inline constexpr driver::PciAddress elsewhere = {.domain = 0, .bus = 0xee, .device = 0};

}

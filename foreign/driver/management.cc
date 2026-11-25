// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:management;

import std;
import engine.core;
import :abi;
import :error;

namespace engine::driver {

/** The link reports no peer at all: the card has no such link, or no nvlink. */
export struct LinkMissing {
	[[nodiscard]] constexpr auto operator==(LinkMissing const&) const -> bool = default;
};

/** The link exists and is not trained: no bridge, an unseated bridge or card, or persistence mode off. */
export struct LinkDown {
	[[nodiscard]] constexpr auto operator==(LinkDown const&) const -> bool = default;
};

/** The link is active and its far end is the device at `remote`. */
export struct LinkUp {
	PciAddress remote;

	[[nodiscard]] constexpr auto operator==(LinkUp const&) const -> bool = default;
};

/** One nvlink link of one card, as NVML reports it. */
export using LinkReading = std::variant<LinkMissing, LinkDown, LinkUp>;

/** NVML's performance states; P0 is the highest, and GeForce Ampere runs CUDA work in P2. */
export enum class PerformanceState : std::uint8_t {
	P0,
	P1,
	P2,
	P3,
	P4,
	P5,
	P6,
	P7,
	P8,
	P9,
	P10,
	P11,
	P12,
	P13,
	P14,
	P15,
};

/**
 * A card's memory clock in NVML's units (MHz, as nvidia-smi prints it: 9751 is the 3090's P0 maximum, 9501 its P2
 * clock under CUDA) and its performance state, absent when NVML cannot tell.
 */
export struct MemoryClock {
	std::uint32_t megahertz;
	std::optional<PerformanceState> state;

	[[nodiscard]] constexpr auto operator==(MemoryClock const&) const -> bool = default;
};

/**
 * An initialized NVML: the session that answers questions about cards by their pci address. NVML's own handles are
 * looked up per question and never kept, so nothing outlives the session.
 */
export class ManagementSession {
public:
	[[nodiscard]] static auto open(Management const& api) -> core::Result<ManagementSession, DeviceError> {
		return call(api.init).transform([&] {
			return ManagementSession{api};
		});
	}

	ManagementSession(ManagementSession const&) = delete;
	auto operator=(ManagementSession const&) -> ManagementSession& = delete;

	ManagementSession(ManagementSession&& other) noexcept : api_{std::exchange(other.api_, Management{})} {}

	auto operator=(ManagementSession&& other) noexcept -> ManagementSession& {
		std::swap(api_, other.api_);
		return *this;
	}

	~ManagementSession() {
		// A moved-from session holds no entry points; shutting down cannot fail in a way the engine could act on.
		if (api_.shutdown.function != nullptr) {
			std::ignore = api_.shutdown();
		}
	}

	[[nodiscard]] auto link(PciAddress card, std::uint32_t link) const -> core::Result<LinkReading, DeviceError> {
		auto const device = find(card);
		if (!device) {
			return std::unexpected(device.error());
		}
		auto state = EnableState::Disabled;
		// NVML answers "not supported" for a link index the card does not have, and for a card without nvlink.
		if (auto const status = api_.nvlink_state(*device, link, &state); status == ManagementStatus::NotSupported) {
			return LinkReading{LinkMissing{}};
		} else if (status != ManagementStatus::Success) {
			return std::unexpected(DeviceError{.reason = ManagementFailure{.symbol = api_.nvlink_state.symbol, .status = status}});
		}
		if (state != EnableState::Enabled) {
			return LinkReading{LinkDown{}};
		}
		auto remote = PciInfo{};
		return call(api_.nvlink_remote_pci_info, *device, link, &remote).transform([&] {
			return LinkReading{LinkUp{.remote = {.domain = remote.domain, .bus = remote.bus, .device = remote.device}}};
		});
	}

	[[nodiscard]] auto memory_clock(PciAddress card) const -> core::Result<MemoryClock, DeviceError> {
		auto const device = find(card);
		if (!device) {
			return std::unexpected(device.error());
		}
		auto megahertz = 0U;
		auto state = 0;
		return call(api_.clock_info, *device, ClockType::Memory, &megahertz)
		    .and_then([&] {
			    return call(api_.performance_state, *device, &state);
		    })
		    .transform([&] {
			    // NVML's states run 0 to 15; anything else is its "unknown".
			    auto const known = state >= 0 && state <= 15;
			    return MemoryClock{
			        .megahertz = megahertz,
			        .state = known ? std::optional{static_cast<PerformanceState>(state)} : std::nullopt,
			    };
		    });
	}

private:
	explicit ManagementSession(Management const& api) : api_{api} {}

	[[nodiscard]] auto find(PciAddress card) const -> core::Result<nvmlDevice_t, DeviceError> {
		auto const bus_id = std::format("{:08x}:{:02x}:{:02x}.0", card.domain, card.bus, card.device);
		auto* device = nvmlDevice_t{};
		return call(api_.device_by_pci_bus_id, bus_id.c_str(), &device).transform([&] {
			return device;
		});
	}

	Management api_;
};

}

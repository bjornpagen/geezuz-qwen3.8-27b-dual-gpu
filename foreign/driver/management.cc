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

/** A card's SM clock in MHz now (NVML_CLOCK_SM). */
export struct SmClock {
	std::uint32_t megahertz;

	[[nodiscard]] constexpr auto operator==(SmClock const&) const -> bool = default;
};

/** A card's power draw now and the limit the driver enforces after every limiter, in milliwatts. */
export struct Power {
	std::uint32_t draw_milliwatts;
	std::uint32_t limit_milliwatts;

	[[nodiscard]] constexpr auto operator==(Power const&) const -> bool = default;
};

/** The reasons holding a card's clocks down right now. */
export struct ClocksEventReasons {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto contains(ClocksEventReason reason) const -> bool {
		return (bits & std::to_underlying(reason)) != 0;
	}

	[[nodiscard]] constexpr auto operator==(ClocksEventReasons const&) const -> bool = default;
};

/**
 * How long, in total since the driver loaded, a card's clocks have been held down by each reason; absent where NVML
 * does not track that reason for the card. A soak that holds a clock lock watches `sw_power_cap` stay flat.
 */
export struct ClocksEventCounters {
	std::optional<std::chrono::nanoseconds> sw_power_cap;
	std::optional<std::chrono::nanoseconds> sync_boost;
	std::optional<std::chrono::nanoseconds> sw_thermal_slowdown;
	std::optional<std::chrono::nanoseconds> hw_thermal_slowdown;
	std::optional<std::chrono::nanoseconds> hw_power_brake_slowdown;

	[[nodiscard]] constexpr auto operator==(ClocksEventCounters const&) const -> bool = default;
};

/** One nvlink link's version and speed in MBps, the speed absent where NVML has no field for the link. */
export struct NvlinkLinkReading {
	NvlinkVersion version;
	std::optional<std::uint32_t> megabytes_per_second;

	[[nodiscard]] constexpr auto operator==(NvlinkLinkReading const&) const -> bool = default;
};

template<class T>
[[nodiscard]] auto leading(std::array<std::byte, 8> const& bytes) -> T {
	auto part = std::array<std::byte, sizeof(T)>{};
	std::ranges::copy_n(bytes.begin(), sizeof(T), part.begin());
	return std::bit_cast<T>(part);
}

/**
 * A field's value as a count, when NVML answered it with one. A field NVML does not support for the card, or answers
 * with a fraction or a negative number, is no count.
 */
[[nodiscard]] auto count_of(FieldValue const& field) -> std::optional<std::uint64_t> {
	if (field.status != ManagementStatus::Success) {
		return std::nullopt;
	}
	auto const non_negative = [](auto value) -> std::optional<std::uint64_t> {
		return value < 0 ? std::nullopt : std::optional{static_cast<std::uint64_t>(value)};
	};
	switch (field.value_type) {
	case ValueType::UnsignedInt:
		return leading<unsigned int>(field.value);
	case ValueType::UnsignedLong:
		return leading<unsigned long>(field.value);
	case ValueType::UnsignedLongLong:
		return leading<unsigned long long>(field.value);
	case ValueType::UnsignedShort:
		return leading<unsigned short>(field.value);
	case ValueType::SignedLongLong:
		return non_negative(leading<long long>(field.value));
	case ValueType::SignedInt:
		return non_negative(leading<int>(field.value));
	case ValueType::Double:
		return std::nullopt;
	}
	return std::nullopt;
}

[[nodiscard]] auto nanoseconds_of(FieldValue const& field) -> std::optional<std::chrono::nanoseconds> {
	return count_of(field).transform([](std::uint64_t count) {
		return std::chrono::nanoseconds{core::narrow<std::chrono::nanoseconds::rep>(count)};
	});
}

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

	[[nodiscard]] auto sm_clock(PciAddress card) const -> core::Result<SmClock, DeviceError> {
		auto megahertz = 0U;
		return find(card)
		    .and_then([&](nvmlDevice_t device) {
			    return call(api_.clock_info, device, ClockType::Sm, &megahertz);
		    })
		    .transform([&] {
			    return SmClock{.megahertz = megahertz};
		    });
	}

	[[nodiscard]] auto power(PciAddress card) const -> core::Result<Power, DeviceError> {
		auto draw = 0U;
		auto limit = 0U;
		return find(card)
		    .and_then([&](nvmlDevice_t device) {
			    return call(api_.power_usage, device, &draw).and_then([&] {
				    return call(api_.enforced_power_limit, device, &limit);
			    });
		    })
		    .transform([&] {
			    return Power{.draw_milliwatts = draw, .limit_milliwatts = limit};
		    });
	}

	[[nodiscard]] auto clocks_event_reasons(PciAddress card) const -> core::Result<ClocksEventReasons, DeviceError> {
		auto bits = 0ULL;
		return find(card)
		    .and_then([&](nvmlDevice_t device) {
			    return call(api_.clocks_event_reasons, device, &bits);
		    })
		    .transform([&] {
			    return ClocksEventReasons{.bits = bits};
		    });
	}

	[[nodiscard]] auto clocks_event_counters(PciAddress card) const -> core::Result<ClocksEventCounters, DeviceError> {
		auto fields = std::array{
		    request(Field::SwPowerCap),           request(Field::SyncBoost),
		    request(Field::SwThermalSlowdown),    request(Field::HwThermalSlowdown),
		    request(Field::HwPowerBrakeSlowdown),
		};
		return read_fields(card, fields).transform([&] {
			return ClocksEventCounters{
			    .sw_power_cap = nanoseconds_of(fields[0]),
			    .sync_boost = nanoseconds_of(fields[1]),
			    .sw_thermal_slowdown = nanoseconds_of(fields[2]),
			    .hw_thermal_slowdown = nanoseconds_of(fields[3]),
			    .hw_power_brake_slowdown = nanoseconds_of(fields[4]),
			};
		});
	}

	/** Link `link`'s version and speed. */
	[[nodiscard]] auto nvlink(PciAddress card, std::uint32_t link) const -> core::Result<NvlinkLinkReading, DeviceError> {
		// NVML numbers the speed fields of links 0 to 5 consecutively; the engine's cards have four links.
		core::invariant(link < 6);
		auto version = 0U;
		auto fields = std::array{request(nvlink_speed_field(link))};
		return read_fields(card, fields)
		    .and_then([&] {
			    return find(card);
		    })
		    .and_then([&](nvmlDevice_t device) {
			    return call(api_.nvlink_version, device, link, &version);
		    })
		    .transform([&] {
			    return NvlinkLinkReading{
			        .version = static_cast<NvlinkVersion>(version),
			        .megabytes_per_second = count_of(fields[0]).transform([](std::uint64_t value) {
				        return core::narrow<std::uint32_t>(value);
			        }),
			    };
		    });
	}

private:
	[[nodiscard]] static auto request(Field field) -> FieldValue {
		return FieldValue{
		    .field = field,
		    .scope = 0,
		    .timestamp_microseconds = 0,
		    .latency_microseconds = 0,
		    .value_type = ValueType::UnsignedLongLong,
		    .status = ManagementStatus::Unknown,
		    .value = {},
		};
	}

	/** Fills each field's value; a field NVML cannot answer reports that in its own status, not as a failure. */
	[[nodiscard]] auto read_fields(PciAddress card, std::span<FieldValue> fields) const -> core::Result<void, DeviceError> {
		return find(card).and_then([&](nvmlDevice_t device) {
			return call(api_.field_values, device, core::narrow<int>(fields.size()), fields.data());
		});
	}

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

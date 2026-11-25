// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:probe;

import std;
import engine.core;
import :abi;
import :error;

namespace engine::driver {

/** What one card says about itself, before anything is decided about it. */
export struct CardFacts {
	CUdevice ordinal;
	int compute_major;
	int compute_minor;
	int sm_count;
	std::uint64_t memory_bytes;
	PciAddress pci;

	[[nodiscard]] constexpr auto operator==(CardFacts const&) const -> bool = default;
};

/** What the driver says about the machine: its CUDA version and every card it sees, in ordinal order. */
export struct Census {
	int driver_version;
	std::vector<CardFacts> cards;
};

[[nodiscard]] auto attribute(Api const& api, CUdevice device, Attribute which) -> core::Result<int, DeviceError> {
	auto value = 0;
	return call(api.device_get_attribute, &value, which, device).transform([&] {
		return value;
	});
}

[[nodiscard]] auto card_facts(Api const& api, int ordinal) -> core::Result<CardFacts, DeviceError> {
	auto device = CUdevice{};
	if (auto got = call(api.device_get, &device, ordinal); !got) {
		return std::unexpected(got.error());
	}
	auto memory = std::size_t{0};
	if (auto got = call(api.device_total_mem, &memory, device); !got) {
		return std::unexpected(got.error());
	}
	auto const major = attribute(api, device, Attribute::ComputeCapabilityMajor);
	auto const minor = major.and_then([&](int) {
		return attribute(api, device, Attribute::ComputeCapabilityMinor);
	});
	auto const sms = minor.and_then([&](int) {
		return attribute(api, device, Attribute::MultiprocessorCount);
	});
	auto const domain = sms.and_then([&](int) {
		return attribute(api, device, Attribute::PciDomainId);
	});
	auto const bus = domain.and_then([&](int) {
		return attribute(api, device, Attribute::PciBusId);
	});
	auto const slot = bus.and_then([&](int) {
		return attribute(api, device, Attribute::PciDeviceId);
	});
	if (!slot) {
		return std::unexpected(slot.error());
	}
	return CardFacts{
	    .ordinal = device,
	    .compute_major = *major,
	    .compute_minor = *minor,
	    .sm_count = *sms,
	    .memory_bytes = memory,
	    .pci =
	        {
	            .domain = core::narrow<std::uint32_t>(*domain),
	            .bus = core::narrow<std::uint32_t>(*bus),
	            .device = core::narrow<std::uint32_t>(*slot),
	        },
	};
}

/** Initializes the driver and reads the machine's census. */
export [[nodiscard]] auto census(Api const& api) -> core::Result<Census, DeviceError> {
	auto version = 0;
	if (auto got = call(api.driver_get_version, &version); !got) {
		return std::unexpected(got.error());
	}
	// The driver reports a machine without cards as a failure to initialize; to the engine it is a census of zero.
	if (auto const status = api.init(0); status == Status::NoDevice) {
		return Census{.driver_version = version, .cards = {}};
	} else if (status != Status::Success) {
		return std::unexpected(DeviceError{.reason = Failure{.symbol = api.init.symbol, .status = status}});
	}
	auto count = 0;
	if (auto got = call(api.device_get_count, &count); !got) {
		return std::unexpected(got.error());
	}
	auto census = Census{.driver_version = version, .cards = {}};
	for (auto ordinal = 0; ordinal < count; ++ordinal) {
		auto facts = card_facts(api, ordinal);
		if (!facts) {
			return std::unexpected(facts.error());
		}
		census.cards.push_back(*facts);
	}
	return census;
}

/** Whether the driver can map `to`'s memory into `from`'s address space. */
export [[nodiscard]] auto can_access_peer(Api const& api, CUdevice from, CUdevice to) -> core::Result<bool, DeviceError> {
	auto can = 0;
	return call(api.device_can_access_peer, &can, from, to).transform([&] {
		return can == 1;
	});
}

}

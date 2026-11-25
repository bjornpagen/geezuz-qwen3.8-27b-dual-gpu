// SPDX-License-Identifier: AGPL-3.0-only
//
// The fake NVML: sessions, card lookup by pci address, nvlink links and the memory clock.
export module engine.driver_double:nvml;

import std;
import engine.core;
import engine.driver;
import :machine;
import :ledger;

namespace engine::driver_double {

using driver::CUcontext;
using driver::CUdevice;
using driver::CUdeviceptr;
using driver::CUevent;
using driver::CUfunction;
using driver::CUmodule;
using driver::CUstream;
using driver::ManagementStatus;

auto fake_nvml_init() -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	++state().sessions;
	return ManagementStatus::Success;
}

auto fake_nvml_shutdown() -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	if (state().sessions == 0) {
		violate(Violation::DoubleRelease);
		return ManagementStatus::Uninitialized;
	}
	--state().sessions;
	return ManagementStatus::Success;
}

[[nodiscard]] auto nvml_card(driver::nvmlDevice_t device) -> NvmlCard const* {
	for (auto& candidate : state().nvml_cards) {
		if (reinterpret_cast<driver::nvmlDevice_t>(&candidate) == device) {
			return &candidate;
		}
	}
	violate(Violation::UnknownHandle);
	return nullptr;
}

auto fake_nvml_device_by_pci_bus_id(char const* bus_id, driver::nvmlDevice_t* device) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto& machine = state();
	if (machine.sessions == 0) {
		return ManagementStatus::Uninitialized;
	}
	for (auto& candidate : machine.nvml_cards) {
		auto const pci = pci_of(candidate.ordinal);
		if (std::format("{:08x}:{:02x}:{:02x}.0", pci.domain, pci.bus, pci.device) == bus_id) {
			*device = reinterpret_cast<driver::nvmlDevice_t>(&candidate);
			return ManagementStatus::Success;
		}
	}
	return ManagementStatus::NotFound;
}

[[nodiscard]] auto link_model(driver::nvmlDevice_t device, unsigned int link) -> std::optional<LinkModel> {
	auto const* const found = nvml_card(device);
	if (found == nullptr || state().sessions == 0) {
		return std::nullopt;
	}
	auto const& links = card(found->ordinal).links;
	return link < links.size() ? links[link] : LinkModel::Missing;
}

auto fake_nvml_nvlink_state(driver::nvmlDevice_t device, unsigned int link, driver::EnableState* active) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const model = link_model(device, link);
	if (!model) {
		return ManagementStatus::InvalidArgument;
	}
	switch (*model) {
	case LinkModel::ToPeer:
	case LinkModel::Elsewhere:
		*active = driver::EnableState::Enabled;
		return ManagementStatus::Success;
	case LinkModel::Down:
		*active = driver::EnableState::Disabled;
		return ManagementStatus::Success;
	case LinkModel::Missing:
		return ManagementStatus::NotSupported;
	}
	return ManagementStatus::Unknown;
}

auto fake_nvml_nvlink_remote_pci_info(driver::nvmlDevice_t device, unsigned int link, driver::PciInfo* info) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const model = link_model(device, link);
	if (!model || (*model != LinkModel::ToPeer && *model != LinkModel::Elsewhere)) {
		return ManagementStatus::InvalidArgument;
	}
	auto const& machine = state();
	auto const ordinal = nvml_card(device)->ordinal;
	auto const peer = (ordinal + 1) % core::narrow<int>(machine.machine.cards.size());
	auto const remote = *model == LinkModel::ToPeer ? pci_of(peer) : elsewhere;
	*info = driver::PciInfo{};
	info->domain = remote.domain;
	info->bus = remote.bus;
	info->device = remote.device;
	return ManagementStatus::Success;
}

auto fake_nvml_clock_info(driver::nvmlDevice_t device, driver::ClockType type, unsigned int* clock) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const found = nvml_card(device);
	if (found == nullptr || state().sessions == 0) {
		return ManagementStatus::InvalidArgument;
	}
	if (type != driver::ClockType::Memory) {
		return ManagementStatus::NotSupported;
	}
	*clock = card(found->ordinal).memory_clock;
	return ManagementStatus::Success;
}

auto fake_nvml_performance_state(driver::nvmlDevice_t device, int* performance_state) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const found = nvml_card(device);
	if (found == nullptr || state().sessions == 0) {
		return ManagementStatus::InvalidArgument;
	}
	*performance_state = card(found->ordinal).performance_state;
	return ManagementStatus::Success;
}

}

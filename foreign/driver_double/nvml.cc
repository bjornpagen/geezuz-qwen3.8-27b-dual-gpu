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
	switch (type) {
	case driver::ClockType::Memory:
		*clock = card(found->ordinal).memory_clock;
		return ManagementStatus::Success;
	case driver::ClockType::Sm:
		*clock = card(found->ordinal).sm_clock;
		return ManagementStatus::Success;
	case driver::ClockType::Graphics:
	case driver::ClockType::Video:
		return ManagementStatus::NotSupported;
	}
	return ManagementStatus::InvalidArgument;
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

/** The card a live session's handle names; null after a refusal already reported. */
[[nodiscard]] auto session_card(driver::nvmlDevice_t device) -> CardModel const* {
	auto const* const found = nvml_card(device);
	return found == nullptr || state().sessions == 0 ? nullptr : &card(found->ordinal);
}

auto fake_nvml_power_usage(driver::nvmlDevice_t device, unsigned int* milliwatts) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const model = session_card(device);
	if (model == nullptr) {
		return ManagementStatus::InvalidArgument;
	}
	*milliwatts = model->power_draw_milliwatts;
	return ManagementStatus::Success;
}

auto fake_nvml_enforced_power_limit(driver::nvmlDevice_t device, unsigned int* milliwatts) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const model = session_card(device);
	if (model == nullptr) {
		return ManagementStatus::InvalidArgument;
	}
	*milliwatts = model->power_limit_milliwatts;
	return ManagementStatus::Success;
}

auto fake_nvml_temperature(driver::nvmlDevice_t device, driver::TemperatureSensor sensor, unsigned int* celsius) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const model = session_card(device);
	if (model == nullptr || sensor != driver::TemperatureSensor::Gpu) {
		return ManagementStatus::InvalidArgument;
	}
	*celsius = model->gpu_celsius;
	return ManagementStatus::Success;
}

auto fake_nvml_clocks_event_reasons(driver::nvmlDevice_t device, unsigned long long* reasons) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const model = session_card(device);
	if (model == nullptr) {
		return ManagementStatus::InvalidArgument;
	}
	*reasons = model->clocks_event_reasons;
	return ManagementStatus::Success;
}

auto fake_nvml_nvlink_version(driver::nvmlDevice_t device, unsigned int link, unsigned int* version) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const model = link_model(device, link);
	if (!model) {
		return ManagementStatus::InvalidArgument;
	}
	if (*model == LinkModel::Missing) {
		return ManagementStatus::NotSupported;
	}
	*version = session_card(device)->nvlink_version;
	return ManagementStatus::Success;
}

/** One field's answer: a count of the given type, or the field unsupported. */
auto answer(driver::FieldValue& field, std::optional<std::uint64_t> count, driver::ValueType type) -> void {
	if (!count) {
		field.status = ManagementStatus::NotSupported;
		return;
	}
	field.status = ManagementStatus::Success;
	field.value_type = type;
	field.value = {};
	if (type == driver::ValueType::UnsignedInt) {
		auto const bytes = std::bit_cast<std::array<std::byte, 4>>(core::narrow<unsigned int>(*count));
		std::ranges::copy(bytes, field.value.begin());
	} else {
		field.value = std::bit_cast<std::array<std::byte, 8>>(static_cast<unsigned long long>(*count));
	}
}

[[nodiscard]] auto nanoseconds(std::optional<std::chrono::nanoseconds> counter) -> std::optional<std::uint64_t> {
	return counter.transform([](std::chrono::nanoseconds value) {
		return core::narrow<std::uint64_t>(value.count());
	});
}

auto fake_nvml_field_values(driver::nvmlDevice_t device, int count, driver::FieldValue* values) -> ManagementStatus {
	if (injected()) {
		return ManagementStatus::Unknown;
	}
	auto const* const model = session_card(device);
	if (model == nullptr || count < 0 || values == nullptr) {
		return ManagementStatus::InvalidArgument;
	}
	auto const& counters = model->clocks_event_counters;
	constexpr auto counter_type = driver::ValueType::UnsignedLongLong;
	for (auto& field : std::span{values, static_cast<std::size_t>(count)}) {
		auto const id = std::to_underlying(field.field);
		auto const first_speed = std::to_underlying(driver::Field::NvlinkSpeedLink0);
		if (id >= first_speed && id < first_speed + 6) {
			auto const link = id - first_speed;
			auto const present = link < model->links.size() && model->links[link] != LinkModel::Missing;
			answer(field, present ? std::optional<std::uint64_t>{model->nvlink_megabytes_per_second} : std::nullopt,
			       driver::ValueType::UnsignedInt);
			continue;
		}
		switch (field.field) {
		case driver::Field::SwPowerCap:
			answer(field, nanoseconds(counters.sw_power_cap), counter_type);
			break;
		case driver::Field::SyncBoost:
			answer(field, nanoseconds(counters.sync_boost), counter_type);
			break;
		case driver::Field::SwThermalSlowdown:
			answer(field, nanoseconds(counters.sw_thermal_slowdown), counter_type);
			break;
		case driver::Field::HwThermalSlowdown:
			answer(field, nanoseconds(counters.hw_thermal_slowdown), counter_type);
			break;
		case driver::Field::HwPowerBrakeSlowdown:
			answer(field, nanoseconds(counters.hw_power_brake_slowdown), counter_type);
			break;
		case driver::Field::NvlinkSpeedLink0:
			break;
		}
	}
	return ManagementStatus::Success;
}

}

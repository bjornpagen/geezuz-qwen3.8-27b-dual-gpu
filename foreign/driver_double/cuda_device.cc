// SPDX-License-Identifier: AGPL-3.0-only
//
// The fake driver's initialization, card queries, primary contexts and peer access.
export module engine.driver_double:cuda_device;

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
using driver::Status;

auto fake_init(unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (flags != 0) {
		return Status::InvalidValue;
	}
	auto& machine = state();
	machine.initialized = machine.machine.init == Status::Success;
	return machine.machine.init;
}

auto fake_driver_get_version(int* version) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	*version = state().machine.driver_version;
	return Status::Success;
}

auto fake_device_get_count(int* count) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!state().initialized) {
		return Status::NotInitialized;
	}
	*count = core::narrow<int>(state().machine.cards.size());
	return Status::Success;
}

auto fake_device_get(CUdevice* device, int ordinal) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!card_exists(ordinal)) {
		return Status::InvalidDevice;
	}
	*device = ordinal;
	return Status::Success;
}

auto fake_device_get_attribute(int* value, driver::Attribute attribute, CUdevice device) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!card_exists(device)) {
		return Status::InvalidDevice;
	}
	auto const& model = card(device);
	auto const pci = pci_of(device);
	switch (attribute) {
	case driver::Attribute::MultiprocessorCount:
		*value = model.sm_count;
		return Status::Success;
	case driver::Attribute::PciBusId:
		*value = core::narrow<int>(pci.bus);
		return Status::Success;
	case driver::Attribute::PciDeviceId:
		*value = core::narrow<int>(pci.device);
		return Status::Success;
	case driver::Attribute::PciDomainId:
		*value = core::narrow<int>(pci.domain);
		return Status::Success;
	case driver::Attribute::ComputeCapabilityMajor:
		*value = model.compute_major;
		return Status::Success;
	case driver::Attribute::ComputeCapabilityMinor:
		*value = model.compute_minor;
		return Status::Success;
	}
	return Status::InvalidValue;
}

auto fake_device_total_mem(std::size_t* bytes, CUdevice device) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!card_exists(device)) {
		return Status::InvalidDevice;
	}
	*bytes = card(device).memory_bytes;
	return Status::Success;
}

auto fake_device_can_access_peer(int* can, CUdevice from, CUdevice to) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!card_exists(from) || !card_exists(to) || from == to) {
		return Status::InvalidDevice;
	}
	*can = card(from).reaches_peer ? 1 : 0;
	return Status::Success;
}

auto fake_primary_ctx_retain(CUcontext* context, CUdevice device) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!card_exists(device)) {
		return Status::InvalidDevice;
	}
	auto& record = state().contexts[static_cast<std::size_t>(device)];
	++record.retains;
	*context = reinterpret_cast<CUcontext>(&record);
	return Status::Success;
}

auto fake_primary_ctx_release(CUdevice device) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (!card_exists(device)) {
		return Status::InvalidDevice;
	}
	auto& record = state().contexts[static_cast<std::size_t>(device)];
	if (record.retains == 0) {
		violate(Violation::DoubleRelease);
		return Status::InvalidContext;
	}
	if (--record.retains == 0) {
		destroy_context(device);
	}
	return Status::Success;
}

auto fake_ctx_set_current(CUcontext context) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto& machine = state();
	if (context == nullptr) {
		machine.current.reset();
		return Status::Success;
	}
	for (auto index = std::size_t{0}; index < machine.contexts.size(); ++index) {
		if (reinterpret_cast<CUcontext>(&machine.contexts[index]) == context) {
			if (machine.contexts[index].retains == 0) {
				violate(Violation::DestroyedCurrentContext);
				return Status::InvalidContext;
			}
			machine.current = core::narrow<CUdevice>(index);
			return Status::Success;
		}
	}
	violate(Violation::UnknownHandle);
	return Status::InvalidContext;
}

auto fake_ctx_enable_peer_access(CUcontext peer_context, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const from = current();
	if (!from) {
		return Status::InvalidContext;
	}
	auto& machine = state();
	auto const found = std::ranges::find_if(machine.contexts, [&](ContextRecord& record) {
		return reinterpret_cast<CUcontext>(&record) == peer_context;
	});
	if (found == machine.contexts.end() || found->retains == 0) {
		violate(Violation::UnknownHandle);
		return Status::InvalidContext;
	}
	auto const to = core::narrow<CUdevice>(found - machine.contexts.begin());
	if (flags != 0 || to == *from) {
		return Status::InvalidValue;
	}
	if (!card(*from).reaches_peer) {
		return Status::PeerAccessUnsupported;
	}
	if (!machine.peers.insert({*from, to}).second) {
		return Status::PeerAccessAlreadyEnabled;
	}
	return Status::Success;
}

}

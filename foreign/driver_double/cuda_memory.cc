// SPDX-License-Identifier: AGPL-3.0-only
//
// The fake driver's device and pinned host memory.
export module engine.driver_double:cuda_memory;

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

auto fake_mem_alloc(CUdeviceptr* address, std::size_t bytes) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto& machine = state();
	auto in_use = std::uint64_t{0};
	for (auto const& [start, block] : machine.device_blocks) {
		in_use += block.ordinal == *ordinal && block.life == Life::Live ? block.bytes : 0;
	}
	if (bytes == 0 || bytes > card(*ordinal).memory_bytes - in_use) {
		return bytes == 0 ? Status::InvalidValue : Status::OutOfMemory;
	}
	*address = machine.next_address;
	machine.device_blocks.emplace(*address, DeviceBlock{.ordinal = *ordinal, .bytes = bytes, .life = Life::Live});
	machine.next_address += (bytes + 0xffff) & ~std::uint64_t{0xffff};
	return Status::Success;
}

auto fake_mem_free(CUdeviceptr address) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto& machine = state();
	auto const found = machine.device_blocks.find(address);
	if (found == machine.device_blocks.end()) {
		violate(Violation::UnknownHandle);
		return Status::InvalidValue;
	}
	return release(found->second);
}

auto fake_mem_host_alloc(void** host, std::size_t bytes, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	if (bytes == 0 || flags != driver::host_alloc_device_map) {
		return Status::InvalidValue;
	}
	auto pages = std::make_unique<Page[]>((bytes + sizeof(Page) - 1) / sizeof(Page));
	*host = pages.get();
	state().host_blocks.emplace(*host, HostBlock{.ordinal = *ordinal, .pages = std::move(pages), .life = Life::Live});
	return Status::Success;
}

auto fake_mem_free_host(void* host) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto& machine = state();
	auto const found = machine.host_blocks.find(host);
	if (found == machine.host_blocks.end()) {
		violate(Violation::UnknownHandle);
		return Status::InvalidValue;
	}
	auto const status = release(found->second);
	if (status == Status::Success) {
		found->second.pages.reset();
	}
	return status;
}

auto fake_mem_host_get_device_pointer(CUdeviceptr* device, void* host, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto& machine = state();
	auto const found = machine.host_blocks.find(host);
	if (flags != 0 || found == machine.host_blocks.end() || found->second.life != Life::Live) {
		return Status::InvalidValue;
	}
	if (found->second.ordinal != *ordinal) {
		violate(Violation::WrongContext);
		return Status::InvalidValue;
	}
	// Unified addressing: the card reaches pinned memory at the host's own address.
	*device = reinterpret_cast<CUdeviceptr>(host);
	return Status::Success;
}

}

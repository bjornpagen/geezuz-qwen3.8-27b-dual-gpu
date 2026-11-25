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
	machine.device_blocks.emplace(*address, DeviceBlock{.ordinal = *ordinal, .bytes = bytes, .life = Life::Live, .contents = {}});
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

/** The live block holding `[address, address + bytes)` on `ordinal`, with its contents materialized. */
[[nodiscard]] auto device_range(CUdeviceptr address, std::size_t bytes, CUdevice ordinal) -> std::optional<std::span<std::byte>> {
	auto& blocks = state().device_blocks;
	auto found = blocks.upper_bound(address);
	if (found == blocks.begin()) {
		violate(Violation::OutOfBounds);
		return std::nullopt;
	}
	--found;
	auto& [start, block] = *found;
	auto const offset = address - start;
	if (block.life != Life::Live || block.ordinal != ordinal || offset > block.bytes || block.bytes - offset < bytes) {
		violate(block.ordinal != ordinal ? Violation::WrongContext : Violation::OutOfBounds);
		return std::nullopt;
	}
	block.contents.resize(block.bytes);
	return std::span{block.contents}.subspan(core::narrow<std::size_t>(offset), bytes);
}

/** Whether `[host, host + bytes)` lies inside one live registration: the copies the engine makes are all pinned. */
[[nodiscard]] auto pinned_range(void const* host, std::size_t bytes) -> bool {
	auto const& registrations = state().registrations;
	auto const address = std::bit_cast<std::uintptr_t>(host);
	for (auto const& [base, registration] : registrations) {
		auto const start = std::bit_cast<std::uintptr_t>(base);
		if (registration.life == Life::Live && address >= start && address - start <= registration.bytes &&
		    registration.bytes - (address - start) >= bytes) {
			return true;
		}
	}
	violate(Violation::Unpinned);
	return false;
}

auto fake_mem_host_register(void* host, std::size_t bytes, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	constexpr auto known = driver::host_register_portable | driver::host_register_device_map | driver::host_register_read_only;
	if (host == nullptr || bytes == 0 || (flags & ~known) != 0) {
		return Status::InvalidValue;
	}
	auto& registrations = state().registrations;
	auto const start = std::bit_cast<std::uintptr_t>(host);
	auto const overlaps = std::ranges::any_of(registrations, [&](auto const& entry) {
		auto const other = std::bit_cast<std::uintptr_t>(entry.first);
		return entry.second.life == Life::Live && start < other + entry.second.bytes && other < start + bytes;
	});
	if (overlaps) {
		return Status::HostMemoryAlreadyRegistered;
	}
	registrations.insert_or_assign(host, Registration{.ordinal = *ordinal, .bytes = bytes, .flags = flags, .life = Life::Live});
	return Status::Success;
}

auto fake_mem_host_unregister(void* host) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto& registrations = state().registrations;
	auto const found = registrations.find(host);
	if (found == registrations.end()) {
		violate(Violation::UnknownHandle);
		return Status::HostMemoryNotRegistered;
	}
	return release(found->second);
}

auto fake_mem_host_get_device_pointer(CUdeviceptr* device, void* host, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto& registrations = state().registrations;
	auto const found = registrations.find(host);
	if (flags != 0 || found == registrations.end() || found->second.life != Life::Live ||
	    (found->second.flags & driver::host_register_device_map) == 0) {
		return Status::InvalidValue;
	}
	if (found->second.ordinal != *ordinal && (found->second.flags & driver::host_register_portable) == 0) {
		violate(Violation::WrongContext);
		return Status::InvalidValue;
	}
	// Unified addressing: the card reaches pinned memory at the host's own address.
	*device = std::bit_cast<CUdeviceptr>(host);
	return Status::Success;
}

/** The ordinal of a live stream that belongs to the current context. */
[[nodiscard]] auto stream_ordinal(CUstream stream) -> std::optional<CUdevice> {
	auto const ordinal = current();
	auto const* const queue = find_object(stream, Kind::Stream);
	if (!ordinal || !live_object(queue)) {
		return std::nullopt;
	}
	if (queue->ordinal != *ordinal) {
		violate(Violation::WrongContext);
		return std::nullopt;
	}
	return ordinal;
}

// The fake's copies happen when enqueued: the fake runs no work, so ordering is moot and completion is the events'.

auto fake_memcpy_htod_async(CUdeviceptr destination, void const* source, std::size_t bytes, CUstream stream) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = stream_ordinal(stream);
	if (!ordinal) {
		return Status::InvalidHandle;
	}
	auto const to = device_range(destination, bytes, *ordinal);
	if (!to || !pinned_range(source, bytes)) {
		return Status::InvalidValue;
	}
	std::ranges::copy(std::span{static_cast<std::byte const*>(source), bytes}, to->begin());
	advance_clock();
	return Status::Success;
}

auto fake_memcpy_dtoh_async(void* destination, CUdeviceptr source, std::size_t bytes, CUstream stream) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = stream_ordinal(stream);
	if (!ordinal) {
		return Status::InvalidHandle;
	}
	auto const from = device_range(source, bytes, *ordinal);
	if (!from || !pinned_range(destination, bytes)) {
		return Status::InvalidValue;
	}
	std::ranges::copy(*from, static_cast<std::byte*>(destination));
	advance_clock();
	return Status::Success;
}

[[nodiscard]] auto context_ordinal(CUcontext context) -> std::optional<CUdevice> {
	auto& contexts = state().contexts;
	for (auto index = std::size_t{0}; index < contexts.size(); ++index) {
		if (reinterpret_cast<CUcontext>(&contexts[index]) == context && contexts[index].retains > 0) {
			return core::narrow<CUdevice>(index);
		}
	}
	violate(Violation::UnknownHandle);
	return std::nullopt;
}

auto fake_memcpy_peer_async(CUdeviceptr destination, CUcontext destination_context, CUdeviceptr source, CUcontext source_context,
                            std::size_t bytes, CUstream stream) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = stream_ordinal(stream);
	auto const to_card = context_ordinal(destination_context);
	auto const from_card = context_ordinal(source_context);
	if (!ordinal || !to_card || !from_card) {
		return Status::InvalidHandle;
	}
	auto const to = device_range(destination, bytes, *to_card);
	auto const from = device_range(source, bytes, *from_card);
	if (!to || !from) {
		return Status::InvalidValue;
	}
	std::ranges::copy(*from, to->begin());
	advance_clock();
	return Status::Success;
}

}

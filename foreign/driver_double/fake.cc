// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver_double:fake;

import std;
import engine.core;
import engine.driver;
import :machine;
import :ledger;
import :cuda_device;
import :cuda_memory;
import :cuda_work;
import :nvml;

namespace engine::driver_double {

using driver::CUcontext;
using driver::CUdevice;
using driver::CUdeviceptr;
using driver::CUevent;
using driver::CUfunction;
using driver::CUmodule;
using driver::CUstream;

/**
 * The fake driver, installed for this owner's lifetime: at most one exists at a time. Tests build the engine over
 * `driver()` exactly as the product builds it over the loaded libraries, then read back what the driver saw.
 */
export class Fake {
public:
	explicit Fake(Machine machine) : state_{std::make_unique<State>()} {
		core::invariant(installed == nullptr);
		state_->contexts.assign(machine.cards.size(), ContextRecord{.retains = 0});
		for (auto ordinal = 0; std::cmp_less(ordinal, machine.cards.size()); ++ordinal) {
			state_->nvml_cards.push_back(NvmlCard{.ordinal = ordinal});
		}
		state_->machine = std::move(machine);
		installed = state_.get();
	}

	Fake(Fake const&) = delete;
	Fake(Fake&&) = delete;
	auto operator=(Fake const&) -> Fake& = delete;
	auto operator=(Fake&&) -> Fake& = delete;

	~Fake() {
		installed = nullptr;
	}

	[[nodiscard]] auto driver() const -> driver::Driver {
		return driver::Driver{
		    .cuda =
		        {
		            .init = {.function = fake_init},
		            .driver_get_version = {.function = fake_driver_get_version},
		            .device_get_count = {.function = fake_device_get_count},
		            .device_get = {.function = fake_device_get},
		            .device_get_attribute = {.function = fake_device_get_attribute},
		            .device_total_mem = {.function = fake_device_total_mem},
		            .device_can_access_peer = {.function = fake_device_can_access_peer},
		            .primary_ctx_retain = {.function = fake_primary_ctx_retain},
		            .primary_ctx_release = {.function = fake_primary_ctx_release},
		            .ctx_set_current = {.function = fake_ctx_set_current},
		            .ctx_enable_peer_access = {.function = fake_ctx_enable_peer_access},
		            .mem_alloc = {.function = fake_mem_alloc},
		            .mem_free = {.function = fake_mem_free},
		            .mem_host_alloc = {.function = fake_mem_host_alloc},
		            .mem_free_host = {.function = fake_mem_free_host},
		            .mem_host_get_device_pointer = {.function = fake_mem_host_get_device_pointer},
		            .stream_create = {.function = fake_stream_create},
		            .stream_destroy = {.function = fake_stream_destroy},
		            .stream_wait_event = {.function = fake_stream_wait_event},
		            .event_create = {.function = fake_event_create},
		            .event_destroy = {.function = fake_event_destroy},
		            .event_record = {.function = fake_event_record},
		            .event_query = {.function = fake_event_query},
		            .module_load_data_ex = {.function = fake_module_load_data_ex},
		            .module_unload = {.function = fake_module_unload},
		            .module_get_function = {.function = fake_module_get_function},
		            .func_get_param_info = {.function = fake_func_get_param_info},
		            .func_set_attribute = {.function = fake_func_set_attribute},
		            .launch_kernel = {.function = fake_launch_kernel},
		            .launch_cooperative_kernel = {.function = fake_launch_cooperative_kernel},
		        },
		    .management =
		        {
		            .init = {.function = fake_nvml_init},
		            .shutdown = {.function = fake_nvml_shutdown},
		            .device_by_pci_bus_id = {.function = fake_nvml_device_by_pci_bus_id},
		            .nvlink_state = {.function = fake_nvml_nvlink_state},
		            .nvlink_remote_pci_info = {.function = fake_nvml_nvlink_remote_pci_info},
		            .clock_info = {.function = fake_nvml_clock_info},
		            .performance_state = {.function = fake_nvml_performance_state},
		        },
		};
	}

	/** How many vendor calls the engine has made so far. */
	[[nodiscard]] auto calls() const -> std::size_t {
		return state_->calls;
	}

	/** How many resources of a kind are alive: retained contexts count once per card. */
	[[nodiscard]] auto live(Kind kind) const -> std::size_t {
		auto const is_live = [](auto const& record) {
			return record.life == Life::Live;
		};
		switch (kind) {
		case Kind::Context:
			return core::narrow<std::size_t>(std::ranges::count_if(state_->contexts, [](ContextRecord const& record) {
				return record.retains > 0;
			}));
		case Kind::DeviceMemory:
			return core::narrow<std::size_t>(std::ranges::count_if(state_->device_blocks | std::views::values, is_live));
		case Kind::HostMemory:
			return core::narrow<std::size_t>(std::ranges::count_if(state_->host_blocks | std::views::values, is_live));
		case Kind::Stream:
		case Kind::Event:
			return core::narrow<std::size_t>(std::ranges::count_if(state_->objects, [&](Object const& object) {
				return object.kind == kind && object.life == Life::Live;
			}));
		case Kind::Module:
			return core::narrow<std::size_t>(std::ranges::count_if(state_->modules, is_live));
		case Kind::ManagementSession:
			return core::narrow<std::size_t>(state_->sessions);
		}
		return 0;
	}

	/** Every resource kind's live count is zero: nothing the engine acquired is left. */
	[[nodiscard]] auto nothing_live() const -> bool {
		constexpr auto kinds = std::array{Kind::Context, Kind::DeviceMemory, Kind::HostMemory,       Kind::Stream,
		                                  Kind::Event,   Kind::Module,       Kind::ManagementSession};
		return std::ranges::all_of(kinds, [&](Kind kind) {
			return live(kind) == 0;
		});
	}

	[[nodiscard]] auto violations() const -> std::span<Violation const> {
		return state_->violations;
	}

	[[nodiscard]] auto launches() const -> std::span<Launch const> {
		return state_->launches;
	}

	/** Whether `from` has `to`'s memory mapped. */
	[[nodiscard]] auto peer_access(CUdevice from, CUdevice to) const -> bool {
		return state_->peers.contains({from, to});
	}

private:
	std::unique_ptr<State> state_;
};

}

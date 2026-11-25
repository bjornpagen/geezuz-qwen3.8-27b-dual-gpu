// SPDX-License-Identifier: AGPL-3.0-only
//
// The fake driver's streams, events, modules and launches, and the jit's view of a ptx image.
export module engine.driver_double:cuda_work;

import std;
import engine.core;
import engine.driver;
import engine.prims;
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

/** The byte size a ptx `.param` declaration takes: an array of `.b8`, or a scalar. */
[[nodiscard]] auto parameter_size(std::string_view declaration) -> std::size_t {
	if (auto const open = declaration.find('['); open != std::string_view::npos) {
		auto size = std::size_t{0};
		auto const digits = declaration.substr(open + 1);
		std::ignore = std::from_chars(digits.data(), digits.data() + digits.size(), size);
		return size;
	}
	auto const wide = declaration.find("64") != std::string_view::npos;
	return wide ? 8 : 4;
}

struct EntryModel {
	std::string name;
	std::vector<std::size_t> parameters;
};

/** The jit's view of a ptx image: every `.entry`, with the sizes of the parameters it declares. */
[[nodiscard]] auto entries_of(std::string_view text) -> std::vector<EntryModel> {
	auto entries = std::vector<EntryModel>{};
	constexpr auto keyword = std::string_view{".entry "};
	for (auto at = text.find(keyword); at != std::string_view::npos; at = text.find(keyword)) {
		text.remove_prefix(at + keyword.size());
		auto const open = text.find('(');
		auto const close = text.find(')');
		if (open == std::string_view::npos || close == std::string_view::npos || close < open) {
			break;
		}
		auto entry = EntryModel{.name = std::string{text.substr(0, open)}, .parameters = {}};
		auto list = text.substr(open + 1, close - open - 1);
		for (auto param = list.find(".param"); param != std::string_view::npos; param = list.find(".param")) {
			list.remove_prefix(param + 1);
			entry.parameters.push_back(parameter_size(list.substr(0, list.find(','))));
		}
		entries.push_back(std::move(entry));
		text.remove_prefix(close);
	}
	return entries;
}

auto fake_stream_create(CUstream* stream, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	return flags != driver::stream_non_blocking ? Status::InvalidValue : create_object(stream, Kind::Stream);
}

auto fake_stream_destroy(CUstream stream) -> Status {
	return injected() ? Status::Unknown : destroy_object(stream, Kind::Stream);
}

auto fake_stream_wait_event(CUstream stream, CUevent event, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const* const waiting = find_object(stream, Kind::Stream);
	auto const* const awaited = find_object(event, Kind::Event);
	return flags != 0 || !live_object(waiting) || !live_object(awaited) ? Status::InvalidHandle : Status::Success;
}

auto fake_event_create(CUevent* event, unsigned int flags) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	return flags != driver::event_disable_timing ? Status::InvalidValue : create_object(event, Kind::Event);
}

auto fake_event_destroy(CUevent event) -> Status {
	return injected() ? Status::Unknown : destroy_object(event, Kind::Event);
}

auto fake_event_record(CUevent event, CUstream stream) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const* const marker = find_object(event, Kind::Event);
	auto const* const queue = find_object(stream, Kind::Stream);
	if (!live_object(marker) || !live_object(queue)) {
		return Status::InvalidHandle;
	}
	if (marker->ordinal != queue->ordinal) {
		violate(Violation::WrongContext);
		return Status::InvalidHandle;
	}
	return Status::Success;
}

auto fake_event_query(CUevent event) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	// The fake runs no work, so whatever an event was recorded after has finished.
	return live_object(find_object(event, Kind::Event)) ? Status::Success : Status::InvalidHandle;
}

auto fake_module_load_data_ex(CUmodule* module, void const* image, unsigned int count, driver::JitOption* options, void** values)
    -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto& machine = state();
	if (auto const* const rejects = std::get_if<JitRejects>(&machine.machine.jit)) {
		auto log = std::span<char>{};
		auto log_slot = std::optional<std::size_t>{};
		for (auto index = std::size_t{0}; index < count; ++index) {
			if (options[index] == driver::JitOption::ErrorLogBuffer) {
				log = std::span{static_cast<char*>(values[index]), log.size()};
			} else if (options[index] == driver::JitOption::ErrorLogBufferSizeBytes) {
				log = std::span{log.data(), std::bit_cast<std::uintptr_t>(values[index])};
				log_slot = index;
			}
		}
		if (log_slot && !log.empty()) {
			auto const written = std::min(rejects->log.size(), log.size() - 1);
			std::ranges::copy(std::string_view{rejects->log}.substr(0, written), log.begin());
			log[written] = '\0';
			values[*log_slot] = std::bit_cast<void*>(std::uintptr_t{written + 1});
		}
		return rejects->status;
	}
	auto& record = machine.modules.emplace_back(ModuleRecord{.ordinal = *ordinal, .life = Life::Live});
	for (auto& entry : entries_of(std::string_view{static_cast<char const*>(image)})) {
		machine.functions.push_back(FunctionRecord{
		    .module = &record,
		    .name = std::move(entry.name),
		    .parameters = std::move(entry.parameters),
		    .dynamic_shared = 0,
		});
	}
	*module = reinterpret_cast<CUmodule>(&record);
	return Status::Success;
}

auto fake_module_unload(CUmodule module) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto* const record = find_module(module);
	return record == nullptr ? Status::InvalidHandle : release(*record);
}

auto fake_module_get_function(CUfunction* function, CUmodule module, char const* name) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto* const record = find_module(module);
	if (record == nullptr || record->life != Life::Live) {
		return Status::InvalidHandle;
	}
	for (auto& candidate : state().functions) {
		if (candidate.module == record && candidate.name == name) {
			*function = reinterpret_cast<CUfunction>(&candidate);
			return Status::Success;
		}
	}
	return Status::NotFound;
}

auto fake_func_get_param_info(CUfunction function, std::size_t index, std::size_t* offset, std::size_t* size) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto const* const record = live_function(function);
	if (record == nullptr) {
		return Status::InvalidHandle;
	}
	if (index >= record->parameters.size()) {
		return Status::InvalidValue;
	}
	auto const before = std::span{record->parameters}.first(index);
	*offset = std::reduce(before.begin(), before.end(), std::size_t{0});
	*size = record->parameters[index];
	return Status::Success;
}

auto fake_func_set_attribute(CUfunction function, driver::FunctionAttribute attribute, int value) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	auto* const record = live_function(function);
	if (record == nullptr) {
		return Status::InvalidHandle;
	}
	if (attribute != driver::FunctionAttribute::MaxDynamicSharedSizeBytes || value < 0 ||
	    std::cmp_greater(value, device::prims::sm86.max_shared_per_block)) {
		return Status::InvalidValue;
	}
	record->dynamic_shared = value;
	return Status::Success;
}

/** The co-residency limit the fake applies: the threads one sm_86 SM holds. */
[[nodiscard]] auto co_resident_blocks(CUdevice ordinal, std::uint64_t block_threads) -> std::uint64_t {
	return core::narrow<std::uint64_t>(card(ordinal).sm_count) *
	       (core::narrow<std::uint64_t>(device::prims::sm86.max_threads_per_sm) / block_threads);
}

[[nodiscard]] auto launch(CUfunction function, driver::Dim grid, driver::Dim block, unsigned int shared, CUstream stream, void** parameters,
                          driver::Residency residency) -> Status {
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto const* const record = live_function(function);
	auto const* const queue = find_object(stream, Kind::Stream);
	if (record == nullptr || !live_object(queue)) {
		return Status::InvalidHandle;
	}
	if (record->module->ordinal != *ordinal || queue->ordinal != *ordinal) {
		violate(Violation::WrongContext);
		return Status::InvalidHandle;
	}
	auto const threads = block.count();
	constexpr auto default_shared = 48U * 1024;
	if (grid.count() == 0 || threads == 0 || std::cmp_greater(threads, device::prims::sm86.max_threads_per_block) ||
	    shared > std::max(default_shared, static_cast<unsigned int>(record->dynamic_shared))) {
		return Status::InvalidValue;
	}
	if (residency == driver::Residency::CoResident && grid.count() > co_resident_blocks(*ordinal, threads)) {
		return Status::CooperativeLaunchTooLarge;
	}
	auto bytes = std::vector<std::byte>{};
	for (auto index = std::size_t{0}; index < record->parameters.size(); ++index) {
		auto const* const first = static_cast<std::byte const*>(parameters[index]);
		bytes.insert(bytes.end(), first, first + record->parameters[index]);
	}
	state().launches.push_back(Launch{
	    .ordinal = *ordinal,
	    .entry = record->name,
	    .grid = grid,
	    .block = block,
	    .shared_bytes = shared,
	    .residency = residency,
	    .parameters = std::move(bytes),
	});
	return Status::Success;
}

auto fake_launch_kernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z, unsigned int block_x,
                        unsigned int block_y, unsigned int block_z, unsigned int shared, CUstream stream, void** parameters, void** extra)
    -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	if (extra != nullptr) {
		return Status::InvalidValue;
	}
	return launch(function, {.x = grid_x, .y = grid_y, .z = grid_z}, {.x = block_x, .y = block_y, .z = block_z}, shared, stream, parameters,
	              driver::Residency::Ordinary);
}

auto fake_launch_cooperative_kernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                                    unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared, CUstream stream,
                                    void** parameters) -> Status {
	if (injected()) {
		return Status::Unknown;
	}
	return launch(function, {.x = grid_x, .y = grid_y, .z = grid_z}, {.x = block_x, .y = block_y, .z = block_z}, shared, stream, parameters,
	              driver::Residency::CoResident);
}

}

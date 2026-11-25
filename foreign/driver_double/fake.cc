// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver_double:fake;

import std;
import engine.core;
import engine.driver;
import engine.prims;
import :machine;

// A test double of the CUDA driver and NVML: plain functions with the vendors' C signatures, filled into the same
// tables the loader fills in the product. It models the rules a real driver enforces (contexts, current context,
// resource lifetimes, peer access, the jit's view of a ptx image, launch limits) so that tests can check the engine
// against them, and it records every misuse a real driver would punish, sometimes silently, as a violation.
//
// C function pointers carry no closure, so the functions reach the one modelled machine through a single installed
// state slot that a `Fake` owns for its lifetime. Driver calls through the fake come from one thread.

namespace engine::driver_double {

using driver::CUcontext;
using driver::CUdevice;
using driver::CUdeviceptr;
using driver::CUevent;
using driver::CUfunction;
using driver::CUmodule;
using driver::CUstream;
using driver::ManagementStatus;
using driver::Status;

enum class Life : std::uint8_t {
	Live,
	Released,
	Orphaned,
};

struct alignas(4096) Page {
	std::array<std::byte, 4096> bytes;
};

struct ContextRecord {
	int retains;
};

struct DeviceBlock {
	CUdevice ordinal;
	std::size_t bytes;
	Life life;
};

struct HostBlock {
	CUdevice ordinal;
	std::unique_ptr<Page[]> pages;
	Life life;
};

struct Object {
	Kind kind;
	CUdevice ordinal;
	Life life;
};

struct ModuleRecord {
	CUdevice ordinal;
	Life life;
};

struct FunctionRecord {
	ModuleRecord* module;
	std::string name;
	std::vector<std::size_t> parameters;
	int dynamic_shared;
};

struct NvmlCard {
	int ordinal;
};

struct State {
	Machine machine;
	std::vector<ContextRecord> contexts;
	std::vector<NvmlCard> nvml_cards;
	bool initialized = false;
	std::size_t calls = 0;
	std::optional<CUdevice> current;
	std::map<CUdeviceptr, DeviceBlock> device_blocks;
	CUdeviceptr next_address = 0x7f00'0000'0000;
	std::map<void const*, HostBlock> host_blocks;
	std::deque<Object> objects;
	std::deque<ModuleRecord> modules;
	std::deque<FunctionRecord> functions;
	std::set<std::pair<CUdevice, CUdevice>> peers;
	int sessions = 0;
	std::vector<Violation> violations;
	std::vector<Launch> launches;
};

State* installed = nullptr;

[[nodiscard]] auto state() -> State& {
	core::invariant(installed != nullptr);
	return *installed;
}

/** Counts a vendor call; true when it is the one the machine says fails. */
[[nodiscard]] auto injected() -> bool {
	auto& machine = state();
	auto const index = machine.calls++;
	return machine.machine.failing_call == index;
}

auto violate(Violation violation) -> void {
	state().violations.push_back(violation);
}

[[nodiscard]] auto card_exists(CUdevice ordinal) -> bool {
	return ordinal >= 0 && std::cmp_less(ordinal, state().machine.cards.size());
}

[[nodiscard]] auto card(CUdevice ordinal) -> CardModel const& {
	return state().machine.cards[static_cast<std::size_t>(ordinal)];
}

/** The calling thread's current context, when it names a live one. */
[[nodiscard]] auto current() -> std::optional<CUdevice> {
	auto& machine = state();
	if (!machine.current) {
		violate(Violation::NoCurrentContext);
		return std::nullopt;
	}
	if (machine.contexts[static_cast<std::size_t>(*machine.current)].retains == 0) {
		violate(Violation::DestroyedCurrentContext);
		return std::nullopt;
	}
	return machine.current;
}

template<class Record>
[[nodiscard]] auto release(Record& record) -> Status {
	switch (record.life) {
	case Life::Live:
		record.life = Life::Released;
		return Status::Success;
	case Life::Released:
		violate(Violation::DoubleRelease);
		return Status::InvalidValue;
	case Life::Orphaned:
		violate(Violation::ReleasedAfterContext);
		return Status::ContextIsDestroyed;
	}
	return Status::Unknown;
}

template<class Record>
auto orphan(Record& record, CUdevice ordinal) -> bool {
	if (record.ordinal != ordinal || record.life != Life::Live) {
		return false;
	}
	violate(Violation::OutlivedContext);
	record.life = Life::Orphaned;
	return true;
}

/** What the real driver does when a primary context's last retain goes: destroy everything still in it. */
auto destroy_context(CUdevice ordinal) -> void {
	auto& machine = state();
	for (auto& [address, block] : machine.device_blocks) {
		std::ignore = orphan(block, ordinal);
	}
	for (auto& [address, block] : machine.host_blocks) {
		if (orphan(block, ordinal)) {
			block.pages.reset();
		}
	}
	for (auto& object : machine.objects) {
		std::ignore = orphan(object, ordinal);
	}
	for (auto& module : machine.modules) {
		std::ignore = orphan(module, ordinal);
	}
	std::erase_if(machine.peers, [&](auto const& pair) {
		return pair.first == ordinal || pair.second == ordinal;
	});
}

template<class Handle>
[[nodiscard]] auto find_object(Handle handle, Kind kind) -> Object* {
	for (auto& object : state().objects) {
		/* SAFETY: the fake's handles are the addresses of its own records, compared, never dereferenced as handles. */
		if (reinterpret_cast<Handle>(&object) == handle && object.kind == kind) {
			return &object;
		}
	}
	violate(Violation::UnknownHandle);
	return nullptr;
}

[[nodiscard]] auto find_module(CUmodule handle) -> ModuleRecord* {
	for (auto& module : state().modules) {
		if (reinterpret_cast<CUmodule>(&module) == handle) {
			return &module;
		}
	}
	violate(Violation::UnknownHandle);
	return nullptr;
}

[[nodiscard]] auto find_function(CUfunction handle) -> FunctionRecord* {
	for (auto& function : state().functions) {
		if (reinterpret_cast<CUfunction>(&function) == handle) {
			return &function;
		}
	}
	violate(Violation::UnknownHandle);
	return nullptr;
}

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

template<class Handle>
[[nodiscard]] auto create_object(Handle* handle, Kind kind) -> Status {
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto& object = state().objects.emplace_back(Object{.kind = kind, .ordinal = *ordinal, .life = Life::Live});
	*handle = reinterpret_cast<Handle>(&object);
	return Status::Success;
}

template<class Handle>
[[nodiscard]] auto destroy_object(Handle handle, Kind kind) -> Status {
	auto* const object = find_object(handle, kind);
	return object == nullptr ? Status::InvalidHandle : release(*object);
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

[[nodiscard]] auto live_object(Object const* object) -> bool {
	return object != nullptr && object->life == Life::Live;
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

[[nodiscard]] auto live_function(CUfunction function) -> FunctionRecord* {
	auto* const record = find_function(function);
	return record != nullptr && record->module->life == Life::Live ? record : nullptr;
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

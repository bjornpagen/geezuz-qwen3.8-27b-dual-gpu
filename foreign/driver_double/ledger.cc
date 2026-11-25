// SPDX-License-Identifier: AGPL-3.0-only
//
// A test double of the CUDA driver and NVML: plain functions with the vendors' C signatures, filled into the same
// tables the loader fills in the product. It models the rules a real driver enforces (contexts, current context,
// resource lifetimes, peer access, the jit's view of a ptx image, launch limits) so that tests can check the engine
// against them, and it records every misuse a real driver would punish, sometimes silently, as a violation.
//
// C function pointers carry no closure, so the functions reach the one modelled machine through a single installed
// state slot that a `Fake` owns for its lifetime. Driver calls through the fake come from one thread.
export module engine.driver_double:ledger;

import std;
import engine.core;
import engine.driver;
import engine.prims;
import :machine;

namespace engine::driver_double {

using driver::CUcontext;
using driver::CUdevice;
using driver::CUdeviceptr;
using driver::CUevent;
using driver::CUfunction;
using driver::CUmodule;
using driver::CUstream;
using driver::Status;

enum class Life : std::uint8_t {
	Live,
	Released,
	Orphaned,
};

struct ContextRecord {
	int retains;
};

/** A device allocation; its contents exist only once a copy touches it, so planning-sized blocks cost nothing. */
struct DeviceBlock {
	CUdevice ordinal;
	std::size_t bytes;
	Life life;
	std::vector<std::byte> contents;
};

/** A pinned host range: the memory is the caller's, the fake only records the pin. */
struct Registration {
	CUdevice ordinal;
	std::size_t bytes;
	unsigned int flags;
	Life life;
};

/**
 * A stream or an event. A recorded event answers "not ready" to its next `pending` queries, and keeps the fake clock's
 * time when it was recorded unless it was created without timing.
 */
struct Object {
	Kind kind;
	CUdevice ordinal;
	Life life;
	std::size_t pending;
	unsigned int flags;
	std::optional<float> stamp;
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
	std::map<void const*, Registration> registrations;
	std::deque<Object> objects;
	std::deque<ModuleRecord> modules;
	std::deque<FunctionRecord> functions;
	std::set<std::pair<CUdevice, CUdevice>> peers;
	int sessions = 0;
	std::vector<Violation> violations;
	std::vector<Launch> launches;
	float clock = 0.0F;
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
	for (auto& [address, registration] : machine.registrations) {
		std::ignore = orphan(registration, ordinal);
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

/** The fake runs work when it is enqueued; each copy or launch moves its clock on by the machine's step. */
auto advance_clock() -> void {
	state().clock += state().machine.milliseconds_per_operation;
}

template<class Handle>
[[nodiscard]] auto create_object(Handle* handle, Kind kind, unsigned int flags) -> Status {
	auto const ordinal = current();
	if (!ordinal) {
		return Status::InvalidContext;
	}
	auto& object = state().objects.emplace_back(
	    Object{.kind = kind, .ordinal = *ordinal, .life = Life::Live, .pending = 0, .flags = flags, .stamp = std::nullopt});
	*handle = reinterpret_cast<Handle>(&object);
	return Status::Success;
}

template<class Handle>
[[nodiscard]] auto destroy_object(Handle handle, Kind kind) -> Status {
	auto* const object = find_object(handle, kind);
	return object == nullptr ? Status::InvalidHandle : release(*object);
}

[[nodiscard]] auto live_object(Object const* object) -> bool {
	return object != nullptr && object->life == Life::Live;
}

[[nodiscard]] auto live_function(CUfunction function) -> FunctionRecord* {
	auto* const record = find_function(function);
	return record != nullptr && record->module->life == Life::Live ? record : nullptr;
}

}

// SPDX-License-Identifier: AGPL-3.0-only
module;

// The dynamic loader: the engine has no link-time dependency on either vendor library.
#include <dlfcn.h>

export module engine.driver:load;

import std;
import engine.core;
import :abi;
import :error;

namespace engine::driver {

template<Symbol Name, class Signature>
[[nodiscard]] auto bind(void* library, Entry<Name, Signature>& entry) -> bool {
	auto* const address = dlsym(library, Name.text.data());
	if (address == nullptr) {
		return false;
	}
	/* SAFETY: the vendor exports this versioned symbol with exactly the signature its entry declares. */
	entry.function = reinterpret_cast<decltype(entry.function)>(address);
	return true;
}

/** Resolves entries in order, stopping at the first symbol the library lacks; that symbol, if any. */
template<class... Entries>
[[nodiscard]] auto resolve_entries(void* library, Entries&... entries) -> std::optional<std::string_view> {
	auto missing = std::optional<std::string_view>{};
	auto const found = [&](auto& entry) {
		if (bind(library, entry)) {
			return true;
		}
		missing = entry.symbol;
		return false;
	};
	std::ignore = (found(entries) && ...);
	return missing;
}

/** Resolves every entry of a table, each of its members in declaration order. */
template<class Table>
[[nodiscard]] auto resolve(void* library, Table& table) -> std::optional<std::string_view> {
	auto& [... entries] = table;
	return resolve_entries(library, entries...);
}

template<class Table>
[[nodiscard]] auto load_library(Library library, char const* soname, Table& table) -> core::Result<void, DeviceError> {
	// Neither library is ever closed: its entry points live on as plain values in tables for the life of the process,
	// and the CUDA driver does not support being unloaded under contexts it created.
	auto* const handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
	if (handle == nullptr) {
		return std::unexpected(DeviceError{
		    .reason = DriverUnavailable{.library = library, .stage = LoadStage::Open, .name = soname},
		});
	}
	if (auto const missing = resolve(handle, table)) {
		return std::unexpected(DeviceError{
		    .reason = DriverUnavailable{.library = library, .stage = LoadStage::EntryPoint, .name = *missing},
		});
	}
	return {};
}

/**
 * The product's one way to obtain the vendor tables: `libcuda.so.1` and `libnvidia-ml.so.1`, opened at run time. On
 * macOS neither exists and this fails with `DriverUnavailable`.
 */
export [[nodiscard]] auto load() -> core::Result<Driver, DeviceError> {
	auto driver = Driver{};
	return load_library(Library::Cuda, "libcuda.so.1", driver.cuda)
	    .and_then([&] {
		    return load_library(Library::Management, "libnvidia-ml.so.1", driver.management);
	    })
	    .transform([&] {
		    return driver;
	    });
}

}

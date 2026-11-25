// SPDX-License-Identifier: AGPL-3.0-only
//
// The one launch every microbenchmark kernel goes through: a kernel declaration says, in its `params`, which typed
// owners it reads and writes, and its parameter struct is built from exactly those, so a launch cannot pass a buffer
// of the wrong type or size.
export module engine.ubench_device:launch;

import std;
import engine.core;
import engine.driver;

namespace engine::ubench_device {

/** A kernel whose parameter struct is built from typed owners by its declaration. */
export template<class K, class... Arguments>
concept Typed = driver::Kernel<K> && requires(Arguments const&... arguments) {
	{ K::params(arguments...) } -> std::same_as<typename K::Params>;
};

/** Launches `K` from `program` on `stream`, its parameters built from `arguments` by its declaration. */
export template<class K, class... Ks, class... Arguments>
    requires Typed<K, Arguments...>
[[nodiscard]] auto launch(driver::Context const& context, driver::Program<Ks...> const& program, driver::Stream const& stream,
                          Arguments const&... arguments) -> core::Result<void, driver::DeviceError> {
	return context.launch<K>(program, stream, K::params(arguments...));
}

}

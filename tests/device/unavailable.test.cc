// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.testing;

namespace {

namespace device = engine::device;

[[nodiscard]] auto starting_without_libcuda_is_driver_unavailable() -> bool {
	auto const started = device::start();
	return !started &&
	       started.error() == device::DeviceError{.reason =
	                                                  device::DriverUnavailable{
	                                                      .library = device::Library::Cuda,
	                                                      .stage = device::LoadStage::Open,
	                                                      .name = "libcuda.so.1",
	                                                  }} &&
	       started.error().retry() == engine::core::Retry::Permanent;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "starting the engine where libcuda.so.1 does not exist is refused as driver unavailable",
	        .holds = starting_without_libcuda_is_driver_unavailable(),
	    },
	};
	return engine::testing::report(cases);
}

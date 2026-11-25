# SPDX-License-Identifier: AGPL-3.0-only
engine_test(device SOURCES devices.test.cc LINK engine_device engine_driver_double)
engine_test(device.memory SOURCES memory.test.cc LINK engine_device engine_driver_double)
engine_test(device.program SOURCES program.test.cc LINK engine_device engine_driver_double)
engine_test(device.ring SOURCES ring.test.cc LINK engine_device engine_driver_double)

# The one platform difference at the driver boundary: macOS has no libcuda.so.1, so starting the engine there is
# refused as driver unavailable. On linux the same start is the bring-up on the cards themselves.
if(ENGINE_PLATFORM STREQUAL "darwin")
	engine_test(device.unavailable SOURCES unavailable.test.cc LINK engine_device)
endif()

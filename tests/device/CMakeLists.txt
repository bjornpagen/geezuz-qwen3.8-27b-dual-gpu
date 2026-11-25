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

engine_compile_fail(kernel_grid_exceeds_cards SOURCE crowded_kernel.cc LINK engine_device MATCH "a co-resident grid fits the cards at its budget")
engine_test(device.transfer SOURCES transfer.test.cc LINK engine_device engine_driver_double engine_file engine_hugepage)
engine_death_test(device.transfer_pending_aborts SOURCES transfer_pending.test.cc LINK engine_device engine_driver_double)
engine_test(device.timing SOURCES timing.test.cc LINK engine_device engine_driver_double)
engine_compile_fail(untimed_events_have_no_elapsed_time SOURCE untimed_since.cc LINK engine_device MATCH "constraints not satisfied")

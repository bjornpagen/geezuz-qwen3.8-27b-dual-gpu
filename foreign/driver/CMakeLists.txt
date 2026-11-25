# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_driver)
target_sources(engine_driver PUBLIC FILE_SET CXX_MODULES FILES
	driver.cc abi.cc error.cc load.cc probe.cc management.cc memory.cc stream.cc program.cc context.cc ring.cc
)
target_link_libraries(engine_driver PUBLIC engine_language_profile engine_core engine_ptx engine_prims)
# The resident ring's slot is a device header; the ring partition is its one host includer.
target_include_directories(engine_driver PRIVATE "${PROJECT_SOURCE_DIR}/device")
# dlopen lives in libdl on older glibc; the variable is empty where it is part of libc.
target_link_libraries(engine_driver PRIVATE ${CMAKE_DL_LIBS})

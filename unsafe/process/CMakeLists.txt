# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_process)

# The platform seam: how a child's descriptors are kept from it, which darwin and linux spell differently. Both define
# the partition engine.process:spawn with the same surface.
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(spawn_backend spawn.darwin.cc)
else()
	set(spawn_backend spawn.linux.cc)
endif()

target_sources(engine_process PUBLIC FILE_SET CXX_MODULES FILES process.cc error.cc ${spawn_backend} child.cc)
target_link_libraries(engine_process PUBLIC engine_language_profile engine_core)

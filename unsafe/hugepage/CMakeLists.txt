# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_hugepage)

# The platform seam: one page backend per machine, both defining the partition engine.hugepage:pages with the same
# surface. Nothing above it tests the platform.
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(pages_backend pages.darwin.cc)
else()
	set(pages_backend pages.linux.cc)
endif()

target_sources(engine_hugepage PUBLIC FILE_SET CXX_MODULES FILES hugepage.cc error.cc ${pages_backend} region.cc)
target_link_libraries(engine_hugepage PUBLIC engine_language_profile engine_core)

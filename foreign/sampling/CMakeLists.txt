# SPDX-License-Identifier: AGPL-3.0-only
# The device half of sampling: `engine.sampling` (src/sampling/) is the dialect half and owns the component name.
add_library(engine_sampling_device)
target_sources(engine_sampling_device PUBLIC FILE_SET CXX_MODULES FILES sampling_device.cc)
target_link_libraries(engine_sampling_device PUBLIC engine_language_profile engine_core)
# The sampling arithmetic is a device header; this unit is its one host includer.
target_include_directories(engine_sampling_device PRIVATE "${PROJECT_SOURCE_DIR}/device")

# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_sampling)
target_sources(engine_sampling PUBLIC FILE_SET CXX_MODULES FILES sampling.cc device.cc params.cc reference.cc)
target_link_libraries(engine_sampling PUBLIC engine_language_profile engine_core engine_sampling_device)

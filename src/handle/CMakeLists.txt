# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_handle)
target_sources(engine_handle PUBLIC FILE_SET CXX_MODULES FILES handle.cc)
target_link_libraries(engine_handle PUBLIC engine_language_profile engine_core)

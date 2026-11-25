# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_prims)
target_sources(engine_prims PUBLIC FILE_SET CXX_MODULES FILES prims.cc)
target_link_libraries(engine_prims PUBLIC engine_language_profile)
# The host-meaningful device primitives are device headers; this unit is their one host includer.
target_include_directories(engine_prims PRIVATE "${PROJECT_SOURCE_DIR}/device")

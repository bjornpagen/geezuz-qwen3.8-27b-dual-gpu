# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_schedule)
target_sources(engine_schedule PUBLIC FILE_SET CXX_MODULES FILES schedule.cc)
target_link_libraries(engine_schedule PUBLIC engine_language_profile engine_model engine_prims)
# The cycle schedule is a device header; this unit is its one host includer.
target_include_directories(engine_schedule PRIVATE "${PROJECT_SOURCE_DIR}/device")

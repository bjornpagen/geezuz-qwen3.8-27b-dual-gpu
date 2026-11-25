# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_model)
target_sources(engine_model PUBLIC FILE_SET CXX_MODULES FILES model.cc)
target_link_libraries(engine_model PUBLIC engine_language_profile)
# The model description is a device header; this unit is its one host includer.
target_include_directories(engine_model PRIVATE "${PROJECT_SOURCE_DIR}/device")

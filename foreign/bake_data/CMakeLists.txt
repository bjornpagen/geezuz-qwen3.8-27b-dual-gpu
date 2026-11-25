# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_bake_data)
target_sources(engine_bake_data PUBLIC FILE_SET CXX_MODULES FILES bake_data.cc)
target_link_libraries(engine_bake_data PUBLIC engine_language_profile)
set(digests "${PROJECT_SOURCE_DIR}/data/checkpoint/sha256.txt")
set_property(SOURCE bake_data.cc APPEND PROPERTY OBJECT_DEPENDS "${digests}")
# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
set_property(SOURCE bake_data.cc APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions)
target_compile_options(engine_bake_data PRIVATE "--embed-dir=${PROJECT_SOURCE_DIR}/data/checkpoint")

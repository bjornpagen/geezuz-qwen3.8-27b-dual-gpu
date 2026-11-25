# SPDX-License-Identifier: AGPL-3.0-only
# Test-only: the tokenizer's reference fixtures, built into the tokenizer test.
add_library(engine_tokenizer_fixtures)
target_sources(engine_tokenizer_fixtures PUBLIC FILE_SET CXX_MODULES FILES tokenizer_fixtures.cc)
target_link_libraries(engine_tokenizer_fixtures PUBLIC engine_language_profile)
set(fixtures "${PROJECT_SOURCE_DIR}/data/tokenizer/fixtures")
set_property(SOURCE tokenizer_fixtures.cc APPEND PROPERTY OBJECT_DEPENDS "${fixtures}/encode.fix" "${fixtures}/decode.fix")
# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
set_property(SOURCE tokenizer_fixtures.cc APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions)
target_compile_options(engine_tokenizer_fixtures PRIVATE "--embed-dir=${fixtures}")

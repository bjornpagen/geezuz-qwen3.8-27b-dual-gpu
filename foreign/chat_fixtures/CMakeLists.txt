# SPDX-License-Identifier: AGPL-3.0-only
# Test-only: the chat format's reference fixtures, built into the chat tests.
add_library(engine_chat_fixtures)
target_sources(engine_chat_fixtures PUBLIC FILE_SET CXX_MODULES FILES chat_fixtures.cc)
target_link_libraries(engine_chat_fixtures PUBLIC engine_language_profile)
set(fixtures "${PROJECT_SOURCE_DIR}/data/chat/fixtures")
set_property(SOURCE chat_fixtures.cc APPEND PROPERTY OBJECT_DEPENDS "${fixtures}/render.json" "${fixtures}/parse.json" "${fixtures}/numbers.json")
# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
set_property(SOURCE chat_fixtures.cc APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions)
target_compile_options(engine_chat_fixtures PRIVATE "--embed-dir=${fixtures}")

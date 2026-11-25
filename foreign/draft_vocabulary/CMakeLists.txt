# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_draft_vocabulary)
target_sources(engine_draft_vocabulary PUBLIC FILE_SET CXX_MODULES FILES draft_vocabulary.cc)
target_link_libraries(engine_draft_vocabulary PUBLIC engine_language_profile)
set(list "${PROJECT_SOURCE_DIR}/data/draft_vocab_32k.bin")
set_property(SOURCE draft_vocabulary.cc APPEND PROPERTY OBJECT_DEPENDS "${list}")
# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
set_property(SOURCE draft_vocabulary.cc APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions)
target_compile_options(engine_draft_vocabulary PRIVATE "--embed-dir=${PROJECT_SOURCE_DIR}/data")

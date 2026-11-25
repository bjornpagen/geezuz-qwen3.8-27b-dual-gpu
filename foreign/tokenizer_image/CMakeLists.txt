# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_tokenizer_image)
target_sources(engine_tokenizer_image PUBLIC FILE_SET CXX_MODULES FILES tokenizer_image.cc)
target_link_libraries(engine_tokenizer_image PUBLIC engine_language_profile)
set(image "${PROJECT_SOURCE_DIR}/data/tokenizer/qwen.tok")
set_property(SOURCE tokenizer_image.cc APPEND PROPERTY OBJECT_DEPENDS "${image}")
# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
set_property(SOURCE tokenizer_image.cc APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions)
target_compile_options(engine_tokenizer_image PRIVATE "--embed-dir=${PROJECT_SOURCE_DIR}/data/tokenizer")

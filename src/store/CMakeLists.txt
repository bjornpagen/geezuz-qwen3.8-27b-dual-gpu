# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_store)
target_sources(engine_store PUBLIC FILE_SET CXX_MODULES FILES store.cc)
target_link_libraries(engine_store PUBLIC engine_language_profile engine_core engine_json engine_tokenizer)

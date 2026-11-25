# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_prefix)
target_sources(engine_prefix PUBLIC FILE_SET CXX_MODULES FILES prefix.cc point.cc tree.cc pending.cc cache.cc)
target_link_libraries(engine_prefix PUBLIC engine_language_profile engine_core engine_handle engine_kv engine_model engine_tokenizer)

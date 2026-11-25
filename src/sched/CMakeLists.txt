# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_sched)
target_sources(engine_sched PUBLIC FILE_SET CXX_MODULES FILES sched.cc request.cc replay.cc)
target_link_libraries(engine_sched PUBLIC engine_language_profile engine_core engine_kv engine_sampling engine_sha256 engine_tokenizer)

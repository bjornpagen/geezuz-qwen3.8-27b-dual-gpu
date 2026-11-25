# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_prefix_world)
target_sources(engine_prefix_world PUBLIC FILE_SET CXX_MODULES FILES world.cc)
target_link_libraries(engine_prefix_world PUBLIC engine_language_profile engine_core engine_kv engine_kv_harness engine_model engine_prefix)

engine_test(prefix.scenarios SOURCES scenarios.test.cc LINK engine_kv engine_prefix engine_prefix_world)
engine_test(prefix.search SOURCES search.test.cc LINK engine_kv engine_prefix engine_prefix_world)

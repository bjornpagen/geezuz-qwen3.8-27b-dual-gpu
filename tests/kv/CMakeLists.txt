# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_kv_harness)
target_sources(engine_kv_harness PUBLIC FILE_SET CXX_MODULES FILES fake_device.cc harness.cc)
target_link_libraries(engine_kv_harness PUBLIC engine_language_profile engine_core engine_kv engine_model)

engine_test(kv.fuzz SOURCES fuzz.test.cc LINK engine_kv engine_kv_harness)
engine_test(kv.exhaustive SOURCES exhaustive.test.cc LINK engine_kv engine_kv_harness)
engine_test(kv.properties SOURCES properties.test.cc LINK engine_core engine_kv engine_kv_harness engine_model)
engine_death_test(kv.stale_release_aborts SOURCES stale_release.death.cc LINK engine_kv engine_kv_harness)

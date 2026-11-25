# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_sched_world)
target_sources(engine_sched_world PUBLIC FILE_SET CXX_MODULES FILES world.cc)
target_link_libraries(engine_sched_world
	PUBLIC engine_language_profile engine_core engine_kv engine_kv_harness engine_model engine_prefix engine_sampling engine_schedule
	engine_sched engine_tokenizer)

engine_test(sched.sim SOURCES sim.test.cc LINK engine_sched engine_sched_world engine_tokenizer)
engine_test(sched.scenarios SOURCES scenarios.test.cc LINK engine_sched engine_sched_world engine_tokenizer)
engine_test(sched.replay SOURCES replay.test.cc LINK engine_sched engine_tokenizer)
engine_death_test(sched.settle_unplanned_aborts SOURCES settle_unplanned.death.cc LINK engine_sched engine_kv_harness)

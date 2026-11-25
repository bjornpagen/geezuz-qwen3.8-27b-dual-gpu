# SPDX-License-Identifier: AGPL-3.0-only

# The per-card scheduler and the engine's worker thread, over the fake driver.
engine_test(pipeline.cards SOURCES cards.test.cc LINK engine_card_scheduler engine_device engine_driver_double engine_exec engine_net)

engine_test(pipeline.masks SOURCES masks.test.cc LINK engine_pipeline engine_vocabulary)

# The parts the engine composes, as test doubles: the cards over the fake driver and the content-checking kv device,
# the grammar, and the api in front.
add_library(engine_pipeline_doubles)
target_sources(engine_pipeline_doubles PUBLIC FILE_SET CXX_MODULES FILES doubles.cc)
target_link_libraries(engine_pipeline_doubles
	PUBLIC engine_language_profile engine_card_scheduler engine_core engine_device engine_exec engine_http engine_kv engine_kv_harness
	engine_model engine_net engine_pipeline engine_sampling engine_sched engine_tokenizer)

engine_test(pipeline.engine SOURCES engine.test.cc
	LINK engine_pipeline engine_pipeline_doubles engine_device engine_driver_double engine_http engine_kv engine_net engine_sched
	engine_tokenizer)

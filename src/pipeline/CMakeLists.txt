# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_pipeline)
target_sources(engine_pipeline PUBLIC FILE_SET CXX_MODULES FILES pipeline.cc masks.cc seams.cc outbox.cc engine.cc facade.cc)
target_link_libraries(engine_pipeline
	PUBLIC engine_language_profile engine_card_scheduler engine_core engine_device engine_exec engine_grammar engine_http engine_kv engine_model
	engine_net engine_sched engine_tokenizer engine_vocabulary)

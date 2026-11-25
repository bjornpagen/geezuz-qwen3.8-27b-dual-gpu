# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_prefill)
target_sources(engine_prefill PUBLIC FILE_SET CXX_MODULES FILES prefill.cc memories.cc steps.cc chunks.cc)
target_link_libraries(engine_prefill
	PUBLIC engine_language_profile engine_attn engine_card_scheduler engine_core engine_deltanet engine_device engine_driver engine_exec
	engine_kv engine_model engine_prefill_device engine_sched engine_schedule)

# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_cards)
target_sources(engine_cards PUBLIC FILE_SET CXX_MODULES FILES cards.cc adapter.cc start.cc)
target_link_libraries(engine_cards
	PUBLIC engine_language_profile engine_bake engine_card_scheduler engine_core engine_decode engine_device engine_exec engine_image engine_kv
	engine_model engine_pipeline engine_platform engine_prefill engine_sched)

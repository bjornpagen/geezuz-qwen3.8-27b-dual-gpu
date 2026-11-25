# SPDX-License-Identifier: AGPL-3.0-only
engine_test(cards SOURCES cards.test.cc
	LINK engine_card_scheduler engine_cards engine_core engine_decode engine_decode_device engine_device engine_driver_double engine_exec
	engine_file_write engine_image engine_image_synthetic engine_kv engine_model engine_pipeline engine_sampling engine_sched)

# The engine as the programs start it, where the cards are absent: the bring-up's refusal and how the server's engine
# slot reports it. Where the cards exist the same run would start them, so the test belongs to darwin's build graph.
if(ENGINE_PLATFORM STREQUAL "darwin")
	engine_test(cards.serving SOURCES serving.test.cc
		LINK engine_cards engine_core engine_driver engine_http engine_net engine_openai)
endif()

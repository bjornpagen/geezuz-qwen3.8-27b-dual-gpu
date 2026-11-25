# SPDX-License-Identifier: AGPL-3.0-only
engine_test(cards SOURCES cards.test.cc
	LINK engine_card_scheduler engine_cards engine_core engine_decode engine_decode_device engine_device engine_driver_double engine_exec
	engine_file_write engine_image engine_image_synthetic engine_kv engine_model engine_pipeline engine_sampling engine_sched)

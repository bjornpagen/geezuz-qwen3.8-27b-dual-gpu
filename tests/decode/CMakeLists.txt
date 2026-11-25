# SPDX-License-Identifier: AGPL-3.0-only
engine_test(decode.walk SOURCES walk.test.cc LINK engine_decode_device engine_model engine_schedule)
engine_test(decode.key SOURCES key.test.cc LINK engine_decode_device engine_model engine_sampling)
engine_test(decode.commands SOURCES commands.test.cc LINK engine_decode engine_decode_device engine_kv engine_model)
engine_test(decode.ledger SOURCES ledger.test.cc LINK engine_decode engine_decode_device engine_kv engine_model engine_sampling engine_sched engine_tokenizer)
engine_test(decode.session SOURCES session.test.cc
	LINK engine_core engine_decode engine_decode_device engine_device engine_driver_double engine_kv engine_model engine_sampling engine_sched
	engine_schedule engine_vocabulary)
engine_test(decode.engine SOURCES engine.test.cc
	LINK engine_core engine_decode engine_decode_device engine_device engine_driver engine_driver_double engine_file_write engine_image
	engine_image_synthetic engine_model)

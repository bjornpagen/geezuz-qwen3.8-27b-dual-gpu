# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_decode)
target_sources(engine_decode PUBLIC FILE_SET CXX_MODULES FILES decode.cc error.cc ledger.cc commands.cc session.cc engine.cc)
target_link_libraries(engine_decode
	PUBLIC engine_language_profile engine_attn engine_core engine_decode_device engine_deltanet engine_device engine_file engine_image engine_kv engine_model
	engine_sched engine_vocabulary)

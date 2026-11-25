# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_bake)
target_sources(engine_bake PUBLIC FILE_SET CXX_MODULES FILES
	bake.cc
	bound.cc
	error.cc
	plan.cc
	produce.cc
	qwen.cc
	source.cc
	verify.cc
	write.cc
)
target_link_libraries(engine_bake PUBLIC
	engine_language_profile
	engine_bake_data
	engine_core
	engine_file
	engine_file_write
	engine_image
	engine_model
	engine_prims
	engine_safetensors
	engine_schedule
	engine_sha256
	engine_vocabulary
)

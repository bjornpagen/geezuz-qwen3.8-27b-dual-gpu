# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_openai)
target_sources(engine_openai PUBLIC FILE_SET CXX_MODULES FILES
	openai.cc
	text.cc
	error.cc
	stop.cc
	engine.cc
	validate.cc
	render.cc
	stream.cc
	gather.cc
	service.cc
)
target_link_libraries(engine_openai PUBLIC engine_language_profile engine_core engine_http engine_json engine_model engine_sampling)

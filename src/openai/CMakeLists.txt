# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_openai)
target_sources(engine_openai PUBLIC FILE_SET CXX_MODULES FILES
	openai.cc
	error.cc
	boundary.cc
	compile.cc
	front.cc
	validate.cc
	assembly.cc
	stream.cc
	routes.cc
	service.cc
)
target_link_libraries(engine_openai
	PUBLIC engine_language_profile engine_chat engine_core engine_grammar engine_http engine_json engine_model engine_net engine_pipeline
	engine_regex engine_sampling engine_sched engine_store engine_tokenizer)

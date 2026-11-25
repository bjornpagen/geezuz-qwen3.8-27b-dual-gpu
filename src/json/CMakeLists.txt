# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_json)
target_sources(engine_json PUBLIC FILE_SET CXX_MODULES FILES
	json.cc
	bounds.cc
	error.cc
	text.cc
	value.cc
	containers.cc
	reader.cc
	write.cc
	request.cc
	decode_support.cc
	decode_neutral.cc
	decode_messages.cc
	decode_tools.cc
	decode.cc
	response.cc
)
target_link_libraries(engine_json PUBLIC engine_language_profile engine_core engine_simdjson)

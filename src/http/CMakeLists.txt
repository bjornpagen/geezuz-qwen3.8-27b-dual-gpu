# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_http)
target_sources(engine_http PUBLIC FILE_SET CXX_MODULES FILES
	http.cc
	alternative.cc
	limits.cc
	status.cc
	error.cc
	request.cc
	head.cc
	inbound.cc
	write.cc
	outbound.cc
	exchange.cc
	connections.cc
	server.cc
)
target_link_libraries(engine_http PUBLIC engine_language_profile engine_core engine_handle engine_net)

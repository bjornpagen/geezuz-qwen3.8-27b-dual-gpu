# SPDX-License-Identifier: AGPL-3.0-only
engine_test(openai.stop SOURCES stop.test.cc LINK engine_openai)

# The engine behind the service in its tests: scripted outputs posted through the server's mailbox.
add_library(engine_openai_engine_double)
target_sources(engine_openai_engine_double PUBLIC FILE_SET CXX_MODULES FILES engine_double.cc)
target_link_libraries(engine_openai_engine_double PUBLIC engine_language_profile engine_core engine_http engine_net engine_openai)

# Not `openai.service`: macOS system policy refuses to run an executable whose name ends in `.service`.
engine_test(openai.api SOURCES api.test.cc
	LINK engine_openai engine_openai_engine_double engine_http engine_http_test_client engine_json engine_net engine_sampling)

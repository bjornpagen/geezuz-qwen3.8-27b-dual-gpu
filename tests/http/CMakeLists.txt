# SPDX-License-Identifier: AGPL-3.0-only
engine_test(http.inbound SOURCES inbound.test.cc LINK engine_http)

# The tests' http client: a response reader and event-stream decoder written from the protocol, the oracle the
# server's bytes are checked against.
add_library(engine_http_test_client)
target_sources(engine_http_test_client PUBLIC FILE_SET CXX_MODULES FILES client.cc)
target_link_libraries(engine_http_test_client PUBLIC engine_language_profile engine_net)

# The refusal bodies are read back with the engine's own JSON reader: engine.http writes them without it.
engine_test(http.server SOURCES server.test.cc LINK engine_http engine_http_test_client engine_net engine_json)

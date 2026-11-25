# SPDX-License-Identifier: AGPL-3.0-only

# The engine behind the service in its tests: scripted outputs posted through the server's mailbox.
add_library(engine_openai_engine_double)
target_sources(engine_openai_engine_double PUBLIC FILE_SET CXX_MODULES FILES engine_double.cc)
target_link_libraries(engine_openai_engine_double
	PUBLIC engine_language_profile engine_core engine_http engine_net engine_openai)

engine_test(openai.front SOURCES front.test.cc
	LINK engine_openai engine_chat engine_grammar engine_json engine_pipeline engine_sampling engine_sched engine_tokenizer)

# The tests' side of the service: the server turned by the test's thread, requests as clients write them, and a reader
# of what the server sends.
add_library(engine_openai_harness)
target_sources(engine_openai_harness PUBLIC FILE_SET CXX_MODULES FILES harness.cc)
target_link_libraries(engine_openai_harness PUBLIC engine_language_profile engine_core engine_http engine_http_test_client engine_json engine_net engine_openai)

# Not `openai.service`: macOS system policy refuses to run an executable whose name ends in `.service`.
engine_test(openai.api SOURCES api.test.cc
	LINK engine_openai engine_openai_engine_double engine_openai_harness engine_chat engine_http engine_http_test_client engine_json engine_net
	engine_sampling engine_sched engine_tokenizer)
engine_test(openai.flow SOURCES flow.test.cc
	LINK engine_openai engine_openai_engine_double engine_openai_harness engine_grammar engine_http engine_http_test_client engine_json engine_net engine_pipeline
	engine_sched engine_tokenizer)

# End to end through the real engine loop, with the pipeline's own doubles of the cards and the grammar.
engine_test(openai.pipeline SOURCES pipeline.test.cc
	LINK engine_openai engine_openai_harness engine_pipeline engine_pipeline_doubles engine_chat engine_device engine_driver_double engine_grammar engine_http
	engine_http_test_client engine_json engine_kv engine_model engine_net engine_sched engine_tokenizer)

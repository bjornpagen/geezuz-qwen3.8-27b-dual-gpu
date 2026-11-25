# SPDX-License-Identifier: AGPL-3.0-only

# The server run as a program on loopback, on the machine without the cards: it must answer /health while its engine
# fails to start with the driver's refusal. Where the cards exist the same run would start them, so the test belongs
# to darwin's build graph alone.
if(ENGINE_PLATFORM STREQUAL "darwin")
	add_executable(engine_test_apps.server server.test.cc)
	target_link_libraries(engine_test_apps.server
		PRIVATE engine_testing engine_language_profile engine_cards engine_core engine_driver engine_http_test_client engine_json engine_openai
		engine_process)
	add_test(NAME apps.server COMMAND engine_test_apps.server $<TARGET_FILE:geezuz-server>)
	set_tests_properties(apps.server PROPERTIES TIMEOUT 120)
endif()

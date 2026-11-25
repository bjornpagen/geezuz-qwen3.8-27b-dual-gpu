# SPDX-License-Identifier: AGPL-3.0-only
add_executable(geezuz main.cc)
target_link_libraries(geezuz PRIVATE engine_language_profile engine_cards engine_core engine_file engine_http engine_json engine_net engine_openai engine_process)

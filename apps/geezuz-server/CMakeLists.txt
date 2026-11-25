# SPDX-License-Identifier: AGPL-3.0-only
add_executable(geezuz-server main.cc)
target_link_libraries(geezuz-server PRIVATE engine_language_profile engine_cards engine_core engine_net engine_openai engine_process)

# SPDX-License-Identifier: AGPL-3.0-only
add_executable(geezuz-ubench main.cc)
target_link_libraries(geezuz-ubench PRIVATE engine_language_profile engine_core engine_device engine_ubench)

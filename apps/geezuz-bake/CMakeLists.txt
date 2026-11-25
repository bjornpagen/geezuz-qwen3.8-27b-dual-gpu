# SPDX-License-Identifier: AGPL-3.0-only
add_executable(geezuz-bake main.cc)
target_link_libraries(geezuz-bake PRIVATE engine_language_profile engine_bake engine_image engine_sha256)

# SPDX-License-Identifier: AGPL-3.0-only
add_executable(geezuz-soak main.cc)
target_link_libraries(geezuz-soak PRIVATE engine_language_profile engine_core engine_bake engine_device engine_image engine_model engine_platform engine_soak)

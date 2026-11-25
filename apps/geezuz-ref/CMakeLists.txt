# SPDX-License-Identifier: AGPL-3.0-only
add_executable(geezuz-ref main.cc)
target_link_libraries(geezuz-ref PRIVATE engine_language_profile engine_core engine_file engine_file_write engine_model engine_ref engine_sha256 engine_tokenizer)

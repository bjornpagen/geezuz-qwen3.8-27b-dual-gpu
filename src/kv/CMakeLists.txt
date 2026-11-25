# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_kv)
target_sources(engine_kv PUBLIC FILE_SET CXX_MODULES FILES kv.cc geometry.cc device.cc error.cc pool.cc)
target_link_libraries(engine_kv PUBLIC engine_language_profile engine_core engine_handle engine_model)

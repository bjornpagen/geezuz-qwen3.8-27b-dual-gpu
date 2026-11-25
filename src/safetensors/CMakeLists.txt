# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_safetensors)
target_sources(engine_safetensors PUBLIC FILE_SET CXX_MODULES FILES safetensors.cc error.cc layout.cc decode.cc)
target_link_libraries(engine_safetensors PUBLIC engine_language_profile engine_core engine_json)

# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_file)
target_sources(engine_file PUBLIC FILE_SET CXX_MODULES FILES file.cc)
target_link_libraries(engine_file PUBLIC engine_language_profile engine_core)

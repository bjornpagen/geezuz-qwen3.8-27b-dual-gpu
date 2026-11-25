# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_file_write)
target_sources(engine_file_write PUBLIC FILE_SET CXX_MODULES FILES file_write.cc)
target_link_libraries(engine_file_write PUBLIC engine_language_profile engine_core)

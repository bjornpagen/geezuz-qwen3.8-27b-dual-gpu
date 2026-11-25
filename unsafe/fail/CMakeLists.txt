# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_fail)
target_sources(engine_fail PUBLIC FILE_SET CXX_MODULES FILES fail.cc)
target_link_libraries(engine_fail PUBLIC engine_language_profile)

# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_testing)
target_sources(engine_testing PUBLIC FILE_SET CXX_MODULES FILES testing.cc)
target_link_libraries(engine_testing PUBLIC engine_language_profile)

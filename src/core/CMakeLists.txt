# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_core)
target_sources(engine_core PUBLIC FILE_SET CXX_MODULES FILES core.cc)
target_link_libraries(engine_core PUBLIC engine_language_profile engine_fail)

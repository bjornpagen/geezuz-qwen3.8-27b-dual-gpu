# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_regex)
target_sources(engine_regex PUBLIC FILE_SET CXX_MODULES FILES regex.cc expr.cc utf8.cc syntax.cc dfa.cc)
target_link_libraries(engine_regex PUBLIC engine_language_profile engine_core)

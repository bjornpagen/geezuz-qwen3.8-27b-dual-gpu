# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_thread_pool)
target_sources(engine_thread_pool PUBLIC FILE_SET CXX_MODULES FILES thread_pool.cc)
target_link_libraries(engine_thread_pool PUBLIC engine_language_profile engine_core)

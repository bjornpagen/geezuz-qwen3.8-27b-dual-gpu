# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_vocabulary)
target_sources(engine_vocabulary PUBLIC FILE_SET CXX_MODULES FILES vocabulary.cc)
target_link_libraries(engine_vocabulary PUBLIC engine_language_profile engine_core engine_draft_vocabulary engine_model engine_sha256)

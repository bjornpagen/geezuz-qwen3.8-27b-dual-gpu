# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_mma_model)
target_sources(engine_mma_model PUBLIC FILE_SET CXX_MODULES FILES mma_model.cc)
target_link_libraries(engine_mma_model PUBLIC engine_language_profile engine_core engine_prims)

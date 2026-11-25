# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_ptx)
target_sources(engine_ptx PUBLIC FILE_SET CXX_MODULES FILES ptx.cc image.cc abi.cc)
target_link_libraries(engine_ptx PUBLIC engine_language_profile engine_core)
engine_embed_ptx(engine_ptx SOURCE abi.cc IMAGES abi)

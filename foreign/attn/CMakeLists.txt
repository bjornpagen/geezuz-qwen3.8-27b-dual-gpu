# SPDX-License-Identifier: AGPL-3.0-only
# Attention's device headers re-exported to host code, its ptx image with the kernels' declarations, and the host
# emulation of its arithmetic.
add_library(engine_attn)
target_sources(engine_attn PUBLIC FILE_SET CXX_MODULES FILES attn.cc spec.cc image.cc emulation.cc)
target_link_libraries(engine_attn PUBLIC engine_language_profile engine_core engine_model engine_prims engine_ptx engine_driver engine_mma_model)
# The attention arithmetic and geometry are device headers; this component is their one host includer.
target_include_directories(engine_attn PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_attn SOURCE image.cc IMAGES attn)

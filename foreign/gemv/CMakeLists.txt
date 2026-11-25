# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_gemv)
target_sources(engine_gemv PUBLIC FILE_SET CXX_MODULES FILES gemv.cc geometry.cc emulation.cc images.cc)
target_link_libraries(engine_gemv PUBLIC engine_language_profile engine_core engine_model engine_prims engine_mma_model engine_ptx)
# The geometry is device headers; this unit is their one host includer.
target_include_directories(engine_gemv PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_gemv SOURCE images.cc IMAGES gemv_check gemv_bench)

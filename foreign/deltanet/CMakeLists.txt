# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_deltanet)
target_sources(engine_deltanet PUBLIC FILE_SET CXX_MODULES FILES deltanet.cc arithmetic.cc replay.cc emulation.cc kernels.cc)
target_link_libraries(engine_deltanet PUBLIC engine_language_profile engine_core engine_model engine_prims engine_ptx engine_driver)
# The deltanet's arithmetic and layouts are device headers; this component is their one host includer.
target_include_directories(engine_deltanet PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_deltanet SOURCE kernels.cc IMAGES deltanet)

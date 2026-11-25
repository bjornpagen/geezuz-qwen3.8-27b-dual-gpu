# SPDX-License-Identifier: AGPL-3.0-only
# The device half of the soak: embeds the integrity kernels' image, declares each kernel, re-exports the layout both
# ends compile, and emulates the card's digest on the host. `engine.platform` (src/platform/) runs the soak over it.
add_library(engine_soak)
target_sources(engine_soak PUBLIC FILE_SET CXX_MODULES FILES soak.cc layout.cc images.cc kernels.cc emulation.cc)
target_link_libraries(engine_soak PUBLIC engine_language_profile engine_core engine_ptx engine_prims engine_driver engine_sha256 engine_sha256_block)
# The layout and the digest's steps are device headers; this component is their one host includer.
target_include_directories(engine_soak PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_soak SOURCE images.cc IMAGES soak)

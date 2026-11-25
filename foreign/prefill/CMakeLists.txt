# SPDX-License-Identifier: AGPL-3.0-only
# The device half of prefill: `engine.prefill` (src/prefill/) is the dialect half and owns the component name.
add_library(engine_prefill_device)
target_sources(engine_prefill_device PUBLIC FILE_SET CXX_MODULES FILES prefill_device.cc layout.cc kernels.cc)
target_link_libraries(engine_prefill_device PUBLIC engine_language_profile engine_core engine_model engine_prims engine_ptx engine_driver engine_schedule)
# Prefill's workspace, parameters and image table are device headers; this component is their one host includer.
target_include_directories(engine_prefill_device PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_prefill_device SOURCE kernels.cc IMAGES prefill)

# SPDX-License-Identifier: AGPL-3.0-only
# The device half of decode: `engine.decode` (src/decode/) is the dialect half and owns the component name.
add_library(engine_decode_device)
target_sources(engine_decode_device PUBLIC FILE_SET CXX_MODULES FILES decode_device.cc)
target_link_libraries(engine_decode_device PUBLIC engine_language_profile engine_core engine_model engine_prims engine_ptx engine_driver)
# The decode kernel's wire types, walk and layouts are device headers; this unit is their one host includer.
target_include_directories(engine_decode_device PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_decode_device SOURCE decode_device.cc IMAGES decode)

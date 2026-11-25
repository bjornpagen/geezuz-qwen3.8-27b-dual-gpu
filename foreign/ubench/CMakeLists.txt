# SPDX-License-Identifier: AGPL-3.0-only
# The device half of the microbenchmarks: `engine.ubench` (bench/ubench/) is the dialect half and owns the component
# name. This one embeds the kernels' images and the primitives' self-check image, declares every kernel, and embeds
# the adversarial review whose microbenchmark table the drivers answer.
add_library(engine_ubench_device)
target_sources(engine_ubench_device PUBLIC FILE_SET CXX_MODULES FILES
	ubench_device.cc shapes.cc images.cc names.cc launch.cc stream.cc sync.cc compute.cc link.cc checks.cc gemv.cc adversarial.cc
)
target_link_libraries(engine_ubench_device PUBLIC engine_language_profile engine_core engine_ptx engine_prims engine_driver engine_gemv)
# The shapes are a device header; this unit is its one host includer.
target_include_directories(engine_ubench_device PRIVATE "${PROJECT_SOURCE_DIR}/device")
engine_embed_ptx(engine_ubench_device SOURCE images.cc IMAGES ubench_stream ubench_sync ubench_compute ubench_link prims_check)
# The review's microbenchmark table is the list the drivers must answer, read from the document itself.
set_property(SOURCE adversarial.cc APPEND PROPERTY OBJECT_DEPENDS "${PROJECT_SOURCE_DIR}/docs/ADVERSARIAL.md")
# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
set_property(SOURCE adversarial.cc APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions "--embed-dir=${PROJECT_SOURCE_DIR}/docs")

# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_device)
target_sources(engine_device PUBLIC FILE_SET CXX_MODULES FILES device.cc vocabulary.cc devices.cc ring.cc describe.cc)
target_link_libraries(engine_device PUBLIC engine_language_profile engine_core engine_ptx engine_model engine_driver engine_hugepage)

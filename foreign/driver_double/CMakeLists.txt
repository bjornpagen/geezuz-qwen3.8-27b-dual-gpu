# SPDX-License-Identifier: AGPL-3.0-only

# The CUDA driver and NVML test double. Only tests link it: the product's one source of driver tables is the loader.
add_library(engine_driver_double)
target_sources(engine_driver_double PUBLIC FILE_SET CXX_MODULES FILES driver_double.cc machine.cc fake.cc ring.cc)
target_link_libraries(engine_driver_double PUBLIC engine_language_profile engine_core engine_driver)

# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_platform)
target_sources(engine_platform PUBLIC FILE_SET CXX_MODULES FILES platform.cc clock.cc telemetry.cc soak.cc bench.cc start.cc report.cc)
target_link_libraries(engine_platform PUBLIC engine_language_profile engine_core engine_device engine_file engine_image engine_model engine_sha256 engine_soak)

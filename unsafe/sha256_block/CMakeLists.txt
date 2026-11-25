# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_sha256_block)

# The platform seam: one compression function per machine on its own SHA instructions, both defining the partition
# engine.sha256_block:compress with the same surface. Nothing above it tests the platform.
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(compress_backend compress.darwin.cc)
else()
	set(compress_backend compress.linux.cc)
endif()

target_sources(engine_sha256_block PUBLIC FILE_SET CXX_MODULES FILES
	sha256_block.cc
	constants.cc
	${compress_backend}
)
target_link_libraries(engine_sha256_block PUBLIC engine_language_profile engine_core)

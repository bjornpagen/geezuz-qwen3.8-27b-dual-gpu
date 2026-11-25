# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_image)
target_sources(engine_image PUBLIC FILE_SET CXX_MODULES FILES
	image.cc
	catalog.cc
	checkpoint.cc
	description.cc
	error.cc
	layout.cc
	model_catalog.cc
	read.cc
	write.cc
)
target_link_libraries(engine_image PUBLIC engine_language_profile engine_core engine_file engine_model engine_sha256)

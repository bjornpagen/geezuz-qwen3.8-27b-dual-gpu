# SPDX-License-Identifier: AGPL-3.0-only

# The tiny plan and its synthetic checkpoints, and the bake's inversion: test doubles and the oracle, shared by the
# unit tests and the checkpoint round trip.
add_library(engine_bake_synthetic)
target_sources(engine_bake_synthetic PUBLIC FILE_SET CXX_MODULES FILES synthetic.cc)
target_link_libraries(engine_bake_synthetic PUBLIC engine_language_profile engine_bake engine_core engine_file_write engine_image engine_model
	engine_schedule engine_sha256 engine_testing)

add_library(engine_bake_inverse)
target_sources(engine_bake_inverse PUBLIC FILE_SET CXX_MODULES FILES inverse.cc)
target_link_libraries(engine_bake_inverse PUBLIC engine_language_profile engine_bake engine_image engine_model engine_prims engine_schedule)

engine_test(bake SOURCES bake.test.cc LINK engine_bake engine_bake_inverse engine_bake_synthetic engine_core engine_file engine_file_write engine_image
	engine_model)
# Synthetic checkpoints and their images go where the test runs: a scratch directory of its own under the build tree.
set_tests_properties(bake PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

engine_checkpoint_test(bake.checkpoint SOURCES checkpoint.test.cc LINK engine_bake engine_bake_inverse engine_image)
set_tests_properties(bake.checkpoint PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

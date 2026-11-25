# SPDX-License-Identifier: AGPL-3.0-only

# Small test-double catalogs and a baker double, shared by the image tests.
add_library(engine_image_synthetic)
target_sources(engine_image_synthetic PUBLIC FILE_SET CXX_MODULES FILES synthetic.cc)
target_link_libraries(engine_image_synthetic PUBLIC engine_language_profile engine_image engine_sha256 engine_testing)

engine_test(image SOURCES image.test.cc LINK engine_image engine_image_synthetic engine_model)
engine_test(image.identity SOURCES identity.test.cc LINK engine_image engine_model)

engine_death_test(image.wrong_extent SOURCES wrong_extent.death.cc LINK engine_image engine_image_synthetic)

# The mapped path needs a real file: `image.emit` writes the specimen image where `image.file` runs, because a dialect
# program can only print its bytes.
add_executable(engine_image_emit emit.cc)
target_link_libraries(engine_image_emit PRIVATE engine_language_profile engine_image_synthetic)
add_test(NAME image.emit
	COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:engine_image_emit>" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/specimen.image"
		-P "${CMAKE_CURRENT_SOURCE_DIR}/emit.cmake"
)
set_tests_properties(image.emit PROPERTIES FIXTURES_SETUP image_specimen TIMEOUT 60)

engine_test(image.file SOURCES file.test.cc LINK engine_file engine_image engine_image_synthetic)
set_tests_properties(image.file PROPERTIES FIXTURES_REQUIRED image_specimen WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

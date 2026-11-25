# SPDX-License-Identifier: AGPL-3.0-only
engine_test(file_write SOURCES file_write.test.cc LINK engine_file_write engine_file engine_core)
# The writer's files go where the test runs: a scratch directory of its own under the build tree.
set_tests_properties(file_write PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

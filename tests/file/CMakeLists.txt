# SPDX-License-Identifier: AGPL-3.0-only
engine_test(file SOURCES file.test.cc LINK engine_file)

# The test's inputs, written where ctest runs it: dialect code cannot write a file, and a mapping test needs a file
# with known bytes, an empty one and an unreadable one.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/known" "mapped bytes\n\tare exactly the file")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty" "")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/unreadable" "x")
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/unreadable" PERMISSIONS OWNER_WRITE)

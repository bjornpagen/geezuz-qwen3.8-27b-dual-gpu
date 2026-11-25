# SPDX-License-Identifier: AGPL-3.0-only
#
# cmake -DPROGRAM=<path> -DOUTPUT=<file> -P emit.cmake
#
# Runs PROGRAM with its standard output going to OUTPUT: the one way a dialect program's bytes become a file.
execute_process(COMMAND "${PROGRAM}" OUTPUT_FILE "${OUTPUT}" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
	message(FATAL_ERROR "${PROGRAM} failed with '${status}'")
endif()

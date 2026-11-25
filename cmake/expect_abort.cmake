# SPDX-License-Identifier: AGPL-3.0-only
#
# cmake -DPROGRAM=<path> -P expect_abort.cmake
#
# Runs PROGRAM and succeeds only when it ends by abort. A normal exit, with any status, is a failure: the program
# under test is a violation that must never be survivable.
execute_process(COMMAND "${PROGRAM}" RESULT_VARIABLE status)
if(NOT status MATCHES "abort|Abort|SIGABRT")
	message(FATAL_ERROR "expected ${PROGRAM} to abort; it ended with '${status}'")
endif()

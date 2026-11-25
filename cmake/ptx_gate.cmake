# SPDX-License-Identifier: AGPL-3.0-only
#
# cmake -DPTX=<file.ptx> -DSTAMP=<file> [-DPTXAS=<ptxas> -DCUBIN=<file>] [-DPROBE=<entry>] -P ptx_gate.cmake
#
# The static gate on one device image, run by the build after the ptx is produced; any finding fails the build.
#
# Rounding is pinned in the ptx itself, so neither llvm nor the driver's jit can change a result: no `.approx` and no
# `.ftz` anywhere, and every floating-point add, sub, mul, fma, mad, div, sqrt and rcp carries `.rn` (an fp32 add or
# mul without it may be fused by the jit).
#
# Every kernel X declares its budget twice in one breath, and the gate holds the two to each other: the launch bounds
# (`.maxntid`, `.minnctapersm`) and the budget record `engine_budget_X`, a 16-byte constant {threads,
# registers_per_thread, shared_bytes, blocks_per_sm} the kernel's translation unit emits from the same `Budget` it
# `static_assert`s. With PTXAS (linux, where a cuda toolkit exists) the image also passes `ptxas -v` with spills as
# errors, and each kernel's registers and static shared memory must be within its record.
#
# With PROBE the image is the contraction probe: entry PROBE must compute a * b + c as `mul.rn.f32` then
# `add.rn.f32` and hold no fma, which proves device code is built with contraction off.

if(NOT PTX OR NOT STAMP)
	message(FATAL_ERROR "ptx_gate: -DPTX=<file> -DSTAMP=<file> are required")
endif()

file(STRINGS "${PTX}" lines)
set(findings "")
set(entry "")
set(entries "")
set(line_number 0)
foreach(line IN LISTS lines)
	math(EXPR line_number "${line_number} + 1")
	# Comments carry no instructions; llvm writes `//` comments only.
	string(REGEX REPLACE "//.*$" "" code "${line}")
	if(code MATCHES "\\.entry[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)")
		set(entry "${CMAKE_MATCH_1}")
		list(APPEND entries "${entry}")
		set(entry_${entry}_maxntid "")
		set(entry_${entry}_minnctapersm "")
		set(entry_${entry}_body "")
	endif()
	if(code MATCHES "^[ \t]*\\.maxntid[ \t]+([0-9]+)")
		set(entry_${entry}_maxntid "${CMAKE_MATCH_1}")
	endif()
	if(code MATCHES "^[ \t]*\\.minnctapersm[ \t]+([0-9]+)")
		set(entry_${entry}_minnctapersm "${CMAKE_MATCH_1}")
	endif()
	if(code MATCHES "\\.approx")
		string(APPEND findings "${PTX}:${line_number}: .approx: ${line}\n")
	endif()
	if(code MATCHES "\\.ftz")
		string(APPEND findings "${PTX}:${line_number}: .ftz: ${line}\n")
	endif()
	# An instruction: optional guard predicate, then the opcode with its dotted modifiers.
	if(code MATCHES "^[ \t]*(@!?%?[A-Za-z0-9_]+[ \t]+)?(add|sub|mul|fma|mad|div|sqrt|rcp)((\\.[A-Za-z0-9_:]+)+)[ \t]")
		set(opcode "${CMAKE_MATCH_2}")
		set(modifiers "${CMAKE_MATCH_3}.")
		if(modifiers MATCHES "\\.(f16|f16x2|bf16|bf16x2|f32|f64)\\." AND NOT modifiers MATCHES "\\.rn\\.")
			string(APPEND findings "${PTX}:${line_number}: floating-point ${opcode} without .rn: ${line}\n")
		endif()
	endif()
	if(code MATCHES "engine_budget_([A-Za-z_$][A-Za-z0-9_$]*)\\[16\\][ \t]*=[ \t]*{([0-9, \t]*)}")
		set(record "${CMAKE_MATCH_1}")
		string(REGEX REPLACE "[ \t]" "" record_bytes "${CMAKE_MATCH_2}")
		string(REPLACE "," ";" record_bytes "${record_bytes}")
		list(LENGTH record_bytes record_length)
		while(record_length LESS 16)
			list(APPEND record_bytes 0)
			math(EXPR record_length "${record_length} + 1")
		endwhile()
		# Four little-endian 32-bit fields: threads, registers per thread, shared bytes, blocks per SM.
		foreach(field IN ITEMS threads registers shared blocks)
			list(POP_FRONT record_bytes b0 b1 b2 b3)
			math(EXPR value "${b0} + 256 * ${b1} + 65536 * ${b2} + 16777216 * ${b3}")
			set(budget_${record}_${field} "${value}")
		endforeach()
		set(budget_${record}_declared TRUE)
	endif()
	if(entry AND code MATCHES "^[ \t]*(mul|add|fma|mad)\\.[A-Za-z0-9_.:]+")
		string(APPEND entry_${entry}_body "${CMAKE_MATCH_0};")
	endif()
endforeach()

foreach(name IN LISTS entries)
	if(entry_${name}_maxntid STREQUAL "" OR entry_${name}_minnctapersm STREQUAL "")
		string(APPEND findings "${PTX}: kernel ${name} states no occupancy: give it launch_bounds(budget.threads, budget.blocks_per_sm)\n")
	elseif(NOT budget_${name}_declared)
		string(APPEND findings "${PTX}: kernel ${name} emits no budget record: define "
			"extern \"C\" __attribute__((device, used)) constexpr Budget engine_budget_${name} = <its budget>\n")
	elseif(NOT entry_${name}_maxntid EQUAL budget_${name}_threads OR NOT entry_${name}_minnctapersm EQUAL budget_${name}_blocks)
		string(APPEND findings "${PTX}: kernel ${name} launches with ${entry_${name}_maxntid} threads and "
			"${entry_${name}_minnctapersm} block(s) per SM, but its budget says ${budget_${name}_threads} and ${budget_${name}_blocks}\n")
	endif()
endforeach()

if(PROBE)
	if(NOT PROBE IN_LIST entries)
		string(APPEND findings "${PTX}: the contraction probe ${PROBE} is missing\n")
	else()
		set(body "${entry_${PROBE}_body}")
		if(NOT body MATCHES "mul\\.rn\\.f32" OR NOT body MATCHES "add\\.rn\\.f32" OR body MATCHES "(fma|mad)(\\.[a-z0-9]+)*\\.f32")
			string(APPEND findings "${PTX}: ${PROBE} does not round a * b + c as mul.rn then add.rn (got '${body}'): "
				"device code is not compiled with -ffp-contract=off\n")
		endif()
	endif()
endif()

if(PTXAS)
	execute_process(
		COMMAND "${PTXAS}" -arch=sm_${SM} -v --warn-on-spills --Werror -o "${CUBIN}" "${PTX}"
		RESULT_VARIABLE status
		OUTPUT_VARIABLE report
		ERROR_VARIABLE report
	)
	if(NOT status EQUAL 0)
		string(APPEND findings "${PTX}: ptxas failed (${status}):\n${report}\n")
	endif()
	string(REPLACE "\n" ";" report_lines "${report}")
	set(compiling "")
	foreach(report_line IN LISTS report_lines)
		if(report_line MATCHES "Compiling entry function '([^']+)'")
			set(compiling "${CMAKE_MATCH_1}")
		elseif(compiling AND report_line MATCHES "Used ([0-9]+) registers")
			set(used "${CMAKE_MATCH_1}")
			set(static_shared 0)
			if(report_line MATCHES "([0-9]+) bytes smem")
				set(static_shared "${CMAKE_MATCH_1}")
			endif()
			if(budget_${compiling}_declared)
				if(used GREATER budget_${compiling}_registers)
					string(APPEND findings "${PTX}: kernel ${compiling} uses ${used} registers; its budget allows "
						"${budget_${compiling}_registers}\n")
				endif()
				if(static_shared GREATER budget_${compiling}_shared)
					string(APPEND findings "${PTX}: kernel ${compiling} has ${static_shared} bytes of static shared memory; its "
						"budget allows ${budget_${compiling}_shared}\n")
				endif()
			endif()
			set(compiling "")
		endif()
	endforeach()
endif()

if(findings)
	# The findings print verbatim, one per line; FATAL_ERROR would rewrap them.
	message(NOTICE "${findings}")
	message(FATAL_ERROR "ptx gate: ${PTX} has findings")
endif()
file(TOUCH "${STAMP}")

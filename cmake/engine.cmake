# SPDX-License-Identifier: AGPL-3.0-only
#
# The build rules every component shares. Each exists because plain target declarations cannot express it once:
# device ptx is a custom command, compile-fail cases and test executables need identical wiring per case. Ordinary
# components are declared directly with add_library; there is no component macro.

# The ptx isa the device rule emits: the newest this LLVM emits for `.target sm_86`. One constant; the clang
# feature flag and the device-side constants derive from it.
set(ENGINE_PTX_ISA 9.3)
set(ENGINE_SM 86)

string(REPLACE "." ";" ENGINE_PTX_ISA_PARTS "${ENGINE_PTX_ISA}")
list(GET ENGINE_PTX_ISA_PARTS 0 ENGINE_PTX_ISA_MAJOR)
list(GET ENGINE_PTX_ISA_PARTS 1 ENGINE_PTX_ISA_MINOR)
set(ENGINE_PTX_DIR "${CMAKE_BINARY_DIR}/ptx")

# ptxas exists exactly where a cuda toolkit does: the production linux box. There it is required and gates every
# ptx; on darwin the gate is absent by platform.
if(ENGINE_PLATFORM STREQUAL "linux")
	find_program(ENGINE_PTXAS ptxas REQUIRED)
endif()

if(ENGINE_DEVICE_CHECKS)
	set(ENGINE_DEVICE_CHECKS_VALUE 1)
else()
	set(ENGINE_DEVICE_CHECKS_VALUE 0)
endif()
if(ENGINE_PROFILER)
	set(ENGINE_DEVICE_PROFILER_VALUE 1)
else()
	set(ENGINE_DEVICE_PROFILER_VALUE 0)
endif()

set(ENGINE_DEVICE_FLAGS
	-x cuda
	--cuda-gpu-arch=sm_${ENGINE_SM}
	--cuda-device-only
	-nocudainc
	-nocudalib
	-std=c++26
	-fno-exceptions
	-fno-rtti
	-O3
	# Rounding is pinned in the ptx: clang's cuda default fuses products into sums (and ignores the contraction
	# pragma while doing so), and an fp32 add or mul without `.rn` leaves the driver's jit free to fuse it again.
	# Off, every fp32 operation is emitted with its rounding mode and every fma is one the source spelled.
	-ffp-contract=off
	-Xclang -target-feature -Xclang +ptx${ENGINE_PTX_ISA_MAJOR}${ENGINE_PTX_ISA_MINOR}
	-DENGINE_PTX_ISA=${ENGINE_PTX_ISA_MAJOR}${ENGINE_PTX_ISA_MINOR}
	-DENGINE_DEVICE_CHECKS=${ENGINE_DEVICE_CHECKS_VALUE}
	-DENGINE_DEVICE_PROFILER=${ENGINE_DEVICE_PROFILER_VALUE}
	-I${PROJECT_SOURCE_DIR}/device
	${ENGINE_WARNINGS}
)

# engine_device_ptx(<name> SOURCE <tu.cu>)
#
# Compiles one device translation unit to `<build>/ptx/<name>.ptx` (one loadable image per translation unit) under
# target `engine_ptx_<name>`. The ptx then passes the static gate (cmake/ptx_gate.cmake), on linux including
# `ptxas -v` with spills as errors; in the lint build the translation unit also passes clang-tidy with the device
# flags.
function(engine_device_ptx name)
	cmake_parse_arguments(PARSE_ARGV 1 arg "" "SOURCE" "")
	if(NOT arg_SOURCE OR arg_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "engine_device_ptx(<name> SOURCE <tu.cu>)")
	endif()
	cmake_path(ABSOLUTE_PATH arg_SOURCE BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE source)
	set(ptx "${ENGINE_PTX_DIR}/${name}.ptx")
	add_custom_command(
		OUTPUT "${ptx}"
		COMMAND "${CMAKE_CXX_COMPILER}" ${ENGINE_DEVICE_FLAGS} -MD -MF "${ptx}.d" -MT "${ptx}" -S -o "${ptx}" "${source}"
		DEPENDS "${source}"
		DEPFILE "${ptx}.d"
		COMMENT "device ptx ${name}"
		COMMAND_EXPAND_LISTS
		VERBATIM
	)
	# The static gate: rounding pinned (no .approx, no .ftz, .rn on every floating-point operation) and every kernel's
	# occupancy stated; on linux the same ptx also passes ptxas -v, spill-free, with registers its occupancy holds.
	set(gate "${ENGINE_PTX_DIR}/${name}.gate.stamp")
	set(gate_arguments -DPTX=${ptx} -DSTAMP=${gate})
	set(gate_outputs "${gate}")
	if(ENGINE_PLATFORM STREQUAL "linux")
		set(cubin "${ENGINE_PTX_DIR}/${name}.ptxas.cubin")
		list(APPEND gate_arguments -DPTXAS=${ENGINE_PTXAS} -DSM=${ENGINE_SM} -DCUBIN=${cubin})
		list(APPEND gate_outputs "${cubin}")
	endif()
	add_custom_command(
		OUTPUT ${gate_outputs}
		COMMAND "${CMAKE_COMMAND}" ${gate_arguments} -P "${PROJECT_SOURCE_DIR}/cmake/ptx_gate.cmake"
		DEPENDS "${ptx}" "${PROJECT_SOURCE_DIR}/cmake/ptx_gate.cmake"
		COMMENT "ptx gate ${name}"
		VERBATIM
	)
	set(outputs "${ptx}" ${gate_outputs})
	if(ENGINE_BUILD STREQUAL "lint")
		set(stamp "${ENGINE_PTX_DIR}/${name}.tidy.stamp")
		add_custom_command(
			OUTPUT "${stamp}"
			COMMAND ${CMAKE_CXX_CLANG_TIDY} --quiet "${source}" -- ${ENGINE_DEVICE_FLAGS}
			COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
			DEPENDS "${source}"
			IMPLICIT_DEPENDS CXX "${source}"
			COMMENT "clang-tidy device ${name}"
			COMMAND_EXPAND_LISTS
			VERBATIM
		)
		list(APPEND outputs "${stamp}")
	endif()
	add_custom_target(engine_ptx_${name} ALL DEPENDS ${outputs})
	set_target_properties(engine_ptx_${name} PROPERTIES ENGINE_PTX_FILE "${ptx}")
endfunction()

# engine_embed_ptx(<target> SOURCE <unit.cc> IMAGES <name>...)
#
# Lets the foreign module unit <unit.cc> of <target> `#embed <name.ptx>` for each image: the images are built before
# the unit is scanned, and a changed image rebuilds the unit.
function(engine_embed_ptx target)
	cmake_parse_arguments(PARSE_ARGV 1 arg "" "SOURCE" "IMAGES")
	if(NOT arg_SOURCE OR NOT arg_IMAGES OR arg_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "engine_embed_ptx(<target> SOURCE <unit.cc> IMAGES <name>...)")
	endif()
	set(files "")
	foreach(image IN LISTS arg_IMAGES)
		add_dependencies(${target} engine_ptx_${image})
		list(APPEND files "${ENGINE_PTX_DIR}/${image}.ptx")
	endforeach()
	set_property(SOURCE "${arg_SOURCE}" APPEND PROPERTY OBJECT_DEPENDS ${files})
	# PIN(clang-embed-extension): clang still calls #embed a C23 extension in C++26; see PINS.md
	set_property(SOURCE "${arg_SOURCE}" APPEND PROPERTY COMPILE_OPTIONS -Wno-c23-extensions)
	target_compile_options(${target} PRIVATE "--embed-dir=${ENGINE_PTX_DIR}")
endfunction()

# engine_test(<name> SOURCES <file>... LINK <target>...)
#
# One dialect test executable `engine_test_<name>`, registered with ctest as `<name>`. Every test runs under a
# deadline, so a hang is a failure.
function(engine_test name)
	cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LINK")
	if(NOT arg_SOURCES OR arg_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "engine_test(<name> SOURCES <file>... LINK <target>...)")
	endif()
	add_executable(engine_test_${name} ${arg_SOURCES})
	target_link_libraries(engine_test_${name} PRIVATE engine_testing engine_language_profile ${arg_LINK})
	add_test(NAME ${name} COMMAND engine_test_${name})
	set_tests_properties(${name} PROPERTIES TIMEOUT 120)
endfunction()

# engine_checkpoint_test(<name> SOURCES <file>... LINK <target>...)
#
# A test that reads the real checkpoint, registered with ctest as `<name>` under the `checkpoint` label and given the
# checkpoint directory as its one argument. The default test presets exclude the label, because the checkpoint is 29 GB
# that a build machine need not hold; `ctest --preset checkpoint` runs exactly these.
set(ENGINE_CHECKPOINT_DIR "$ENV{HOME}/models/Qwen3.8-27B-FP8" CACHE PATH
	"the Qwen3.8-27B-FP8 checkpoint directory the checkpoint-labelled tests read")
function(engine_checkpoint_test name)
	cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LINK")
	if(NOT arg_SOURCES OR arg_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "engine_checkpoint_test(<name> SOURCES <file>... LINK <target>...)")
	endif()
	add_executable(engine_test_${name} ${arg_SOURCES})
	target_link_libraries(engine_test_${name} PRIVATE engine_testing engine_language_profile ${arg_LINK})
	add_test(NAME ${name} COMMAND engine_test_${name} "${ENGINE_CHECKPOINT_DIR}")
	set_tests_properties(${name} PROPERTIES TIMEOUT 3600 LABELS checkpoint)
endfunction()

# engine_death_test(<name> SOURCES <file>... LINK <target>...)
#
# A program that must abort, registered with ctest as `<name>`: an invariant violation must end the process in every
# build, and a normal exit with any status fails the case.
function(engine_death_test name)
	cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LINK")
	if(NOT arg_SOURCES OR arg_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "engine_death_test(<name> SOURCES <file>... LINK <target>...)")
	endif()
	add_executable(engine_test_${name} ${arg_SOURCES})
	target_link_libraries(engine_test_${name} PRIVATE engine_language_profile ${arg_LINK})
	add_test(NAME ${name}
		COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:engine_test_${name}>" -P "${PROJECT_SOURCE_DIR}/cmake/expect_abort.cmake"
	)
	set_tests_properties(${name} PROPERTIES TIMEOUT 120)
endfunction()

# engine_compile_fail(<name> SOURCE <file> LINK <target>... MATCH <regex>)
#
# A program that must not compile, registered with ctest as `compile_fail.<name>`. It passes only when building it
# fails with the pinned diagnostic: compiling, or failing for another reason, fails the case.
function(engine_compile_fail name)
	cmake_parse_arguments(PARSE_ARGV 1 arg "" "SOURCE;MATCH" "LINK")
	if(NOT arg_SOURCE OR NOT arg_MATCH OR arg_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "engine_compile_fail(<name> SOURCE <file> LINK <target>... MATCH <regex>)")
	endif()
	add_executable(engine_cf_${name} EXCLUDE_FROM_ALL ${arg_SOURCE})
	target_link_libraries(engine_cf_${name} PRIVATE engine_language_profile ${arg_LINK})
	set_target_properties(engine_cf_${name} PROPERTIES CXX_CLANG_TIDY "")
	add_test(NAME compile_fail.${name} COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target engine_cf_${name})
	set_tests_properties(compile_fail.${name} PROPERTIES
		PASS_REGULAR_EXPRESSION "${arg_MATCH}"
		RUN_SERIAL TRUE
		TIMEOUT 300
	)
endfunction()

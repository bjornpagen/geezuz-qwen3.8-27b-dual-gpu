# SPDX-License-Identifier: AGPL-3.0-only
cmake_minimum_required(VERSION 4.4)

# The configure gate: the one place that decides whether a machine and toolchain can build this engine. Translation
# units never negotiate features.

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
	message(FATAL_ERROR "Ninja is the only generator (got ${CMAKE_GENERATOR}); configure through a preset")
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.5)
	message(FATAL_ERROR "the import-std gate value below belongs to the CMake 4.4 series (got ${CMAKE_VERSION}); "
		"read the new series' experimental-feature table, update it and this bound together")
endif()

# PIN(cmake-import-std-uuid): the 4.4-series value; see PINS.md
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

# The compiler is found by name, searching the preset's program path before PATH, and cached as an absolute path.
if(NOT CMAKE_CXX_COMPILER)
	find_program(ENGINE_CLANGXX NAMES clang++ REQUIRED)
	set(CMAKE_CXX_COMPILER "${ENGINE_CLANGXX}")
endif()

# libc++ is the standard library on both machines, so compiler detection and `import std` discovery must already see
# it; on linux clang's default would be libstdc++.
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")

# The std module metadata and the libc++ the binaries load are located once, before compiler detection, from the
# compiler's own installation: the toolchain is one tuple.
if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
	file(REAL_PATH "${CMAKE_CXX_COMPILER}" ENGINE_CXX_PATH)
	execute_process(
		COMMAND "${ENGINE_CXX_PATH}" -stdlib=libc++ -print-file-name=libc++.modules.json
		OUTPUT_VARIABLE ENGINE_MODULES_JSON
		OUTPUT_STRIP_TRAILING_WHITESPACE
		COMMAND_ERROR_IS_FATAL ANY
	)
	if(NOT IS_ABSOLUTE "${ENGINE_MODULES_JSON}")
		cmake_path(GET ENGINE_CXX_PATH PARENT_PATH ENGINE_CXX_BIN)
		# PIN(macports-libcxx-layout): libc++ in lib/libc++/, outside the darwin driver's search; see PINS.md
		set(ENGINE_MODULES_JSON "${ENGINE_CXX_BIN}/../lib/libc++/libc++.modules.json")
	endif()
	if(NOT EXISTS "${ENGINE_MODULES_JSON}")
		message(FATAL_ERROR "libc++.modules.json must ship with the compiler (looked for ${ENGINE_MODULES_JSON})")
	endif()
	file(REAL_PATH "${ENGINE_MODULES_JSON}" ENGINE_MODULES_JSON)
	set(CMAKE_CXX_STDLIB_MODULES_JSON "${ENGINE_MODULES_JSON}" CACHE FILEPATH "libc++'s std module metadata")
endif()
cmake_path(GET CMAKE_CXX_STDLIB_MODULES_JSON PARENT_PATH ENGINE_LIBCXX_DIR)

project(geezuz LANGUAGES CXX)

execute_process(
	COMMAND "${CMAKE_MAKE_PROGRAM}" --version
	OUTPUT_VARIABLE ENGINE_NINJA_VERSION
	OUTPUT_STRIP_TRAILING_WHITESPACE
	COMMAND_ERROR_IS_FATAL ANY
)
if(ENGINE_NINJA_VERSION VERSION_LESS 1.13)
	message(FATAL_ERROR "Ninja 1.13 or later is required (got ${ENGINE_NINJA_VERSION})")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
	set(ENGINE_PLATFORM darwin)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
	set(ENGINE_PLATFORM linux)
else()
	message(FATAL_ERROR "the engine builds on exactly two machines, darwin arm64 (development) and linux x86_64 "
		"(production); got ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.1)
	message(FATAL_ERROR "LLVM clang 23.1 or later is the only compiler (got ${CMAKE_CXX_COMPILER_ID} "
		"${CMAKE_CXX_COMPILER_VERSION} at ${CMAKE_CXX_COMPILER})")
endif()

set(ENGINE_BUILDS dev release asan-ubsan profile lint)
set(ENGINE_BUILD "" CACHE STRING "the build personality; set only by CMakePresets.json")
set_property(CACHE ENGINE_BUILD PROPERTY STRINGS ${ENGINE_BUILDS})
if(NOT ENGINE_BUILD IN_LIST ENGINE_BUILDS)
	message(FATAL_ERROR "ENGINE_BUILD must be one of ${ENGINE_BUILDS} (got '${ENGINE_BUILD}'); configure through a preset")
endif()
if(CMAKE_BUILD_TYPE)
	message(FATAL_ERROR "CMAKE_BUILD_TYPE is unused: ENGINE_BUILD is the build personality (got '${CMAKE_BUILD_TYPE}')")
endif()

if(NOT CMAKE_CXX_STDLIB_MODULES_JSON MATCHES "/libc\\+\\+\\.modules\\.json$" OR NOT "26" IN_LIST CMAKE_CXX_COMPILER_IMPORT_STD)
	message(FATAL_ERROR "libc++ with `import std` for C++26 is required (module metadata '${CMAKE_CXX_STDLIB_MODULES_JSON}', "
		"import std levels '${CMAKE_CXX_COMPILER_IMPORT_STD}': ${CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE})")
endif()

# Every LLVM tool comes from the compiler's own installation, so the toolchain is one tuple.
file(REAL_PATH "${CMAKE_CXX_COMPILER}" ENGINE_LLVM_CXX)
cmake_path(GET ENGINE_LLVM_CXX PARENT_PATH ENGINE_LLVM_BIN)
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(ENGINE_LIBCXX "${ENGINE_LIBCXX_DIR}/libc++.dylib")
else()
	set(ENGINE_LIBCXX "${ENGINE_LIBCXX_DIR}/libc++.so")
endif()
if(NOT EXISTS "${ENGINE_LIBCXX}")
	message(FATAL_ERROR "libc++ must sit beside its module metadata (looked for ${ENGINE_LIBCXX})")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_LINKER_TYPE LLD)

# Host and device must round shared arithmetic identically and the cpu reference must be reproducible, so no
# compiler-chosen fused multiply-add: clang contracts a*b+c by default on both arm64 and x86-64.
string(APPEND CMAKE_CXX_FLAGS " -ffp-contract=off")

# The production host is one known cpu, an epyc 7v13 (zen 3), so host code targets exactly it and vector paths compile
# in without runtime dispatch. Every arm64 cpu has the neon the development mac's vector paths need.
if(ENGINE_PLATFORM STREQUAL "linux")
	string(APPEND CMAKE_CXX_FLAGS " -march=znver3")
endif()

# Module-ABI flags are global: CMake's synthesized std module sees only CMAKE_CXX_FLAGS, and its BMI must match
# every importer. That includes the sanitizer and optimization choice of each personality.
string(APPEND CMAKE_CXX_FLAGS
	" -fno-exceptions -fno-rtti"
	" -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE"
)
# PIN(libcxx-std-module-name): the synthesized std module is compiled with libc++'s module directory as a plain
# include path, so its own `export module std;` trips the reserved-name warning; see PINS.md
string(APPEND CMAKE_CXX_FLAGS " -Wno-reserved-module-identifier")
# libc++ is linked dynamically from the compiler's installation, not the system's older copy.
set(ENGINE_RUNTIME_DIRS "${ENGINE_LIBCXX_DIR}")
if(EXISTS "${ENGINE_LIBCXX_DIR}/../libunwind")
	# PIN(macports-libcxx-layout): libc++ loads libunwind by rpath from its own lib/libunwind/; see PINS.md
	file(REAL_PATH "${ENGINE_LIBCXX_DIR}/../libunwind" ENGINE_LIBUNWIND_DIR)
	list(APPEND ENGINE_RUNTIME_DIRS "${ENGINE_LIBUNWIND_DIR}")
endif()
string(APPEND CMAKE_EXE_LINKER_FLAGS " -L${ENGINE_LIBCXX_DIR}")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " -L${ENGINE_LIBCXX_DIR}")
foreach(dir IN LISTS ENGINE_RUNTIME_DIRS)
	string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-rpath,${dir}")
	string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-rpath,${dir}")
endforeach()

if(ENGINE_BUILD STREQUAL "release" OR ENGINE_BUILD STREQUAL "profile")
	string(APPEND CMAKE_CXX_FLAGS " -O3")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
	set(ENGINE_DEVICE_CHECKS Off)
elseif(ENGINE_BUILD STREQUAL "asan-ubsan")
	string(APPEND CMAKE_CXX_FLAGS " -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all")
	string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address,undefined")
	set(ENGINE_DEVICE_CHECKS On)
else()
	string(APPEND CMAKE_CXX_FLAGS " -O1 -g -fsanitize=undefined -fsanitize-trap=all")
	string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=undefined -fsanitize-trap=all")
	set(ENGINE_DEVICE_CHECKS On)
endif()

if(ENGINE_BUILD STREQUAL "profile")
	set(ENGINE_PROFILER On)
else()
	set(ENGINE_PROFILER Off)
endif()

if(ENGINE_BUILD STREQUAL "lint")
	if(NOT EXISTS "${ENGINE_LLVM_BIN}/clang-tidy")
		message(FATAL_ERROR "clang-tidy must sit beside the compiler (looked for ${ENGINE_LLVM_BIN}/clang-tidy)")
	endif()
	# Every source in the tree has a profile through its zone's .clang-tidy. The one source outside it, the std module
	# CMake synthesizes (and cannot be configured per target), has none, and is not ours to lint.
	set(CMAKE_CXX_CLANG_TIDY "${ENGINE_LLVM_BIN}/clang-tidy" --allow-no-checks)
endif()

# The warning set is one list, shared by host targets and the device ptx rule.
set(ENGINE_WARNINGS
	-Werror
	-Wall
	-Wextra
	-Wreserved-identifier
	-Wpedantic
	-Wconversion
	-Wsign-conversion
	-Wshadow-all
	-Wold-style-cast
	-Wcast-align
	-Wnull-dereference
	-Wdouble-promotion
	-Wformat=2
	-Wimplicit-fallthrough
	-Wextra-semi
	-Wcomma
	-Wunreachable-code-aggressive
	-Wloop-analysis
	-Wundef
)

add_library(engine_language_profile INTERFACE)
target_compile_options(engine_language_profile INTERFACE ${ENGINE_WARNINGS})

include(cmake/engine.cmake)

include(FetchContent)
# The latest stdexec release, pinned by commit; it backs std::execution until libc++ ships it.
FetchContent_Declare(stdexec
	GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
	GIT_TAG fc5937a544db908edbde37c9aa67b8d308e6a788 # v0.11.0
	SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(stdexec)

enable_testing()

# Components are discovered, never registered: a lane adds one by creating `<zone>/<name>/CMakeLists.txt`, and no
# shared file changes. Target names resolve at generate time, so discovery order carries no meaning.
foreach(zone IN ITEMS device foreign unsafe src apps tests bench)
	file(GLOB ENGINE_ZONE_COMPONENTS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/${zone}/*/CMakeLists.txt")
	list(SORT ENGINE_ZONE_COMPONENTS)
	foreach(component IN LISTS ENGINE_ZONE_COMPONENTS)
		cmake_path(GET component PARENT_PATH component_dir)
		add_subdirectory("${component_dir}")
	endforeach()
endforeach()

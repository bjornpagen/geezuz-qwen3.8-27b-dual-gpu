# SPDX-License-Identifier: AGPL-3.0-only
include(FetchContent)

# simdjson v5.3.0 (2026-11-22), pinned by the SHA-256 of the tagged commit's source archive. Only its amalgamated
# `singleheader/` pair is built; SOURCE_SUBDIR names a directory without a CMakeLists.txt so none of simdjson's own
# build (options, tests, tools) enters the graph.
FetchContent_Declare(simdjson
	URL https://github.com/simdjson/simdjson/archive/34f8a72e512a4b3e1c88797d4152b474b6a26dc2.tar.gz
	URL_HASH SHA256=bdf285924b9607a1543db90d9e9b9c56ae8fe02d6a8b68ad99e8bf550fdaaf2d
	SOURCE_SUBDIR singleheader/engine-none
)
FetchContent_MakeAvailable(simdjson)

# Exactly one simdjson kernel per platform, chosen here and never at run time: NEON on darwin arm64, AVX2 (haswell)
# on the linux EPYC (Zen 3: AVX2, no AVX-512). With one kernel compiled simdjson binds it statically and never asks
# the CPU; the fallback and every other kernel are compiled out. The module unit checks the choice at compile time.
set(simdjson_kernels ARM64 HASWELL ICELAKE WESTMERE PPC64 LSX LASX RVV_VLS FALLBACK)
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(simdjson_kernel ARM64)
	set(simdjson_builtin arm64)
else()
	# The engine-wide -march=znver3 turns on every feature the AVX2 kernel needs.
	set(simdjson_kernel HASWELL)
	set(simdjson_builtin haswell)
endif()
set(simdjson_definitions SIMDJSON_EXCEPTIONS=0 SIMDJSON_BUILTIN_IMPLEMENTATION=${simdjson_builtin})
foreach(kernel IN LISTS simdjson_kernels)
	if(kernel STREQUAL simdjson_kernel)
		list(APPEND simdjson_definitions SIMDJSON_IMPLEMENTATION_${kernel}=1)
	else()
		list(APPEND simdjson_definitions SIMDJSON_IMPLEMENTATION_${kernel}=0)
	endif()
endforeach()

# Vendor code: the module-ABI flags every unit shares, none of the dialect's warnings, no lint.
add_library(engine_simdjson_vendor STATIC "${simdjson_SOURCE_DIR}/singleheader/simdjson.cpp")
target_include_directories(engine_simdjson_vendor SYSTEM PUBLIC "${simdjson_SOURCE_DIR}/singleheader")
target_compile_definitions(engine_simdjson_vendor PUBLIC ${simdjson_definitions})
set_target_properties(engine_simdjson_vendor PROPERTIES CXX_CLANG_TIDY "")

add_library(engine_simdjson)
target_sources(engine_simdjson PUBLIC FILE_SET CXX_MODULES FILES simdjson.cc)
target_link_libraries(engine_simdjson PUBLIC engine_language_profile engine_core PRIVATE engine_simdjson_vendor)

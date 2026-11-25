# SPDX-License-Identifier: AGPL-3.0-only

# stdexec's own `stdexec` module, built from its pinned source: vendor code, so it gets the module-ABI flags every
# BMI shares and none of the dialect's warnings.
add_library(engine_stdexec_vendor)
target_sources(engine_stdexec_vendor PUBLIC
	FILE_SET CXX_MODULES BASE_DIRS "${stdexec_SOURCE_DIR}/modules" FILES "${stdexec_SOURCE_DIR}/modules/stdexec.cppm"
)
target_include_directories(engine_stdexec_vendor SYSTEM PUBLIC "${stdexec_SOURCE_DIR}/include")
target_compile_definitions(engine_stdexec_vendor PUBLIC STDEXEC_BUILD_MODULES=1)
set_target_properties(engine_stdexec_vendor PROPERTIES CXX_CLANG_TIDY "")

add_library(engine_exec)
target_sources(engine_exec PUBLIC FILE_SET CXX_MODULES FILES exec.cc)
target_link_libraries(engine_exec PUBLIC engine_language_profile engine_stdexec_vendor engine_core)

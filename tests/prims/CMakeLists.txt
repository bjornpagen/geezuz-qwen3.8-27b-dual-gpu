# SPDX-License-Identifier: AGPL-3.0-only
engine_test(prims.budget SOURCES budget.test.cc LINK engine_prims)
engine_test(prims.tile SOURCES tile.test.cc LINK engine_prims)
engine_test(prims.numeric SOURCES numeric.test.cc LINK engine_prims)
engine_test(prims.fragment SOURCES fragment.test.cc LINK engine_prims)
engine_test(prims.stage_ring SOURCES stage_ring.test.cc LINK engine_prims)

# The transcendentals are proven over all 2^32 inputs, too many for one deadline: one executable, one ctest case per
# function and eighth of the inputs, run in parallel.
add_executable(engine_test_prims.transcendental transcendental.test.cc)
target_link_libraries(engine_test_prims.transcendental PRIVATE engine_testing engine_language_profile engine_prims)
foreach(function IN ITEMS exp log rsqrt sigmoid silu softplus)
	foreach(shard RANGE 7)
		add_test(NAME prims.transcendental.${function}.${shard} COMMAND engine_test_prims.transcendental ${function} ${shard})
		set_tests_properties(prims.transcendental.${function}.${shard} PROPERTIES TIMEOUT 120)
	endforeach()
endforeach()

# The ptx gate rejects each thing it exists to reject: a test passes only when the gate fails with that finding.
foreach(fixture IN ITEMS unrounded approximate flushed unbounded undeclared mismatched fused)
	add_test(NAME prims.ptx_gate.rejects_${fixture}
		COMMAND "${CMAKE_COMMAND}" -DPTX=${CMAKE_CURRENT_SOURCE_DIR}/gate/${fixture}.ptx -DSTAMP=${CMAKE_CURRENT_BINARY_DIR}/${fixture}.stamp
			-DPROBE=probe -P "${PROJECT_SOURCE_DIR}/cmake/ptx_gate.cmake"
	)
endforeach()
set_tests_properties(prims.ptx_gate.rejects_unrounded PROPERTIES PASS_REGULAR_EXPRESSION "floating-point add without \\.rn")
set_tests_properties(prims.ptx_gate.rejects_approximate PROPERTIES PASS_REGULAR_EXPRESSION "\\.approx")
set_tests_properties(prims.ptx_gate.rejects_flushed PROPERTIES PASS_REGULAR_EXPRESSION "\\.ftz")
set_tests_properties(prims.ptx_gate.rejects_unbounded PROPERTIES PASS_REGULAR_EXPRESSION "states no occupancy")
set_tests_properties(prims.ptx_gate.rejects_undeclared PROPERTIES PASS_REGULAR_EXPRESSION "emits no budget record")
set_tests_properties(prims.ptx_gate.rejects_mismatched PROPERTIES PASS_REGULAR_EXPRESSION "launches with 32 threads and 1 block\\(s\\) per SM, but its budget says 64 and 1")
set_tests_properties(prims.ptx_gate.rejects_fused PROPERTIES PASS_REGULAR_EXPRESSION "does not round a \\* b \\+ c")

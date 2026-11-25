# SPDX-License-Identifier: AGPL-3.0-only
# The primitives' self-check kernels, run at bring-up by the microbenchmark drivers.
engine_device_ptx(prims_check SOURCE check.cu)
# The contraction probe: the ptx gate proves on its image that the device build rounds a*b+c as two operations.
engine_device_ptx(prims_contraction SOURCE contraction.cu)
set(probe_stamp "${ENGINE_PTX_DIR}/prims_contraction.probe.stamp")
add_custom_command(
	OUTPUT "${probe_stamp}"
	COMMAND "${CMAKE_COMMAND}" -DPTX=${ENGINE_PTX_DIR}/prims_contraction.ptx -DSTAMP=${probe_stamp} -DPROBE=prims_contraction_probe
		-P "${PROJECT_SOURCE_DIR}/cmake/ptx_gate.cmake"
	DEPENDS "${ENGINE_PTX_DIR}/prims_contraction.ptx" "${PROJECT_SOURCE_DIR}/cmake/ptx_gate.cmake"
	COMMENT "ptx gate: contraction off"
	VERBATIM
)
add_custom_target(engine_ptx_prims_contraction_probe ALL DEPENDS "${probe_stamp}")

# SPDX-License-Identifier: AGPL-3.0-only
# The weight matmuls' standalone kernels, run at bring-up by the microbenchmark drivers: the self-checks against the
# host's emulation, and the U5 and U22 microbenchmarks.
engine_device_ptx(gemv_check SOURCE check.cu)
engine_device_ptx(gemv_bench SOURCE bench.cu)

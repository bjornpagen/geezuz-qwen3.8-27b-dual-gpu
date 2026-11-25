# SPDX-License-Identifier: AGPL-3.0-only
# The microbenchmark kernels, one image per concern; `engine.ubench_device` embeds them and the drivers in
# bench/ubench/ run them at bring-up.
engine_device_ptx(ubench_stream SOURCE stream.cu)
engine_device_ptx(ubench_sync SOURCE sync.cu)
engine_device_ptx(ubench_compute SOURCE compute.cu)
engine_device_ptx(ubench_link SOURCE link.cu)

# SPDX-License-Identifier: AGPL-3.0-only
# Each kernel family's host emulation (the device's exact order) against the cpu reference, bit for bit.
engine_test(numerics.projection SOURCES projection.test.cc LINK engine_gemv engine_model engine_prims engine_ref engine_sampling engine_sampling_device engine_schedule)
engine_test(numerics.attention SOURCES attention.test.cc LINK engine_attn engine_model engine_prims engine_ref)
engine_test(numerics.deltanet SOURCES deltanet.test.cc LINK engine_deltanet engine_model engine_prims engine_ref)

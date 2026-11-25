# SPDX-License-Identifier: AGPL-3.0-only
# Each kernel family's host emulation (the device's exact order) against the cpu reference, bit for bit. The costliest
# cases (the real widths of the products, attention's long context) are tests of their own, so each runs under its
# own deadline however loaded the machine is.
add_library(engine_numerics_problems)
target_sources(engine_numerics_problems PUBLIC FILE_SET CXX_MODULES FILES problems.cc)
target_link_libraries(engine_numerics_problems PUBLIC engine_language_profile engine_gemv engine_model engine_ref engine_testing)

add_library(engine_numerics_attention)
target_sources(engine_numerics_attention PUBLIC FILE_SET CXX_MODULES FILES attention_cases.cc)
target_link_libraries(engine_numerics_attention PUBLIC engine_language_profile engine_attn engine_model engine_prims engine_ref engine_testing)

engine_test(numerics.projection SOURCES projection.test.cc LINK engine_gemv engine_model engine_numerics_problems engine_prims engine_ref)
engine_test(numerics.down_proj SOURCES down_proj.test.cc LINK engine_gemv engine_model engine_numerics_problems)
engine_test(numerics.lm_head SOURCES lm_head.test.cc LINK engine_gemv engine_model engine_numerics_problems engine_sampling engine_sampling_device engine_schedule)
engine_test(numerics.attention SOURCES attention.test.cc LINK engine_attn engine_numerics_attention)
engine_test(numerics.attention_long SOURCES attention_long.test.cc LINK engine_numerics_attention)
engine_test(numerics.deltanet SOURCES deltanet.test.cc LINK engine_deltanet engine_model engine_prims engine_ref)

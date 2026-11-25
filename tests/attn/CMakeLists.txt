# SPDX-License-Identifier: AGPL-3.0-only
engine_test(attn.layout SOURCES layout.test.cc LINK engine_attn engine_model engine_prims)
engine_test(attn.plan SOURCES plan.test.cc LINK engine_attn engine_model)
engine_test(attn.split SOURCES split.test.cc LINK engine_attn engine_prims)
engine_test(attn.rope SOURCES rope.test.cc LINK engine_attn engine_model engine_prims)
engine_test(attn.reference SOURCES reference.test.cc LINK engine_attn engine_prims)

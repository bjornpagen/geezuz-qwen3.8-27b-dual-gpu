# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_ref)
target_sources(engine_ref PUBLIC FILE_SET CXX_MODULES FILES
	ref.cc accumulation.cc attention.cc checkpoint.cc deltanet.cc dump.cc forward.cc golden.cc model.cc numeric.cc plan.cc project.cc rulers.cc sums.cc)
target_link_libraries(engine_ref PUBLIC engine_language_profile engine_core engine_file engine_mma_model engine_model engine_prims engine_safetensors
	engine_sampling engine_sampling_device engine_schedule engine_sha256 engine_thread_pool engine_vocabulary)

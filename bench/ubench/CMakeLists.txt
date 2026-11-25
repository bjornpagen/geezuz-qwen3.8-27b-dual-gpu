# SPDX-License-Identifier: AGPL-3.0-only
# The microbenchmark drivers: each runs its kernels under the measurement protocol, reads their results back and
# evaluates them; the runner (apps/geezuz-ubench) prints every outcome beside the review's expectation.
add_library(engine_ubench)
target_sources(engine_ubench PUBLIC FILE_SET CXX_MODULES FILES ubench.cc outcome.cc catalogue.cc protocol.cc stream.cc sync.cc compute.cc link.cc checks.cc runner.cc)
target_link_libraries(engine_ubench PUBLIC engine_language_profile engine_core engine_device engine_model engine_prims engine_ubench_device)

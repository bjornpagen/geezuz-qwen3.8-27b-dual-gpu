# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_deltanet_fixtures)
target_sources(engine_deltanet_fixtures PUBLIC FILE_SET CXX_MODULES FILES fixtures.cc)
target_link_libraries(engine_deltanet_fixtures PUBLIC engine_language_profile engine_deltanet)

engine_test(deltanet.layout SOURCES layout.test.cc LINK engine_deltanet engine_deltanet_fixtures engine_model)
engine_test(deltanet.arithmetic SOURCES arithmetic.test.cc LINK engine_deltanet engine_deltanet_fixtures engine_model)
engine_test(deltanet.replay SOURCES replay.test.cc LINK engine_deltanet engine_deltanet_fixtures)
engine_test(deltanet.prefill SOURCES prefill.test.cc LINK engine_deltanet engine_deltanet_fixtures)
engine_death_test(deltanet.overaccept_aborts SOURCES overaccept.death.cc LINK engine_deltanet)

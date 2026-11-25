# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_card_scheduler)
target_sources(engine_card_scheduler PUBLIC FILE_SET CXX_MODULES FILES card_scheduler.cc card.cc worker.cc)
target_link_libraries(engine_card_scheduler PUBLIC engine_language_profile engine_core engine_driver engine_exec engine_net)

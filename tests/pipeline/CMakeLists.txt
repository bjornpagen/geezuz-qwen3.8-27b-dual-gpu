# SPDX-License-Identifier: AGPL-3.0-only

# The per-card scheduler and the engine's worker thread, over the fake driver.
engine_test(pipeline.cards SOURCES cards.test.cc LINK engine_card_scheduler engine_device engine_driver_double engine_exec engine_net)

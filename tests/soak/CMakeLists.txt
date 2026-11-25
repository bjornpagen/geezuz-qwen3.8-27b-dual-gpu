# SPDX-License-Identifier: AGPL-3.0-only
engine_test(soak.emulation SOURCES emulation.test.cc LINK engine_soak engine_sha256)
engine_test(soak.loop SOURCES loop.test.cc LINK engine_platform engine_soak engine_sha256)
engine_test(soak.clock SOURCES clock.test.cc LINK engine_platform engine_soak engine_driver_double)
engine_test(soak.cards SOURCES cards.test.cc LINK engine_platform engine_soak engine_driver_double engine_file_write engine_image engine_sha256)

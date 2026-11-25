# SPDX-License-Identifier: AGPL-3.0-only
engine_test(core SOURCES core.test.cc LINK engine_core)
engine_death_test(core.invariant_aborts SOURCES invariant.test.cc LINK engine_core)

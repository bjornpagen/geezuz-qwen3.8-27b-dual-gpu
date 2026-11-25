# SPDX-License-Identifier: AGPL-3.0-only
engine_test(sampling.params SOURCES params.test.cc LINK engine_sampling)
engine_test(sampling.rng SOURCES rng.test.cc LINK engine_sampling)
engine_test(sampling.exp SOURCES exp.test.cc LINK engine_sampling)
engine_test(sampling.distribution SOURCES distribution.test.cc LINK engine_sampling)
engine_test(sampling.verify SOURCES verify.test.cc LINK engine_sampling)
engine_death_test(sampling.nan_logit_aborts SOURCES nan_logit.test.cc LINK engine_sampling)

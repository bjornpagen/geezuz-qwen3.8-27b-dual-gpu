# SPDX-License-Identifier: AGPL-3.0-only
engine_test(ubench.catalogue SOURCES catalogue.test.cc LINK engine_ubench)
engine_test(ubench.kernels SOURCES kernels.test.cc LINK engine_ubench_device engine_device engine_driver_double)
engine_test(ubench.drivers SOURCES drivers.test.cc LINK engine_ubench engine_device engine_driver_double)
engine_test(ubench.evaluate SOURCES evaluate.test.cc LINK engine_ubench engine_ubench_device)

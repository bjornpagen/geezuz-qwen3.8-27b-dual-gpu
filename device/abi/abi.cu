// SPDX-License-Identifier: AGPL-3.0-only
//
// The device-side half of the host/device contract, checked by the device compiler itself: the host passes device
// addresses, sizes and kernel parameters as 64-bit integers and plain 4- and 2-byte values, which is sound only while
// the device agrees on every width and alignment below. The image this unit produces holds no code; the driver loads
// it as the cheapest proof that it can jit the engine's ptx isa.
#include "target.h"

static_assert(sizeof(void*) == 8 && alignof(void*) == 8, "device addresses travel as 64-bit integers");
static_assert(sizeof(decltype(sizeof(0))) == 8, "sizes travel as 64-bit integers");
static_assert(sizeof(long long) == 8 && sizeof(int) == 4 && sizeof(short) == 2, "kernel parameters are fixed-width");
static_assert(sizeof(float) == 4 && sizeof(double) == 8, "fp32 partials cross the cards as 4-byte words");
static_assert(alignof(long long) == 8 && alignof(int) == 4 && alignof(float) == 4, "parameter structs share one layout");
static_assert(engine::device::ptx_isa >= 70, "cp.async and mbarrier need ptx 7.0");

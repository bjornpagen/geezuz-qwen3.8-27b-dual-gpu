// SPDX-License-Identifier: AGPL-3.0-only
//
// The build graph's facts about the device target, entering device code here and nowhere else. The ptx isa and the
// build personality's choices arrive as defines from the device ptx rule; every other device header reads these
// constants, never the defines.
#pragma once

namespace engine::device {

enum class Checks : unsigned char { Off, On };

enum class Profiler : unsigned char { Off, On };

// The ptx isa as major * 10 + minor, ptx's own spelling of its versions (ptx93 is 9.3); a primitive needing isa 7.0
// states `static_assert(ptx_isa >= 70)`.
inline constexpr int ptx_isa = ENGINE_PTX_ISA;
inline constexpr int sm = __CUDA_ARCH__ / 10;
inline constexpr Checks checks = ENGINE_DEVICE_CHECKS == 1 ? Checks::On : Checks::Off;
inline constexpr Profiler profiler = ENGINE_DEVICE_PROFILER == 1 ? Profiler::On : Profiler::Off;

static_assert(sm == 86, "the two cards are sm_86; nothing else is a target");

}

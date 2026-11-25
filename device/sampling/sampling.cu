// SPDX-License-Identifier: AGPL-3.0-only
//
// The device compiler's proof over the sampling arithmetic the decode kernels and the host reference share: it
// compiles the headers for sm_86 and evaluates their known-answer checks itself. The image holds no code.
#include "sampling/draw.h"

namespace engine::device::sampling {

static_assert(sizeof(Candidate) == 8, "a candidate crosses the cards as one 8-byte word");
static_assert(sizeof(Rule) == 48 && alignof(Rule) == 8, "the host fills each slot's rule in this layout");

}

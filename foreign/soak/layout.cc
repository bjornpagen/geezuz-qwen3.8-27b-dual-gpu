// SPDX-License-Identifier: AGPL-3.0-only
//
// The soak kernels' parameters, result layouts and geometry re-exported to host code from the device headers the
// kernels compile, so the host launches what each kernel was built for and reads results by the layout that wrote them.
module;

#include "soak/layout.h"
#include "soak/sha256.h"

export module engine.soak:layout;

namespace engine::soak {

namespace device = engine::device::soak;

export using device::AbsorbParams;
export using device::Address;
export using device::CheckParams;
export using device::Chunk;
export using device::FinishParams;
export using device::HashState;
export using device::Pass;
export using device::Slice;
export using device::Tally;
export using device::WriteParams;

export using device::block_bytes;
export using device::hash_budget;
export using device::pass_of;
export using device::pattern_blocks;
export using device::pattern_budget;
export using device::pattern_warps;
export using device::pattern_word;
export using device::region_bytes;
export using device::region_chunks;
export using device::slice;
export using device::slice_blocks;
export using device::slices;
export using device::tallies;

/** The digest's state between blocks, and the state a pass starts from. */
export using State = device::sha256::State;
export inline constexpr State initial_state = device::sha256::initial;

/** The card's round constants, for the host to hold against its own. */
export inline constexpr auto const& round_constants = device::sha256::round_constants;

}

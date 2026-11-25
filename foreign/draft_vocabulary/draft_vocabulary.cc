// SPDX-License-Identifier: AGPL-3.0-only
export module engine.draft_vocabulary;

import std;

namespace engine::draft_vocabulary {

// The draft vocabulary decides how every card orders its lm_head rows, so it is part of the one model the engine runs
// and is built in rather than found at run time.
constexpr unsigned char list[] = {
#embed "draft_vocab_32k.bin"
};

/** data/draft_vocab_32k.bin as built in, readable by the build: little-endian 32-bit ids. Program-lifetime storage. */
export inline constexpr std::span<unsigned char const, sizeof(list)> bytes{list};

}

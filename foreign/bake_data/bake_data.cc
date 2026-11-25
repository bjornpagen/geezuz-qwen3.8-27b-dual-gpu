// SPDX-License-Identifier: AGPL-3.0-only
export module engine.bake_data;

import std;

namespace engine::bake_data {

// The baker refuses any checkpoint file whose SHA-256 is not the published one, so the published list is part of the
// baker, not an input it could be handed.
constexpr char digests[] = {
#embed "sha256.txt"
};

/** data/checkpoint/sha256.txt as built in: `<64 hex digits>  <file>` lines. Program-lifetime storage. */
export inline constexpr std::string_view checkpoint_digests{digests, sizeof(digests)};

}

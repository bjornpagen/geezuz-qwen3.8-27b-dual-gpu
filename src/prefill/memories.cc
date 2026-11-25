// SPDX-License-Identifier: AGPL-3.0-only
//
// What a prefill chunk reads and writes that the decode engine owns: each card's image, kv pool, block tables, slot
// state images, rope table and the row the prompt's end reads, and the embedding table both cards map. The decode
// side builds it and hands it to each run; prefill never keeps it.
export module engine.prefill:memories;

import std;
import engine.model;
import engine.prefill_device;

namespace engine::prefill {

export using Address = prefill_device::Address;

/** One card's memories, as device addresses on that card. */
export struct CardMemories {
	/** The card's baked image, as the schedule lays it out. */
	Address image;
	/** The kv pool's blocks. */
	Address pool;
	/** The slots' block tables, u32 [slots][kv_table_blocks]. */
	Address tables;
	/** The slots' deltanet state images, slot s at s * slot_state_bytes. */
	Address states;
	/** The rope table every attention kernel reads (`attn::rope_table_bytes`), filled before any chunk runs. */
	Address rotations;
	/** bf16 [hidden]: where the prompt end's tail reads the final norm of the prompt's last row. */
	Address prompt;

	[[nodiscard]] constexpr auto operator==(CardMemories const&) const -> bool = default;
};

/** Both cards' memories, in rank order, and the host image's embedding table both cards map. */
export struct Memories {
	Address embedding;
	std::array<CardMemories, model::cards> cards;

	[[nodiscard]] constexpr auto operator==(Memories const&) const -> bool = default;
};

}

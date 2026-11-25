// SPDX-License-Identifier: AGPL-3.0-only
//
// The kv block layout, the shared tiles and the groups: every element has one place, a chunk's plane is one contiguous
// tile in exactly the swizzle `ldmatrix` reads, a block's prefix is a prefix of each plane, the weight tiles read
// without bank conflicts, and the groups tile the chunks.
import std;
import engine.attn;
import engine.model;
import engine.prims;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace prims = engine::device::prims;

using attn::Count;
using attn::Kind;

inline constexpr std::array<Kind, 2> kinds = {Kind::Key, Kind::Value};

/** Every (layer, head, kind, position, feature) lands on its own two bytes of the block, and together they fill it. */
[[nodiscard]] auto elements_tile_the_block() -> bool {
	auto taken = std::vector<bool>(engine::model::kv_block_bytes / 2, false);
	auto sound = true;
	for (auto layer = Count{0}; layer < attn::kv_layers; ++layer) {
		for (auto head = Count{0}; head < attn::kv_heads; ++head) {
			for (auto const kind : kinds) {
				for (auto position = Count{0}; position < attn::block_positions; ++position) {
					for (auto feature = Count{0}; feature < attn::head_dim; ++feature) {
						auto const offset = attn::element_offset(layer, head, kind, position, feature);
						sound = sound && offset % 2 == 0 && offset < engine::model::kv_block_bytes && !taken.at(offset / 2);
						if (offset < engine::model::kv_block_bytes) {
							taken.at(offset / 2) = true;
						}
					}
				}
			}
		}
	}
	return sound && std::ranges::all_of(taken, std::identity{});
}

/** A chunk of a plane is 16 KiB at `tile_offset`, laid out exactly as the swizzled tile `ldmatrix` reads. */
[[nodiscard]] auto chunks_are_swizzled_tiles() -> bool {
	auto sound = true;
	for (auto layer = Count{0}; layer < attn::kv_layers; ++layer) {
		for (auto head = Count{0}; head < attn::kv_heads; ++head) {
			for (auto const kind : kinds) {
				for (auto chunk = Count{0}; chunk < 4; ++chunk) {
					auto const tile = attn::tile_offset(layer, head, kind, chunk);
					for (auto row = Count{0}; row < attn::chunk_positions; ++row) {
						auto const position = chunk * attn::chunk_positions + row;
						for (auto feature = Count{0}; feature < attn::head_dim; ++feature) {
							auto const offset = attn::element_offset(layer, head, kind, position, feature);
							auto const expected =
							    tile + static_cast<Count>(attn::KvTile::offset(static_cast<int>(row), static_cast<int>(feature / 8))) + (feature % 8) * 2;
							sound = sound && offset == expected;
						}
					}
					sound = sound && attn::table_entry(chunk) == chunk / 2;
				}
			}
		}
	}
	return sound;
}

/** Positions [0, n) of every plane are its first n rows: a block's prefix copies as one run per plane. */
[[nodiscard]] auto prefixes_are_plane_prefixes() -> bool {
	auto sound = true;
	for (auto layer = Count{0}; layer < attn::kv_layers; ++layer) {
		for (auto head = Count{0}; head < attn::kv_heads; ++head) {
			for (auto const kind : kinds) {
				auto const plane = attn::plane_offset(layer, head, kind);
				for (auto position = Count{0}; position < attn::block_positions; ++position) {
					auto const row = attn::row_offset(layer, head, kind, position);
					sound = sound && row == plane + position * attn::row_bytes;
					for (auto feature = Count{0}; feature < attn::head_dim; ++feature) {
						auto const offset = attn::element_offset(layer, head, kind, position, feature);
						sound = sound && offset >= row && offset < row + attn::row_bytes;
					}
				}
			}
		}
	}
	return sound;
}

/** The weight tiles: every (row, chunk) its own 16 bytes, and each 8-row `ldmatrix` phase on 8 distinct bank groups. */
[[nodiscard]] auto weight_tiles_are_conflict_free() -> bool {
	constexpr auto rows = 32U;
	auto seen = std::set<unsigned>{};
	auto sound = true;
	for (auto row = 0U; row < rows; ++row) {
		for (auto chunk = 0U; chunk < 4; ++chunk) {
			auto const offset = attn::weight_offset(row, chunk);
			sound = sound && offset % 16 == 0 && offset < rows * 64 && seen.insert(offset).second;
		}
	}
	for (auto first = 0U; first < rows; first += 8) {
		for (auto chunk = 0U; chunk < 4; ++chunk) {
			auto groups = 0U;
			for (auto row = first; row < first + 8; ++row) {
				groups |= 1U << static_cast<unsigned>(prims::fragment::bank_group(static_cast<int>(attn::weight_offset(row, chunk))));
			}
			sound = sound && groups == 0xffU;
		}
	}
	return sound;
}

/** Every chunk lies in the group `group_of` names, and the groups are 8 chunks near the start and 64 beyond. */
[[nodiscard]] auto groups_tile_the_chunks() -> bool {
	auto sound = attn::groups_covering(attn::max_chunks) == attn::max_groups && attn::group_begin(0) == 0;
	for (auto chunk = Count{0}; chunk < attn::max_chunks; ++chunk) {
		auto const group = attn::group_of(chunk);
		sound = sound && attn::group_begin(group) <= chunk && chunk < attn::group_begin(group + 1);
		sound = sound && attn::groups_covering(chunk + 1) == group + 1;
	}
	for (auto group = Count{0}; group < attn::max_groups; ++group) {
		auto const length = attn::group_begin(group + 1) - attn::group_begin(group);
		sound = sound && length == (attn::group_begin(group) < attn::near_chunks ? attn::near_group_chunks : attn::far_group_chunks);
	}
	for (auto chunk = Count{0}; chunk <= attn::max_chunks; ++chunk) {
		auto const first = attn::first_group_from(chunk);
		sound = sound && attn::group_begin(first) >= chunk && (first == 0 || attn::group_begin(first - 1) < chunk);
	}
	return sound;
}

static_assert(prims::fragment::conflict_free<attn::KvTile>());
static_assert(prims::fits(attn::core_budget<attn::RowTile<1>>()) && prims::fits(attn::core_budget<attn::RowTile<5>>()));
static_assert(prims::fits(attn::core_budget<attn::PromptTile>()));

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every kv element has its own two bytes and together they fill the block", .holds = elements_tile_the_block()},
	    engine::testing::Case{.name = "a chunk of a plane is one 16 KiB tile in the ldmatrix swizzle", .holds = chunks_are_swizzled_tiles()},
	    engine::testing::Case{.name = "a block's prefix is a prefix of each plane", .holds = prefixes_are_plane_prefixes()},
	    engine::testing::Case{.name = "the weight part tiles are a bijection and conflict-free", .holds = weight_tiles_are_conflict_free()},
	    engine::testing::Case{.name = "the groups tile the chunks, 8 near the start and 64 beyond", .holds = groups_tile_the_chunks()},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.kv;
import engine.kv_fake;
import engine.model;
import engine.testing;

namespace {

namespace kv = engine::kv;
namespace model = engine::model;

using kv::Count;
using kv::Geometry;
using kv::Slot;

constexpr auto tiny = Geometry{.block_tokens = 4, .pool_blocks = 40, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 5};

/** The device indices of every block the slot tables hold, with repeats. */
template<Geometry G>
[[nodiscard]] auto table_indices(kv::BasicPool<G> const& pool) -> std::vector<kv::BlockIndex> {
	auto indices = std::vector<kv::BlockIndex>{};
	for (auto slot = Count{0}; slot < model::slots; ++slot) {
		for (auto entry = Count{0}; entry < pool.table_size(Slot::at(slot)); ++entry) {
			indices.push_back(pool.index(pool.entry(Slot::at(slot), entry)));
		}
	}
	return indices;
}

/**
 * Slots grow privately in random steps until the pool refuses; then random slots shrink, and it starts again. Each
 * time the pool runs dry, the tables hold every block of the pool exactly once (the free list gave each block out
 * once and kept none back), and the refusal changed nothing; at the end everything comes back.
 */
[[nodiscard]] auto drains_to_every_block_once(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	auto pool = std::make_unique<kv::BasicPool<tiny>>();
	auto device = std::make_unique<engine::kv_fake::FakeDevice<tiny>>();
	auto lengths = std::array<Count, model::slots>{};
	auto holds = true;
	for (auto round = 0; round < 40 && holds; ++round) {
		while (true) {
			auto const slot = engine::core::narrow<Count>(rng() % model::slots);
			auto const until = std::min(lengths[slot] + 1 + engine::core::narrow<Count>(rng() % 9), kv::table_positions(tiny));
			auto const extend = std::array{kv::Extend{.slot = Slot::at(slot), .length = lengths[slot], .until = until}};
			auto const before = table_indices(*pool);
			auto const done = pool->ensure(*device, extend);
			if (!done) {
				auto indices = table_indices(*pool);
				std::ranges::sort(indices);
				auto const distinct = std::ranges::adjacent_find(indices) == indices.end();
				holds = holds && before == table_indices(*pool) && distinct && done.error().free == pool->free_blocks() &&
				        indices.size() + pool->free_blocks() == tiny.pool_blocks && done.error().needed > done.error().free;
				break;
			}
			lengths[slot] = until;
		}
		for (auto slot = Count{0}; slot < model::slots; ++slot) {
			if (rng() % 2 == 0) {
				lengths[slot] = lengths[slot] == 0 ? 0 : engine::core::narrow<Count>(rng() % lengths[slot]);
				pool->trim(Slot::at(slot), lengths[slot]);
			}
		}
	}
	for (auto slot = Count{0}; slot < model::slots; ++slot) {
		pool->trim(Slot::at(slot), 0);
	}
	return holds && pool->free_blocks() == tiny.pool_blocks && device->violations().empty();
}

/** In the model's pool, one slot takes the longest sequence the model reads, and gives every block back. */
[[nodiscard]] auto longest_sequence_fits_the_model_pool() -> bool {
	auto pool = std::make_unique<kv::Pool>();
	auto device = std::make_unique<engine::kv_fake::FakeDevice<kv::model_geometry>>();
	auto const slot = Slot::at(0);
	auto const extend = std::array{kv::Extend{.slot = slot, .length = 0, .until = model::max_positions}};
	auto const done = pool->ensure(*device, extend);
	auto const taken = pool->table_size(slot);
	auto const free = pool->free_blocks();
	pool->trim(slot, 0);
	return done.has_value() && taken == model::kv_table_blocks && free == model::pool_blocks - model::kv_table_blocks &&
	       pool->free_blocks() == model::pool_blocks && device->violations().empty();
}

/** In the model's pool, the snapshot cap is reached by count while blocks remain free, and the cap is the share. */
[[nodiscard]] auto snapshot_cap_holds_in_the_model_pool() -> bool {
	auto pool = std::make_unique<kv::Pool>();
	auto device = std::make_unique<engine::kv_fake::FakeDevice<kv::model_geometry>>();
	auto taken = std::vector<kv::SnapshotHandle>{};
	while (auto const snapshot = pool->save_snapshot(*device, Slot::at(taken.size() % model::slots))) {
		taken.push_back(*snapshot);
	}
	auto const refused = pool->save_snapshot(*device, Slot::at(0));
	auto const held = model::pool_blocks - pool->free_blocks();
	for (auto const snapshot : taken) {
		pool->drop_snapshot(snapshot);
	}
	return taken.size() == model::snapshot_capacity && !refused &&
	       refused.error() == kv::SnapshotRefused{.shortage = kv::SnapshotShortage::Capacity} && held == model::snapshot_cap_blocks &&
	       pool->free_blocks() == model::pool_blocks;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "a drained pool has handed out every block exactly once, and refusing changed nothing (seeds 1-20)",
	        .holds = std::ranges::all_of(std::views::iota(std::uint64_t{1}, std::uint64_t{21}), drains_to_every_block_once),
	    },
	    engine::testing::Case{
	        .name = "one slot holds the model's longest sequence in the model's pool and gives every block back",
	        .holds = longest_sequence_fits_the_model_pool(),
	    },
	    engine::testing::Case{
	        .name = "the model's pool takes snapshot_capacity snapshots, exactly snapshot_cap_blocks blocks, then refuses",
	        .holds = snapshot_cap_holds_in_the_model_pool(),
	    },
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.kv;
import engine.kv_fake;
import engine.model;
import engine.sched;

// An outcome with no cycle in flight is a pipeline that lost track of the cards: the scheduler aborts rather than
// commit tokens no slot was asked for.
auto main() -> int {
	namespace sched = engine::sched;
	constexpr auto geometry = engine::kv::Geometry{.block_tokens = 4, .pool_blocks = 64, .table_blocks = 16, .snapshot_blocks = 2, .snapshot_capacity = 8};
	auto scheduler = std::make_unique<sched::BasicScheduler<geometry, engine::model::RowBudget::ThreeTiles, sched::tuning>>();
	auto device = std::make_unique<engine::kv_fake::FakeDevice<geometry>>();
	auto const events = scheduler->settle(*device, sched::Done{sched::BatchDone{.rows = {}, .took = {}}});
	return events.empty() ? 0 : 1;
}

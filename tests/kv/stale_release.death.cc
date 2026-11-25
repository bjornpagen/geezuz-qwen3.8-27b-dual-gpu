// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.kv;
import engine.kv_fake;

// Giving back a reference to a block that has since been freed is a bug in the holder: the pool aborts instead of
// letting the count of whatever block now holds that index go wrong.
auto main() -> int {
	namespace kv = engine::kv;
	auto pool = std::make_unique<kv::Pool>();
	auto device = std::make_unique<engine::kv_fake::FakeDevice<kv::model_geometry>>();
	auto const slot = kv::Slot::at(0);
	auto const extend = std::array{kv::Extend{.slot = slot, .length = 0, .until = 1}};
	if (!pool->ensure(*device, extend)) {
		return 0;
	}
	auto const block = pool->retain(slot, 0);
	pool->release(block);
	pool->trim(slot, 0);
	pool->release(block);
	return 0;
}

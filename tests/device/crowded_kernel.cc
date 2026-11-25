// SPDX-License-Identifier: AGPL-3.0-only
//
// A co-resident kernel whose grid is larger than the cards hold at its budget's occupancy: the driver would refuse the
// cooperative launch at run time, so the declaration must not compile.
import std;
import engine.device;
import engine.prims;
import engine.ptx;

namespace {

struct Crowded {
	static constexpr engine::ptx::Image image = engine::ptx::abi;
	static constexpr std::string_view entry = "crowded";
	static constexpr engine::device::Dim grid = {.x = static_cast<std::uint32_t>(engine::device::prims::sm86.sms) + 1, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = {
	    .threads = 256, .registers_per_thread = 64, .shared_bytes = 0, .blocks_per_sm = 1};
	static constexpr engine::device::Residency residency = engine::device::Residency::CoResident;
	using Params = std::uint64_t;
};

}

auto main() -> int {
	return sizeof(engine::device::Program<Crowded>) > 0 ? 0 : 1;
}

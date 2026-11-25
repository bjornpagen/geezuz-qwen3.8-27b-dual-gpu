// SPDX-License-Identifier: AGPL-3.0-only
//
// Dropping a transfer while its copy is still running would free buffers the copy engine is reading: it must abort.
import std;
import engine.device;
import engine.driver_double;

auto main() -> int {
	auto const driver = engine::driver_double::Fake{engine::driver_double::Machine{.pending_queries = 1'000}};
	auto const opened = engine::device::Devices::open(driver.driver());
	if (!opened) {
		return 0;
	}
	auto const& context = opened->context(engine::device::Rank::Zero);
	auto staging = context.allocate_mapped<std::uint32_t, 64>();
	auto array = context.allocate<std::uint32_t, 64>();
	auto const stream = context.create_stream();
	if (!staging || !array || !stream) {
		return 0;
	}
	{
		auto const transfer = context.upload(*stream, std::move(*array), std::move(*staging), 0);
	}
	return 0;
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.prims;
import engine.testing;

namespace {

namespace prims = engine::device::prims;

using prims::Budget;
using prims::Fit;

/**
 * The oracle: make `blocks_per_sm` blocks resident one warp at a time, each block placing its warps round-robin from
 * scheduler 0, and charge every resource as it is taken. The budget fits when nothing is overdrawn.
 */
[[nodiscard]] auto resident(Budget const& budget) -> bool {
	if (budget.threads < 32 || budget.threads % 32 != 0 || budget.threads > 1024) {
		return false;
	}
	if (budget.registers_per_thread < 1 || budget.registers_per_thread > 255) {
		return false;
	}
	if (budget.shared_bytes < 0 || budget.shared_bytes > 99 * 1024 || budget.blocks_per_sm < 1 || budget.blocks_per_sm > 16) {
		return false;
	}
	auto registers = std::array<int, 4>{};
	auto threads = 0;
	auto shared = 0;
	for (auto block = 0; block < budget.blocks_per_sm; ++block) {
		for (auto warp = 0; warp < budget.threads / 32; ++warp) {
			// Registers are granted per warp in blocks of 256.
			registers.at(static_cast<std::size_t>(warp % 4)) += (budget.registers_per_thread * 32 + 255) / 256 * 256;
			threads += 32;
		}
		shared += (budget.shared_bytes + 127) / 128 * 128 + 1024;
	}
	return threads <= 1536 && shared <= 100 * 1024 && std::ranges::all_of(registers, [](int used) { return used <= 16'384; });
}

/** Every thread count, occupancy and register count, at shared sizes around each block count's share of the SM. */
[[nodiscard]] auto fit_agrees_with_residency_everywhere() -> bool {
	auto holds = true;
	for (auto threads = 0; threads <= 1056; threads += 16) {
		for (auto blocks = 0; blocks <= 17; ++blocks) {
			for (auto registers = 0; registers <= 256; ++registers) {
				auto const share = blocks == 0 ? 0 : 100 * 1024 / blocks - 1024;
				for (auto const shared : {0, 1, share - 1, share, share + 1, 99 * 1024, 99 * 1024 + 1}) {
					auto const budget = Budget{.threads = threads, .registers_per_thread = registers, .shared_bytes = shared, .blocks_per_sm = blocks};
					holds = holds && (prims::check(budget) == Fit::Fits) == resident(budget);
				}
			}
		}
	}
	return holds;
}

/** The cap is exactly the largest register count that still fits, wherever threads and occupancy fit at all. */
[[nodiscard]] auto register_cap_is_the_largest_fitting_count() -> bool {
	auto holds = true;
	for (auto threads = 32; threads <= 1024; threads += 32) {
		for (auto blocks = 1; blocks * threads <= 1536 && blocks <= 16; ++blocks) {
			auto const at = [&](int registers) {
				return Budget{.threads = threads, .registers_per_thread = registers, .shared_bytes = 0, .blocks_per_sm = blocks};
			};
			auto const cap = prims::register_cap(at(1));
			holds = holds && resident(at(cap)) && (cap == 255 || !resident(at(cap + 1)));
		}
	}
	return holds;
}

// The budgets the hardware study names: one resident block of 288 threads keeps 224 registers per thread only when its
// nine warps share the four schedulers evenly, so the conservative placement grants 168.
static_assert(prims::fits({.threads = 256, .registers_per_thread = 255, .shared_bytes = 99 * 1024, .blocks_per_sm = 1}));
static_assert(prims::check({.threads = 288, .registers_per_thread = 224, .shared_bytes = 0, .blocks_per_sm = 1}) == Fit::RegisterFile);
static_assert(prims::fits({.threads = 288, .registers_per_thread = 168, .shared_bytes = 0, .blocks_per_sm = 1}));
static_assert(prims::check({.threads = 256, .registers_per_thread = 32, .shared_bytes = 50 * 1024, .blocks_per_sm = 2}) == Fit::SharedPerSm);

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "a budget fits exactly when its blocks can all be made resident on one sm_86 SM",
	        .holds = fit_agrees_with_residency_everywhere(),
	    },
	    engine::testing::Case{
	        .name = "the register cap is the largest per-thread count the declared occupancy can hold",
	        .holds = register_cap_is_the_largest_fitting_count(),
	    },
	};
	return engine::testing::report(cases);
}

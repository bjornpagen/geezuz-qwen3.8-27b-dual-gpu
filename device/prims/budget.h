// SPDX-License-Identifier: AGPL-3.0-only
//
// What one sm_86 SM can hold, held once, and the resource budget every kernel declares against it. Pure constexpr with
// no device dependency: the kernel `static_assert`s its budget, and the host's typed launch function (through
// `engine.prims`) launches with exactly the budget's threads and shared bytes, so neither side can disagree.
#pragma once

namespace engine::device::prims {

/** The limits of one sm_86 SM (rtx 3090: 82 of GA102's 84 enabled). Counts, bytes and registers are 32-bit words. */
struct Machine {
	int sms;
	int warp_threads;
	int max_threads_per_block;
	int max_threads_per_sm;
	int max_warps_per_sm;
	int max_blocks_per_sm;
	int schedulers_per_sm;
	int registers_per_sm;
	int registers_per_scheduler;
	int max_registers_per_thread;
	int register_unit_per_warp;
	int shared_per_sm;
	int max_shared_per_block;
	int shared_reserved_per_block;
	int shared_unit;
	int named_barriers;
};

// A warp lives on one scheduler and takes its registers from that scheduler's quarter of the file, in units of 256
// per warp. Which scheduler a warp lands on is not documented, so the check assumes the worst placement consistent
// with each block spreading its own warps evenly: every block starting at the same scheduler. Each resident block
// also costs 1 KiB of shared memory the system reserves, which is why one block may hold at most 99 of the SM's
// 100 KiB.
inline constexpr Machine sm86 = {
    .sms = 82,
    .warp_threads = 32,
    .max_threads_per_block = 1024,
    .max_threads_per_sm = 1536,
    .max_warps_per_sm = 48,
    .max_blocks_per_sm = 16,
    .schedulers_per_sm = 4,
    .registers_per_sm = 65'536,
    .registers_per_scheduler = 16'384,
    .max_registers_per_thread = 255,
    .register_unit_per_warp = 256,
    .shared_per_sm = 100 * 1024,
    .max_shared_per_block = 99 * 1024,
    .shared_reserved_per_block = 1024,
    .shared_unit = 128,
    .named_barriers = 16,
};

/**
 * One kernel's resources: threads per block, the registers each thread may use, the block's shared memory in bytes
 * (the whole of it, a compile-time constant the launch passes as the dynamic size), and the blocks it keeps resident
 * per SM. The kernel spells `launch_bounds(budget.threads, budget.blocks_per_sm)`, so ptxas caps its registers at
 * what that occupancy leaves, which `fits` proves is at least `registers_per_thread`; on linux the ptxas gate checks
 * the registers ptxas used against the same bound and fails on any spill.
 */
struct Budget {
	int threads;
	int registers_per_thread;
	int shared_bytes;
	int blocks_per_sm;
};

/** The first limit a budget breaks, or `Fits`; `static_assert(check(b) == Fit::Fits)` names it in the diagnostic. */
enum class Fit : unsigned char {
	Fits,
	ThreadsNotWholeWarps,
	ThreadsPerBlock,
	BlocksPerSm,
	ThreadsPerSm,
	RegistersPerThread,
	RegisterFile,
	SharedPerBlock,
	SharedPerSm,
};

[[nodiscard]] constexpr auto round_up(int value, int unit) -> int {
	return (value + unit - 1) / unit * unit;
}

[[nodiscard]] constexpr auto warps(Budget const& budget) -> int {
	return budget.threads / sm86.warp_threads;
}

/** The registers one warp of the kernel occupies: its threads' registers rounded up to the allocation unit. */
[[nodiscard]] constexpr auto registers_per_warp(Budget const& budget) -> int {
	return round_up(budget.registers_per_thread * sm86.warp_threads, sm86.register_unit_per_warp);
}

/**
 * The most warps one scheduler may hold when `blocks_per_sm` blocks are resident, each block spreading its warps
 * round-robin from the same scheduler.
 */
[[nodiscard]] constexpr auto warps_per_scheduler(Budget const& budget) -> int {
	return budget.blocks_per_sm * (round_up(warps(budget), sm86.schedulers_per_sm) / sm86.schedulers_per_sm);
}

/**
 * The registers per thread the occupancy leaves, which is what ptxas caps the kernel at under its launch bounds:
 * the scheduler's quarter of the file over its warps, in whole allocation units, at most 255.
 */
[[nodiscard]] constexpr auto register_cap(Budget const& budget) -> int {
	int const units = sm86.registers_per_scheduler / warps_per_scheduler(budget) / sm86.register_unit_per_warp;
	int const cap = units * sm86.register_unit_per_warp / sm86.warp_threads;
	return cap < sm86.max_registers_per_thread ? cap : sm86.max_registers_per_thread;
}

/** The shared memory `blocks_per_sm` resident blocks take from the SM, system reservation included. */
[[nodiscard]] constexpr auto shared_per_sm(Budget const& budget) -> int {
	return budget.blocks_per_sm * (round_up(budget.shared_bytes, sm86.shared_unit) + sm86.shared_reserved_per_block);
}

[[nodiscard]] constexpr auto check(Budget const& budget) -> Fit {
	if (budget.threads < sm86.warp_threads || budget.threads % sm86.warp_threads != 0) {
		return Fit::ThreadsNotWholeWarps;
	}
	if (budget.threads > sm86.max_threads_per_block) {
		return Fit::ThreadsPerBlock;
	}
	if (budget.blocks_per_sm < 1 || budget.blocks_per_sm > sm86.max_blocks_per_sm) {
		return Fit::BlocksPerSm;
	}
	if (budget.threads * budget.blocks_per_sm > sm86.max_threads_per_sm) {
		return Fit::ThreadsPerSm;
	}
	if (budget.registers_per_thread < 1 || budget.registers_per_thread > sm86.max_registers_per_thread) {
		return Fit::RegistersPerThread;
	}
	if (budget.registers_per_thread > register_cap(budget)) {
		return Fit::RegisterFile;
	}
	if (budget.shared_bytes < 0 || budget.shared_bytes > sm86.max_shared_per_block) {
		return Fit::SharedPerBlock;
	}
	if (shared_per_sm(budget) > sm86.shared_per_sm) {
		return Fit::SharedPerSm;
	}
	return Fit::Fits;
}

/** Whether `budget` fits one sm_86 SM at its declared occupancy. */
[[nodiscard]] consteval auto fits(Budget const& budget) -> bool {
	return check(budget) == Fit::Fits;
}

static_assert(sm86.max_warps_per_sm * sm86.warp_threads == sm86.max_threads_per_sm);
static_assert(sm86.registers_per_scheduler * sm86.schedulers_per_sm == sm86.registers_per_sm);
static_assert(sm86.max_shared_per_block + sm86.shared_reserved_per_block == sm86.shared_per_sm);

}

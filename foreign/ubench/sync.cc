// SPDX-License-Identifier: AGPL-3.0-only
//
// The synchronisation kernels as the host launches them.
export module engine.ubench_device:sync;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :shapes;
import :images;
import :names;
import :stream;

namespace engine::ubench_device {

namespace ub = engine::device::ubench;
namespace sync = engine::device::ubench::sync;

export using Cycles = driver::DeviceArray<unsigned long long, 1>;

export struct ClockKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = "ubench_clock";
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = sync::clock_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = sync::ClockParams;

	[[nodiscard]] static auto params(BlockClocks const& clocks) -> Params {
		return {.clocks = clocks.address()};
	}
};

export template<ub::Wait wait>
struct MbarrierKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = wait == ub::Wait::Backoff ? "ubench_mbarrier_backoff" : "ubench_mbarrier_spin";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = sync::pingpong_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = sync::CyclesParams;

	[[nodiscard]] static auto params(Cycles const& cycles) -> Params {
		return {.cycles = cycles.address()};
	}
};

export struct NamedBarrierKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = "ubench_named_barrier";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = sync::pingpong_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = sync::CyclesParams;

	[[nodiscard]] static auto params(Cycles const& cycles) -> Params {
		return {.cycles = cycles.address()};
	}
};

export using CopySource = driver::DeviceArray<ub::Chunk, 32>;

export struct CopyCompletionKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = "ubench_copy_completion";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = sync::copy_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = sync::CopyParams;

	[[nodiscard]] static auto params(CopySource const& source, Cycles const& cycles) -> Params {
		return {.source = source.address(), .cycles = cycles.address()};
	}
};

export using SleepReadings = driver::DeviceArray<sync::SleepReading, sync::sleep_kinds>;

export struct SleepGrainKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = "ubench_sleep_grain";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = sync::sleep_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = sync::SleepParams;

	[[nodiscard]] static auto params(SleepReadings const& readings) -> Params {
		return {.readings = readings.address()};
	}
};

export using HopScratch = driver::DeviceArray<sync::HopScratch, 1>;
export using HopReading = driver::DeviceArray<sync::HopReading, 1>;

/** The hop's two blocks wait on each other, so both must be resident at once. */
export template<ub::Wait wait>
struct HopKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = wait == ub::Wait::Backoff ? "ubench_hop_backoff" : "ubench_hop_spin";
	static constexpr driver::Dim grid = {.x = 2, .y = 1, .z = 1};
	static constexpr device::prims::Budget budget = sync::hop_budget;
	static constexpr driver::Residency residency = driver::Residency::CoResident;
	using Params = sync::HopParams;

	[[nodiscard]] static auto params(HopScratch const& scratch, HopReading const& reading) -> Params {
		return {.scratch = scratch.address(), .reading = reading.address()};
	}
};

export struct StageHandoffKernel {
	static constexpr ptx::Image image = sync_image;
	static constexpr std::string_view entry = "ubench_stage_handoff";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = sync::handoff_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = sync::CyclesParams;

	[[nodiscard]] static auto params(Cycles const& cycles) -> Params {
		return {.cycles = cycles.address()};
	}
};

}

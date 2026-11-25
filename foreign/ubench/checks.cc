// SPDX-License-Identifier: AGPL-3.0-only
//
// The primitives' self-check kernels as the bring-up drivers launch them. Every launch fact (parameter struct, budget,
// grid, buffer layout) is the prims component's own, from the self-check header the kernels compile.
export module engine.ubench_device:checks;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :images;
import :names;

namespace engine::ubench_device {

namespace selfcheck = engine::device::prims::selfcheck;
namespace prims = engine::device::prims;

/** A self-check's report, which the kernel expects zeroed. */
export using CheckReport = driver::DeviceArray<selfcheck::Report, 1>;

[[nodiscard]] constexpr auto blocks(unsigned count) -> driver::Dim {
	return {.x = count, .y = 1, .z = 1};
}

/** A self-check kernel that writes one report. */
template<std::string_view const& entry_, prims::Budget budget_, unsigned blocks_>
struct ReportingCheck {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = entry_;
	static constexpr driver::Dim grid = blocks(blocks_);
	static constexpr prims::Budget budget = budget_;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = selfcheck::ReportParams;

	[[nodiscard]] static auto params(CheckReport const& report) -> Params {
		return {.report = report.address()};
	}
};

inline constexpr std::string_view check_decode = "prims_check_decode";
inline constexpr std::string_view check_fragments = "prims_check_fragments";
inline constexpr std::string_view check_reduce = "prims_check_reduce";

/** The exact decode through `decode_word` and through the `mma`, every code. */
export using CheckDecodeKernel = ReportingCheck<check_decode, selfcheck::decode_budget, selfcheck::decode_blocks>;
/** The fragment maps against `ldmatrix` and `.trans` on a swizzled tile. */
export using CheckFragmentsKernel = ReportingCheck<check_fragments, selfcheck::fragment_budget, selfcheck::fragment_blocks>;
/** The fixed-order float butterfly and the integer `redux.sync` forms. */
export using CheckReduceKernel = ReportingCheck<check_reduce, selfcheck::reduce_budget, selfcheck::reduce_blocks>;

/** Every (function, block) digest the transcendental check folds, which the kernel expects zeroed. */
export using Digests = driver::DeviceArray<unsigned long long, static_cast<std::size_t>(selfcheck::functions) * selfcheck::digest_blocks>;

export struct DigestKernel {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = "prims_digest_transcendentals";
	static constexpr driver::Dim grid = blocks(selfcheck::digest_blocks);
	static constexpr prims::Budget budget = selfcheck::digest_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = selfcheck::DigestParams;

	[[nodiscard]] static auto params(Digests const& digests) -> Params {
		return {.digests = digests.address()};
	}
};

export using MmaProbes = driver::DeviceArray<selfcheck::MmaProbe, selfcheck::mma_probes>;
export using MmaResults = driver::DeviceArray<prims::Accumulator, static_cast<std::size_t>(selfcheck::mma_probes) * 32>;

/** U6: one host-crafted probe per warp through `mma.sync`, each lane's D written back. */
export struct ProbeMmaKernel {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = "prims_probe_mma";
	static constexpr driver::Dim grid = blocks(selfcheck::mma_blocks);
	static constexpr prims::Budget budget = selfcheck::mma_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = selfcheck::MmaParams;

	[[nodiscard]] static auto params(MmaProbes const& probes, MmaResults const& results) -> Params {
		return {.probes = probes.address(), .results = results.address()};
	}
};

export using RingSource = driver::DeviceArray<prims::Chunk, selfcheck::stage_ring_source_chunks>;
export using RingElapsed = driver::DeviceArray<unsigned long long, selfcheck::stage_ring_blocks>;

/** U8's ring under the hardware: a producer warp's copies through a 4-stage ring, every word checked. */
export struct StageRingCheckKernel {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = "prims_check_stage_ring";
	static constexpr driver::Dim grid = blocks(selfcheck::stage_ring_blocks);
	static constexpr prims::Budget budget = selfcheck::stage_ring_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = selfcheck::StageRingParams;

	[[nodiscard]] static auto params(RingSource const& source, CheckReport const& report, RingElapsed const& elapsed) -> Params {
		return {.source = source.address(), .report = report.address(), .elapsed = elapsed.address()};
	}
};

export using OrderScratch = driver::DeviceArray<selfcheck::OrderScratch, 1>;

/** Memory order across SMs and the remaining primitives; its grid barrier needs every block resident at once. */
export struct OrderCheckKernel {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = "prims_check_order";
	static constexpr driver::Dim grid = blocks(selfcheck::order_blocks);
	static constexpr prims::Budget budget = selfcheck::order_budget;
	static constexpr driver::Residency residency = driver::Residency::CoResident;
	using Params = selfcheck::OrderParams;

	[[nodiscard]] static auto params(OrderScratch const& scratch, CheckReport const& report) -> Params {
		return {.scratch = scratch.address(), .report = report.address()};
	}
};

}

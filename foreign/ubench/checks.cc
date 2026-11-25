// SPDX-License-Identifier: AGPL-3.0-only
//
// The primitives' self-check kernels as the bring-up drivers launch them. Their budgets come from the records their
// image carries, so the host never restates what the kernels declare. Only the kernels that take one parameter can be
// launched by the engine's typed launch, which passes exactly one parameter struct; `prims_probe_mma`,
// `prims_check_stage_ring` and `prims_check_order` take several and are not declared here.
export module engine.ubench_device:checks;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :images;
import :names;
import :stream;

namespace engine::ubench_device {

namespace selfcheck = engine::device::prims::selfcheck;

/** A self-check's report, which the kernel expects zeroed. */
export using CheckReport = driver::DeviceArray<selfcheck::Report, 1>;

/** The one parameter of a reporting self-check: where its report is. */
export struct ReportParams {
	std::uint64_t report;
};

/** A self-check kernel that writes one report from one block. */
template<std::string_view const& entry_>
struct ReportingCheck {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = entry_;
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = recorded_budget(prims_check_image, entry_);
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = ReportParams;

	[[nodiscard]] static auto params(CheckReport const& report) -> Params {
		return {.report = report.address()};
	}
};

inline constexpr std::string_view check_decode = "prims_check_decode";
inline constexpr std::string_view check_fragments = "prims_check_fragments";
inline constexpr std::string_view check_reduce = "prims_check_reduce";

/** The exact decode through `decode_word` and through the `mma`, every code. */
export using CheckDecodeKernel = ReportingCheck<check_decode>;
/** The fragment maps against `ldmatrix` and `.trans` on a swizzled tile. */
export using CheckFragmentsKernel = ReportingCheck<check_fragments>;
/** The fixed-order float butterfly and the integer `redux.sync` forms. */
export using CheckReduceKernel = ReportingCheck<check_reduce>;

/** Every (function, block) digest the transcendental check folds, which the kernel expects zeroed. */
export using Digests = driver::DeviceArray<unsigned long long, static_cast<std::size_t>(selfcheck::functions) * selfcheck::digest_blocks>;

export struct DigestParams {
	std::uint64_t digests;
};

export struct DigestKernel {
	static constexpr ptx::Image image = prims_check_image;
	static constexpr std::string_view entry = "prims_digest_transcendentals";
	static constexpr driver::Dim grid = {.x = selfcheck::digest_blocks, .y = 1, .z = 1};
	static constexpr device::prims::Budget budget = selfcheck::digest_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = DigestParams;

	[[nodiscard]] static auto params(Digests const& digests) -> Params {
		return {.digests = digests.address()};
	}
};

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight-stream kernels as the host launches them, and the sweeps the image holds a kernel for.
export module engine.ubench_device:stream;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :shapes;
import :images;
import :names;

namespace engine::ubench_device {

namespace ub = engine::device::ubench;
namespace stream = engine::device::ubench::stream;

/** The stream every weight-stream kernel reads: each SM's run of whole tiles, one card's worth. */
export using StreamSource = driver::DeviceArray<ub::Chunk, stream::chunks>;
export using BlockClocks = driver::DeviceArray<ub::Clocks, ub::sms>;
export using BlockRates = driver::DeviceArray<ub::Rate, ub::sms>;

export inline constexpr driver::Dim every_sm = {.x = static_cast<std::uint32_t>(ub::sms), .y = 1, .z = 1};
export inline constexpr driver::Dim one_block = {.x = 1, .y = 1, .z = 1};

export template<class Shape>
struct ReadKernel {
	static constexpr ptx::Image image = stream_image;
	static constexpr EntryName name = Shape::order == stream::TileOrder::Contiguous
	                                      ? entry_name("ubench_read_p", Shape::producer_warps, "_d", Shape::depth_kib)
	                                      : entry_name("ubench_read_p", Shape::producer_warps, "_d", Shape::depth_kib, "_row_major");
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = Shape::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = stream::ReadParams;

	[[nodiscard]] static auto params(StreamSource const& source) -> Params {
		return {.source = source.address()};
	}
};

/** The read sweep: producer warps by in-flight KiB, every point where a stage holds a whole issue per producer. */
export using ReadSweep = std::tuple<
    stream::Read<1, 2, stream::TileOrder::Contiguous>, stream::Read<1, 4, stream::TileOrder::Contiguous>,
    stream::Read<1, 8, stream::TileOrder::Contiguous>, stream::Read<1, 16, stream::TileOrder::Contiguous>,
    stream::Read<1, 32, stream::TileOrder::Contiguous>, stream::Read<1, 64, stream::TileOrder::Contiguous>,
    stream::Read<2, 4, stream::TileOrder::Contiguous>, stream::Read<2, 8, stream::TileOrder::Contiguous>,
    stream::Read<2, 16, stream::TileOrder::Contiguous>, stream::Read<2, 32, stream::TileOrder::Contiguous>,
    stream::Read<2, 64, stream::TileOrder::Contiguous>, stream::Read<4, 8, stream::TileOrder::Contiguous>,
    stream::Read<4, 16, stream::TileOrder::Contiguous>, stream::Read<4, 32, stream::TileOrder::Contiguous>,
    stream::Read<4, 64, stream::TileOrder::Contiguous>, stream::Read<8, 16, stream::TileOrder::Contiguous>,
    stream::Read<8, 32, stream::TileOrder::Contiguous>, stream::Read<8, 64, stream::TileOrder::Contiguous>>;

/** The canonical stream stored row-major: the layout comparison's other half. */
export using RowMajorRead = stream::Read<1, 64, stream::TileOrder::RowMajor>;

export template<class Shape>
struct StallKernel {
	static constexpr ptx::Image image = stream_image;
	static constexpr EntryName name = entry_name("ubench_stall_r", Shape::ring_kib, "_t", Shape::stall_us);
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = Shape::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = stream::ReadParams;

	[[nodiscard]] static auto params(StreamSource const& source) -> Params {
		return {.source = source.address()};
	}
};

/** The stall sweep: ring KiB per SM, each by every stall; the zero stall is each ring's baseline. */
export template<int ring_kib>
using StallRow = std::tuple<stream::Stall<ring_kib, 0>, stream::Stall<ring_kib, 4>, stream::Stall<ring_kib, 8>, stream::Stall<ring_kib, 12>>;
export using StallSweep = std::tuple<StallRow<32>, StallRow<48>, StallRow<64>, StallRow<80>>;

export struct SustainKernel {
	static constexpr ptx::Image image = stream_image;
	static constexpr std::string_view entry = "ubench_sustain";
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = stream::Sustain::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = stream::SustainParams;

	[[nodiscard]] static auto params(StreamSource const& source, BlockClocks const& clocks) -> Params {
		return {.source = source.address(), .clocks = clocks.address()};
	}
};

export using Activations = driver::DeviceArray<ub::Chunk, stream::Reuse<stream::Priorities::Split>::activation_chunks>;

export template<stream::Priorities priorities>
struct ReuseKernel {
	using Shape = stream::Reuse<priorities>;
	static constexpr ptx::Image image = stream_image;
	static constexpr std::string_view entry = priorities == stream::Priorities::Split ? "ubench_reuse_split" : "ubench_reuse_uniform";
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = Shape::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = stream::ReuseParams;

	[[nodiscard]] static auto params(StreamSource const& source, Activations const& activations, BlockRates const& rates) -> Params {
		return {.source = source.address(), .activations = activations.address(), .rates = rates.address()};
	}
};

export using L2Region = driver::DeviceArray<ub::Chunk, ub::l2::region_chunks>;
export using SegmentRates = driver::DeviceArray<ub::Rate, ub::l2::segments>;
inline constexpr driver::Dim every_segment = {.x = static_cast<std::uint32_t>(ub::l2::segments), .y = 1, .z = 1};

export struct L2PrefetchKernel {
	static constexpr ptx::Image image = stream_image;
	static constexpr std::string_view entry = "ubench_l2_prefetch";
	static constexpr driver::Dim grid = every_segment;
	static constexpr device::prims::Budget budget = ub::l2::prefetch_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = ub::l2::PrefetchParams;

	[[nodiscard]] static auto params(L2Region const& region) -> Params {
		return {.region = region.address()};
	}
};

export struct L2ProbeKernel {
	static constexpr ptx::Image image = stream_image;
	static constexpr std::string_view entry = "ubench_l2_probe";
	static constexpr driver::Dim grid = every_segment;
	static constexpr device::prims::Budget budget = ub::l2::probe_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = ub::l2::ProbeParams;

	[[nodiscard]] static auto params(L2Region const& region, SegmentRates const& cycles) -> Params {
		return {.region = region.address(), .cycles = cycles.address()};
	}
};

}

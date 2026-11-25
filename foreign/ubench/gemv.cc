// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight matmuls' microbenchmark kernels as the drivers launch them: the gemv component's image and parameter
// structs, its kernels' budgets read from the records the image carries, and the buffers each takes sized by the
// gemv's own geometry.
export module engine.ubench_device:gemv;

import std;
import engine.driver;
import engine.gemv;
import engine.prims;
import engine.ptx;
import :names;

namespace engine::ubench_device {

namespace selfcheck = engine::gemv::selfcheck;
namespace prims = engine::device::prims;

/** The chunks of a matrix's placement: every unit's tiles, whole 16 KiB tiles in stream order. */
export template<engine::gemv::Matrix const& matrix>
inline constexpr std::size_t placement_chunks =
    static_cast<std::size_t>(engine::gemv::placement_units(matrix) * matrix.tensors) * static_cast<std::size_t>(prims::tile::bytes / 16);

/** What a gemv over `matrix` at `row_tiles` n8 tiles reads and writes, sized by the gemv's geometry. */
template<engine::gemv::Matrix const& matrix, int row_tiles>
inline constexpr engine::gemv::selfcheck::Buffers sizes = engine::gemv::selfcheck::buffers(matrix, row_tiles);

export using DownTiles = driver::DeviceArray<prims::Chunk, placement_chunks<engine::gemv::down>>;
export using GateUpTiles = driver::DeviceArray<prims::Chunk, placement_chunks<engine::gemv::gate_up>>;
export using BlockElapsed = driver::DeviceArray<unsigned, selfcheck::blocks>;

/** The largest row sweep point, which every buffer of the gemv sweep is sized for. */
inline constexpr int widest = 5;

export using StreamActivations = driver::DeviceArray<prims::Chunk, static_cast<std::size_t>(sizes<engine::gemv::down, widest>.activation_chunks)>;
export using StreamScales = driver::DeviceArray<unsigned short, static_cast<std::size_t>(sizes<engine::gemv::down, widest>.scales)>;
export using StreamFixups = driver::DeviceArray<float, static_cast<std::size_t>(sizes<engine::gemv::down, widest>.fixup_floats)>;
export using FixupFlags = driver::DeviceArray<unsigned long long, selfcheck::blocks>;
export using StreamOut = driver::DeviceArray<float, static_cast<std::size_t>(sizes<engine::gemv::down, widest>.out_floats)>;

/**
 * U5: down_proj's whole placement streamed against 8 x `row_tiles` rows. The blocks wait on each other's fixups across
 * SMs, so every block must be resident at once.
 */
export template<int row_tiles_>
struct GemvStreamKernel {
	static constexpr int row_tiles = row_tiles_;
	static constexpr ptx::Image image = engine::gemv::bench_image;
	static constexpr EntryName name = entry_name("gemv_stream_r", row_tiles);
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = {.x = selfcheck::blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = recorded_budget(engine::gemv::bench_image, entry);
	static constexpr driver::Residency residency = driver::Residency::CoResident;
	using Params = selfcheck::GemvStreamParams;

	static_assert(row_tiles >= 1 && row_tiles <= widest);
	static_assert(budget.threads == selfcheck::threads, "the record and the gemv's warp roles agree");

	[[nodiscard]] static auto params(DownTiles const& tiles, StreamActivations const& activations, StreamScales const& scales,
	                                 StreamFixups const& fixups, FixupFlags const& flags, StreamOut const& out, BlockElapsed const& elapsed) -> Params {
		return {
		    .tiles = tiles.address(),
		    .activations = activations.address(),
		    .scales = scales.address(),
		    .fixups = fixups.address(),
		    .fixup_flags = flags.address(),
		    .out = out.address(),
		    .elapsed = elapsed.address(),
		};
	}
};

export using GemvSweep = std::tuple<GemvStreamKernel<1>, GemvStreamKernel<2>, GemvStreamKernel<3>, GemvStreamKernel<4>, GemvStreamKernel<5>>;

/** Which prefill gemm: down_proj, or gate_proj with up_proj into silu(gate) * up. */
export enum class PrefillMatrix : std::uint8_t { Down, GateUp };

inline constexpr int prefill_row_tiles = static_cast<int>(selfcheck::prefill_rows / 8);

export template<PrefillMatrix which>
inline constexpr engine::gemv::Matrix const& prefill_matrix = which == PrefillMatrix::Down ? engine::gemv::down : engine::gemv::gate_up;

/** A gemm's activation rows, scales and outputs at the prefill chunk's rows. */
export template<PrefillMatrix which>
struct PrefillBuffers {
	static constexpr auto sized = sizes<prefill_matrix<which>, prefill_row_tiles>;
	using Tiles = driver::DeviceArray<prims::Chunk, placement_chunks<prefill_matrix<which>>>;
	using Activations = driver::DeviceArray<prims::Chunk, static_cast<std::size_t>(sized.activation_chunks)>;
	using Scales = driver::DeviceArray<unsigned short, static_cast<std::size_t>(sized.scales)>;
	using Out = driver::DeviceArray<unsigned, static_cast<std::size_t>(which == PrefillMatrix::Down ? sized.out_floats : sized.product_words)>;
};

/** U22: the prefill chunk's rows through a whole placement. */
export template<PrefillMatrix which_>
struct GemmPrefillKernel {
	static constexpr PrefillMatrix which = which_;
	static constexpr ptx::Image image = engine::gemv::bench_image;
	static constexpr std::string_view entry = which == PrefillMatrix::Down ? "gemm_prefill_down" : "gemm_prefill_gate_up";
	static constexpr driver::Dim grid = {.x = selfcheck::blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = recorded_budget(engine::gemv::bench_image, entry);
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = selfcheck::GemmBenchParams;
	using Buffers = PrefillBuffers<which>;

	/** One tensor passes its scales twice; gate and up pass each's. */
	[[nodiscard]] static auto params(typename Buffers::Tiles const& tiles, typename Buffers::Activations const& activations,
	                                 typename Buffers::Scales const& gate_scales, typename Buffers::Scales const& up_scales,
	                                 typename Buffers::Out const& out, BlockElapsed const& elapsed) -> Params {
		return {
		    .tiles = tiles.address(),
		    .activations = activations.address(),
		    .gate_scales = gate_scales.address(),
		    .up_scales = up_scales.address(),
		    .out = out.address(),
		    .elapsed = elapsed.address(),
		};
	}
};

}

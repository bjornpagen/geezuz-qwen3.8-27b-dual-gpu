// SPDX-License-Identifier: AGPL-3.0-only
//
// down_proj's real geometry against the cpu reference, bit for bit: the first two output tiles of the trunk's
// down_proj with its own run, so the real k (ffn_per_card) and the real split of those tiles over the SMs, through the
// emulation of every warp of the decode gemv and the prefill gemm. It is the costliest product, so it has its own
// deadline.
import std;
import engine.gemv;
import engine.model;
import engine.numerics_problems;
import engine.testing;

namespace {

namespace gemv = engine::gemv;
namespace model = engine::model;

/** The first two output tiles of down_proj, with down_proj's own run. */
inline constexpr gemv::Matrix down_head = {
    .units = gemv::down.units, .first = 0, .output_tiles = 2, .k_tiles = gemv::down.k_tiles, .tensors = 1, .format = model::Format::E4m3};

static_assert(gemv::k_extent(down_head) == model::ffn_per_card);

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "down_proj's real k and split: decode at 8 rows and prefill equal the reference (seed 4)",
		                      .holds = engine::numerics_problems::product_is_the_reference<1>(down_head, 4)},
	};
	return engine::testing::report(cases);
}

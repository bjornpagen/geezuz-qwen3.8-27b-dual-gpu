// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight products' kernels against the cpu reference, bit for bit, on small placements whose output tiles
// straddle two and three SMs. The host emulation of every warp of the decode gemv and the prefill gemm (the baked
// tiles, the decode of each lane's bytes, the swizzled activations, every `mma.sync` through the model of the sm_86
// tensor core, the scale folds and the SM partials in the device's order) must equal what the reference computes for
// the same weights and activations through its own route (`ref::tile_sums` over the same split, from plain row-major
// weights). On top of the products: silu(gate) * up, and the residual update with the hidden rms norm it feeds. The
// real layer widths have their own tests (`numerics.down_proj`, `numerics.lm_head`), each under its own deadline.
import std;
import engine.gemv;
import engine.model;
import engine.numerics_problems;
import engine.prims;
import engine.ref;
import engine.testing;

namespace {

namespace gemv = engine::gemv;
namespace model = engine::model;
namespace prims = engine::device::prims;
namespace ref = engine::ref;

using engine::numerics_problems::make_problem;
using engine::numerics_problems::product_is_the_reference;
using engine::numerics_problems::reference_rows;
using engine::numerics_problems::same_bits;
using gemv::Matrix;

/** A small placement at odd places in its run, so tiles straddle two and three SMs. */
inline constexpr Matrix straddling_e4m3 = {
    .units = 193, .first = 11, .output_tiles = 3, .k_tiles = 7, .tensors = 1, .format = model::Format::E4m3};
inline constexpr Matrix straddling_pair = {
    .units = 247, .first = 5, .output_tiles = 2, .k_tiles = 9, .tensors = 2, .format = model::Format::E4m3};
inline constexpr Matrix straddling_bf16 = {
    .units = 193, .first = 50, .output_tiles = 2, .k_tiles = 9, .tensors = 1, .format = model::Format::Bf16};

/** gate_proj and up_proj interleaved, into silu(gate) * up: the emulation's product words against the reference's rows. */
[[nodiscard]] auto mlp_product_is_the_reference(std::uint64_t seed) -> bool {
	auto const problem = make_problem(straddling_pair, 24, seed);
	auto const emulated = gemv::product(problem, gemv::rows_of(problem, gemv::decode<3>(problem)));
	auto const rows = reference_rows(problem);
	auto const width = static_cast<std::size_t>(straddling_pair.output_tiles * gemv::tile_rows);
	auto const count = static_cast<std::size_t>(problem.rows);
	auto gate = ref::Rows<float>::zeros(count, width);
	auto up = ref::Rows<float>::zeros(count, width);
	std::ranges::copy(std::span{rows}.first(count * width), gate.values.begin());
	std::ranges::copy(std::span{rows}.subspan(count * width), up.values.begin());
	auto const product = ref::mlp_products(gate, up);
	auto holds = true;
	for (auto index = std::size_t{0}; index < product.values.size(); ++index) {
		auto const word = emulated.at(index / 2);
		holds = holds && product.values.at(index).bits == static_cast<std::uint16_t>(word >> (16U * (index % 2)));
	}
	return holds;
}

/**
 * The residual update after a row-parallel phase and the rms norm it feeds: the emulation's new residual and per-tile
 * squares (each tile's tree, folded tile by tile in the next phase), then the model's norm on them, against the
 * reference's addition and its hidden norm, bit for bit.
 */
[[nodiscard]] auto norm_is_the_reference(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	constexpr auto rows = 5;
	auto const hidden = static_cast<std::size_t>(model::hidden);
	auto const draw = [&](float scale) {
		return std::normal_distribution<float>{0.0F, scale}(random);
	};
	auto own = std::vector<float>(hidden * rows);
	auto peer = std::vector<float>(hidden * rows);
	auto residual = std::vector<float>(hidden * rows);
	std::ranges::generate(own, [&] {
		return draw(3.0F);
	});
	std::ranges::generate(peer, [&] {
		return draw(3.0F);
	});
	std::ranges::generate(residual, [&] {
		return draw(40.0F);
	});
	auto weight = std::vector<ref::Bf16>(hidden);
	std::ranges::generate(weight, [&] {
		return ref::round(draw(0.3F));
	});
	auto const update = gemv::residual_update(own, peer, residual, rows);
	auto const tiles = hidden / static_cast<std::size_t>(gemv::tile_rows);
	auto reference = ref::Rows<float>{.width = hidden, .values = residual};
	auto sum = ref::Rows<float>::zeros(rows, hidden);
	for (auto row = std::size_t{0}; row < rows; ++row) {
		for (auto feature = std::size_t{0}; feature < hidden; ++feature) {
			// The cards' partials as the epilogue lays them out, [tile][row][128], added card 0 first.
			auto const from = (feature / gemv::tile_rows * rows + row) * gemv::tile_rows + feature % gemv::tile_rows;
			sum.row(row)[feature] = own.at(from) + peer.at(from);
		}
	}
	ref::add_into(reference, sum);
	auto holds = same_bits(update.residual, reference.values);
	for (auto row = std::size_t{0}; row < rows; ++row) {
		auto squares = update.squares.at(row);
		for (auto tile = std::size_t{1}; tile < tiles; ++tile) {
			squares = squares + update.squares.at(tile * rows + row);
		}
		auto const x = std::span<float const, model::hidden>{reference.row(row)};
		holds = holds && std::bit_cast<std::uint32_t>(squares) == std::bit_cast<std::uint32_t>(ref::tiled_squares<model::hidden>(x));
		auto const factor = model::rms_factor(squares, model::hidden);
		auto const normed = ref::hidden_norm<model::hidden>(x, std::span<ref::Bf16 const, model::hidden>{weight});
		for (auto feature = std::size_t{0}; feature < hidden; ++feature) {
			auto const emulated = prims::round_to_bf16(
			    model::one_plus_scaled(update.residual.at(row * hidden + feature), factor, ref::widen(weight.at(feature))));
			holds = holds && emulated.bits == normed.at(feature).bits;
		}
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "e4m3 tiles straddling SMs: decode at 8 and 40 rows and prefill equal the reference (seeds 1, 2)",
		                      .holds = product_is_the_reference<1>(straddling_e4m3, 1) && product_is_the_reference<5>(straddling_e4m3, 2)},
	    engine::testing::Case{.name = "bf16 tiles straddling SMs: decode at 16 rows and prefill equal the reference (seed 3)",
		                      .holds = product_is_the_reference<2>(straddling_bf16, 3)},
	    engine::testing::Case{.name = "gate_proj and up_proj interleaved, then silu(gate) * up, equal the reference (seed 5)",
		                      .holds = mlp_product_is_the_reference(5)},
	    engine::testing::Case{.name = "the residual update's squares and the hidden rms norm equal the reference (seed 6)",
		                      .holds = norm_is_the_reference(6)},
	};
	return engine::testing::report(cases);
}

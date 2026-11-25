// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.gemv;
import engine.model;
import engine.prims;
import engine.testing;

namespace {

namespace gemv = engine::gemv;
namespace model = engine::model;
namespace prims = engine::device::prims;

using gemv::Count;
using gemv::Matrix;
using gemv::Problem;

// ---------------------------------------------------------------------------------------------------------------------
// The split and the summation order.

/**
 * Every output tile's k tiles are cut into one contiguous segment per holding SM, in SM order, covering every k tile
 * once; every unit of the placement is held by exactly one SM; and the fixup law holds.
 */
[[nodiscard]] auto split_is_a_partition(Matrix const& matrix) -> bool {
	auto holds = gemv::fixups_fit(matrix);
	for (auto tile = Count{0}; tile < matrix.output_tiles; ++tile) {
		auto const holders = gemv::tile_holders(matrix, tile);
		auto next = Count{0};
		for (auto sm = holders.begin; sm < holders.begin + holders.length; ++sm) {
			auto const part = gemv::segment(matrix, tile, sm);
			holds = holds && part.begin == next && part.length > 0;
			next += part.length;
		}
		holds = holds && next == matrix.k_tiles;
		holds = holds && gemv::holding(matrix, tile, holders.begin + holders.length - 1) != gemv::Holding::Contributes;
	}
	auto covered = Count{0};
	auto expected_begin = matrix.first;
	for (auto sm = Count{0}; sm < gemv::selfcheck::blocks; ++sm) {
		auto const mine = gemv::held(matrix, sm);
		if (mine.length == 0) {
			continue;
		}
		holds = holds && mine.begin == expected_begin;
		expected_begin += mine.length;
		covered += mine.length;
	}
	return holds && covered == gemv::placement_units(matrix);
}

/** Matrices whose placement sits at odd places in a run, so tiles straddle two and three SMs. */
inline constexpr Matrix straddling_e4m3 = {
    .units = 193, .first = 11, .output_tiles = 3, .k_tiles = 7, .tensors = 1, .format = model::Format::E4m3};
inline constexpr Matrix straddling_pair = {
    .units = 247, .first = 5, .output_tiles = 2, .k_tiles = 9, .tensors = 2, .format = model::Format::E4m3};
inline constexpr Matrix straddling_bf16 = {
    .units = 193, .first = 50, .output_tiles = 2, .k_tiles = 9, .tensors = 1, .format = model::Format::Bf16};
/** The first two output tiles of down_proj, with down_proj's own run: the real split of those tiles. */
inline constexpr Matrix down_head = {
    .units = gemv::down.units, .first = 0, .output_tiles = 2, .k_tiles = gemv::down.k_tiles, .tensors = 1, .format = model::Format::E4m3};

// ---------------------------------------------------------------------------------------------------------------------
// Problems.

enum class Values : unsigned char {
	/** Codes, scales and activations drawn across many binades. */
	Spread,
	/** Each unit's second half of k is the first half negated against the same activations: exact sums of zero. */
	Cancelling,
};

[[nodiscard]] auto draw_code(std::mt19937_64& random) -> std::uint16_t {
	auto code = static_cast<std::uint16_t>(random() & 0xffU);
	while ((code & 0x7fU) == 0x7fU) {
		code = static_cast<std::uint16_t>(random() & 0xffU);
	}
	return code;
}

/** A bf16 with a random sign and mantissa and an exponent field in [low, high]. */
[[nodiscard]] auto draw_bf16(std::mt19937_64& random, unsigned low, unsigned high) -> std::uint16_t {
	auto const field = low + static_cast<unsigned>(random() % (high - low + 1));
	return static_cast<std::uint16_t>((random() & 0x807fU) | (field << 7U));
}

[[nodiscard]] auto make_problem(Matrix const& matrix, int rows, std::uint64_t seed, Values values) -> Problem {
	auto random = engine::testing::generator(seed);
	auto const span = static_cast<std::size_t>(gemv::unit_k(matrix));
	auto const half = span / 2;
	auto const e4m3 = matrix.format == model::Format::E4m3;
	auto problem = Problem{.matrix = matrix, .rows = rows, .weights = {}, .scales = {}, .activations = {}};
	auto const units = static_cast<std::size_t>(gemv::placement_units(matrix));
	for (auto unit = std::size_t{0}; unit < units * static_cast<std::size_t>(matrix.tensors); ++unit) {
		auto tile = std::vector<std::uint16_t>(static_cast<std::size_t>(gemv::tile_rows) * span);
		for (auto row = std::size_t{0}; row < static_cast<std::size_t>(gemv::tile_rows); ++row) {
			for (auto k = std::size_t{0}; k < span; ++k) {
				auto& stored = tile.at(row * span + k);
				if (values == Values::Cancelling && k >= half) {
					stored = static_cast<std::uint16_t>(tile.at(row * span + k - half) ^ (e4m3 ? 0x80U : 0x8000U));
				} else {
					stored = e4m3 ? draw_code(random) : draw_bf16(random, 110, 140);
				}
			}
		}
		problem.weights.push_back(std::move(tile));
	}
	if (e4m3) {
		for (auto tensor = std::size_t{0}; tensor < static_cast<std::size_t>(matrix.tensors); ++tensor) {
			auto& scales = problem.scales.at(tensor);
			for (auto index = Count{0}; index < matrix.output_tiles * matrix.k_tiles; ++index) {
				// Positive, as the checkpoint's are, from 2^-27 to 2^13.
				scales.push_back(static_cast<std::uint16_t>(draw_bf16(random, 100, 140) & 0x7fffU));
			}
		}
	}
	auto const extent = static_cast<std::size_t>(gemv::k_extent(matrix));
	problem.activations.resize(static_cast<std::size_t>(rows) * extent);
	for (auto row = std::size_t{0}; std::cmp_less(row, rows); ++row) {
		for (auto k = std::size_t{0}; k < extent; ++k) {
			auto& value = problem.activations.at(row * extent + k);
			if (values == Values::Cancelling && k % span >= half) {
				value = problem.activations.at(row * extent + k - half);
			} else {
				value = random() % 16 == 0 ? std::uint16_t{0} : draw_bf16(random, 112, 136);
			}
		}
	}
	return problem;
}

/** The problem's first `rows` rows of activations, as a problem of its own. */
[[nodiscard]] auto first_rows(Problem const& problem, int rows) -> Problem {
	auto smaller = problem;
	smaller.rows = rows;
	smaller.activations.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(gemv::k_extent(problem.matrix)));
	return smaller;
}

[[nodiscard]] auto same_bits(std::span<float const> left, std::span<float const> right) -> bool {
	return std::ranges::equal(left, right, [](float a, float b) {
		return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
	});
}

// ---------------------------------------------------------------------------------------------------------------------
// Properties.

/** Every warp of the decode, emulated through the baked tiles, the swizzle and the mma model, is the reference. */
template<int row_tiles>
[[nodiscard]] auto decode_is_the_reference(Matrix const& matrix, std::uint64_t seed, Values values) -> bool {
	auto const problem = make_problem(matrix, 8 * row_tiles, seed, values);
	auto const decoded = gemv::rows_of(problem, gemv::decode<row_tiles>(problem));
	return same_bits(decoded, gemv::reference(problem));
}

/** Every warp of the prefill gemm is the reference too, so prefill rows are the decode's bits. */
[[nodiscard]] auto prefill_is_the_reference(Matrix const& matrix, std::uint64_t seed, Values values) -> bool {
	auto const problem = make_problem(matrix, 64, seed, values);
	auto const rows = gemv::prefill(problem);
	auto const decoded = gemv::rows_of(first_rows(problem, 8), gemv::decode<1>(first_rows(problem, 8)));
	auto const width = static_cast<std::size_t>(matrix.output_tiles * gemv::tile_rows);
	auto decode_agrees = true;
	for (auto tensor = std::size_t{0}; tensor < static_cast<std::size_t>(matrix.tensors); ++tensor) {
		auto const from_prefill = std::span<float const>{rows}.subspan(tensor * 64 * width, 8 * width);
		auto const from_decode = std::span<float const>{decoded}.subspan(tensor * 8 * width, 8 * width);
		decode_agrees = decode_agrees && same_bits(from_prefill, from_decode);
	}
	return same_bits(rows, gemv::reference(problem)) && decode_agrees;
}

/** silu(gate) * up from the decode's tiles and from the prefill's rows is the reference's product, word for word. */
[[nodiscard]] auto products_agree(std::uint64_t seed) -> bool {
	auto const problem = make_problem(straddling_pair, 32, seed, Values::Spread);
	auto const reference = gemv::product(problem, gemv::reference(problem));
	auto const decoded = gemv::product(problem, gemv::rows_of(problem, gemv::decode<4>(problem)));
	auto const prefilled = gemv::product(problem, gemv::prefill(problem));
	return decoded == reference && prefilled == reference;
}

/**
 * The reference is the exact product within the tensor core's truncation and fp32's rounding: every block loses less
 * than 2^-18 of the magnitudes it sums, so the error stays below that times the blocks and folds an element passes
 * through, times the sum of the terms' magnitudes.
 */
[[nodiscard]] auto reference_is_the_product(Matrix const& matrix, std::uint64_t seed, Values values) -> bool {
	auto const problem = make_problem(matrix, 8, seed, values);
	auto const result = gemv::reference(problem);
	auto const span = static_cast<std::size_t>(gemv::unit_k(matrix));
	auto const extent = static_cast<std::size_t>(gemv::k_extent(matrix));
	auto const width = static_cast<std::size_t>(matrix.output_tiles * gemv::tile_rows);
	auto const e4m3 = matrix.format == model::Format::E4m3;
	auto const steps = static_cast<double>(extent) / 8.0 + static_cast<double>(matrix.k_tiles) + 2.0;
	for (auto tensor = std::size_t{0}; tensor < static_cast<std::size_t>(matrix.tensors); ++tensor) {
		for (auto row = std::size_t{0}; row < 8; ++row) {
			for (auto output = std::size_t{0}; output < width; ++output) {
				auto const tile = output / static_cast<std::size_t>(gemv::tile_rows);
				auto const within = output % static_cast<std::size_t>(gemv::tile_rows);
				auto exact = 0.0;
				auto magnitude = 0.0;
				for (auto k = std::size_t{0}; k < extent; ++k) {
					auto const k_tile = k / span;
					auto const unit = tile * static_cast<std::size_t>(matrix.k_tiles) + k_tile;
					auto const stored =
					    problem.weights.at(unit * static_cast<std::size_t>(matrix.tensors) + tensor).at(within * span + k % span);
					auto const w = e4m3 ? static_cast<double>(prims::value(prims::E4m3{.bits = static_cast<std::uint8_t>(stored)}))
					                    : static_cast<double>(prims::widen(prims::Bf16{.bits = stored}));
					auto const s =
					    e4m3 ? static_cast<double>(prims::widen(prims::Bf16{
					               .bits = problem.scales.at(tensor).at(tile * static_cast<std::size_t>(matrix.k_tiles) + k_tile)}))
						     : 1.0;
					auto const a = static_cast<double>(prims::widen(prims::Bf16{.bits = problem.activations.at(row * extent + k)}));
					exact += w * s * a;
					magnitude += std::abs(w * s * a);
				}
				auto const got = static_cast<double>(result.at((tensor * 8 + row) * width + output));
				if (std::abs(got - exact) > steps * std::ldexp(magnitude, -18)) {
					return false;
				}
			}
		}
	}
	return true;
}

/**
 * The static split shows in the bits: summing each output tile in one chain of folds, ignoring the SM segments, gives
 * a different result for some element. So the emulation and the kernels must honour the split, and the order is a
 * fact both sides hold, not an accident one side could drop.
 */
[[nodiscard]] auto split_order_shows(std::uint64_t seed) -> bool {
	auto const problem = make_problem(down_head, 8, seed, Values::Spread);
	auto const result = gemv::reference(problem);
	auto unsplit = problem;
	// A run of 82 tiles' units gives every SM one whole tile: tile 0 is one segment on SM 0.
	unsplit.matrix.units = gemv::selfcheck::blocks * unsplit.matrix.k_tiles;
	unsplit.matrix.output_tiles = 1;
	auto const whole = gemv::reference(unsplit);
	auto const width = static_cast<std::size_t>(gemv::tile_rows);
	auto differs = false;
	for (auto row = std::size_t{0}; row < 8; ++row) {
		for (auto output = std::size_t{0}; output < width; ++output) {
			differs = differs || std::bit_cast<std::uint32_t>(whole.at(row * width + output)) !=
			                         std::bit_cast<std::uint32_t>(result.at(row * 2 * width + output));
		}
	}
	return differs && gemv::tile_holders(down_head, 0).length > 1 && gemv::tile_holders(unsplit.matrix, 0).length == 1;
}

/**
 * The residual update: each element is residual + (own + peer), the same bits whichever card's sum is `own`, and
 * each tile's squares are the exact sum of squares of its 128 new values within fp32's rounding of 128 terms.
 */
[[nodiscard]] auto residual_update_holds(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	constexpr auto rows = gemv::selfcheck::residual_rows;
	auto const hidden = static_cast<std::size_t>(model::hidden);
	auto const values = hidden * static_cast<std::size_t>(rows);
	auto const draw = [&] {
		return prims::widen(prims::Bf16{.bits = draw_bf16(random, 110, 140)}) * 1.5F;
	};
	auto own = std::vector<float>(values);
	auto peer = std::vector<float>(values);
	auto residual = std::vector<float>(values);
	std::ranges::generate(own, draw);
	std::ranges::generate(peer, draw);
	std::ranges::generate(residual, draw);
	auto const update = gemv::residual_update(own, peer, residual, rows);
	auto const swapped = gemv::residual_update(peer, own, residual, rows);
	auto const width = static_cast<std::size_t>(gemv::tile_rows);
	auto holds = same_bits(update.residual, swapped.residual) && same_bits(update.squares, swapped.squares);
	for (auto tile = std::size_t{0}; tile < hidden / width; ++tile) {
		for (auto row = std::size_t{0}; row < static_cast<std::size_t>(rows); ++row) {
			auto exact = 0.0;
			for (auto output = std::size_t{0}; output < width; ++output) {
				auto const from = (tile * static_cast<std::size_t>(rows) + row) * width + output;
				auto const at = row * hidden + tile * width + output;
				auto const expected = residual.at(at) + (own.at(from) + peer.at(from));
				holds = holds && std::bit_cast<std::uint32_t>(update.residual.at(at)) == std::bit_cast<std::uint32_t>(expected);
				exact += static_cast<double>(expected) * static_cast<double>(expected);
			}
			auto const got = static_cast<double>(update.squares.at(tile * static_cast<std::size_t>(rows) + row));
			holds = holds && std::abs(got - exact) <= std::ldexp(exact, -17);
		}
	}
	return holds;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "the split partitions down_proj and gate_up",
		     .holds = split_is_a_partition(gemv::down) && split_is_a_partition(gemv::gate_up)},
	    Case{.name = "the split partitions straddling placements",
		     .holds =
		         split_is_a_partition(straddling_e4m3) && split_is_a_partition(straddling_pair) && split_is_a_partition(straddling_bf16)},
	    Case{.name = "decode warps are the reference at 8 rows (seed 1)",
		     .holds = decode_is_the_reference<1>(straddling_e4m3, 1, Values::Spread)},
	    Case{.name = "decode warps are the reference at 16 rows (seed 2)",
		     .holds = decode_is_the_reference<2>(straddling_e4m3, 2, Values::Spread)},
	    Case{.name = "decode warps are the reference at 24 rows (seed 3)",
		     .holds = decode_is_the_reference<3>(straddling_e4m3, 3, Values::Spread)},
	    Case{.name = "decode warps are the reference at 32 rows (seed 4)",
		     .holds = decode_is_the_reference<4>(straddling_e4m3, 4, Values::Spread)},
	    Case{.name = "decode warps are the reference under cancellation (seed 5)",
		     .holds = decode_is_the_reference<3>(straddling_e4m3, 5, Values::Cancelling)},
	    Case{.name = "decode warps are the reference for interleaved tensors (seed 6)",
		     .holds = decode_is_the_reference<2>(straddling_pair, 6, Values::Spread)},
	    Case{.name = "decode warps are the reference for bf16 tiles (seed 7)",
		     .holds = decode_is_the_reference<4>(straddling_bf16, 7, Values::Spread)},
	    Case{.name = "decode warps are the reference over down_proj's split (seed 8)",
		     .holds = decode_is_the_reference<1>(down_head, 8, Values::Spread)},
	    Case{.name = "prefill warps are the reference and the decode (seed 9)",
		     .holds = prefill_is_the_reference(straddling_e4m3, 9, Values::Spread)},
	    Case{.name = "prefill warps are the reference under cancellation (seed 10)",
		     .holds = prefill_is_the_reference(straddling_pair, 10, Values::Cancelling)},
	    Case{.name = "prefill warps are the reference for bf16 tiles (seed 11)",
		     .holds = prefill_is_the_reference(straddling_bf16, 11, Values::Spread)},
	    Case{.name = "silu(gate) * up agrees across decode, prefill and reference (seed 12)", .holds = products_agree(12)},
	    Case{.name = "the reference is the product within the mma's rounding (seed 13)",
		     .holds = reference_is_the_product(straddling_e4m3, 13, Values::Spread)},
	    Case{.name = "the reference is the product under cancellation (seed 14)",
		     .holds = reference_is_the_product(straddling_e4m3, 14, Values::Cancelling)},
	    Case{.name = "the reference is the product for bf16 tiles (seed 15)",
		     .holds = reference_is_the_product(straddling_bf16, 15, Values::Spread)},
	    Case{.name = "the static split shows in the bits (seed 16)", .holds = split_order_shows(16)},
	    Case{.name = "the residual update adds both cards and sums its squares (seed 17)", .holds = residual_update_holds(17)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight-product problems for the numerics tests: weights, block scales and activations drawn across many binades,
// the reference's own reading of them (plain row-major weights, each code decoded exactly) over the same split of k
// into SM segments, and the comparison of the emulation's rows against the reference's, bit for bit.
export module engine.numerics_problems;

import std;
import engine.gemv;
import engine.model;
import engine.ref;
import engine.testing;

namespace engine::numerics_problems {

namespace gemv = engine::gemv;
namespace model = engine::model;
namespace ref = engine::ref;

using gemv::Count;
using gemv::Matrix;
using gemv::Problem;

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

/** Weights, positive block scales and activations across many binades, some activations zero. */
export [[nodiscard]] auto make_problem(Matrix const& matrix, int rows, std::uint64_t seed) -> Problem {
	auto random = engine::testing::generator(seed);
	auto const span = static_cast<std::size_t>(gemv::unit_k(matrix));
	auto const e4m3 = matrix.format == model::Format::E4m3;
	auto problem = Problem{.matrix = matrix, .rows = rows, .weights = {}, .scales = {}, .activations = {}};
	auto const units = static_cast<std::size_t>(gemv::placement_units(matrix));
	for (auto unit = std::size_t{0}; unit < units * static_cast<std::size_t>(matrix.tensors); ++unit) {
		auto tile = std::vector<std::uint16_t>(static_cast<std::size_t>(gemv::tile_rows) * span);
		std::ranges::generate(tile, [&] {
			return e4m3 ? draw_code(random) : draw_bf16(random, 110, 140);
		});
		problem.weights.push_back(std::move(tile));
	}
	if (e4m3) {
		for (auto tensor = std::size_t{0}; tensor < static_cast<std::size_t>(matrix.tensors); ++tensor) {
			for (auto index = Count{0}; index < matrix.output_tiles * matrix.k_tiles; ++index) {
				problem.scales.at(tensor).push_back(static_cast<std::uint16_t>(draw_bf16(random, 100, 140) & 0x7fffU));
			}
		}
	}
	problem.activations.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(gemv::k_extent(matrix)));
	std::ranges::generate(problem.activations, [&] {
		return random() % 16 == 0 ? std::uint16_t{0} : draw_bf16(random, 112, 136);
	});
	return problem;
}

/** The reference's reader of a problem's tiles: plain row-major weights, each code decoded exactly, each row its scale. */
struct ProblemTiles {
	Problem const& problem;
	std::size_t tensor;

	[[nodiscard]] auto read(ref::TileWork const& work, Count unit) const -> ref::TileData {
		auto const& matrix = problem.matrix;
		auto const at = static_cast<std::size_t>(work.block * matrix.k_tiles + unit);
		auto const& stored = problem.weights.at(at * static_cast<std::size_t>(matrix.tensors) + tensor);
		auto tile = ref::TileData{.weights = {}, .scales = {}};
		for (auto const bits : stored) {
			tile.weights.push_back(matrix.format == model::Format::E4m3
			                           ? ref::decode_e4m3(std::byte{static_cast<std::uint8_t>(bits & 0xffU)})
									   : ref::Bf16{.bits = bits});
		}
		if (matrix.format == model::Format::E4m3) {
			tile.scales.assign(static_cast<std::size_t>(gemv::tile_rows), ref::widen(ref::Bf16{.bits = problem.scales.at(tensor).at(at)}));
		}
		return tile;
	}
};

static_assert(ref::TileReader<ProblemTiles>);

/** Output tile `tile`'s k units as the SMs holding them split them, in SM order. */
[[nodiscard]] auto work_of(Matrix const& matrix, Count tile) -> ref::TileWork {
	auto work = ref::TileWork{.card = model::Card::Zero, .block = tile, .segments = {}};
	auto const holders = gemv::tile_holders(matrix, tile);
	for (auto sm = holders.begin; sm < holders.begin + holders.length; ++sm) {
		auto const part = gemv::segment(matrix, tile, sm);
		work.segments.push_back({.first = part.begin, .count = part.length});
	}
	return work;
}

/** The reference's product of `problem`, laid out as the emulation lays its rows: [tensor][row][output]. */
export [[nodiscard]] auto reference_rows(Problem const& problem) -> std::vector<float> {
	auto const& matrix = problem.matrix;
	auto x = ref::Rows<ref::Bf16>::zeros(static_cast<std::size_t>(problem.rows), gemv::k_extent(matrix));
	std::ranges::transform(problem.activations, x.values.begin(), [](std::uint16_t bits) {
		return ref::Bf16{.bits = bits};
	});
	auto const width = static_cast<std::size_t>(matrix.output_tiles * gemv::tile_rows);
	auto const rows = static_cast<std::size_t>(problem.rows);
	auto out = std::vector<float>(static_cast<std::size_t>(matrix.tensors) * rows * width);
	for (auto tensor = std::size_t{0}; tensor < static_cast<std::size_t>(matrix.tensors); ++tensor) {
		auto const reader = ProblemTiles{.problem = problem, .tensor = tensor};
		for (auto tile = Count{0}; tile < matrix.output_tiles; ++tile) {
			auto const sums = ref::tile_sums<ref::Tensor>(work_of(matrix, tile), matrix.format, reader, x);
			for (auto row = std::size_t{0}; row < rows; ++row) {
				for (auto output = std::size_t{0}; output < static_cast<std::size_t>(gemv::tile_rows); ++output) {
					out.at((tensor * rows + row) * width + tile * gemv::tile_rows + output) = sums.row(row)[output];
				}
			}
		}
	}
	return out;
}

export [[nodiscard]] auto same_bits(std::span<float const> left, std::span<float const> right) -> bool {
	return std::ranges::equal(left, right, [](float a, float b) {
		return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
	});
}

/** The decode gemv's warps at `row_tiles` n8 tiles of rows, and the prefill gemm's at 32 rows, against the reference. */
export template<int row_tiles>
[[nodiscard]] auto product_is_the_reference(Matrix const& matrix, std::uint64_t seed) -> bool {
	auto const decode = make_problem(matrix, 8 * row_tiles, seed);
	auto const prefill = make_problem(matrix, gemv::layout::prefill::block_rows, seed + 1);
	return same_bits(gemv::rows_of(decode, gemv::decode<row_tiles>(decode)), reference_rows(decode)) &&
	       same_bits(gemv::prefill(prefill), reference_rows(prefill));
}

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The weight products' kernels against the cpu reference, bit for bit. The host emulation of every warp of the decode
// gemv and the prefill gemm (the baked tiles, the decode of each lane's bytes, the swizzled activations, every
// `mma.sync` through the model of the sm_86 tensor core, the scale folds and the SM partials in the device's order) must
// equal what the reference computes for the same weights and activations through its own route (`ref::tile_sums`
// over the same split, from plain row-major weights). On top of the products: silu(gate) * up, the residual update and
// the hidden rms norm it feeds, and the lm_head's top-k candidates merged from both cards' rows.
import std;
import engine.gemv;
import engine.model;
import engine.prims;
import engine.ref;
import engine.sampling;
import engine.sampling_device;
import engine.schedule;
import engine.testing;

namespace {

namespace gemv = engine::gemv;
namespace model = engine::model;
namespace prims = engine::device::prims;
namespace ref = engine::ref;
namespace sampling = engine::sampling;

using gemv::Count;
using gemv::Matrix;
using gemv::Problem;

/** A small placement at odd places in its run, so tiles straddle two and three SMs. */
inline constexpr Matrix straddling_e4m3 = {
    .units = 193, .first = 11, .output_tiles = 3, .k_tiles = 7, .tensors = 1, .format = model::Format::E4m3};
inline constexpr Matrix straddling_pair = {
    .units = 247, .first = 5, .output_tiles = 2, .k_tiles = 9, .tensors = 2, .format = model::Format::E4m3};
inline constexpr Matrix straddling_bf16 = {
    .units = 193, .first = 50, .output_tiles = 2, .k_tiles = 9, .tensors = 1, .format = model::Format::Bf16};
/** The first two output tiles of down_proj, with down_proj's own run: the real k and the real split of those tiles. */
inline constexpr Matrix down_head = {
    .units = gemv::down.units, .first = 0, .output_tiles = 2, .k_tiles = gemv::down.k_tiles, .tensors = 1, .format = model::Format::E4m3};
/** The first output tiles of the lm_head, with the head phase's own run: bf16 tiles over the whole hidden size. */
inline constexpr Matrix head_phase = gemv::matrix(gemv::phase_of(engine::schedule::Op::Head), 0);
inline constexpr Matrix head_front = {
    .units = head_phase.units, .first = 0, .output_tiles = 4, .k_tiles = head_phase.k_tiles, .tensors = 1, .format = model::Format::Bf16};

static_assert(head_phase.format == model::Format::Bf16 && gemv::k_extent(head_phase) == model::hidden);

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
[[nodiscard]] auto make_problem(Matrix const& matrix, int rows, std::uint64_t seed) -> Problem {
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
[[nodiscard]] auto reference_rows(Problem const& problem) -> std::vector<float> {
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

[[nodiscard]] auto same_bits(std::span<float const> left, std::span<float const> right) -> bool {
	return std::ranges::equal(left, right, [](float a, float b) {
		return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
	});
}

/** The decode gemv's warps at `row_tiles` n8 tiles of rows, and the prefill gemm's at 32 rows, against the reference. */
template<int row_tiles>
[[nodiscard]] auto product_is_the_reference(Matrix const& matrix, std::uint64_t seed) -> bool {
	auto const decode = make_problem(matrix, 8 * row_tiles, seed);
	auto const prefill = make_problem(matrix, gemv::layout::prefill::block_rows, seed + 1);
	return same_bits(gemv::rows_of(decode, gemv::decode<row_tiles>(decode)), reference_rows(decode)) &&
	       same_bits(gemv::prefill(prefill), reference_rows(prefill));
}

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

/**
 * The lm_head's logits on both cards (each card's first rows of its permuted vocabulary) and the top-k candidates the
 * two cards' halves merge to: the emulation's against the reference's, logit bits and ids alike.
 */
[[nodiscard]] auto head_candidates_are_the_reference(std::uint64_t seed) -> bool {
	constexpr auto rows = 8;
	auto const width = static_cast<std::size_t>(head_front.output_tiles * gemv::tile_rows);
	auto emulated = std::array<std::vector<float>, model::cards>{};
	auto referenced = std::array<std::vector<float>, model::cards>{};
	auto problems = std::vector<Problem>{};
	for (auto card = std::size_t{0}; card < model::cards; ++card) {
		auto problem = make_problem(head_front, rows, seed + card);
		if (card == 1) {
			problem.activations = problems.front().activations;
		}
		emulated.at(card) = gemv::rows_of(problem, gemv::decode<1>(problem));
		referenced.at(card) = reference_rows(problem);
		problems.push_back(std::move(problem));
	}
	auto holds = same_bits(emulated[0], referenced[0]) && same_bits(emulated[1], referenced[1]);
	for (auto row = std::size_t{0}; row < rows; ++row) {
		// Logits by token id: card c's row r is the token `token(c, r)`, here numbered densely in card-major order.
		auto const logits = [&](std::array<std::vector<float>, model::cards> const& cards) {
			auto out = std::vector<float>{};
			for (auto const& card : cards) {
				auto const row_logits = std::span{card}.subspan(row * width, width);
				out.insert(out.end(), row_logits.begin(), row_logits.end());
			}
			return out;
		};
		auto const from_emulation = sampling::candidates(logits(emulated));
		auto const from_reference = sampling::candidates(logits(referenced));
		holds = holds && std::ranges::equal(from_emulation, from_reference, [](auto const& a, auto const& b) {
			        return a.id == b.id && std::bit_cast<std::uint32_t>(a.logit) == std::bit_cast<std::uint32_t>(b.logit);
		        });
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
	    engine::testing::Case{.name = "down_proj's real k and split: decode at 8 rows and prefill equal the reference (seed 4)",
		                      .holds = product_is_the_reference<1>(down_head, 4)},
	    engine::testing::Case{.name = "gate_proj and up_proj interleaved, then silu(gate) * up, equal the reference (seed 5)",
		                      .holds = mlp_product_is_the_reference(5)},
	    engine::testing::Case{.name = "the residual update's squares and the hidden rms norm equal the reference (seed 6)",
		                      .holds = norm_is_the_reference(6)},
	    engine::testing::Case{.name = "lm_head logits over the real hidden and their top-k candidates equal the reference (seed 7)",
		                      .holds = head_candidates_are_the_reference(7)},
	};
	return engine::testing::report(cases);
}

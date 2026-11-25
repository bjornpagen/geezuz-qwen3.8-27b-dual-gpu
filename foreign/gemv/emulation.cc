// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's emulation of every warp of the weight matmuls, and the straightforward product they must equal. The
// emulation reads what the kernels read where they read it: the baked tiles through the bake's own encoder, each
// lane's 16 bytes at the warps' own offsets decoded by the bit-exact model of the decode's instructions, each
// activation fragment through the swizzle and the `ldmatrix` lane map, every `mma.sync` through the bit-exact tensor
// core model, every fold and sum in fp32 exactly as the order says. The reference instead takes the plain row-major
// weights and activations and applies the same mma model to the same k order, one output element at a time. The two
// agree bit for bit when the warps compute what they claim; the drivers hand the emulation's outputs to the
// self-check kernels as the expected bits.
export module engine.gemv:emulation;

import std;
import engine.core;
import engine.mma_model;
import engine.model;
import engine.prims;
import :geometry;

namespace engine::gemv {

namespace prims = engine::device::prims;
namespace mma = engine::mma_model;

using engine::prims::TileWords;

/** One matmul's inputs, as the bake and the activation producers would leave them. */
export struct Problem {
	Matrix matrix;
	/** Activation rows: 8 to 40 for the decode (whole n8 tiles), whole blocks of 32 for the prefill. */
	int rows;
	/**
	 * Per unit of the placement (output tile by output tile, k ascending) and tensor, at [unit * tensors + tensor]:
	 * the weight tile row-major, 128 rows by `unit_k`; e4m3 codes in the low byte, or bf16 bit patterns.
	 */
	std::vector<std::vector<std::uint16_t>> weights;
	/** Per tensor: bf16 block scales [tile * k_tiles + k] (e4m3 only). */
	std::array<std::vector<std::uint16_t>, 2> scales;
	/** bf16 activation rows [row][k_extent]. */
	std::vector<std::uint16_t> activations;
};

[[nodiscard]] auto count(Count value) -> std::size_t {
	return static_cast<std::size_t>(value);
}

[[nodiscard]] auto is_e4m3(Matrix const& matrix) -> bool {
	return matrix.format == model::Format::E4m3;
}

/** The placement as the bake streams it: every tile encoded, at [unit * tensors + tensor]. */
export [[nodiscard]] auto bake(Problem const& problem) -> std::vector<TileWords> {
	auto tiles = std::vector<TileWords>{};
	tiles.reserve(problem.weights.size());
	for (auto const& weights : problem.weights) {
		if (is_e4m3(problem.matrix)) {
			auto codes = std::vector<std::uint8_t>(weights.size());
			std::ranges::transform(weights, codes.begin(), [](std::uint16_t code) {
				return static_cast<std::uint8_t>(code & 0xffU);
			});
			auto const baked = engine::prims::bake_e4m3_tile(std::span<std::uint8_t const, prims::tile::bytes>{codes});
			core::invariant(baked.has_value());
			tiles.push_back(baked.value_or(TileWords{}));
		} else {
			tiles.push_back(engine::prims::bake_bf16_tile(std::span<std::uint16_t const, prims::tile::bytes / 2>{weights}));
		}
	}
	return tiles;
}

/** A warp's accumulators for one n8 tile: per lane, the four fp32 registers. */
using Fragment = mma::WarpResult;

/** A swizzled activation tile, byte for byte: one unit's k span of `Tile::rows` rows starting at `first_row`. */
template<class Tile>
[[nodiscard]] auto stage_activations(Problem const& problem, int first_row, unsigned k_tile) -> std::vector<std::uint8_t> {
	auto bytes = std::vector<std::uint8_t>(static_cast<std::size_t>(Tile::bytes));
	auto const extent = count(k_extent(problem.matrix));
	auto const span = count(unit_k(problem.matrix));
	for (auto row = 0; row < Tile::rows; ++row) {
		for (auto chunk = 0; chunk < Tile::chunks_per_row; ++chunk) {
			auto const at = static_cast<std::size_t>(Tile::offset(row, chunk));
			auto const source = static_cast<std::size_t>(first_row + row) * extent + k_tile * span + static_cast<std::size_t>(chunk) * 8;
			for (auto element = std::size_t{0}; element < 8; ++element) {
				auto const bits = problem.activations.at(source + element);
				bytes.at(at + 2 * element) = static_cast<std::uint8_t>(bits & 0xffU);
				bytes.at(at + 2 * element + 1) = static_cast<std::uint8_t>(bits >> 8U);
			}
		}
	}
	return bytes;
}

/** The two k16 steps' b-fragments lane by lane, as `ldmatrix.x4` hands them for step pair `pair`. */
template<class Tile>
[[nodiscard]] auto b_fragments(std::span<std::uint8_t const> activations, int first_row, int pair)
    -> std::array<std::array<std::array<std::uint32_t, 2>, 32>, 2> {
	auto matrices = std::array<std::array<std::uint32_t, 4>, 32>{};
	for (auto lane = 0; lane < 32; ++lane) {
		for (auto matrix = 0; matrix < 4; ++matrix) {
			for (auto half = 0; half < 2; ++half) {
				auto const at = prims::fragment::ldmatrix(lane, half, prims::fragment::Transpose::Plain);
				auto const supplier = static_cast<unsigned>(8 * matrix + at.row);
				auto const address = static_cast<std::size_t>(layout::activation_offset<Tile>(first_row, pair, supplier)) +
				                     2 * static_cast<std::size_t>(at.column);
				auto const bits = std::uint32_t{activations[address]} | (std::uint32_t{activations[address + 1]} << 8U);
				matrices.at(static_cast<std::size_t>(lane)).at(static_cast<std::size_t>(matrix)) |=
				    bits << (16U * static_cast<std::uint32_t>(half));
			}
		}
	}
	auto steps = std::array<std::array<std::array<std::uint32_t, 2>, 32>, 2>{};
	for (auto step = std::size_t{0}; step < 2; ++step) {
		for (auto lane = std::size_t{0}; lane < 32; ++lane) {
			steps.at(step).at(lane) = {matrices.at(lane).at(2 * step), matrices.at(lane).at(2 * step + 1)};
		}
	}
	return steps;
}

/** One lane's 16 bytes of the tile at `offset`, as four little-endian words. */
[[nodiscard]] auto lane_chunk(TileWords const& tile, unsigned offset) -> std::array<std::uint32_t, 4> {
	auto const first = static_cast<std::size_t>(offset / 4U);
	return {tile.at(first), tile.at(first + 1), tile.at(first + 2), tile.at(first + 3)};
}

/** The a-fragments of chunk `chunk` of m16 slab `slab`, lane by lane, for the k16 steps it holds (two e4m3, one bf16). */
[[nodiscard]] auto a_fragments(Matrix const& matrix, TileWords const& tile, unsigned slab, unsigned chunk)
    -> std::vector<std::array<std::array<std::uint32_t, 4>, 32>> {
	auto steps = std::vector<std::array<std::array<std::uint32_t, 4>, 32>>(is_e4m3(matrix) ? 2 : 1);
	for (auto lane = 0U; lane < 32; ++lane) {
		auto const words = lane_chunk(tile, layout::weight_offset(slab, chunk, lane));
		if (is_e4m3(matrix)) {
			for (auto step = std::size_t{0}; step < 2; ++step) {
				auto const low = prims::decode::model(words.at(2 * step));
				auto const high = prims::decode::model(words.at(2 * step + 1));
				steps.at(step).at(lane) = {low.top.bits, low.bottom.bits, high.top.bits, high.bottom.bits};
			}
		} else {
			steps.front().at(lane) = words;
		}
	}
	return steps;
}

/**
 * One warp's m16 slab `slab` of one weight tile against the n8 tiles whose first rows are `first_rows`: every mma in
 * ascending k, each n8 tile's chain continuing in `chains`.
 */
template<class Tile>
void warp_tile(Matrix const& matrix, TileWords const& tile, std::span<std::uint8_t const> activations, unsigned slab,
               std::span<int const> first_rows, std::span<Fragment> chains) {
	// One ldmatrix.x4 covers a pair of k16 steps: an e4m3 tile's 8 steps are four pairs (one chunk each), a bf16 tile's 4
	// steps two pairs (two chunks each).
	auto const pairs = is_e4m3(matrix) ? 4 : 2;
	for (auto pair = 0; pair < pairs; ++pair) {
		auto a = std::array<std::array<std::array<std::uint32_t, 4>, 32>, 2>{};
		if (is_e4m3(matrix)) {
			auto const decoded = a_fragments(matrix, tile, slab, static_cast<unsigned>(pair));
			a = {decoded.at(0), decoded.at(1)};
		} else {
			a = {a_fragments(matrix, tile, slab, static_cast<unsigned>(2 * pair)).front(),
			     a_fragments(matrix, tile, slab, static_cast<unsigned>(2 * pair + 1)).front()};
		}
		for (auto n = std::size_t{0}; n < first_rows.size(); ++n) {
			auto const b = b_fragments<Tile>(activations, first_rows[n], pair);
			for (auto step = std::size_t{0}; step < 2; ++step) {
				chains[n] = mma::warp_mma({.a = a.at(step), .b = b.at(step), .c = chains[n]});
			}
		}
	}
}

/** partial = fma(chain, scale, partial), register by register. */
void fold(Fragment& partial, Fragment const& chain, float scale) {
	for (auto lane = std::size_t{0}; lane < 32; ++lane) {
		for (auto reg = std::size_t{0}; reg < 4; ++reg) {
			auto& value = partial.at(lane).at(reg);
			value = prims::math::fma(chain.at(lane).at(reg), scale, value);
		}
	}
}

/** total = partial for the first segment, total + partial after it. */
void join(Fragment& total, Fragment const& partial, bool first_segment) {
	for (auto lane = std::size_t{0}; lane < 32; ++lane) {
		for (auto reg = std::size_t{0}; reg < 4; ++reg) {
			auto& value = total.at(lane).at(reg);
			value = first_segment ? partial.at(lane).at(reg) : value + partial.at(lane).at(reg);
		}
	}
}

[[nodiscard]] auto scale(Problem const& problem, std::size_t tensor, unsigned tile, unsigned k_tile) -> float {
	auto const at = static_cast<std::size_t>(tile) * count(problem.matrix.k_tiles) + k_tile;
	return prims::widen(prims::Bf16{.bits = problem.scales.at(tensor).at(at)});
}

/** Per warp, per tensor, per slab, per n8 tile: a set of warps' accumulators over one output tile. */
using Fragments = std::vector<std::vector<std::vector<std::vector<Fragment>>>>;

/**
 * The fragments of one output tile as a set of warps computes them over the decode's segments: per warp, `slabs`
 * lists its m16 blocks of the tile and `first_rows` the first rows of its n8 tiles; `stage(k_tile)` gives that unit's
 * swizzled activation tile.
 */
template<class Tile, class Stage>
[[nodiscard]] auto tile_fragments(Problem const& problem, std::vector<TileWords> const& tiles, unsigned tile,
                                  std::span<std::vector<unsigned> const> slabs, std::span<std::vector<int> const> first_rows,
                                  Stage const& stage) -> Fragments {
	auto const& matrix = problem.matrix;
	auto const tensors = count(matrix.tensors);
	auto const warps = slabs.size();
	auto shape = [&] {
		auto value = Fragments(warps);
		for (auto warp = std::size_t{0}; warp < warps; ++warp) {
			value.at(warp).assign(tensors, std::vector<std::vector<Fragment>>(slabs[warp].size(),
			                                                                  std::vector<Fragment>(first_rows[warp].size(), Fragment{})));
		}
		return value;
	};
	auto total = shape();
	auto const holders = tile_holders(matrix, tile);
	for (auto sm = holders.begin; sm < holders.begin + holders.length; ++sm) {
		auto partial = shape();
		auto const part = segment(matrix, tile, sm);
		for (auto k_tile = static_cast<unsigned>(part.begin); k_tile < part.begin + part.length; ++k_tile) {
			auto const activations = stage(k_tile);
			auto const unit = static_cast<std::size_t>(tile) * count(matrix.k_tiles) + k_tile;
			for (auto tensor = std::size_t{0}; tensor < tensors; ++tensor) {
				auto const& weights = tiles.at(unit * tensors + tensor);
				for (auto warp = std::size_t{0}; warp < warps; ++warp) {
					for (auto slab = std::size_t{0}; slab < slabs[warp].size(); ++slab) {
						auto& running = partial.at(warp).at(tensor).at(slab);
						if (is_e4m3(matrix)) {
							auto chain = std::vector<Fragment>(running.size(), Fragment{});
							warp_tile<Tile>(matrix, weights, activations, slabs[warp].at(slab), first_rows[warp], chain);
							for (auto n = std::size_t{0}; n < chain.size(); ++n) {
								fold(running.at(n), chain.at(n), scale(problem, tensor, tile, k_tile));
							}
						} else {
							warp_tile<Tile>(matrix, weights, activations, slabs[warp].at(slab), first_rows[warp], running);
						}
					}
				}
			}
		}
		for (auto warp = std::size_t{0}; warp < warps; ++warp) {
			for (auto tensor = std::size_t{0}; tensor < tensors; ++tensor) {
				for (auto slab = std::size_t{0}; slab < slabs[warp].size(); ++slab) {
					for (auto n = std::size_t{0}; n < first_rows[warp].size(); ++n) {
						join(total.at(warp).at(tensor).at(slab).at(n), partial.at(warp).at(tensor).at(slab).at(n), sm == holders.begin);
					}
				}
			}
		}
	}
	return total;
}

/**
 * What the decode gemv writes for `problem` (rows = 8 `row_tiles`): every output tile's fp32 sums as
 * [tile][tensor][row][128], each element computed by the warp and lane that owns it on the SMs that hold its units.
 */
export template<int row_tiles>
[[nodiscard]] auto decode(Problem const& problem) -> std::vector<float> {
	auto const& matrix = problem.matrix;
	core::invariant(problem.rows == 8 * row_tiles);
	auto const tiles = bake(problem);
	auto const tensors = count(matrix.tensors);
	auto const tile_floats = tensors * static_cast<std::size_t>(problem.rows) * count(tile_rows);
	auto out = std::vector<float>(count(matrix.output_tiles) * tile_floats);
	auto slabs = std::vector<std::vector<unsigned>>{};
	auto first_rows = std::vector<std::vector<int>>{};
	for (auto warp = 0U; warp < 8; ++warp) {
		slabs.push_back({warp});
		auto rows = std::vector<int>{};
		for (auto n = 0; n < row_tiles; ++n) {
			rows.push_back(8 * n);
		}
		first_rows.push_back(rows);
	}
	auto run = [&]<class Tile>() {
		for (auto tile = 0U; tile < matrix.output_tiles; ++tile) {
			auto const fragments = tile_fragments<Tile>(problem, tiles, tile, slabs, first_rows, [&](unsigned k_tile) {
				return stage_activations<Tile>(problem, 0, k_tile);
			});
			for (auto warp = 0U; warp < 8; ++warp) {
				for (auto tensor = std::size_t{0}; tensor < tensors; ++tensor) {
					for (auto n = 0; n < row_tiles; ++n) {
						for (auto lane = 0U; lane < 32; ++lane) {
							for (auto reg = 0; reg < 4; ++reg) {
								auto const at = layout::element(warp, lane, n, reg);
								auto const index = layout::slot_index(static_cast<int>(tensor), problem.rows, at);
								out.at(tile * tile_floats + static_cast<std::size_t>(index)) = fragments.at(warp)
								                                                                   .at(tensor)
								                                                                   .front()
								                                                                   .at(static_cast<std::size_t>(n))
								                                                                   .at(lane)
								                                                                   .at(static_cast<std::size_t>(reg));
							}
						}
					}
				}
			}
		}
	};
	if (is_e4m3(matrix)) {
		run.template operator()<layout::ActivationTile<model::Format::E4m3, row_tiles>>();
	} else {
		run.template operator()<layout::ActivationTile<model::Format::Bf16, row_tiles>>();
	}
	return out;
}

/**
 * What the prefill gemm writes for `problem` (rows a multiple of 32): fp32 rows [tensor][row][output], each element
 * computed by the warp and lane that owns it in its work item, over the decode's segments.
 */
export [[nodiscard]] auto prefill(Problem const& problem) -> std::vector<float> {
	auto const& matrix = problem.matrix;
	core::invariant(problem.rows % layout::prefill::block_rows == 0);
	auto const tiles = bake(problem);
	auto const tensors = count(matrix.tensors);
	auto const width = count(matrix.output_tiles * tile_rows);
	auto out = std::vector<float>(tensors * static_cast<std::size_t>(problem.rows) * width);
	auto slabs = std::vector<std::vector<unsigned>>{};
	auto first_rows = std::vector<std::vector<int>>{};
	for (auto warp = 0U; warp < 8; ++warp) {
		slabs.push_back({layout::prefill::tile_slab(warp, 0), layout::prefill::tile_slab(warp, 1)});
		first_rows.push_back({layout::prefill::first_row(warp, 0), layout::prefill::first_row(warp, 1)});
	}
	auto run = [&]<class Tile>() {
		for (auto block = 0; block < problem.rows / layout::prefill::block_rows; ++block) {
			for (auto tile = 0U; tile < matrix.output_tiles; ++tile) {
				auto const fragments = tile_fragments<Tile>(problem, tiles, tile, slabs, first_rows, [&](unsigned k_tile) {
					return stage_activations<Tile>(problem, block * layout::prefill::block_rows, k_tile);
				});
				for (auto warp = 0U; warp < 8; ++warp) {
					for (auto tensor = std::size_t{0}; tensor < tensors; ++tensor) {
						for (auto slab = 0; slab < layout::prefill::warp_slabs; ++slab) {
							for (auto n = 0; n < layout::prefill::warp_row_tiles; ++n) {
								for (auto lane = 0U; lane < 32; ++lane) {
									for (auto reg = 0; reg < 4; ++reg) {
										auto const at = layout::prefill::element(warp, lane, slab, n, reg);
										auto const row =
										    static_cast<std::size_t>(block) * static_cast<std::size_t>(layout::prefill::block_rows) +
										    static_cast<std::size_t>(at.row);
										auto const column =
										    static_cast<std::size_t>(tile) * count(tile_rows) + static_cast<std::size_t>(at.output);
										auto const& fragment = fragments.at(warp)
										                           .at(tensor)
										                           .at(static_cast<std::size_t>(slab))
										                           .at(static_cast<std::size_t>(n));
										out.at((tensor * static_cast<std::size_t>(problem.rows) + row) * width + column) =
										    fragment.at(lane).at(static_cast<std::size_t>(reg));
									}
								}
							}
						}
					}
				}
			}
		}
	};
	if (is_e4m3(matrix)) {
		run.template operator()<layout::prefill::ActivationTile<model::Format::E4m3>>();
	} else {
		run.template operator()<layout::prefill::ActivationTile<model::Format::Bf16>>();
	}
	return out;
}

/** A weight as the tensor core takes it: an e4m3 code converted exactly, or the bf16 as stored. */
[[nodiscard]] auto weight(Matrix const& matrix, std::uint16_t stored) -> prims::Bf16 {
	if (is_e4m3(matrix)) {
		return prims::to_bf16(prims::E4m3{.bits = static_cast<std::uint8_t>(stored & 0xffU)});
	}
	return {.bits = stored};
}

/**
 * The straightforward product, one output element at a time from the plain row-major inputs, in the one summation
 * order: per SM segment, each e4m3 unit an mma chain of its 128 k from +0 folded by fma with its scale, or a bf16
 * segment one chain over all its k; the segments' partials added left to right. Layout [tensor][row][output].
 */
export [[nodiscard]] auto reference(Problem const& problem) -> std::vector<float> {
	auto const& matrix = problem.matrix;
	auto const tensors = count(matrix.tensors);
	auto const rows = static_cast<std::size_t>(problem.rows);
	auto const span = count(unit_k(matrix));
	auto const extent = count(k_extent(matrix));
	auto const width = count(matrix.output_tiles * tile_rows);
	auto out = std::vector<float>(tensors * rows * width);
	for (auto tile = 0U; tile < matrix.output_tiles; ++tile) {
		auto const holders = tile_holders(matrix, tile);
		for (auto tensor = std::size_t{0}; tensor < tensors; ++tensor) {
			for (auto output = std::size_t{0}; output < count(tile_rows); ++output) {
				for (auto row = std::size_t{0}; row < rows; ++row) {
					auto total = 0.0F;
					for (auto sm = holders.begin; sm < holders.begin + holders.length; ++sm) {
						auto const part = segment(matrix, tile, sm);
						auto partial = 0.0F;
						for (auto k_tile = static_cast<unsigned>(part.begin); k_tile < part.begin + part.length; ++k_tile) {
							auto const unit = static_cast<std::size_t>(tile) * count(matrix.k_tiles) + k_tile;
							auto const& weights = problem.weights.at(unit * tensors + tensor);
							auto w = std::vector<prims::Bf16>(span);
							auto a = std::vector<prims::Bf16>(span);
							for (auto k = std::size_t{0}; k < span; ++k) {
								w.at(k) = weight(matrix, weights.at(output * span + k));
								a.at(k) = {.bits = problem.activations.at(row * extent + k_tile * span + k)};
							}
							if (is_e4m3(matrix)) {
								auto const chain =
								    mma::chain<128>(0.0F, std::span<prims::Bf16 const, 128>{w}, std::span<prims::Bf16 const, 128>{a});
								partial = prims::math::fma(chain, scale(problem, tensor, tile, k_tile), partial);
							} else {
								partial = mma::chain<64>(partial, std::span<prims::Bf16 const, 64>{w}, std::span<prims::Bf16 const, 64>{a});
							}
						}
						total = sm == holders.begin ? partial : total + partial;
					}
					out.at((tensor * rows + row) * width + static_cast<std::size_t>(tile) * count(tile_rows) + output) = total;
				}
			}
		}
	}
	return out;
}

/** The decode's [tile][tensor][row][128] in the rows layout [tensor][row][output]. */
export [[nodiscard]] auto rows_of(Problem const& problem, std::span<float const> tiles) -> std::vector<float> {
	auto const& matrix = problem.matrix;
	auto const tensors = count(matrix.tensors);
	auto const rows = static_cast<std::size_t>(problem.rows);
	auto const width = count(matrix.output_tiles * tile_rows);
	auto const tile_floats = tensors * rows * count(tile_rows);
	auto out = std::vector<float>(tiles.size());
	for (auto tile = std::size_t{0}; tile < count(matrix.output_tiles); ++tile) {
		for (auto tensor = std::size_t{0}; tensor < tensors; ++tensor) {
			for (auto row = std::size_t{0}; row < rows; ++row) {
				for (auto output = std::size_t{0}; output < count(tile_rows); ++output) {
					out.at((tensor * rows + row) * width + tile * count(tile_rows) + output) =
					    tiles[tile * tile_floats + (tensor * rows + row) * count(tile_rows) + output];
				}
			}
		}
	}
	return out;
}

/**
 * silu(gate) * up rounded once to bf16, from rows [tensor][row][output] of interleaved gate (tensor 0) and up
 * (tensor 1): the product rows [row][output] as words of two adjacent outputs, the first in the low half.
 */
export [[nodiscard]] auto product(Problem const& problem, std::span<float const> rows) -> std::vector<std::uint32_t> {
	core::invariant(problem.matrix.tensors == 2);
	auto const count_rows = static_cast<std::size_t>(problem.rows);
	auto const width = count(problem.matrix.output_tiles * tile_rows);
	auto words = std::vector<std::uint32_t>(count_rows * width / 2);
	for (auto row = std::size_t{0}; row < count_rows; ++row) {
		for (auto output = std::size_t{0}; output < width; ++output) {
			auto const gate = rows[row * width + output];
			auto const up = rows[(count_rows + row) * width + output];
			auto const bits = std::uint32_t{prims::round_to_bf16(model::mlp_product(gate, up)).bits};
			words.at((row * width + output) / 2) |= bits << (16U * static_cast<std::uint32_t>(output % 2));
		}
	}
	return words;
}

/** The residual after one row-parallel phase, and the squares its next norm adds. */
export struct ResidualUpdate {
	/** [row][hidden] fp32. */
	std::vector<float> residual;
	/** [tile][row] fp32: each output tile's sum of squares of the new residual, per row. */
	std::vector<float> squares;
};

/**
 * What `add_to_residual` leaves, run on every output tile of the hidden state: residual + (own + peer) per element,
 * and per tile and row the squares under `model::hidden_tile` (lane l folds outputs 4l .. 4l + 3 by fma from +0,
 * ascending, then the butterfly at distance 16, 8, 4, 2, 1). `own` and `peer` are [tile][row][128].
 */
export [[nodiscard]] auto residual_update(std::span<float const> own, std::span<float const> peer, std::span<float const> residual,
                                          int rows) -> ResidualUpdate {
	auto const hidden = static_cast<std::size_t>(model::hidden);
	auto const width = count(tile_rows);
	auto const tiles = hidden / width;
	auto const count_rows = static_cast<std::size_t>(rows);
	auto update =
	    ResidualUpdate{.residual = std::vector<float>(residual.begin(), residual.end()), .squares = std::vector<float>(tiles * count_rows)};
	for (auto tile = std::size_t{0}; tile < tiles; ++tile) {
		for (auto row = std::size_t{0}; row < count_rows; ++row) {
			for (auto output = std::size_t{0}; output < width; ++output) {
				auto const from = (tile * count_rows + row) * width + output;
				auto& value = update.residual.at(row * hidden + tile * width + output);
				value = value + (own[from] + peer[from]);
			}
			auto const updated = std::span{update.residual}.subspan(row * hidden, hidden);
			update.squares.at(tile * count_rows + row) = model::lane_sum<model::hidden_tile>(updated, updated, tile * width);
		}
	}
	return update;
}

}

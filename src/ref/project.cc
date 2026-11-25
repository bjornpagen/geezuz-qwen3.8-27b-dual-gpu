// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight products in the engine's order (`model::Sum::Weights`): per SM, per unit, mma chains over the unit's k16 steps
// (a fresh chain per e4m3 unit, folded with its block scale; one chain over a bf16 SM's run), the SMs' partials added in
// SM order, the cards' partial sums added card 0 first.
export module engine.ref:project;

import std;
import engine.core;
import engine.model;
import engine.schedule;
import engine.thread_pool;
import :accumulation;
import :checkpoint;
import :numeric;
import :plan;

namespace engine::ref {

namespace model = engine::model;
namespace schedule = engine::schedule;

using model::Count;

export inline constexpr Count tile_rows = schedule::tile_rows;

/** One unit's weights for one output block, decoded exactly: rows x columns bf16, and each row's block scale. */
export struct TileData {
	std::vector<Bf16> weights;
	/** The e4m3 block scale of each row; empty for a bf16 tile. */
	std::vector<float> scales;
};

/** Reads one unit of one output block. */
export template<class R>
concept TileReader = requires(R const& reader, TileWork const& work, Count unit) {
	{ reader.read(work, unit) } -> std::same_as<TileData>;
};

/**
 * The sums one card computes for one output block, token-major ([token][row of the block]): every segment's partials
 * in the engine's order, then the segments added in SM order. `x` holds the card's activation columns.
 */
export template<Accumulation A, TileReader R>
[[nodiscard]] auto tile_sums(TileWork const& work, model::Format format, R const& reader, Rows<Bf16> const& x) -> Rows<float> {
	auto const columns = schedule::tile_columns(format);
	auto const tokens = x.count();
	auto total = Rows<float>::zeros(tokens, tile_rows);
	for (auto const [index, segment] : std::views::enumerate(work.segments)) {
		auto partial = Rows<float>::zeros(tokens, tile_rows);
		for (auto unit = segment.first; unit < segment.first + segment.count; ++unit) {
			auto const tile = reader.read(work, unit);
			for (auto row = Count{0}; row < tile_rows; ++row) {
				auto const weights = std::span<Bf16 const>{tile.weights}.subspan(row * columns, columns);
				for (auto token = Count{0}; token < tokens; ++token) {
					auto const activations = x.row(token).subspan(unit * columns, columns);
					auto& sum = partial.row(token)[row];
					switch (format) {
					case model::Format::E4m3: {
						auto chain = 0.0F;
						for (auto k = Count{0}; k < columns; k += 16) {
							chain = A::step(chain, step_of(weights, k), step_of(activations, k));
						}
						sum = std::fma(chain, tile.scales[row], sum);
						break;
					}
					case model::Format::Bf16:
						for (auto k = Count{0}; k < columns; k += 16) {
							sum = A::step(sum, step_of(weights, k), step_of(activations, k));
						}
						break;
					}
				}
			}
		}
		if (index == 0) {
			total = std::move(partial);
		} else {
			for (auto const [into, add] : std::views::zip(total.values, partial.values)) {
				into = into + add;
			}
		}
	}
	return total;
}

/** Reads a checkpoint tensor's tiles: rows and columns through the card's shard, codes decoded exactly. */
struct CheckpointTiles {
	Checkpoint const& checkpoint;
	Count tensor;
	schedule::Instance at;

	[[nodiscard]] auto read(TileWork const& work, Count unit) const -> TileData {
		auto const& entry = model::inventory[tensor];
		auto const columns = schedule::tile_columns(entry.format);
		auto const width = entry.shape.extents[1];
		auto const data = checkpoint.data(tensor, at);
		auto const first_column = source_column(tensor, work.card, unit * columns);
		auto tile = TileData{.weights = std::vector<Bf16>(tile_rows * columns), .scales = {}};
		for (auto row = Count{0}; row < tile_rows; ++row) {
			auto const source = source_row(tensor, work.card, (work.block * tile_rows) + row);
			auto const into = std::span{tile.weights}.subspan(row * columns, columns);
			switch (entry.format) {
			case model::Format::E4m3: {
				auto const codes = data.subspan((source * width) + first_column, columns);
				std::ranges::transform(codes, into.begin(), decode_e4m3);
				auto const scale_columns = model::divide_up(width, model::scale_block);
				auto const scale = bf16_at(checkpoint.scales(tensor, at), ((source / model::scale_block) * scale_columns) + (first_column / model::scale_block));
				tile.scales.push_back(widen(scale));
				break;
			}
			case model::Format::Bf16:
				for (auto column = Count{0}; column < columns; ++column) {
					into[column] = bf16_at(data, (source * width) + first_column + column);
				}
				break;
			}
		}
		return tile;
	}
};

/** The activation columns card `card` holds of `tensor`'s input, in its order. */
[[nodiscard]] auto card_columns(Count tensor, model::Card card, Rows<Bf16> const& x) -> Rows<Bf16> {
	auto const width = model::card_shape(model::inventory[tensor], std::to_underlying(card)).extents[1];
	auto out = Rows<Bf16>::zeros(x.count(), width);
	for (auto token = Count{0}; token < x.count(); ++token) {
		for (auto column = Count{0}; column < width; ++column) {
			out.row(token)[column] = x.row(token)[source_column(tensor, card, column)];
		}
	}
	return out;
}

/** One block's sums into the rows of the whole tensor they belong to. */
auto place(Count tensor, TileWork const& work, Rows<float> const& sum, Rows<float>& into) -> void {
	for (auto row = Count{0}; row < tile_rows; ++row) {
		auto const source = source_row(tensor, work.card, (work.block * tile_rows) + row);
		for (auto token = Count{0}; token < sum.count(); ++token) {
			into.row(token)[source] = sum.row(token)[row];
		}
	}
}

auto scatter(Product const& product, std::span<Rows<float> const> sums, Rows<float>& into) -> void {
	for (auto const [work, sum] : std::views::zip(product.tiles, sums)) {
		place(product.tensor, work, sum, into);
	}
}

/**
 * `product` applied to `x` (token-major, the tensor's whole input width): token-major outputs over the tensor's whole
 * output rows, the rows the product selects filled, the rest zero.
 */
export template<Accumulation A>
[[nodiscard]] auto project(Checkpoint const& checkpoint, thread_pool::Pool& pool, Product const& product, Rows<Bf16> const& x) -> Rows<float> {
	auto const& entry = model::inventory[product.tensor];
	core::invariant(x.width == entry.shape.extents[1]);
	auto const inputs = std::array{card_columns(product.tensor, model::Card::Zero, x), card_columns(product.tensor, model::Card::One, x)};
	auto const reader = CheckpointTiles{.checkpoint = checkpoint, .tensor = product.tensor, .at = product.at};
	auto sums = std::vector<Rows<float>>(product.tiles.size());
	pool.share(product.tiles.size(), [&](std::size_t index) {
		auto const& work = product.tiles[index];
		sums[index] = tile_sums<A>(work, entry.format, reader, inputs.at(std::to_underlying(work.card)));
	});
	auto const rows = entry.shape.extents[0];
	if (!summed_across_cards(product.tensor)) {
		// Each row is one card's alone: placed, never added, so a -0 stays -0.
		auto out = Rows<float>::zeros(x.count(), rows);
		scatter(product, sums, out);
		return out;
	}
	auto cards = std::array{Rows<float>::zeros(x.count(), rows), Rows<float>::zeros(x.count(), rows)};
	for (auto const [work, sum] : std::views::zip(product.tiles, sums)) {
		place(product.tensor, work, sum, cards.at(std::to_underlying(work.card)));
	}
	for (auto const [into, other] : std::views::zip(cards[0].values, cards[1].values)) {
		into = into + other;
	}
	return std::move(cards[0]);
}

/**
 * A bf16 matrix read whole by its phase (in_proj_a, in_proj_b): one mma chain over the whole input width per output
 * element, from +0 in ascending k.
 */
export template<Accumulation A>
[[nodiscard]] auto whole_product(std::span<std::byte const> matrix, Count rows, Rows<Bf16> const& x) -> Rows<float> {
	auto const width = x.width;
	core::invariant(matrix.size() == rows * width * 2 && width % 16 == 0);
	auto out = Rows<float>::zeros(x.count(), rows);
	auto weights = std::vector<Bf16>(width);
	for (auto row = Count{0}; row < rows; ++row) {
		for (auto column = Count{0}; column < width; ++column) {
			weights[column] = bf16_at(matrix, (row * width) + column);
		}
		for (auto token = Count{0}; token < x.count(); ++token) {
			auto chain = 0.0F;
			for (auto k = Count{0}; k < width; k += 16) {
				chain = A::step(chain, step_of(weights, k), step_of(x.row(token), k));
			}
			out.row(token)[row] = chain;
		}
	}
	return out;
}

}

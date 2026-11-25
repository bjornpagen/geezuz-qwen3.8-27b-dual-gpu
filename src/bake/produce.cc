// SPDX-License-Identifier: AGPL-3.0-only
//
// The bytes of one placement on one card, in image order, each a copy or a permutation of checkpoint bytes:
//
// - tiles: for each selected 128-row block in selection order, for each tile of k in order, for each tensor the
//   placement interleaves, that tensor's tile of the block, encoded by the tile encoding (e4m3 128 x 128, bf16 128 x 64);
//   so one output block's units are contiguous and a unit holds one tile of every interleaved tensor;
// - scales: for each selected block, the card's row of that block's bf16 scales, one per 128 columns the card holds;
// - whole: the card's part of the tensor, row-major, as the checkpoint stores it.
//
// A card's row or column maps to the checkpoint's through the tensor's split: the ranges a shard holds, laid end to
// end, or for the vocabulary the permutation.
export module engine.bake:produce;

import std;
import engine.core;
import engine.model;
import engine.prims;
import engine.schedule;
import :error;
import :plan;
import :source;

namespace engine::bake {

namespace tile = device::prims::tile;

/** The index `index` of the card's part of a split axis has in the whole axis: the shard's ranges laid end to end. */
[[nodiscard]] constexpr auto through(model::Shard const& part, std::size_t index) -> std::size_t {
	for (auto const& range : std::span{part.ranges}.first(part.count)) {
		if (index < range.length) {
			return range.begin + index;
		}
		index -= range.length;
	}
	core::invariant(false);
	return 0;
}

/** One card's part of one tensor copy: how its rows and columns map to the checkpoint's. */
template<Plan P>
struct CardPart {
	model::Tensor tensor;
	model::Card card;
	model::Shard shard;
	model::Shape shape;

	[[nodiscard]] static constexpr auto of(std::size_t index, model::Card card) -> CardPart {
		auto const& tensor = std::span{P::inventory}[index];
		auto const rank = std::size_t{std::to_underlying(card)};
		return {.tensor = tensor, .card = card, .shard = model::shard(tensor, rank), .shape = model::card_shape(tensor, rank)};
	}

	[[nodiscard]] constexpr auto source_row(std::size_t row) const -> std::size_t {
		if (shard.axis != 0) {
			return row;
		}
		// The vocabulary's rows are a permutation, which no range of the shard can say.
		return tensor.split == model::Split::Vocabulary ? P::token(card, row) : through(shard, row);
	}

	[[nodiscard]] constexpr auto source_column(std::size_t column) const -> std::size_t {
		return shard.axis == 1 ? through(shard, column) : column;
	}

	/** Elements of one row of the whole tensor, every axis after the first. */
	[[nodiscard]] constexpr auto source_row_elements() const -> std::size_t {
		return model::elements(tensor.shape) / tensor.shape.extents[0];
	}

	[[nodiscard]] constexpr auto row_elements() const -> std::size_t {
		return model::elements(shape) / shape.extents[0];
	}
};

/** Each selected block, in selection order. */
template<class Visit>
auto each_block(schedule::Blocks const& blocks, Visit const& visit) -> core::Result<void, BakeError> {
	for (auto group = std::size_t{0}; group < blocks.groups; ++group) {
		for (auto taken = std::size_t{0}; taken < blocks.take; ++taken) {
			if (auto done = visit(blocks.first + (group * blocks.stride) + taken); !done) {
				return done;
			}
		}
	}
	return {};
}

using TileBytes = std::array<std::byte, tile::bytes>;

[[nodiscard]] auto to_bytes(prims::TileWords const& words) -> TileBytes {
	auto bytes = TileBytes{};
	for (auto word = std::size_t{0}; word < words.size(); ++word) {
		for (auto byte = std::size_t{0}; byte < 4; ++byte) {
			bytes.at((4 * word) + byte) = std::byte{core::narrow<std::uint8_t>((words[word] >> (8U * byte)) & 0xffU)};
		}
	}
	return bytes;
}

/** The checkpoint bytes of the `columns` elements of `part`'s card row `row` starting at card column `column`. */
template<Plan P>
[[nodiscard]] auto source_run(CardPart<P> const& part, std::span<std::byte const> whole, std::size_t row, std::size_t column, std::size_t columns)
    -> std::span<std::byte const> {
	auto const first = part.source_column(column);
	// A tile's columns lie inside one range of the shard: every range is whole tiles wide.
	core::invariant(part.source_column(column + columns - 1) == first + columns - 1);
	auto const element = model::element_bytes(part.tensor.format);
	auto const at = (part.source_row(row) * part.source_row_elements()) + first;
	return whole.subspan(at * element, columns * element);
}

/** The encoded e4m3 tile of `part`'s block `block` and k tile `k`; a nan code is refused at its checkpoint position. */
template<Plan P>
[[nodiscard]] auto e4m3_tile(CardPart<P> const& part, std::string const& name, std::span<std::byte const> whole, std::size_t block, std::size_t k)
    -> core::Result<TileBytes, BakeError> {
	auto codes = std::array<std::uint8_t, tile::bytes>{};
	for (auto row = std::size_t{0}; row < tile::e4m3_rows; ++row) {
		auto const run = source_run(part, whole, (block * tile::e4m3_rows) + row, k * tile::e4m3_columns, tile::e4m3_columns);
		std::ranges::transform(run, std::next(codes.begin(), static_cast<std::ptrdiff_t>(row * tile::e4m3_columns)),
		                       [](std::byte code) { return std::to_integer<std::uint8_t>(code); });
	}
	auto const words = prims::bake_e4m3_tile(codes);
	if (!words) {
		auto const at = static_cast<std::size_t>(std::ranges::find_if(codes, [](std::uint8_t code) { return !prims::FiniteE4m3::parse(code); }) - codes.begin());
		auto const row = (block * tile::e4m3_rows) + (at / tile::e4m3_columns);
		return refuse(NanCode{.tensor = name, .row = part.source_row(row), .column = part.source_column((k * tile::e4m3_columns) + (at % tile::e4m3_columns))});
	}
	return to_bytes(*words);
}

template<Plan P>
[[nodiscard]] auto bf16_tile(CardPart<P> const& part, std::span<std::byte const> whole, std::size_t block, std::size_t k) -> TileBytes {
	auto values = std::array<std::uint16_t, tile::bytes / 2>{};
	for (auto row = std::size_t{0}; row < tile::bf16_rows; ++row) {
		auto const run = source_run(part, whole, (block * tile::bf16_rows) + row, k * tile::bf16_columns, tile::bf16_columns);
		for (auto column = std::size_t{0}; column < tile::bf16_columns; ++column) {
			values.at((row * tile::bf16_columns) + column) = core::narrow<std::uint16_t>(
			    std::to_integer<unsigned>(run[2 * column]) | (std::to_integer<unsigned>(run[(2 * column) + 1]) << 8U));
		}
	}
	return to_bytes(prims::bake_bf16_tile(values));
}

[[nodiscard]] constexpr auto tile_columns(model::Format format) -> std::size_t {
	switch (format) {
	case model::Format::E4m3:
		return tile::e4m3_columns;
	case model::Format::Bf16:
		return tile::bf16_columns;
	}
	std::unreachable();
}

/** The tiles of a ring placement on `card`, in image order, each passed to `emit`. */
template<Plan P, class Emit>
[[nodiscard]] auto produce_tiles(Source const& source, schedule::Placement const& placement, model::Card card, Emit const& emit)
    -> core::Result<void, BakeError> {
	auto const& read = placement.read;
	auto parts = std::vector<CardPart<P>>{};
	auto names = std::vector<std::string>{};
	auto wholes = std::vector<std::span<std::byte const>>{};
	for (auto const tensor : std::span{read.tensor}.first(read.tensor_count)) {
		parts.push_back(CardPart<P>::of(tensor, card));
		names.push_back(checkpoint_name(parts.back().tensor, placement.instance));
		wholes.push_back(source.bytes(names.back()));
	}
	auto const k_tiles = parts.front().shape.extents[1] / tile_columns(parts.front().tensor.format);
	return each_block(P::blocks(read.select, read.tensor[0], card), [&](std::size_t block) -> core::Result<void, BakeError> {
		for (auto k = std::size_t{0}; k < k_tiles; ++k) {
			for (auto which = std::size_t{0}; which < parts.size(); ++which) {
				auto const& part = parts[which];
				switch (part.tensor.format) {
				case model::Format::E4m3: {
					auto const tile_bytes = e4m3_tile(part, names[which], wholes[which], block, k);
					if (!tile_bytes) {
						return std::unexpected(tile_bytes.error());
					}
					if (auto done = emit(std::span<std::byte const>{*tile_bytes}); !done) {
						return done;
					}
					break;
				}
				case model::Format::Bf16:
					if (auto done = emit(std::span<std::byte const>{bf16_tile(part, wholes[which], block, k)}); !done) {
						return done;
					}
					break;
				}
			}
		}
		return {};
	});
}

/** The block scales of a scale placement on `card`: each selected block's row of the card's scale columns. */
template<Plan P, class Emit>
[[nodiscard]] auto produce_scales(Source const& source, schedule::Placement const& placement, model::Card card, Emit const& emit)
    -> core::Result<void, BakeError> {
	auto const& read = placement.read;
	auto const part = CardPart<P>::of(read.tensor[0], card);
	auto const scales = source.bytes(scale_name(checkpoint_name(part.tensor, placement.instance)));
	auto const block = model::scale_block;
	auto const columns = model::divide_up(part.shape.extents[1], block);
	auto const stored_columns = model::scale_shape(part.tensor.shape).extents[1];
	auto row_bytes = std::vector<std::byte>(columns * 2);
	return each_block(P::blocks(read.select, read.tensor[0], card), [&](std::size_t selected) -> core::Result<void, BakeError> {
		auto const stored_row = part.source_row(selected * block) / block;
		for (auto column = std::size_t{0}; column < columns; ++column) {
			auto const stored = (stored_row * stored_columns) + (part.source_column(column * block) / block);
			std::ranges::copy(scales.subspan(stored * 2, 2), std::next(row_bytes.begin(), static_cast<std::ptrdiff_t>(column * 2)));
		}
		return emit(std::span<std::byte const>{row_bytes});
	});
}

/** A small tensor's card part, row-major, read whole by the consumers. */
template<Plan P, class Emit>
[[nodiscard]] auto produce_whole(Source const& source, schedule::Placement const& placement, model::Card card, Emit const& emit)
    -> core::Result<void, BakeError> {
	auto const part = CardPart<P>::of(placement.read.tensor[0], card);
	auto const whole = source.bytes(checkpoint_name(part.tensor, placement.instance));
	auto const element = model::element_bytes(part.tensor.format);
	auto const columns = part.row_elements();
	for (auto row = std::size_t{0}; row < part.shape.extents[0]; ++row) {
		// A column split holds whole ranges of each row; any other split holds whole rows.
		auto const runs = part.shard.axis == 1 ? std::span{part.shard.ranges}.first(part.shard.count)
		                                       : std::span<model::Range const>{};
		auto const source_row = part.source_row(row) * part.source_row_elements();
		if (runs.empty()) {
			if (auto done = emit(whole.subspan(source_row * element, columns * element)); !done) {
				return done;
			}
			continue;
		}
		for (auto const& range : runs) {
			if (auto done = emit(whole.subspan((source_row + range.begin) * element, range.length * element)); !done) {
				return done;
			}
		}
	}
	return {};
}

/** The bytes of `placement` on `card`, in image order, passed to `emit` in pieces. */
export template<Plan P, class Emit>
[[nodiscard]] auto produce(Source const& source, schedule::Placement const& placement, model::Card card, Emit const& emit)
    -> core::Result<void, BakeError> {
	switch (placement.read.part) {
	case schedule::Part::Tiles:
		return produce_tiles<P>(source, placement, card, emit);
	case schedule::Part::Scales:
		return produce_scales<P>(source, placement, card, emit);
	case schedule::Part::Whole:
		return produce_whole<P>(source, placement, card, emit);
	}
	std::unreachable();
}

}

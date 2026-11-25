// SPDX-License-Identifier: AGPL-3.0-only
//
// The bake inverted, as an oracle: rebuilds every checkpoint tensor a plan places from the card and host images (each
// card's part through the split, each tile through the device's own decode model, each lm_head row through the
// vocabulary permutation) and compares it with the checkpoint byte for byte, counting how many cards supply each
// element. It walks the images in the direction opposite to the bake: from an image position to the checkpoint
// element it must hold, with the tile encoding's forward slot map where the bake uses the inverse.
export module engine.bake_inverse;

import std;
import engine.bake;
import engine.image;
import engine.model;
import engine.prims;
import engine.schedule;

namespace engine::bake_inverse {

namespace prims = engine::device::prims;
namespace tile = prims::tile;

using bake::cards;
using bake::Plan;

/** A copy of a tensor: its inventory index and which instance. */
struct Copy {
	std::size_t tensor;
	model::Stream stream;
	std::size_t layer;

	[[nodiscard]] auto operator<=>(Copy const&) const = default;
};

/** Where a copy's bytes are in one card image: the placement, and the copy's position among the tensors it interleaves. */
struct Held {
	model::Card card;
	std::size_t placement;
	std::size_t which;
};

/** The checkpoint index of `part`'s card index `index` along its split axis, from the shard's ranges. */
[[nodiscard]] auto from_ranges(model::Shard const& part, std::size_t index) -> std::optional<std::size_t> {
	for (auto const& range : std::span{part.ranges}.first(part.count)) {
		if (index < range.length) {
			return range.begin + index;
		}
		index -= range.length;
	}
	return std::nullopt;
}

/** How card rows and columns of a tensor map to the checkpoint's: the split, read from the model description. */
template<Plan P>
struct Inverse {
	model::Tensor tensor;
	model::Card card;
	model::Shard shard;

	[[nodiscard]] auto row(std::size_t card_row) const -> std::size_t {
		if (tensor.split == model::Split::Vocabulary) {
			return P::token(card, card_row);
		}
		return shard.axis == 0 ? from_ranges(shard, card_row).value_or(tensor.shape.extents[0]) : card_row;
	}

	[[nodiscard]] auto column(std::size_t card_column) const -> std::size_t {
		return shard.axis == 1 ? from_ranges(shard, card_column).value_or(tensor.shape.extents[1]) : card_column;
	}
};

/** A tensor being rebuilt: its bytes, and how many cards supplied each element. */
struct Rebuilt {
	std::vector<std::byte> bytes;
	std::vector<std::uint8_t> supplied;
	std::size_t element_bytes;
	bool conflict = false;

	static auto of(std::size_t elements, std::size_t element_bytes) -> Rebuilt {
		return {.bytes = std::vector<std::byte>(elements * element_bytes), .supplied = std::vector<std::uint8_t>(elements, 0), .element_bytes = element_bytes};
	}

	/** Stores an element a card supplies; a second card must supply the same bytes. */
	auto put(std::size_t element, std::span<std::byte const> value) -> void {
		if (element >= supplied.size()) {
			conflict = true;
			return;
		}
		auto const at = std::span{bytes}.subspan(element * element_bytes, element_bytes);
		if (supplied[element] != 0 && !std::ranges::equal(at, value)) {
			conflict = true;
		}
		std::ranges::copy(value, at.begin());
		supplied[element] = static_cast<std::uint8_t>(supplied[element] + 1);
	}

	[[nodiscard]] auto matches(std::span<std::byte const> stored, std::uint8_t holders) const -> bool {
		return !conflict && std::ranges::equal(bytes, stored) && std::ranges::all_of(supplied, [holders](std::uint8_t count) { return count == holders; });
	}
};

/** The tile encoding's slots, by tile row and column: the forward map, computed once. */
struct Slots {
	std::vector<tile::Slot> e4m3;
	std::vector<tile::Slot> bf16;

	static auto make() -> Slots {
		auto slots = Slots{};
		for (auto row = 0; row < tile::e4m3_rows; ++row) {
			for (auto column = 0; column < tile::e4m3_columns; ++column) {
				slots.e4m3.push_back(tile::e4m3_slot(row, column));
			}
		}
		for (auto row = 0; row < tile::bf16_rows; ++row) {
			for (auto column = 0; column < tile::bf16_columns; ++column) {
				slots.bf16.push_back(tile::bf16_slot(row, column));
			}
		}
		return slots;
	}
};

/** The e4m3 code of each bf16 the device decode can produce, or nothing. */
[[nodiscard]] auto code_table() -> std::vector<std::optional<std::uint8_t>> {
	auto table = std::vector<std::optional<std::uint8_t>>(0x10000);
	for (auto code = 0U; code < 0x100U; ++code) {
		auto const e4m3 = prims::E4m3{.bits = static_cast<unsigned char>(code)};
		if (prims::finite(e4m3)) {
			table.at(prims::to_bf16(e4m3).bits) = static_cast<std::uint8_t>(code);
		}
	}
	return table;
}

[[nodiscard]] auto word_at(std::span<std::byte const> bytes, std::size_t word) -> unsigned {
	auto value = 0U;
	for (auto byte = std::size_t{0}; byte < 4; ++byte) {
		value |= std::to_integer<unsigned>(bytes[(4 * word) + byte]) << (8U * byte);
	}
	return value;
}

[[nodiscard]] auto half(prims::Bf16x2 pair, int place) -> prims::Bf16 {
	return place == 0 ? prims::first(pair) : prims::second(pair);
}

[[nodiscard]] auto le16(unsigned bits) -> std::array<std::byte, 2> {
	return {std::byte{static_cast<unsigned char>(bits & 0xffU)}, std::byte{static_cast<unsigned char>((bits >> 8U) & 0xffU)}};
}

/** The rebuilder's tables: one per run, shared by every tile. */
struct Tables {
	Slots slots;
	std::vector<std::optional<std::uint8_t>> codes;
	bool undecodable = false;
};

/** Puts one tile's elements, decoded through the device decode model, at their checkpoint positions. */
template<Plan P>
auto rebuild_tile(Tables& tables, Inverse<P> const& inverse, std::span<std::byte const> bytes, std::size_t block, std::size_t k, Rebuilt& into)
    -> void {
	auto const columns = inverse.tensor.shape.extents[1];
	switch (inverse.tensor.format) {
	case model::Format::E4m3: {
		auto decoded = std::vector<prims::decode::Pairs>(tile::words);
		for (auto word = std::size_t{0}; word < decoded.size(); ++word) {
			decoded[word] = prims::decode::model(word_at(bytes, word));
		}
		for (auto row = std::size_t{0}; row < tile::e4m3_rows; ++row) {
			auto const source_row = inverse.row((block * tile::e4m3_rows) + row);
			for (auto column = std::size_t{0}; column < tile::e4m3_columns; ++column) {
				auto const slot = tables.slots.e4m3[(row * tile::e4m3_columns) + column];
				auto const pairs = decoded[static_cast<std::size_t>(slot.word)];
				auto const value = half(slot.place < 2 ? pairs.top : pairs.bottom, slot.place % 2);
				auto const code = tables.codes.at(value.bits);
				tables.undecodable = tables.undecodable || !code;
				auto const element = (source_row * columns) + inverse.column((k * tile::e4m3_columns) + column);
				into.put(element, std::array{std::byte{code.value_or(0)}});
			}
		}
		break;
	}
	case model::Format::Bf16:
		for (auto row = std::size_t{0}; row < tile::bf16_rows; ++row) {
			auto const source_row = inverse.row((block * tile::bf16_rows) + row);
			for (auto column = std::size_t{0}; column < tile::bf16_columns; ++column) {
				auto const slot = tables.slots.bf16[(row * tile::bf16_columns) + column];
				auto const pair = prims::Bf16x2{.bits = word_at(bytes, static_cast<std::size_t>(slot.word))};
				auto const element = (source_row * columns) + inverse.column((k * tile::bf16_columns) + column);
				into.put(element, le16(half(pair, slot.place).bits));
			}
		}
		break;
	}
}

[[nodiscard]] auto tile_columns(model::Format format) -> std::size_t {
	return format == model::Format::E4m3 ? tile::e4m3_columns : tile::bf16_columns;
}

/** Each selected block, with its ordinal in the selection. */
template<class Visit>
auto each_block(schedule::Blocks const& blocks, Visit const& visit) -> void {
	auto ordinal = std::size_t{0};
	for (auto group = std::size_t{0}; group < blocks.groups; ++group) {
		for (auto taken = std::size_t{0}; taken < blocks.take; ++taken) {
			visit(ordinal, blocks.first + (group * blocks.stride) + taken);
			++ordinal;
		}
	}
}

/** Puts what one placement holds of a copy at its checkpoint positions: weights into `weights`, scales into `scales`. */
template<Plan P>
auto rebuild_placement(Tables& tables, schedule::Placement const& placement, std::size_t which, model::Card card,
                       std::span<std::byte const> extent, Rebuilt& weights, Rebuilt& scales) -> void {
	auto const& read = placement.read;
	auto const tensor = std::span{read.tensor}[which];
	auto const& entry = std::span{P::inventory}[tensor];
	auto const rank = std::size_t{std::to_underlying(card)};
	auto const inverse = Inverse<P>{.tensor = entry, .card = card, .shard = model::shard(entry, rank)};
	auto const shape = model::card_shape(entry, rank);
	auto const blocks = P::blocks(read.select, read.tensor[0], card);
	switch (read.part) {
	case schedule::Part::Tiles: {
		auto const k_tiles = shape.extents[1] / tile_columns(entry.format);
		each_block(blocks, [&](std::size_t ordinal, std::size_t block) {
			for (auto k = std::size_t{0}; k < k_tiles; ++k) {
				auto const unit = (ordinal * k_tiles) + k;
				auto const at = ((unit * read.tensor_count) + which) * static_cast<std::size_t>(tile::bytes);
				rebuild_tile(tables, inverse, extent.subspan(at, static_cast<std::size_t>(tile::bytes)), block, k, weights);
			}
		});
		break;
	}
	case schedule::Part::Scales: {
		auto const block = model::scale_block;
		auto const columns = model::divide_up(shape.extents[1], block);
		auto const stored_columns = model::scale_shape(entry.shape).extents[1];
		each_block(blocks, [&](std::size_t ordinal, std::size_t selected) {
			for (auto column = std::size_t{0}; column < columns; ++column) {
				auto const element = ((inverse.row(selected * block) / block) * stored_columns) + (inverse.column(column * block) / block);
				scales.put(element, extent.subspan(((ordinal * columns) + column) * 2, 2));
			}
		});
		break;
	}
	case schedule::Part::Whole: {
		auto const row_elements = model::elements(shape) / shape.extents[0];
		auto const stored_row_elements = model::elements(entry.shape) / entry.shape.extents[0];
		auto const element_bytes = model::element_bytes(entry.format);
		for (auto row = std::size_t{0}; row < shape.extents[0]; ++row) {
			for (auto column = std::size_t{0}; column < row_elements; ++column) {
				auto const element = (inverse.row(row) * stored_row_elements) + inverse.column(column);
				weights.put(element, extent.subspan(((row * row_elements) + column) * element_bytes, element_bytes));
			}
		}
		break;
	}
	}
}

/** What the inversion found: copies rebuilt and matched, and the first copy that did not, if any. */
export struct Verdict {
	std::size_t copies;
	std::size_t matched;
	std::vector<std::string> mismatched;
};

/** The payload extents of each image, by catalog index: what the rebuild reads. */
export struct Images {
	std::array<std::vector<std::span<std::byte const>>, model::cards> cards;
	std::vector<std::span<std::byte const>> host;
};

/**
 * Rebuilds every tensor copy `P` places from `images` and compares it with `source`: bit-exact, every element supplied
 * by exactly the cards its split gives it to (every card when replicated, one otherwise), every tile code decodable.
 */
export template<Plan P>
[[nodiscard]] auto invert(bake::Source const& source, Images const& images) -> Verdict {
	auto copies = std::map<Copy, std::vector<Held>>{};
	for (auto const card : cards) {
		auto const placements = P::placements(card);
		for (auto index = std::size_t{0}; index < placements.size(); ++index) {
			auto const& placement = placements[index];
			for (auto which = std::size_t{0}; which < placement.read.tensor_count; ++which) {
				auto const key = Copy{.tensor = std::span{placement.read.tensor}[which], .stream = placement.instance.stream, .layer = placement.instance.layer};
				copies[key].push_back({.card = card, .placement = index, .which = which});
			}
		}
	}
	auto tables = Tables{.slots = Slots::make(), .codes = code_table()};
	auto verdict = Verdict{.copies = 0, .matched = 0, .mismatched = {}};
	for (auto const& [copy, held] : copies) {
		auto const& entry = std::span{P::inventory}[copy.tensor];
		auto const name = bake::checkpoint_name(entry, schedule::Instance{.stream = copy.stream, .layer = copy.layer});
		auto weights = Rebuilt::of(model::elements(entry.shape), model::element_bytes(entry.format));
		auto const e4m3 = entry.format == model::Format::E4m3;
		auto scales = Rebuilt::of(e4m3 ? model::elements(model::scale_shape(entry.shape)) : 0, 2);
		for (auto const& each : held) {
			auto const& placement = P::placements(each.card)[each.placement];
			auto const extent = images.cards.at(std::to_underlying(each.card)).at(each.placement);
			rebuild_placement<P>(tables, placement, each.which, each.card, extent, weights, scales);
		}
		auto const holders = static_cast<std::uint8_t>(entry.split == model::Split::Replicated ? model::cards : 1);
		auto const matched = weights.matches(source.bytes(name), holders) && (!e4m3 || scales.matches(source.bytes(bake::scale_name(name)), holders));
		++verdict.copies;
		if (matched) {
			++verdict.matched;
		} else {
			verdict.mismatched.push_back(name);
		}
	}
	auto const host_tensors = std::span<std::size_t const>{P::host_tensors};
	for (auto index = std::size_t{0}; index < host_tensors.size(); ++index) {
		auto const& entry = std::span{P::inventory}[host_tensors[index]];
		auto const name = bake::checkpoint_name(entry, schedule::Instance{.stream = model::Stream::Head, .layer = 0});
		++verdict.copies;
		if (std::ranges::equal(images.host.at(index), source.bytes(name))) {
			++verdict.matched;
		} else {
			verdict.mismatched.push_back(name);
		}
	}
	if (tables.undecodable) {
		verdict.mismatched.emplace_back("a tile word that decodes to no e4m3 code");
	}
	return verdict;
}

/** The extents of an opened image, by catalog index. */
export template<image::Catalog C>
[[nodiscard]] auto extents_of(image::Image<C> const& opened) -> std::vector<std::span<std::byte const>> {
	auto extents = std::vector<std::span<std::byte const>>{};
	for (auto index = std::size_t{0}; index < std::size(C::extents); ++index) {
		extents.push_back(opened.bytes(index));
	}
	return extents;
}

}

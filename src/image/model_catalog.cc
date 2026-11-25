// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:model_catalog;

import std;
import engine.core;
import engine.model;
import engine.sha256;
import :catalog;
import :description;

namespace engine::image {

/** Which part of a tensor an extent stores. */
export enum class Part : std::uint8_t {
	/** The tensor's elements. */
	Values,
	/** An e4m3 tensor's block scales. */
	Scales,
};

/** Decoder layers in one numbering: the trunk's in order, then the mtp head's. */
export inline constexpr std::size_t decoder_layers = model::trunk_layers + model::mtp_layers;

/** Where an extent of the model catalog comes from. */
export struct Site {
	/** The tensor's position in `model::inventory`. */
	std::uint16_t tensor;
	/** The decoder layer it belongs to; none for a tensor outside the decoder layers. */
	std::optional<std::uint8_t> layer;
	Part part;

	[[nodiscard]] constexpr auto operator==(Site const&) const -> bool = default;
};

[[nodiscard]] constexpr auto mixer_of(std::size_t layer) -> model::Mixer {
	return layer < model::trunk_layers ? std::span{model::trunk_mixers}[layer] : model::mtp_mixer;
}

[[nodiscard]] constexpr auto tensor_at(std::size_t index) -> model::Tensor const& {
	return std::span{model::inventory}[index];
}

/** The shape of a tensor an image holds: one card's part, or the whole of a host tensor. Both cards' parts are alike. */
[[nodiscard]] constexpr auto image_shape(model::Tensor const& tensor) -> model::Shape {
	return tensor.split == model::Split::Host ? tensor.shape : model::card_shape(tensor, 0);
}

[[nodiscard]] constexpr auto parts_of(model::Format format) -> std::size_t {
	switch (format) {
	case model::Format::E4m3:
		return 2;
	case model::Format::Bf16:
		return 1;
	}
	std::unreachable();
}

/** Every extent of the model, in inventory order, then layer order, then values before scales. */
[[nodiscard]] constexpr auto model_sites() -> std::vector<Site> {
	auto sites = std::vector<Site>{};
	auto const add = [&sites](std::size_t tensor, std::optional<std::uint8_t> layer) {
		for (auto part = std::size_t{0}; part < parts_of(tensor_at(tensor).format); ++part) {
			sites.push_back({.tensor = core::narrow<std::uint16_t>(tensor), .layer = layer, .part = part == 0 ? Part::Values : Part::Scales});
		}
	};
	for (auto tensor = std::size_t{0}; tensor < std::size(model::inventory); ++tensor) {
		auto const scope = tensor_at(tensor).scope;
		switch (scope) {
		case model::Scope::Model:
		case model::Scope::Mtp:
			add(tensor, std::nullopt);
			break;
		case model::Scope::DecoderLayer:
		case model::Scope::LinearAttentionLayer:
		case model::Scope::FullAttentionLayer:
			for (auto layer = std::size_t{0}; layer < decoder_layers; ++layer) {
				if (model::in_layer(scope, mixer_of(layer))) {
					add(tensor, core::narrow<std::uint8_t>(layer));
				}
			}
			break;
		}
	}
	return sites;
}

inline constexpr std::size_t model_extent_count = model_sites().size();

[[nodiscard]] constexpr auto extent_of(Site const& site) -> Extent {
	auto const& tensor = tensor_at(site.tensor);
	auto const shape = image_shape(tensor);
	switch (site.part) {
	case Part::Values:
		return {.bytes = model::elements(shape) * model::element_bytes(tensor.format),
		        .encoding = tensor.format == model::Format::E4m3 ? Encoding::E4m3Tiles : Encoding::Bf16};
	case Part::Scales:
		return {.bytes = model::elements(model::scale_shape(shape)) * model::element_bytes(model::Format::Bf16), .encoding = Encoding::Bf16Scales};
	}
	std::unreachable();
}

// The laws the model catalog rests on: both cards hold alike shapes, so one catalog serves both images; and every
// e4m3 part an image holds is whole tiles, so its scales are one per tile and its runs are whole 16 KiB tiles.
[[nodiscard]] consteval auto cards_hold_alike_shapes() -> bool {
	return std::ranges::all_of(model::inventory, [](model::Tensor const& tensor) {
		auto const first = model::card_shape(tensor, 0);
		for (auto card = model::Count{1}; card < model::cards; ++card) {
			auto const other = model::card_shape(tensor, card);
			if (other.rank != first.rank || !std::ranges::equal(other.extents, first.extents)) {
				return false;
			}
		}
		return true;
	});
}

[[nodiscard]] consteval auto e4m3_parts_are_whole_tiles() -> bool {
	return std::ranges::all_of(model::inventory, [](model::Tensor const& tensor) {
		auto const shape = image_shape(tensor);
		return tensor.format != model::Format::E4m3 ||
		       (shape.rank == 2 && shape.extents[0] % model::scale_block == 0 && shape.extents[1] % model::scale_block == 0);
	});
}

static_assert(cards_hold_alike_shapes());
static_assert(e4m3_parts_are_whole_tiles());

/** The catalog of every extent a card's image of this model may hold, and the model description's digest. */
export struct ModelCatalog {
	static constexpr std::array<Site, model_extent_count> sites = [] {
		auto table = std::array<Site, model_extent_count>{};
		std::ranges::copy(model_sites(), table.begin());
		return table;
	}();

	static constexpr std::array<Extent, model_extent_count> extents = [] {
		auto table = std::array<Extent, model_extent_count>{};
		std::ranges::transform(sites, table.begin(), extent_of);
		return table;
	}();

	static constexpr sha256::Digest description = description_digest;
};

static_assert(Catalog<ModelCatalog>);

/** Where the extent `key` of the model catalog comes from. */
export [[nodiscard]] constexpr auto site(Key key) -> Site {
	return std::span{ModelCatalog::sites}[key.index];
}

/** The key of a tensor's part, by its inventory name and decoder layer; none when the model has no such extent. */
export [[nodiscard]] constexpr auto find(std::string_view tensor, std::optional<std::size_t> layer, Part part) -> std::optional<Key> {
	for (auto index = std::size_t{0}; index < ModelCatalog::sites.size(); ++index) {
		auto const& candidate = ModelCatalog::sites.at(index);
		auto const same_layer = candidate.layer.transform([](std::uint8_t value) { return std::size_t{value}; }) == layer;
		if (std::string_view{tensor_at(candidate.tensor).name} == tensor && same_layer && candidate.part == part) {
			return Key{.index = core::narrow<std::uint16_t>(index)};
		}
	}
	return std::nullopt;
}

}

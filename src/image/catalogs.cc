// SPDX-License-Identifier: AGPL-3.0-only
//
// The three images of the model: each card's, laid out by the decode schedule, and the host's, holding what lives in
// pinned host memory. Each image is exactly what its memory holds and is copied there whole.
export module engine.image:catalogs;

import std;
import engine.model;
import engine.schedule;
import :catalog;
import :description;

namespace engine::image {

// An extent's identity packs what the schedule says it reads: its tensors (inventory positions), their count, the part,
// the row-block selection, and the stream and layer of the copy.
static_assert(schedule::tensor_count <= 0x100 && model::trunk_layers + model::mtp_layers <= 0x10000);

[[nodiscard]] constexpr auto identity(schedule::Read const& read, schedule::Instance const& at) -> std::uint64_t {
	auto const tensors = std::span{read.tensor};
	return tensors[0] | (tensors[1] << 8U) | (read.tensor_count << 16U) | (std::uint64_t{std::to_underlying(read.part)} << 20U) |
	       (std::uint64_t{std::to_underlying(read.select)} << 24U) | (std::uint64_t{std::to_underlying(at.stream)} << 32U) |
	       (at.layer << 40U);
}

[[nodiscard]] consteval auto card_extents(model::Card card) -> std::array<Extent, schedule::placements> {
	auto const image = schedule::image(std::to_underlying(card));
	auto extents = std::array<Extent, schedule::placements>{};
	for (auto index = std::size_t{0}; index < extents.size(); ++index) {
		auto const& placement = std::span{image.placements}[index];
		extents.at(index) = {.identity = identity(placement.read, placement.instance), .bytes = placement.bytes, .part = placement.read.part};
	}
	return extents;
}

/** A card's image: the schedule's placements for the card, one extent each, in the schedule's order. */
export template<model::Card card>
struct CardCatalog {
	static constexpr Target target = card_target(card);
	static constexpr std::array<Extent, schedule::placements> extents = card_extents(card);
	static constexpr sha256::Digest description = description_digest;
};

[[nodiscard]] constexpr auto on_host(model::Tensor const& tensor) -> bool {
	return tensor.split == model::Split::Host;
}

inline constexpr std::size_t host_tensors = std::ranges::count_if(model::inventory, on_host);

[[nodiscard]] consteval auto host_extents() -> std::array<Extent, host_tensors> {
	auto extents = std::array<Extent, host_tensors>{};
	auto count = std::size_t{0};
	for (auto tensor = std::size_t{0}; tensor < std::size(model::inventory); ++tensor) {
		auto const& entry = std::span{model::inventory}[tensor];
		if (!on_host(entry)) {
			continue;
		}
		// The schedule places a model-wide tensor's one copy in the head stream; a host tensor is placed nowhere else.
		auto const read = schedule::Read{.tensor = {tensor, tensor}, .tensor_count = 1, .part = Part::Whole, .select = schedule::Select::All};
		auto const at = schedule::Instance{.stream = model::Stream::Head, .layer = 0};
		extents.at(count) = {.identity = identity(read, at), .bytes = model::stored_bytes(entry.format, entry.shape), .part = Part::Whole};
		++count;
	}
	return extents;
}

// A host tensor is one bf16 copy, read whole: it has no block scales and no layers.
static_assert(std::ranges::all_of(model::inventory, [](model::Tensor const& tensor) {
	return !on_host(tensor) || (tensor.format == model::Format::Bf16 && tensor.scope == model::Scope::Model);
}));

/** The host image: every tensor whose split is Host, whole, in inventory order. */
export struct HostCatalog {
	static constexpr Target target = Target::Host;
	static constexpr std::array<Extent, host_tensors> extents = host_extents();
	static constexpr sha256::Digest description = description_digest;
};

static_assert(Catalog<CardCatalog<model::Card::Zero>> && Catalog<CardCatalog<model::Card::One>> && Catalog<HostCatalog>);

/** Whether the layout `place` gives a card's extents is the schedule's own, offset for offset, to the image's end. */
template<model::Card card>
[[nodiscard]] consteval auto layout_is_the_schedules() -> bool {
	auto const image = schedule::image(std::to_underlying(card));
	auto const& placed = layout<CardCatalog<card>>;
	for (auto index = std::size_t{0}; index < placed.size(); ++index) {
		auto const& planned = std::span{image.placements}[index];
		if (placed.at(index).offset != planned.offset || placed.at(index).bytes != planned.bytes) {
			return false;
		}
	}
	return payload_bytes<CardCatalog<card>> == image.bytes;
}

static_assert(layout_is_the_schedules<model::Card::Zero>());
static_assert(layout_is_the_schedules<model::Card::One>());

[[nodiscard]] consteval auto extent_bytes(std::span<Extent const> extents) -> std::uint64_t {
	auto total = std::uint64_t{0};
	for (auto const& extent : extents) {
		total += extent.bytes;
	}
	return total;
}

static_assert(extent_bytes(CardCatalog<model::Card::Zero>::extents) == model::weight_bytes_per_card,
              "a card's image holds exactly the card's weights");
static_assert(extent_bytes(HostCatalog::extents) == model::embedding_host_bytes, "the host image holds exactly the host tensors");

}

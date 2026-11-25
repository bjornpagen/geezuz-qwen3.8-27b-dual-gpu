// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:catalog;

import std;
import engine.model;
import engine.schedule;
import engine.sha256;

namespace engine::image {

/** The memory an image is loaded into whole: one card's, or pinned host memory. */
export enum class Target : std::uint8_t {
	CardZero,
	CardOne,
	Host,
};

static_assert(model::cards == 2, "one card target per tensor-parallel rank");

export [[nodiscard]] constexpr auto card_target(model::Card card) -> Target {
	switch (card) {
	case model::Card::Zero:
		return Target::CardZero;
	case model::Card::One:
		return Target::CardOne;
	}
	std::unreachable();
}

/** How an extent holds its tensor: the schedule's parts, which fix its alignment and the unit its length counts. */
export using schedule::Part;

/** The boundary, in bytes from the payload's start, an extent of `part` begins on. */
export [[nodiscard]] constexpr auto alignment(Part part) -> std::uint64_t {
	switch (part) {
	case Part::Tiles:
		// A run of tiles starts on a tile, so each tile is 64 whole dram chunks, each one bank and row.
		return schedule::tile_bytes;
	case Part::Scales:
	case Part::Whole:
		return schedule::side_alignment;
	}
	std::unreachable();
}

/** The unit an extent's length is a whole number of: tiles, or bf16 elements. */
[[nodiscard]] constexpr auto granule(Part part) -> std::uint64_t {
	switch (part) {
	case Part::Tiles:
		return schedule::tile_bytes;
	case Part::Scales:
	case Part::Whole:
		return model::element_bytes(model::Format::Bf16);
	}
	std::unreachable();
}

/**
 * One extent of an image: what it holds, as a code that tells it from every other extent of its catalog, its length
 * and how it holds its tensor.
 */
export struct Extent {
	std::uint64_t identity;
	std::uint64_t bytes;
	Part part;

	[[nodiscard]] constexpr auto operator==(Extent const&) const -> bool = default;
};

/** Where an extent's bytes lie, from the payload's start. */
export struct Placement {
	std::uint64_t offset;
	std::uint64_t bytes;

	[[nodiscard]] constexpr auto end() const -> std::uint64_t {
		return offset + bytes;
	}

	[[nodiscard]] constexpr auto operator==(Placement const&) const -> bool = default;
};

export [[nodiscard]] constexpr auto align_up(std::uint64_t value, std::uint64_t boundary) -> std::uint64_t {
	return (value + boundary - 1) / boundary * boundary;
}

/**
 * The payload layout of `extents`, in their order: each at the first boundary of its part after the extent before it,
 * so every extent is aligned, they lie in order without overlap, and only padding shorter than a boundary separates
 * them.
 */
export [[nodiscard]] constexpr auto place(std::span<Extent const> extents) -> std::vector<Placement> {
	auto placements = std::vector<Placement>{};
	placements.reserve(extents.size());
	auto end = std::uint64_t{0};
	for (auto const& extent : extents) {
		auto const offset = align_up(end, alignment(extent.part));
		placements.push_back({.offset = offset, .bytes = extent.bytes});
		end = offset + extent.bytes;
	}
	return placements;
}

/** Whether `extents` can be a catalog: each a nonzero whole number of its unit, no identity twice. */
export [[nodiscard]] constexpr auto well_formed(std::span<Extent const> extents) -> bool {
	auto identities = std::vector<std::uint64_t>{};
	for (auto const& extent : extents) {
		if (extent.bytes == 0 || extent.bytes % granule(extent.part) != 0) {
			return false;
		}
		identities.push_back(extent.identity);
	}
	std::ranges::sort(identities);
	return std::ranges::adjacent_find(identities) == identities.end();
}

/**
 * One kind of image: the memory it targets, the extents it holds in the order it holds them, and the digest of the
 * model description they derive from. Everything in a catalog is a compile-time fact; an image's header restates it
 * for the reader to check, and its payload holds the bytes.
 */
export template<class C>
concept Catalog = requires {
	{ C::target } -> std::convertible_to<Target>;
	{ std::span<Extent const>{C::extents} };
	{ C::description } -> std::convertible_to<sha256::Digest>;
} && well_formed(std::span<Extent const>{C::extents});

/** Where each extent of `C` lies in its payload. */
export template<Catalog C>
inline constexpr auto layout = [] {
	auto placements = std::array<Placement, std::size(C::extents)>{};
	std::ranges::copy(place(C::extents), placements.begin());
	return placements;
}();

/** Bytes of the payload of `C`: up to the end of its last extent. */
export template<Catalog C>
inline constexpr std::uint64_t payload_bytes = layout<C>.empty() ? 0 : layout<C>.back().end();

}

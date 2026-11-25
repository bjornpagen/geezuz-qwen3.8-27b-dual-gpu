// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:catalog;

import std;
import engine.core;
import engine.model;
import engine.sha256;

namespace engine::image {

/** How an extent's bytes are stored. It fixes the boundary the extent starts on and the unit its length counts. */
export enum class Encoding : std::uint8_t {
	/** e4m3 weights as whole 128 x 128 tiles, each one contiguous run laid out as the producers stream it. */
	E4m3Tiles,
	/** The bf16 block scales of an e4m3 weight, one per tile. */
	Bf16Scales,
	/** A bf16 tensor. */
	Bf16,
};

/** Bytes of one e4m3 tile: a scale block of one-byte weights. */
export inline constexpr std::uint64_t tile_bytes =
    model::scale_block * model::scale_block * model::element_bytes(model::Format::E4m3);

static_assert(tile_bytes == std::uint64_t{16} * 1024);

/** The boundary, in bytes from the payload's start, an extent of `encoding` begins on. */
export [[nodiscard]] constexpr auto alignment(Encoding encoding) -> std::uint64_t {
	switch (encoding) {
	case Encoding::E4m3Tiles:
		// A tile is 64 whole 256-byte dram chunks, each one bank and row, so streaming it opens each row once.
		return 256;
	case Encoding::Bf16Scales:
	case Encoding::Bf16:
		// cp.async moves 16 bytes from a 16-byte boundary.
		return 16;
	}
	std::unreachable();
}

/** The unit an extent's length is a whole number of. */
[[nodiscard]] constexpr auto granule(Encoding encoding) -> std::uint64_t {
	switch (encoding) {
	case Encoding::E4m3Tiles:
		return tile_bytes;
	case Encoding::Bf16Scales:
	case Encoding::Bf16:
		return model::element_bytes(model::Format::Bf16);
	}
	std::unreachable();
}

/** One stored part of a tensor: its length and its encoding. */
export struct Extent {
	std::uint64_t bytes;
	Encoding encoding;

	[[nodiscard]] constexpr auto operator==(Extent const&) const -> bool = default;
};

/** An extent of a catalog, named by its position there. */
export struct Key {
	std::uint16_t index;

	[[nodiscard]] constexpr auto operator<=>(Key const&) const = default;
};

/** Whether `extents` can be a catalog: every extent reachable by a key, and a nonzero whole number of its unit. */
export [[nodiscard]] constexpr auto well_formed(std::span<Extent const> extents) -> bool {
	return extents.size() <= std::numeric_limits<std::uint16_t>::max() + std::size_t{1} &&
	       std::ranges::all_of(extents, [](Extent const& extent) {
		       return extent.bytes > 0 && extent.bytes % granule(extent.encoding) == 0;
	       });
}

/**
 * Every extent an image may hold, with its length and encoding, and the digest of the description they derive from.
 * Everything in a catalog is a compile-time fact; an image says only which of its extents it holds, in which order.
 */
export template<class C>
concept Catalog = requires {
	{ std::span<Extent const>{C::extents} };
	{ C::description } -> std::convertible_to<sha256::Digest>;
} && well_formed(std::span<Extent const>{C::extents});

/** The extent `key` names in catalog `C`. */
export template<Catalog C>
[[nodiscard]] constexpr auto extent(Key key) -> Extent {
	return std::span<Extent const>{C::extents}[key.index];
}

/**
 * The extents an image holds, in the order it holds them: the order a compiled schedule streams. Each key names an
 * extent of `C`, at most once, which the constructor checks; a constant order that breaks it does not compile. The
 * keys are borrowed: they must outlive the order, as a schedule's program-lifetime table does.
 */
export template<Catalog C>
class Order {
public:
	constexpr explicit Order(std::span<Key const> keys) : keys_{keys} {
		auto seen = std::vector<bool>(std::size(C::extents), false);
		for (auto const key : keys) {
			core::invariant(key.index < seen.size() && !seen[key.index]);
			seen[key.index] = true;
		}
	}

	[[nodiscard]] constexpr auto keys() const -> std::span<Key const> {
		return keys_;
	}

private:
	std::span<Key const> keys_;
};

}

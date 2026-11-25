// SPDX-License-Identifier: AGPL-3.0-only
//
// The file layout of a baked image. All integers are little-endian.
//
//   offset  bytes        field
//   0       8            magic
//   8       4            format version
//   12      4            card
//   16      32           model description digest
//   48      20           checkpoint commit
//   68      32           checkpoint manifest digest
//   100     32           payload digest: SHA-256 of every payload byte, padding included
//   132     4            extent count n
//   136     2 n          section table: the catalog key of each extent, in stream order
//   ...     zeros        to the payload, which starts on a 4 KiB boundary
//   p       ...          payload: the extents in table order, each on its encoding's boundary, zeros between
//
// Nothing else is stored: every extent's length is a fact of the catalog, so its offset follows from the table, and an
// image cannot describe a misaligned or overlapping extent. The file ends where the last extent ends.
export module engine.image:layout;

import std;
import engine.core;
import engine.model;
import engine.sha256;
import :catalog;
import :checkpoint;

namespace engine::image {

/** The card an image is baked for: its tensor-parallel rank. */
export enum class Card : std::uint8_t {
	Zero,
	One,
};

static_assert(model::cards == 2, "one card value per tensor-parallel rank");

/** A file that does not begin with these bytes is not a baked image; the CR LF and SUB catch text-mode mangling. */
export inline constexpr std::array<std::byte, 8> magic = {
    std::byte{0x89}, std::byte{'G'}, std::byte{'Z'}, std::byte{'I'}, std::byte{'M'}, std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1a},
};

/** The format this engine reads and writes. Any change to the layout or to how an extent's bytes are encoded bumps it. */
export inline constexpr std::uint32_t format_version = 1;

/** The payload starts on a page boundary, so the payload can be registered with the driver for direct copies. */
export inline constexpr std::uint64_t payload_alignment = 4096;

struct Field {
	std::size_t offset;
	std::size_t bytes;

	[[nodiscard]] constexpr auto end() const -> std::size_t {
		return offset + bytes;
	}

	[[nodiscard]] constexpr auto next(std::size_t length) const -> Field {
		return {.offset = end(), .bytes = length};
	}
};

inline constexpr Field magic_field = {.offset = 0, .bytes = magic.size()};
inline constexpr Field version_field = magic_field.next(4);
inline constexpr Field card_field = version_field.next(4);
inline constexpr Field description_field = card_field.next(sha256::digest_bytes);
inline constexpr Field commit_field = description_field.next(commit_bytes);
inline constexpr Field manifest_field = commit_field.next(sha256::digest_bytes);
inline constexpr Field payload_digest_field = manifest_field.next(sha256::digest_bytes);
inline constexpr Field count_field = payload_digest_field.next(4);

/** Bytes of the header before the section table. */
export inline constexpr std::size_t fixed_header_bytes = count_field.end();

static_assert(fixed_header_bytes == 136);

inline constexpr std::size_t key_bytes = sizeof(std::uint16_t);

[[nodiscard]] constexpr auto table_entry(std::size_t index) -> Field {
	return {.offset = fixed_header_bytes + (key_bytes * index), .bytes = key_bytes};
}

export [[nodiscard]] constexpr auto align_up(std::uint64_t value, std::uint64_t boundary) -> std::uint64_t {
	return (value + boundary - 1) / boundary * boundary;
}

/** Where the section table of an image of `extents` extents ends. */
[[nodiscard]] constexpr auto table_end(std::uint64_t extents) -> std::uint64_t {
	return fixed_header_bytes + (key_bytes * extents);
}

/** The file offset of the payload of an image of `extents` extents. */
export [[nodiscard]] constexpr auto payload_offset(std::uint64_t extents) -> std::uint64_t {
	return align_up(table_end(extents), payload_alignment);
}

/** Where one extent's bytes lie, from the payload's start. */
export struct Placement {
	std::uint64_t offset;
	std::uint64_t bytes;

	[[nodiscard]] constexpr auto end() const -> std::uint64_t {
		return offset + bytes;
	}

	[[nodiscard]] constexpr auto operator==(Placement const&) const -> bool = default;
};

/** One entry of an image: the extent, and where its bytes lie in the payload. */
export struct Section {
	Key key;
	Placement placement;

	[[nodiscard]] constexpr auto operator==(Section const&) const -> bool = default;
};

/**
 * The payload layout of `order`: each extent at the first boundary of its encoding after the extent before it, so
 * every extent is aligned, they lie in order without overlap, and only padding shorter than a boundary separates them.
 */
export template<Catalog C>
[[nodiscard]] constexpr auto place(Order<C> const& order) -> std::vector<Section> {
	auto sections = std::vector<Section>{};
	sections.reserve(order.keys().size());
	auto end = std::uint64_t{0};
	for (auto const key : order.keys()) {
		auto const stored = extent<C>(key);
		auto const offset = align_up(end, alignment(stored.encoding));
		sections.push_back({.key = key, .placement = {.offset = offset, .bytes = stored.bytes}});
		end = offset + stored.bytes;
	}
	return sections;
}

/** Bytes of the payload of `sections`: up to the end of the last extent. */
export [[nodiscard]] constexpr auto payload_bytes(std::span<Section const> sections) -> std::uint64_t {
	return sections.empty() ? 0 : sections.back().placement.end();
}

/** Writes `value` little-endian into all of `to`. */
constexpr auto store(std::span<std::byte> to, std::uint64_t value) -> void {
	for (auto& byte : to) {
		byte = std::byte{core::narrow<std::uint8_t>(value & 0xffU)};
		value >>= 8U;
	}
}

/** Reads the little-endian integer in all of `from`. */
[[nodiscard]] constexpr auto load(std::span<std::byte const> from) -> std::uint64_t {
	auto value = std::uint64_t{0};
	for (auto const byte : std::views::reverse(from)) {
		value = (value << 8U) | std::to_integer<std::uint64_t>(byte);
	}
	return value;
}

[[nodiscard]] constexpr auto in(std::span<std::byte> header, Field field) -> std::span<std::byte> {
	return header.subspan(field.offset, field.bytes);
}

[[nodiscard]] constexpr auto in(std::span<std::byte const> header, Field field) -> std::span<std::byte const> {
	return header.subspan(field.offset, field.bytes);
}

/** What a header records besides the table. */
struct Stamped {
	Card card;
	Checkpoint checkpoint;
	sha256::Digest description;
	sha256::Digest payload;
};

/** The header of an image of `sections`, through the zero padding before the payload. */
[[nodiscard]] constexpr auto encode_header(Stamped const& stamped, std::span<Section const> sections) -> std::vector<std::byte> {
	auto header = std::vector<std::byte>(payload_offset(sections.size()), std::byte{0});
	auto const bytes = std::span{header};
	std::ranges::copy(magic, in(bytes, magic_field).begin());
	store(in(bytes, version_field), format_version);
	store(in(bytes, card_field), std::to_underlying(stamped.card));
	std::ranges::copy(stamped.description.bytes, in(bytes, description_field).begin());
	std::ranges::copy(stamped.checkpoint.commit.bytes, in(bytes, commit_field).begin());
	std::ranges::copy(stamped.checkpoint.manifest.bytes, in(bytes, manifest_field).begin());
	std::ranges::copy(stamped.payload.bytes, in(bytes, payload_digest_field).begin());
	store(in(bytes, count_field), sections.size());
	for (auto index = std::size_t{0}; index < sections.size(); ++index) {
		store(in(bytes, table_entry(index)), sections[index].key.index);
	}
	return header;
}

}

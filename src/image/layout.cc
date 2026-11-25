// SPDX-License-Identifier: AGPL-3.0-only
//
// The file layout of an image. All integers are little-endian.
//
//   offset  bytes        field
//   0       8            magic
//   8       4            format version
//   12      4            target: 0 card zero, 1 card one, 2 host
//   16      32           model description digest
//   48      20           checkpoint commit
//   68      32           checkpoint manifest digest
//   100     32           payload digest: SHA-256 of every payload byte, padding included
//   132     4            extent count n
//   136     24 n         section table: each extent's identity, payload offset and length, in payload order
//   ...     zeros        to the payload, which starts on a 4 KiB boundary
//   p       ...          payload: the extents in table order, zeros between; the file ends where the last one ends
//
// The section table restates the catalog the baker was built with; the reader requires it to equal its own, entry
// for entry, so an image laid out by another schedule is refused before any extent is read.
export module engine.image:layout;

import std;
import engine.core;
import engine.sha256;
import :catalog;
import :checkpoint;

namespace engine::image {

/** A file that does not begin with these bytes is not an image; the CR LF and SUB catch text-mode mangling. */
export inline constexpr std::array<std::byte, 8> magic = {
    std::byte{0x89}, std::byte{'G'}, std::byte{'Z'}, std::byte{'I'}, std::byte{'M'}, std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1a},
};

/** The format this engine reads and writes. Any change to this layout bumps it. */
export inline constexpr std::uint32_t format_version = 1;

/** The payload starts on a page boundary, so it can be registered with the driver for direct copies. */
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
inline constexpr Field target_field = version_field.next(4);
inline constexpr Field description_field = target_field.next(sha256::digest_bytes);
inline constexpr Field commit_field = description_field.next(commit_bytes);
inline constexpr Field manifest_field = commit_field.next(sha256::digest_bytes);
inline constexpr Field payload_digest_field = manifest_field.next(sha256::digest_bytes);
inline constexpr Field count_field = payload_digest_field.next(4);

/** Bytes of the header before the section table. */
export inline constexpr std::size_t fixed_header_bytes = count_field.end();

static_assert(fixed_header_bytes == 136);

/** One section table entry: identity, offset and length, eight bytes each. */
export inline constexpr std::size_t entry_bytes = 3 * sizeof(std::uint64_t);

[[nodiscard]] constexpr auto entry_field(std::size_t index, std::size_t word) -> Field {
	return {.offset = fixed_header_bytes + (entry_bytes * index) + (sizeof(std::uint64_t) * word), .bytes = sizeof(std::uint64_t)};
}

/** Where the section table of an image of `extents` extents ends. */
[[nodiscard]] constexpr auto table_end(std::uint64_t extents) -> std::uint64_t {
	return fixed_header_bytes + (entry_bytes * extents);
}

/** The file offset of the payload of an image of `extents` extents. */
export [[nodiscard]] constexpr auto payload_offset(std::uint64_t extents) -> std::uint64_t {
	return align_up(table_end(extents), payload_alignment);
}

/** The file offset where the payload of an image of catalog `C` starts, and the file's length. */
export template<Catalog C>
inline constexpr std::uint64_t payload_start = payload_offset(std::size(C::extents));

export template<Catalog C>
inline constexpr std::uint64_t file_bytes = payload_start<C> + payload_bytes<C>;

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

/** The header of an image of catalog `C`, through the zero padding before the payload. */
template<Catalog C>
[[nodiscard]] constexpr auto encode_header(Checkpoint const& checkpoint, sha256::Digest const& payload) -> std::vector<std::byte> {
	auto header = std::vector<std::byte>(payload_start<C>, std::byte{0});
	auto const bytes = std::span{header};
	std::ranges::copy(magic, in(bytes, magic_field).begin());
	store(in(bytes, version_field), format_version);
	store(in(bytes, target_field), std::to_underlying(C::target));
	std::ranges::copy(C::description.bytes, in(bytes, description_field).begin());
	std::ranges::copy(checkpoint.commit.bytes, in(bytes, commit_field).begin());
	std::ranges::copy(checkpoint.manifest.bytes, in(bytes, manifest_field).begin());
	std::ranges::copy(payload.bytes, in(bytes, payload_digest_field).begin());
	store(in(bytes, count_field), std::size(C::extents));
	for (auto index = std::size_t{0}; index < std::size(C::extents); ++index) {
		store(in(bytes, entry_field(index, 0)), C::extents[index].identity);
		store(in(bytes, entry_field(index, 1)), layout<C>[index].offset);
		store(in(bytes, entry_field(index, 2)), layout<C>[index].bytes);
	}
	return header;
}

}

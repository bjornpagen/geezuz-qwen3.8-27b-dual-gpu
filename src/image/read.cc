// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:read;

import std;
import engine.core;
import engine.file;
import engine.sha256;
import :catalog;
import :checkpoint;
import :error;
import :layout;

namespace engine::image {

[[nodiscard]] constexpr auto reject(Failure failure) -> std::unexpected<Rejection> {
	return std::unexpected(Rejection{.failure = failure});
}

[[nodiscard]] constexpr auto holds(std::span<std::byte const> file, Field field, std::span<std::byte const> expected) -> bool {
	return std::ranges::equal(in(file, field), expected);
}

/** Whether the section table in `file` is the catalog's own, entry for entry. */
template<Catalog C>
[[nodiscard]] constexpr auto table_matches(std::span<std::byte const> file) -> bool {
	for (auto index = std::size_t{0}; index < std::size(C::extents); ++index) {
		if (load(in(file, entry_field(index, 0))) != C::extents[index].identity || load(in(file, entry_field(index, 1))) != layout<C>[index].offset ||
		    load(in(file, entry_field(index, 2))) != layout<C>[index].bytes) {
			return false;
		}
	}
	return true;
}

/**
 * Checks every byte of `file` as an image of catalog `C` baked from `checkpoint`. The checks that need only the header
 * run first, so an image for another memory or checkpoint is refused before its payload is hashed.
 */
export template<Catalog C>
[[nodiscard]] auto parse(std::span<std::byte const> file, Checkpoint const& checkpoint) -> core::Result<void, Rejection> {
	if (file.size() < fixed_header_bytes) {
		return reject(Failure::Truncated);
	}
	if (!holds(file, magic_field, magic)) {
		return reject(Failure::BadMagic);
	}
	if (load(in(file, version_field)) != format_version) {
		return reject(Failure::UnsupportedVersion);
	}
	if (!holds(file, description_field, C::description.bytes)) {
		return reject(Failure::ModelMismatch);
	}
	if (!holds(file, commit_field, checkpoint.commit.bytes) || !holds(file, manifest_field, checkpoint.manifest.bytes)) {
		return reject(Failure::CheckpointMismatch);
	}
	if (load(in(file, target_field)) != std::to_underlying(C::target)) {
		return reject(Failure::WrongTarget);
	}
	if (load(in(file, count_field)) != std::size(C::extents)) {
		return reject(Failure::ScheduleMismatch);
	}
	if (file.size() < payload_start<C>) {
		return reject(Failure::Truncated);
	}
	if (!table_matches<C>(file)) {
		return reject(Failure::ScheduleMismatch);
	}
	auto const end_of_table = table_end(std::size(C::extents));
	if (!std::ranges::all_of(file.subspan(end_of_table, payload_start<C> - end_of_table), [](std::byte byte) { return byte == std::byte{0}; })) {
		return reject(Failure::DirtyPadding);
	}
	if (file.size() < file_bytes<C>) {
		return reject(Failure::Truncated);
	}
	if (file.size() > file_bytes<C>) {
		return reject(Failure::Overlong);
	}
	if (!holds(file, payload_digest_field, sha256::digest(file.subspan(payload_start<C>)).bytes)) {
		return reject(Failure::PayloadMismatch);
	}
	return {};
}

/**
 * An image of catalog `C`, mapped and verified whole: it exists only once every header field, the section table and
 * the payload digest have matched. Its spans borrow the mapping: they are valid while the image lives and is not moved
 * from.
 */
export template<Catalog C>
class Image {
public:
	[[nodiscard]] static auto open(std::filesystem::path const& path, Checkpoint const& checkpoint) -> core::Result<Image, ImageError> {
		auto mapping = file::Mapping::open(path);
		if (!mapping) {
			return std::unexpected(ImageError{.cause = mapping.error()});
		}
		if (auto const checked = parse<C>(mapping->bytes(), checkpoint); !checked) {
			return std::unexpected(ImageError{.cause = checked.error()});
		}
		return Image{*std::move(mapping)};
	}

	/** The payload, as it is copied to its memory whole: extent `i` lies at `layout<C>[i]` in it. */
	[[nodiscard]] auto payload() const -> std::span<std::byte const> {
		return mapping_.bytes().subspan(payload_start<C>);
	}

	/** The bytes of extent `index` of the catalog. */
	[[nodiscard]] auto bytes(std::size_t index) const -> std::span<std::byte const> {
		auto const placement = std::span{layout<C>}[index];
		return payload().subspan(placement.offset, placement.bytes);
	}

	/** The bytes of extent `index`, with the length the catalog gives it as part of the type. */
	template<std::size_t index>
	[[nodiscard]] auto extent() const -> std::span<std::byte const, std::get<index>(C::extents).bytes> {
		return std::span<std::byte const, std::get<index>(C::extents).bytes>{bytes(index)};
	}

private:
	explicit Image(file::Mapping mapping) : mapping_{std::move(mapping)} {}

	file::Mapping mapping_;
};

}

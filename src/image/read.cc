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

/** What an engine expects of an image: the card it is for, the checkpoint it comes from, the order it streams. */
export template<Catalog C>
struct Expect {
	Card card;
	Checkpoint checkpoint;
	Order<C> order;
};

[[nodiscard]] constexpr auto reject(Failure failure) -> std::unexpected<Rejection> {
	return std::unexpected(Rejection{.failure = failure});
}

[[nodiscard]] constexpr auto holds(std::span<std::byte const> file, Field field, std::span<std::byte const> expected) -> bool {
	return std::ranges::equal(in(file, field), expected);
}

/**
 * Checks every byte of `file` against `expect` and returns where each extent lies in the payload. The checks that need
 * only the header run first, so an image for the wrong card or checkpoint is refused before its payload is hashed.
 */
export template<Catalog C>
[[nodiscard]] auto parse(std::span<std::byte const> file, Expect<C> const& expect) -> core::Result<std::vector<Section>, Rejection> {
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
	if (!holds(file, commit_field, expect.checkpoint.commit.bytes) || !holds(file, manifest_field, expect.checkpoint.manifest.bytes)) {
		return reject(Failure::CheckpointMismatch);
	}
	if (load(in(file, card_field)) != std::to_underlying(expect.card)) {
		return reject(Failure::WrongCard);
	}
	auto const keys = expect.order.keys();
	if (load(in(file, count_field)) != keys.size()) {
		return reject(Failure::ScheduleMismatch);
	}
	auto const payload_start = payload_offset(keys.size());
	if (file.size() < payload_start) {
		return reject(Failure::Truncated);
	}
	for (auto index = std::size_t{0}; index < keys.size(); ++index) {
		if (load(in(file, table_entry(index))) != keys[index].index) {
			return reject(Failure::ScheduleMismatch);
		}
	}
	auto const padding = file.subspan(table_end(keys.size()), payload_start - table_end(keys.size()));
	if (!std::ranges::all_of(padding, [](std::byte byte) { return byte == std::byte{0}; })) {
		return reject(Failure::DirtyPadding);
	}
	auto sections = place(expect.order);
	auto const end = payload_start + payload_bytes(sections);
	if (file.size() < end) {
		return reject(Failure::Truncated);
	}
	if (file.size() > end) {
		return reject(Failure::Overlong);
	}
	if (!holds(file, payload_digest_field, sha256::digest(file.subspan(payload_start)).bytes)) {
		return reject(Failure::PayloadMismatch);
	}
	return sections;
}

/**
 * A baked image, mapped and verified whole: it exists only once every header field, the section table and the payload
 * digest have matched what the engine expects. Its spans borrow the mapping: they are valid while the image lives and
 * is not moved from.
 */
export template<Catalog C>
class Image {
public:
	[[nodiscard]] static auto open(std::filesystem::path const& path, Expect<C> const& expect) -> core::Result<Image, ImageError> {
		auto mapping = file::Mapping::open(path);
		if (!mapping) {
			return std::unexpected(ImageError{.cause = mapping.error()});
		}
		auto sections = parse(mapping->bytes(), expect);
		if (!sections) {
			return std::unexpected(ImageError{.cause = sections.error()});
		}
		return Image{*std::move(mapping), *std::move(sections)};
	}

	/** The payload, as the device copies it: each section's placement is its offset in here. */
	[[nodiscard]] auto payload() const -> std::span<std::byte const> {
		return mapping_.bytes().subspan(payload_offset(sections_.size()));
	}

	/** The extents in stream order, with their placements. */
	[[nodiscard]] auto sections() const -> std::span<Section const> {
		return sections_;
	}

	/** The bytes of extent `key`, which the expected order holds. */
	[[nodiscard]] auto bytes(Key key) const -> std::span<std::byte const> {
		auto const found = std::ranges::find(sections_, key, &Section::key);
		core::invariant(found != sections_.end());
		return payload().subspan(found->placement.offset, found->placement.bytes);
	}

	/** The bytes of extent `key`, with the length the catalog gives it as part of the type. */
	template<Key key>
	[[nodiscard]] auto extent() const -> std::span<std::byte const, image::extent<C>(key).bytes> {
		return std::span<std::byte const, image::extent<C>(key).bytes>{bytes(key)};
	}

private:
	Image(file::Mapping mapping, std::vector<Section> sections) : mapping_{std::move(mapping)}, sections_{std::move(sections)} {}

	file::Mapping mapping_;
	std::vector<Section> sections_;
};

}

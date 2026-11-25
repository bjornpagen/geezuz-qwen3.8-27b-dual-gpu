// SPDX-License-Identifier: AGPL-3.0-only
//
// Test doubles for the image format: small catalogs with every encoding and lengths that force padding, so a whole
// image is ~100 KB and a test can write, corrupt and read thousands of them; plus a baker double that writes an image
// the way the real baker does, piece by piece at the offsets the writer gives.
export module engine.image_synthetic;

import std;
import engine.core;
import engine.image;
import engine.sha256;
import engine.testing;

namespace engine::image_synthetic {

namespace image = engine::image;

using image::Encoding;
using image::Extent;

inline constexpr std::array<Extent, 10> synthetic_extents = {{
    {.bytes = image::tile_bytes, .encoding = Encoding::E4m3Tiles},
    {.bytes = 2, .encoding = Encoding::Bf16Scales},
    {.bytes = 3 * image::tile_bytes, .encoding = Encoding::E4m3Tiles},
    {.bytes = 6, .encoding = Encoding::Bf16Scales},
    {.bytes = 10, .encoding = Encoding::Bf16},
    {.bytes = 2, .encoding = Encoding::Bf16},
    {.bytes = 4098, .encoding = Encoding::Bf16},
    {.bytes = 2 * image::tile_bytes, .encoding = Encoding::E4m3Tiles},
    {.bytes = 4, .encoding = Encoding::Bf16Scales},
    {.bytes = 34, .encoding = Encoding::Bf16},
}};

/** A test double of a model catalog. */
export struct SyntheticCatalog {
	static constexpr std::array<Extent, 10> extents = synthetic_extents;
	static constexpr sha256::Digest description = sha256::digest("synthetic catalog: a test double");
};

/** The same extents under another description: an image of it stands for one baked from another model. */
export struct OtherCatalog {
	static constexpr std::array<Extent, 10> extents = synthetic_extents;
	static constexpr sha256::Digest description = sha256::digest("another synthetic catalog: a test double");
};

export inline constexpr image::Checkpoint synthetic_checkpoint = {
    .commit = image::checkpoint_commit,
    .manifest = sha256::digest("synthetic checkpoint manifest: a test double"),
};

/** Every key of a catalog, in catalog order. */
export template<image::Catalog C>
[[nodiscard]] auto all_keys() -> std::vector<image::Key> {
	auto keys = std::vector<image::Key>(std::size(C::extents));
	for (auto index = std::size_t{0}; index < keys.size(); ++index) {
		keys[index] = image::Key{.index = engine::core::narrow<std::uint16_t>(index)};
	}
	return keys;
}

/** A random order: a random subset of the catalog's keys, possibly empty, in random sequence. */
export template<image::Catalog C>
[[nodiscard]] auto random_order(std::mt19937_64& random) -> std::vector<image::Key> {
	auto keys = all_keys<C>();
	std::ranges::shuffle(keys, random);
	keys.resize(std::uniform_int_distribution<std::size_t>{0, keys.size()}(random));
	return keys;
}

/** The bytes the baker double writes for `key` under `seed`: distinct per key, offset and seed. */
export template<image::Catalog C>
[[nodiscard]] auto content(image::Key key, std::uint64_t seed) -> std::vector<std::byte> {
	auto bytes = std::vector<std::byte>(image::extent<C>(key).bytes);
	auto random = std::mt19937_64{seed ^ (std::uint64_t{key.index} << 40U)};
	std::ranges::generate(bytes, [&random] { return std::byte{static_cast<std::uint8_t>(random() & 0xffU)}; });
	return bytes;
}

/** Places `bytes` at `offset` of `file`, as a positional write into a file created empty: any gap reads as zeros. */
auto place_at(std::vector<std::byte>& file, std::uint64_t offset, std::span<std::byte const> bytes) -> void {
	if (file.size() < offset + bytes.size()) {
		file.resize(offset + bytes.size(), std::byte{0});
	}
	std::ranges::copy(bytes, std::span{file}.subspan(offset).begin());
}

/** A written image, and what the writer said about it. */
export struct Baked {
	std::vector<std::byte> file;
	image::Finished finished;
};

/**
 * A baker double: writes every extent of `order` in random pieces at the offsets the writer gives, then the header at
 * offset 0, into an in-memory file.
 */
export template<image::Catalog C>
[[nodiscard]] auto bake(image::Stamp stamp, std::span<image::Key const> order, std::uint64_t seed, std::mt19937_64& random) -> Baked {
	auto writer = image::Writer<C>{stamp, image::Order<C>{order}};
	auto file = std::vector<std::byte>{};
	for (auto const key : order) {
		auto const bytes = content<C>(key, seed);
		auto remaining = std::span<std::byte const>{bytes};
		while (!remaining.empty()) {
			auto const piece = remaining.first(std::uniform_int_distribution<std::size_t>{0, remaining.size()}(random));
			place_at(file, writer.write(key, piece), piece);
			remaining = remaining.subspan(piece.size());
		}
	}
	auto finished = std::move(writer).finish();
	place_at(file, 0, finished.header);
	return {.file = std::move(file), .finished = std::move(finished)};
}

/** The keys of the specimen image the file tests share: every synthetic extent, in reverse catalog order. */
export [[nodiscard]] auto specimen_keys() -> std::vector<image::Key> {
	auto keys = all_keys<SyntheticCatalog>();
	std::ranges::reverse(keys);
	return keys;
}

export inline constexpr image::Stamp specimen_stamp = {.card = image::Card::One, .checkpoint = synthetic_checkpoint};

export inline constexpr std::uint64_t specimen_seed = 0x5eed;

/** The specimen image: the same bytes on every run, so one program can write it to a file and another check it. */
export [[nodiscard]] auto specimen_image() -> Baked {
	auto random = engine::testing::generator(specimen_seed);
	return bake<SyntheticCatalog>(specimen_stamp, specimen_keys(), specimen_seed, random);
}

}

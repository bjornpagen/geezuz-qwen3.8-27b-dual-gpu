// SPDX-License-Identifier: AGPL-3.0-only
//
// Test doubles for the image format: small catalogs with every part and lengths that force padding, so a whole image
// is ~100 KB and a test can write, corrupt and read thousands of them; plus a baker double that writes an image the way
// the real baker does, piece by piece at the offsets the writer gives.
export module engine.image_synthetic;

import std;
import engine.image;
import engine.sha256;
import engine.testing;

namespace engine::image_synthetic {

namespace image = engine::image;

using image::Extent;
using image::Part;
using image::Target;

inline constexpr std::array<Extent, 10> synthetic_extents = {{
    {.identity = 11, .bytes = image::alignment(Part::Tiles), .part = Part::Tiles},
    {.identity = 12, .bytes = 2, .part = Part::Scales},
    {.identity = 13, .bytes = 3 * image::alignment(Part::Tiles), .part = Part::Tiles},
    {.identity = 14, .bytes = 6, .part = Part::Scales},
    {.identity = 15, .bytes = 10, .part = Part::Whole},
    {.identity = 16, .bytes = 2, .part = Part::Whole},
    {.identity = 17, .bytes = 4098, .part = Part::Whole},
    {.identity = 18, .bytes = 2 * image::alignment(Part::Tiles), .part = Part::Tiles},
    {.identity = 19, .bytes = 4, .part = Part::Scales},
    {.identity = 20, .bytes = 34, .part = Part::Whole},
}};

/** A test double of a catalog, for any target. */
export template<Target where>
struct SyntheticCatalog {
	static constexpr Target target = where;
	static constexpr std::array<Extent, 10> extents = synthetic_extents;
	static constexpr sha256::Digest description = sha256::digest("synthetic catalog: a test double");
};

/** The same extents under another description: an image of it stands for one baked from another model. */
export struct OtherModelCatalog {
	static constexpr Target target = Target::CardZero;
	static constexpr std::array<Extent, 10> extents = synthetic_extents;
	static constexpr sha256::Digest description = sha256::digest("another synthetic catalog: a test double");
};

/** The same extents in another order: an image of it stands for one laid out by another schedule. */
export struct OtherScheduleCatalog {
	static constexpr Target target = Target::CardZero;
	static constexpr std::array<Extent, 10> extents = [] {
		auto reordered = synthetic_extents;
		std::ranges::rotate(reordered, std::next(reordered.begin(), 3));
		return reordered;
	}();
	static constexpr sha256::Digest description = SyntheticCatalog<Target::CardZero>::description;
};

/** A catalog with no extents: its image is a header alone. */
export struct EmptyCatalog {
	static constexpr Target target = Target::Host;
	static constexpr std::array<Extent, 0> extents = {};
	static constexpr sha256::Digest description = SyntheticCatalog<Target::Host>::description;
};

export inline constexpr image::Checkpoint synthetic_checkpoint = {
    .commit = image::checkpoint_commit,
    .manifest = sha256::digest("synthetic checkpoint manifest: a test double"),
};

/** The bytes the baker double writes for extent `index` under `seed`: distinct per extent, offset and seed. */
export template<image::Catalog C>
[[nodiscard]] auto content(std::size_t index, std::uint64_t seed) -> std::vector<std::byte> {
	auto bytes = std::vector<std::byte>(std::span{C::extents}[index].bytes);
	auto random = engine::testing::generator(seed ^ (std::uint64_t{index} << 40U));
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
 * A baker double: writes every extent of `C` in random pieces at the offsets the writer gives, then the header at
 * offset 0, into an in-memory file.
 */
export template<image::Catalog C>
[[nodiscard]] auto bake(image::Checkpoint const& checkpoint, std::uint64_t seed, std::mt19937_64& random) -> Baked {
	auto writer = image::Writer<C>{checkpoint};
	auto file = std::vector<std::byte>{};
	for (auto index = std::size_t{0}; index < std::size(C::extents); ++index) {
		auto const bytes = content<C>(index, seed);
		auto remaining = std::span<std::byte const>{bytes};
		while (!remaining.empty()) {
			auto const piece = remaining.first(std::uniform_int_distribution<std::size_t>{0, remaining.size()}(random));
			place_at(file, writer.write(index, piece), piece);
			remaining = remaining.subspan(piece.size());
		}
	}
	auto finished = std::move(writer).finish();
	place_at(file, 0, finished.header);
	return {.file = std::move(file), .finished = std::move(finished)};
}

/** The specimen the file tests share: a card-one image, the same bytes on every run. */
export using SpecimenCatalog = SyntheticCatalog<Target::CardOne>;

export inline constexpr std::uint64_t specimen_seed = 0x5eed;

export [[nodiscard]] auto specimen_image() -> Baked {
	auto random = engine::testing::generator(specimen_seed);
	return bake<SpecimenCatalog>(synthetic_checkpoint, specimen_seed, random);
}

}

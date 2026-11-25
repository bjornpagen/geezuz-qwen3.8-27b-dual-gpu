// SPDX-License-Identifier: AGPL-3.0-only
//
// Feeding the writer an extent out of the order's sequence is a baker bug: the write aborts.
import std;
import engine.image;
import engine.image_synthetic;

auto main() -> int {
	namespace image = engine::image;
	namespace synthetic = engine::image_synthetic;
	auto const keys = std::array{image::Key{.index = 4}, image::Key{.index = 5}};
	auto writer = image::Writer<synthetic::SyntheticCatalog>{synthetic::specimen_stamp, image::Order<synthetic::SyntheticCatalog>{keys}};
	auto const bytes = std::array<std::byte, 2>{};
	return static_cast<int>(writer.write(image::Key{.index = 5}, bytes));
}

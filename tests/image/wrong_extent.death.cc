// SPDX-License-Identifier: AGPL-3.0-only
//
// Feeding the writer an extent out of the catalog's order is a baker bug: the write aborts.
import std;
import engine.image;
import engine.image_synthetic;

auto main() -> int {
	namespace synthetic = engine::image_synthetic;
	auto writer = engine::image::Writer<synthetic::SpecimenCatalog>{synthetic::synthetic_checkpoint};
	auto const bytes = std::array<std::byte, 2>{};
	return static_cast<int>(writer.write(1, bytes));
}

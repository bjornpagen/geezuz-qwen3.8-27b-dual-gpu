// SPDX-License-Identifier: AGPL-3.0-only
//
// An order that names one extent twice is a schedule bug: building it aborts.
import std;
import engine.image;
import engine.image_synthetic;

auto main() -> int {
	auto const keys = std::array{engine::image::Key{.index = 2}, engine::image::Key{.index = 5}, engine::image::Key{.index = 2}};
	auto const order = engine::image::Order<engine::image_synthetic::SyntheticCatalog>{keys};
	return static_cast<int>(order.keys().size());
}

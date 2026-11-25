// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet's layouts: every map from a model coordinate to a place is a bijection onto exactly the space it
// claims, the slot image is tiled by its layers' states and windows, and a snapshot's parts carry an image through
// scattered pool blocks and back unchanged.
import std;
import engine.deltanet;
import engine.deltanet_fixtures;
import engine.model;
import engine.testing;

namespace {

namespace dn = engine::deltanet;
namespace fx = engine::deltanet_fixtures;
namespace model = engine::model;

/** Whether `place` sends every coordinate in [0, count) to a distinct place in [0, count). */
template<class Place>
[[nodiscard]] auto bijective(int count, Place place) -> bool {
	auto seen = std::vector<bool>(static_cast<std::size_t>(count), false);
	for (auto coordinate = 0; coordinate < count; ++coordinate) {
		auto const at = place(coordinate);
		if (at < 0 || at >= count || seen.at(static_cast<std::size_t>(at))) {
			return false;
		}
		seen.at(static_cast<std::size_t>(at)) = true;
	}
	return true;
}

/** The state layout under tree `T`: a bijection, and each lane's chunks hold exactly its rows of its column. */
template<class T>
[[nodiscard]] auto state_layout_is_the_lanes_order() -> bool {
	using Layout = dn::StateLayout<T>;
	auto const bijection = bijective(dn::head_floats, [](int coordinate) {
		return Layout::element(coordinate / dn::dim, coordinate % dn::dim);
	});
	auto lanes_hold_their_rows = true;
	for (auto column = 0; column < dn::dim; ++column) {
		for (auto row = 0; row < dn::dim; ++row) {
			auto const lane = Layout::lane(row, column);
			auto const within = row % T::per_lane;
			auto const chunk_start = Layout::lane_chunk(column / Layout::unit_columns, within / 4, lane);
			lanes_hold_their_rows = lanes_hold_their_rows && lane / T::lanes == column % Layout::unit_columns &&
			                        lane % T::lanes == row / T::per_lane && Layout::element(row, column) == chunk_start + within % 4;
		}
	}
	return bijection && lanes_hold_their_rows;
}

template<class T>
[[nodiscard]] auto vector_layout_is_a_bijection() -> bool {
	return bijective(dn::dim, [](int element) {
		return dn::vector_element<T>(element);
	});
}

/** Every card channel belongs to exactly one key head group, and a group's chunks are runs of card channels. */
[[nodiscard]] auto groups_partition_the_channels() -> bool {
	auto const bijection = bijective(dn::channels, [](int coordinate) {
		return dn::card_channel(coordinate / dn::group_channels, coordinate % dn::group_channels);
	});
	auto runs = true;
	for (auto key_head = 0; key_head < dn::key_heads; ++key_head) {
		for (auto local = 0; local < dn::group_channels; ++local) {
			runs = runs && dn::card_channel(key_head, local) - dn::card_channel(key_head, local - local % dn::bf16_per_chunk) ==
			                   local % dn::bf16_per_chunk;
		}
		// The group reads its own query and key head and the value heads that read them.
		runs = runs && dn::card_channel(key_head, 0) == key_head * dn::dim &&
		       dn::card_channel(key_head, dn::dim) == dn::query_channels + key_head * dn::dim &&
		       dn::card_channel(key_head, 2 * dn::dim) == 2 * dn::query_channels + dn::group * key_head * dn::dim;
	}
	return bijection && runs;
}

/** The image's state and window ranges tile it end to end, in layer order. */
[[nodiscard]] auto image_is_tiled() -> bool {
	auto expected = std::size_t{0};
	auto holds = true;
	for (auto layer = 0; layer < dn::layers; ++layer) {
		for (auto head = 0; head < dn::value_heads; ++head) {
			holds = holds && dn::state_offset(layer, head) == expected;
			expected += dn::head_state_bytes;
		}
	}
	for (auto layer = 0; layer < dn::layers; ++layer) {
		for (auto key_head = 0; key_head < dn::key_heads; ++key_head) {
			holds = holds && dn::window_offset(layer, key_head) == expected;
			expected += dn::window_group_bytes;
		}
	}
	return holds && expected == dn::slot_image_bytes && expected == model::slot_state_bytes;
}

[[nodiscard]] auto ring_is_tiled() -> bool {
	auto expected = std::size_t{0};
	auto holds = true;
	for (auto layer = 0; layer < dn::layers; ++layer) {
		for (auto key_head = 0; key_head < dn::key_heads; ++key_head) {
			for (auto entry = 0; entry < dn::max_drafts; ++entry) {
				holds = holds && dn::ring_offset(layer, key_head, entry) == expected;
				expected += dn::entry_bytes;
			}
		}
	}
	return holds && expected == dn::ring_bytes_per_slot && dn::entry_bytes % 16 == 0;
}

/** The deltanet ordinal counts the linear-attention trunk layers before a layer, and names each of them once. */
[[nodiscard]] auto ordinals_count_deltanet_layers() -> bool {
	auto next = 0;
	auto holds = true;
	for (auto layer = std::size_t{0}; layer < model::trunk_layers; ++layer) {
		holds = holds && dn::deltanet_layer(layer) == next;
		next += model::trunk_mixers[layer] == model::Mixer::LinearAttention ? 1 : 0;
	}
	return holds && next == dn::layers;
}

/** The snapshot parts cover the image once, in order, each fitting a block. */
[[nodiscard]] auto parts_cover_the_image() -> bool {
	auto expected = std::size_t{0};
	auto holds = true;
	for (auto part = std::size_t{0}; part < model::snapshot_blocks; ++part) {
		auto const extent = dn::snapshot_part(part);
		holds = holds && extent.offset == expected && extent.bytes > 0 && extent.bytes <= model::kv_block_bytes;
		expected += extent.bytes;
	}
	return holds && expected == dn::slot_image_bytes;
}

/**
 * A slot image saved into scattered pool blocks part by part and loaded back into another slot is the same image,
 * and every layer reads back as written.
 */
[[nodiscard]] auto snapshot_round_trips(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto image = std::vector<std::byte>(dn::slot_image_bytes);
	std::ranges::generate(image, [&] {
		return static_cast<std::byte>(generator() & 0xffU);
	});
	auto layers = std::vector<std::pair<int, dn::LayerBuffers>>{};
	for (auto const layer : {0, 17, dn::layers - 1}) {
		auto buffers = fx::random_layer(generator);
		dn::store_layer(image, layer, buffers);
		layers.emplace_back(layer, std::move(buffers));
	}
	constexpr auto pool_blocks = model::snapshot_blocks * 3;
	auto pool = std::vector<std::byte>(pool_blocks * model::kv_block_bytes);
	auto blocks = std::vector<std::size_t>(pool_blocks);
	std::ranges::iota(blocks, std::size_t{0});
	std::ranges::shuffle(blocks, generator);
	blocks.resize(model::snapshot_blocks);
	for (auto part = std::size_t{0}; part < model::snapshot_blocks; ++part) {
		auto const extent = dn::snapshot_part(part);
		std::ranges::copy(std::span{image}.subspan(extent.offset, extent.bytes),
		                  pool.begin() + static_cast<std::ptrdiff_t>(blocks.at(part) * model::kv_block_bytes));
	}
	auto loaded = std::vector<std::byte>(dn::slot_image_bytes);
	for (auto part = std::size_t{0}; part < model::snapshot_blocks; ++part) {
		auto const extent = dn::snapshot_part(part);
		std::ranges::copy(std::span{pool}.subspan(blocks.at(part) * model::kv_block_bytes, extent.bytes),
		                  loaded.begin() + static_cast<std::ptrdiff_t>(extent.offset));
	}
	auto holds = std::ranges::equal(image, loaded);
	for (auto const& [layer, buffers] : layers) {
		holds = holds && fx::same_stored(dn::load_layer(loaded, layer), buffers);
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the state layout is the eight-lane tree's load order",
		                      .holds = state_layout_is_the_lanes_order<dn::Column>()},
	    engine::testing::Case{.name = "the state layout is the whole-warp tree's load order",
		                      .holds = state_layout_is_the_lanes_order<dn::Tree<32>>()},
	    engine::testing::Case{.name = "the eight-lane vector layout is a bijection", .holds = vector_layout_is_a_bijection<dn::Column>()},
	    engine::testing::Case{.name = "the whole-warp vector layout is a bijection", .holds = vector_layout_is_a_bijection<dn::Tree<32>>()},
	    engine::testing::Case{.name = "key head groups partition the card's channels", .holds = groups_partition_the_channels()},
	    engine::testing::Case{.name = "states then windows tile the slot image", .holds = image_is_tiled()},
	    engine::testing::Case{.name = "ring entries tile a slot's ring", .holds = ring_is_tiled()},
	    engine::testing::Case{.name = "deltanet ordinals count the linear-attention layers", .holds = ordinals_count_deltanet_layers()},
	    engine::testing::Case{.name = "snapshot parts cover the image once, in order", .holds = parts_cover_the_image()},
	    engine::testing::Case{.name = "a snapshot round-trips through scattered blocks, seed 5", .holds = snapshot_round_trips(5)},
	};
	return engine::testing::report(cases);
}

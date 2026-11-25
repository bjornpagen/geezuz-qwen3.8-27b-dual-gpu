// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.image;
import engine.image_synthetic;
import engine.sha256;
import engine.testing;

namespace {

namespace image = engine::image;
namespace sha256 = engine::sha256;
namespace synthetic = engine::image_synthetic;

using image::Failure;
using image::Key;
using synthetic::SyntheticCatalog;

using Expect = image::Expect<SyntheticCatalog>;
using Order = image::Order<SyntheticCatalog>;

[[nodiscard]] auto random_byte(std::mt19937_64& random) -> std::byte {
	return std::byte{static_cast<std::uint8_t>(random() & 0xffU)};
}

[[nodiscard]] auto random_card(std::mt19937_64& random) -> image::Card {
	return (random() & 1U) == 0 ? image::Card::Zero : image::Card::One;
}

[[nodiscard]] auto other_card(image::Card card) -> image::Card {
	return card == image::Card::Zero ? image::Card::One : image::Card::Zero;
}

[[nodiscard]] auto random_checkpoint(std::mt19937_64& random) -> image::Checkpoint {
	auto checkpoint = synthetic::synthetic_checkpoint;
	std::ranges::generate(checkpoint.commit.bytes, [&random] { return random_byte(random); });
	std::ranges::generate(checkpoint.manifest.bytes, [&random] { return random_byte(random); });
	return checkpoint;
}

[[nodiscard]] auto refused_with(engine::core::Result<std::vector<image::Section>, image::Rejection> const& result, Failure failure) -> bool {
	return !result.has_value() && result.error().failure == failure;
}

/** Whatever was written reads back: the stamp is accepted, the layout is the order's, and every extent holds its bytes. */
[[nodiscard]] auto images_round_trip(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 150; ++trial) {
		auto const keys = synthetic::random_order<SyntheticCatalog>(random);
		auto const stamp = image::Stamp{.card = random_card(random), .checkpoint = random_checkpoint(random)};
		auto const content_seed = random();
		auto const baked = synthetic::bake<SyntheticCatalog>(stamp, keys, content_seed, random);
		auto const order = Order{keys};
		auto const sections = image::parse(baked.file, Expect{.card = stamp.card, .checkpoint = stamp.checkpoint, .order = order});
		if (!sections || *sections != image::place(order) || baked.file.size() != baked.finished.file_bytes) {
			return false;
		}
		auto const payload = std::span{baked.file}.subspan(image::payload_offset(keys.size()));
		if (sha256::digest(payload) != baked.finished.payload) {
			return false;
		}
		for (auto const& section : *sections) {
			auto const stored = payload.subspan(section.placement.offset, section.placement.bytes);
			if (!std::ranges::equal(stored, synthetic::content<SyntheticCatalog>(section.key, content_seed))) {
				return false;
			}
		}
	}
	return true;
}

/**
 * The layout laws, for any order of catalog `C`: each extent starts on its encoding's boundary and is a whole number of
 * its unit; e4m3 runs are whole 16 KiB tiles on 256-byte boundaries; extents follow the order without overlap, packed
 * tight (padding shorter than the boundary); the payload starts on a page boundary after the table.
 */
template<image::Catalog C>
[[nodiscard]] auto layout_laws_hold(std::uint64_t seed, int trials) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < trials; ++trial) {
		auto const keys = synthetic::random_order<C>(random);
		auto const sections = image::place(image::Order<C>{keys});
		auto const start = image::payload_offset(keys.size());
		if (sections.size() != keys.size() || start % image::payload_alignment != 0 || start < image::fixed_header_bytes + (2 * keys.size())) {
			return false;
		}
		auto end = std::uint64_t{0};
		for (auto index = std::size_t{0}; index < sections.size(); ++index) {
			auto const& [key, placement] = sections[index];
			auto const stored = image::extent<C>(key);
			auto const boundary = image::alignment(stored.encoding);
			auto const tiles_hold = stored.encoding != image::Encoding::E4m3Tiles ||
			                        (placement.offset % 256 == 0 && placement.bytes % image::tile_bytes == 0);
			auto const in_place = key == keys[index] && placement.bytes == stored.bytes && placement.offset % 16 == 0 &&
			                      placement.offset % boundary == 0 && placement.offset >= end && placement.offset - end < boundary;
			if (!tiles_hold || !in_place) {
				return false;
			}
			end = placement.end();
		}
		if (image::payload_bytes(sections) != end) {
			return false;
		}
	}
	return true;
}

/** A fixed image of every synthetic extent, for the corruption searches. */
struct Specimen {
	std::vector<Key> keys;
	image::Stamp stamp;
	synthetic::Baked baked;

	[[nodiscard]] auto expect() const -> Expect {
		return {.card = stamp.card, .checkpoint = stamp.checkpoint, .order = Order{keys}};
	}
};

[[nodiscard]] auto specimen(std::mt19937_64& random) -> Specimen {
	auto keys = synthetic::all_keys<SyntheticCatalog>();
	std::ranges::shuffle(keys, random);
	auto const stamp = image::Stamp{.card = random_card(random), .checkpoint = random_checkpoint(random)};
	auto baked = synthetic::bake<SyntheticCatalog>(stamp, keys, random(), random);
	return {.keys = std::move(keys), .stamp = stamp, .baked = std::move(baked)};
}

/** Every proper prefix of an image is refused as truncated, from the empty file to one byte short. */
[[nodiscard]] auto every_prefix_is_truncated(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const expect = sample.expect();
	auto const file = std::span<std::byte const>{sample.baked.file};
	for (auto length = std::size_t{0}; length < file.size(); ++length) {
		if (!refused_with(image::parse(file.first(length), expect), Failure::Truncated)) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto appended_bytes_are_refused(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	for (auto extra = std::size_t{1}; extra <= 300; ++extra) {
		auto file = sample.baked.file;
		for (auto index = std::size_t{0}; index < extra; ++index) {
			file.push_back(random_byte(random));
		}
		if (!refused_with(image::parse(file, sample.expect()), Failure::Overlong)) {
			return false;
		}
	}
	return true;
}

/** The header's layout as the format states it: the failure a change to each byte must produce. */
[[nodiscard]] auto failure_at(std::uint64_t position, std::size_t extents) -> Failure {
	struct Region {
		std::uint64_t end;
		Failure failure;
	};
	auto const regions = std::array{
	    Region{.end = 8, .failure = Failure::BadMagic},
	    Region{.end = 12, .failure = Failure::UnsupportedVersion},
	    Region{.end = 16, .failure = Failure::WrongCard},
	    Region{.end = 48, .failure = Failure::ModelMismatch},
	    Region{.end = 100, .failure = Failure::CheckpointMismatch},
	    Region{.end = 132, .failure = Failure::PayloadMismatch},
	    Region{.end = 136 + (2 * extents), .failure = Failure::ScheduleMismatch},
	};
	auto const found = std::ranges::find_if(regions, [position](Region const& region) { return position < region.end; });
	return found == regions.end() ? Failure::DirtyPadding : found->failure;
}

/** A change to any one header byte, to any other value, is refused, and with the failure of the field it lies in. */
[[nodiscard]] auto every_header_byte_is_checked(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const expect = sample.expect();
	auto file = sample.baked.file;
	for (auto position = std::size_t{0}; position < image::payload_offset(sample.keys.size()); ++position) {
		auto const original = file[position];
		file[position] = original ^ std::byte{static_cast<std::uint8_t>(std::uniform_int_distribution<int>{1, 255}(random))};
		auto const refused = refused_with(image::parse(file, expect), failure_at(position, sample.keys.size()));
		file[position] = original;
		if (!refused) {
			return false;
		}
	}
	return image::parse(file, expect).has_value();
}

/** A change to any payload byte, extent or padding between extents, is refused as a payload mismatch. */
[[nodiscard]] auto payload_bytes_are_checked(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const expect = sample.expect();
	auto file = sample.baked.file;
	auto const start = image::payload_offset(sample.keys.size());
	auto pick = std::uniform_int_distribution<std::size_t>{start, file.size() - 1};
	for (auto trial = 0; trial < 600; ++trial) {
		auto const position = pick(random);
		auto const original = file[position];
		file[position] = original ^ std::byte{static_cast<std::uint8_t>(std::uniform_int_distribution<int>{1, 255}(random))};
		auto const refused = refused_with(image::parse(file, expect), Failure::PayloadMismatch);
		file[position] = original;
		if (!refused) {
			return false;
		}
	}
	return true;
}

/** Whole, well-formed images baked for something else: the other card, checkpoint, model or order. */
[[nodiscard]] auto mismatched_images_are_refused(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 100; ++trial) {
		auto const sample = specimen(random);
		auto const expect = sample.expect();
		auto const& file = sample.baked.file;

		auto const wrong_card = Expect{.card = other_card(expect.card), .checkpoint = expect.checkpoint, .order = expect.order};
		auto other_commit = expect.checkpoint;
		other_commit.commit.bytes.at(random() % image::commit_bytes) ^= std::byte{1};
		auto other_manifest = expect.checkpoint;
		other_manifest.manifest.bytes.at(random() % sha256::digest_bytes) ^= std::byte{0x80};
		auto const wrong_commit = Expect{.card = expect.card, .checkpoint = other_commit, .order = expect.order};
		auto const wrong_manifest = Expect{.card = expect.card, .checkpoint = other_manifest, .order = expect.order};

		auto other_keys = synthetic::random_order<SyntheticCatalog>(random);
		if (other_keys == sample.keys) {
			other_keys.pop_back();
		}
		auto const wrong_order = Expect{.card = expect.card, .checkpoint = expect.checkpoint, .order = Order{other_keys}};

		auto const other_model = synthetic::bake<synthetic::OtherCatalog>(sample.stamp, sample.keys, random(), random);

		if (!refused_with(image::parse(file, wrong_card), Failure::WrongCard) ||
		    !refused_with(image::parse(file, wrong_commit), Failure::CheckpointMismatch) ||
		    !refused_with(image::parse(file, wrong_manifest), Failure::CheckpointMismatch) ||
		    !refused_with(image::parse(file, wrong_order), Failure::ScheduleMismatch) ||
		    !refused_with(image::parse(other_model.file, expect), Failure::ModelMismatch)) {
			return false;
		}
	}
	return true;
}

/** Version and card fields holding any other value: a future format, or a card that does not exist. */
[[nodiscard]] auto foreign_field_values_are_refused(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const expect = sample.expect();
	auto const store = [](std::vector<std::byte>& file, std::size_t offset, std::uint32_t value) {
		for (auto index = std::size_t{0}; index < 4; ++index) {
			file.at(offset + index) = std::byte{static_cast<std::uint8_t>((value >> (8 * index)) & 0xffU)};
		}
	};
	for (auto trial = 0; trial < 500; ++trial) {
		auto version = static_cast<std::uint32_t>(random());
		version = version == image::format_version ? 0 : version;
		auto card = static_cast<std::uint32_t>(random() % 8);
		card = card == std::to_underlying(expect.card) ? 1 - card : card;
		auto with_version = sample.baked.file;
		store(with_version, 8, version);
		auto with_card = sample.baked.file;
		store(with_card, 12, card);
		if (!refused_with(image::parse(with_version, expect), Failure::UnsupportedVersion) ||
		    !refused_with(image::parse(with_card, expect), Failure::WrongCard)) {
			return false;
		}
	}
	return true;
}

/**
 * Garbage never crashes the reader and is never accepted: random files, random tails behind a valid header start, and
 * images with random bytes overwritten in the header and table.
 */
[[nodiscard]] auto garbage_is_refused(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const expect = sample.expect();
	auto const start = image::payload_offset(sample.keys.size());
	for (auto trial = 0; trial < 2000; ++trial) {
		auto file = std::vector<std::byte>{};
		switch (trial % 3) {
		case 0:
			file.resize(std::uniform_int_distribution<std::size_t>{0, 9000}(random));
			std::ranges::generate(file, [&random] { return random_byte(random); });
			break;
		case 1: {
			file.assign(sample.baked.file.begin(), sample.baked.file.begin() + std::uniform_int_distribution<std::ptrdiff_t>{0, 200}(random));
			auto const tail = std::uniform_int_distribution<std::size_t>{0, 9000}(random);
			for (auto index = std::size_t{0}; index < tail; ++index) {
				file.push_back(random_byte(random));
			}
			break;
		}
		default: {
			file = sample.baked.file;
			auto const changes = std::uniform_int_distribution<int>{1, 32}(random);
			for (auto change = 0; change < changes; ++change) {
				auto const position = std::uniform_int_distribution<std::size_t>{0, start - 1}(random);
				file[position] ^= std::byte{static_cast<std::uint8_t>(std::uniform_int_distribution<int>{1, 255}(random))};
			}
			if (file == sample.baked.file) {
				continue;
			}
			break;
		}
		}
		if (image::parse(file, expect).has_value()) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "written images read back whole (seed 1)", .holds = images_round_trip(1)},
	    engine::testing::Case{.name = "written images read back whole (seed 2)", .holds = images_round_trip(2)},
	    engine::testing::Case{.name = "layout laws hold for random synthetic orders (seed 3)", .holds = layout_laws_hold<SyntheticCatalog>(3, 2000)},
	    engine::testing::Case{.name = "layout laws hold for random model orders (seed 4)", .holds = layout_laws_hold<image::ModelCatalog>(4, 200)},
	    engine::testing::Case{.name = "every proper prefix is truncated (seed 5)", .holds = every_prefix_is_truncated(5)},
	    engine::testing::Case{.name = "appended bytes are refused (seed 6)", .holds = appended_bytes_are_refused(6)},
	    engine::testing::Case{.name = "every header byte is checked by its field (seed 7)", .holds = every_header_byte_is_checked(7)},
	    engine::testing::Case{.name = "every header byte is checked by its field (seed 8)", .holds = every_header_byte_is_checked(8)},
	    engine::testing::Case{.name = "payload bytes are checked (seed 9)", .holds = payload_bytes_are_checked(9)},
	    engine::testing::Case{.name = "images for another card, checkpoint, model or order are refused (seed 10)",
	                          .holds = mismatched_images_are_refused(10)},
	    engine::testing::Case{.name = "foreign version and card values are refused (seed 11)", .holds = foreign_field_values_are_refused(11)},
	    engine::testing::Case{.name = "garbage is refused without a crash (seed 12)", .holds = garbage_is_refused(12)},
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.image;
import engine.image_synthetic;
import engine.model;
import engine.sha256;
import engine.testing;

namespace {

namespace image = engine::image;
namespace sha256 = engine::sha256;
namespace synthetic = engine::image_synthetic;

using image::Failure;
using image::Target;

using CardZero = synthetic::SyntheticCatalog<Target::CardZero>;
using CardOne = synthetic::SyntheticCatalog<Target::CardOne>;
using Host = synthetic::SyntheticCatalog<Target::Host>;

[[nodiscard]] auto random_byte(std::mt19937_64& random) -> std::byte {
	return std::byte{static_cast<std::uint8_t>(random() & 0xffU)};
}

[[nodiscard]] auto nonzero_byte(std::mt19937_64& random) -> std::byte {
	return std::byte{static_cast<std::uint8_t>(std::uniform_int_distribution<int>{1, 255}(random))};
}

[[nodiscard]] auto random_checkpoint(std::mt19937_64& random) -> image::Checkpoint {
	auto checkpoint = synthetic::synthetic_checkpoint;
	std::ranges::generate(checkpoint.commit.bytes, [&random] { return random_byte(random); });
	std::ranges::generate(checkpoint.manifest.bytes, [&random] { return random_byte(random); });
	return checkpoint;
}

[[nodiscard]] auto refused_with(engine::core::Result<void, image::Rejection> const& result, Failure failure) -> bool {
	return !result.has_value() && result.error().failure == failure;
}

/** Whatever was written reads back: the stamp is accepted and every extent holds its bytes where the layout says. */
template<image::Catalog C>
[[nodiscard]] auto images_round_trip(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 60; ++trial) {
		auto const checkpoint = random_checkpoint(random);
		auto const content_seed = random();
		auto const baked = synthetic::bake<C>(checkpoint, content_seed, random);
		if (!image::parse<C>(baked.file, checkpoint) || baked.file.size() != image::file_bytes<C>) {
			return false;
		}
		auto const payload = std::span{baked.file}.subspan(image::payload_start<C>);
		if (sha256::digest(payload) != baked.finished.payload) {
			return false;
		}
		for (auto index = std::size_t{0}; index < std::size(C::extents); ++index) {
			auto const placement = image::layout<C>[index];
			if (!std::ranges::equal(payload.subspan(placement.offset, placement.bytes), synthetic::content<C>(index, content_seed))) {
				return false;
			}
		}
	}
	return true;
}

/**
 * The layout laws, for the extents of `C` in any order and any subset: each starts on its part's boundary (whole
 * tiles on a tile, side data on a 256-byte chunk; so all on 16 bytes); they follow the order without overlap, packed
 * tight (padding shorter than the boundary); and the payload starts on a page after the table.
 */
template<image::Catalog C>
[[nodiscard]] auto layout_laws_hold(std::uint64_t seed, int trials) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < trials; ++trial) {
		auto extents = std::vector<image::Extent>{std::begin(C::extents), std::end(C::extents)};
		std::ranges::shuffle(extents, random);
		extents.resize(std::uniform_int_distribution<std::size_t>{0, extents.size()}(random));
		auto const placements = image::place(extents);
		auto const start = image::payload_offset(extents.size());
		if (placements.size() != extents.size() || start % image::payload_alignment != 0 ||
		    start < image::fixed_header_bytes + (image::entry_bytes * extents.size())) {
			return false;
		}
		auto end = std::uint64_t{0};
		for (auto index = std::size_t{0}; index < placements.size(); ++index) {
			auto const& placement = placements[index];
			auto const& extent = extents[index];
			auto const boundary = image::alignment(extent.part);
			auto const tiles = extent.part != image::Part::Tiles || (placement.offset % 256 == 0 && placement.bytes % (std::uint64_t{16} * 1024) == 0);
			auto const in_place = placement.bytes == extent.bytes && placement.offset % 16 == 0 && placement.offset % boundary == 0 &&
			                      placement.offset >= end && placement.offset - end < boundary;
			if (!tiles || !in_place) {
				return false;
			}
			end = placement.end();
		}
	}
	return true;
}

/** A fixed image of the card-zero test catalog, for the corruption searches. */
struct Specimen {
	image::Checkpoint checkpoint;
	synthetic::Baked baked;
};

[[nodiscard]] auto specimen(std::mt19937_64& random) -> Specimen {
	auto const checkpoint = random_checkpoint(random);
	return {.checkpoint = checkpoint, .baked = synthetic::bake<CardZero>(checkpoint, random(), random)};
}

/** Every proper prefix of an image is refused as truncated, from the empty file to one byte short. */
[[nodiscard]] auto every_prefix_is_truncated(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const file = std::span<std::byte const>{sample.baked.file};
	for (auto length = std::size_t{0}; length < file.size(); ++length) {
		if (!refused_with(image::parse<CardZero>(file.first(length), sample.checkpoint), Failure::Truncated)) {
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
		if (!refused_with(image::parse<CardZero>(file, sample.checkpoint), Failure::Overlong)) {
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
	    Region{.end = 16, .failure = Failure::WrongTarget},
	    Region{.end = 48, .failure = Failure::ModelMismatch},
	    Region{.end = 100, .failure = Failure::CheckpointMismatch},
	    Region{.end = 132, .failure = Failure::PayloadMismatch},
	    Region{.end = 136 + (24 * extents), .failure = Failure::ScheduleMismatch},
	};
	auto const found = std::ranges::find_if(regions, [position](Region const& region) { return position < region.end; });
	return found == regions.end() ? Failure::DirtyPadding : found->failure;
}

/** A change to any one header byte, to any other value, is refused, with the failure of the field it lies in. */
[[nodiscard]] auto every_header_byte_is_checked(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto file = sample.baked.file;
	for (auto position = std::size_t{0}; position < image::payload_start<CardZero>; ++position) {
		auto const original = file[position];
		file[position] = original ^ nonzero_byte(random);
		auto const refused = refused_with(image::parse<CardZero>(file, sample.checkpoint), failure_at(position, std::size(CardZero::extents)));
		file[position] = original;
		if (!refused) {
			return false;
		}
	}
	return image::parse<CardZero>(file, sample.checkpoint).has_value();
}

/** A change to any payload byte, extent or padding between extents, is refused as a payload mismatch. */
[[nodiscard]] auto payload_bytes_are_checked(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto file = sample.baked.file;
	auto pick = std::uniform_int_distribution<std::size_t>{image::payload_start<CardZero>, file.size() - 1};
	for (auto trial = 0; trial < 600; ++trial) {
		auto const position = pick(random);
		auto const original = file[position];
		file[position] = original ^ nonzero_byte(random);
		auto const refused = refused_with(image::parse<CardZero>(file, sample.checkpoint), Failure::PayloadMismatch);
		file[position] = original;
		if (!refused) {
			return false;
		}
	}
	return true;
}

/** Whole, well-formed images baked for something else: another memory, checkpoint, model or schedule. */
[[nodiscard]] auto mismatched_images_are_refused(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 60; ++trial) {
		auto const sample = specimen(random);
		auto const& file = sample.baked.file;
		auto other_commit = sample.checkpoint;
		other_commit.commit.bytes.at(random() % image::commit_bytes) ^= nonzero_byte(random);
		auto other_manifest = sample.checkpoint;
		other_manifest.manifest.bytes.at(random() % sha256::digest_bytes) ^= nonzero_byte(random);
		auto const other_model = synthetic::bake<synthetic::OtherModelCatalog>(sample.checkpoint, random(), random);
		auto const other_schedule = synthetic::bake<synthetic::OtherScheduleCatalog>(sample.checkpoint, random(), random);
		if (!refused_with(image::parse<CardOne>(file, sample.checkpoint), Failure::WrongTarget) ||
		    !refused_with(image::parse<Host>(file, sample.checkpoint), Failure::WrongTarget) ||
		    !refused_with(image::parse<CardZero>(file, other_commit), Failure::CheckpointMismatch) ||
		    !refused_with(image::parse<CardZero>(file, other_manifest), Failure::CheckpointMismatch) ||
		    !refused_with(image::parse<CardZero>(other_model.file, sample.checkpoint), Failure::ModelMismatch) ||
		    !refused_with(image::parse<CardZero>(other_schedule.file, sample.checkpoint), Failure::ScheduleMismatch)) {
			return false;
		}
	}
	return true;
}

/** Version and target fields holding any other value: a future format, or a memory that does not exist. */
[[nodiscard]] auto foreign_field_values_are_refused(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const sample = specimen(random);
	auto const store = [](std::vector<std::byte>& file, std::size_t offset, std::uint32_t value) {
		for (auto index = std::size_t{0}; index < 4; ++index) {
			file.at(offset + index) = std::byte{static_cast<std::uint8_t>((value >> (8 * index)) & 0xffU)};
		}
	};
	for (auto trial = 0; trial < 500; ++trial) {
		auto const version = static_cast<std::uint32_t>(std::uniform_int_distribution<std::uint64_t>{2, 0xffffffffU}(random));
		auto const target = static_cast<std::uint32_t>(std::uniform_int_distribution<std::uint64_t>{1, 0xffffffffU}(random));
		auto with_version = sample.baked.file;
		store(with_version, 8, trial % 2 == 0 ? 0 : version);
		auto with_target = sample.baked.file;
		store(with_target, 12, trial % 2 == 0 ? std::to_underlying(Target::Host) : target);
		if (!refused_with(image::parse<CardZero>(with_version, sample.checkpoint), Failure::UnsupportedVersion) ||
		    !refused_with(image::parse<CardZero>(with_target, sample.checkpoint), Failure::WrongTarget)) {
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
	for (auto trial = 0; trial < 2000; ++trial) {
		auto file = std::vector<std::byte>{};
		switch (trial % 3) {
		case 0:
			file.resize(std::uniform_int_distribution<std::size_t>{0, 9000}(random));
			std::ranges::generate(file, [&random] { return random_byte(random); });
			break;
		case 1: {
			file.assign(sample.baked.file.begin(), sample.baked.file.begin() + std::uniform_int_distribution<std::ptrdiff_t>{0, 400}(random));
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
				file[std::uniform_int_distribution<std::size_t>{0, image::payload_start<CardZero> - 1}(random)] ^= nonzero_byte(random);
			}
			if (file == sample.baked.file) {
				continue;
			}
			break;
		}
		}
		if (image::parse<CardZero>(file, sample.checkpoint).has_value()) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "card-zero images read back whole (seed 1)", .holds = images_round_trip<CardZero>(1)},
	    Case{.name = "card-one images read back whole (seed 2)", .holds = images_round_trip<CardOne>(2)},
	    Case{.name = "host images read back whole (seed 3)", .holds = images_round_trip<Host>(3)},
	    Case{.name = "an image of no extents reads back (seed 4)", .holds = images_round_trip<synthetic::EmptyCatalog>(4)},
	    Case{.name = "layout laws hold for the test catalog in any order (seed 5)", .holds = layout_laws_hold<CardZero>(5, 2000)},
	    Case{.name = "layout laws hold for the card catalog in any order (seed 6)",
	         .holds = layout_laws_hold<image::CardCatalog<engine::model::Card::Zero>>(6, 100)},
	    Case{.name = "every proper prefix is truncated (seed 7)", .holds = every_prefix_is_truncated(7)},
	    Case{.name = "appended bytes are refused (seed 8)", .holds = appended_bytes_are_refused(8)},
	    Case{.name = "every header byte is checked by its field (seed 9)", .holds = every_header_byte_is_checked(9)},
	    Case{.name = "every header byte is checked by its field (seed 10)", .holds = every_header_byte_is_checked(10)},
	    Case{.name = "payload bytes are checked (seed 11)", .holds = payload_bytes_are_checked(11)},
	    Case{.name = "images for another memory, checkpoint, model or schedule are refused (seed 12)",
	         .holds = mismatched_images_are_refused(12)},
	    Case{.name = "foreign version and target values are refused (seed 13)", .holds = foreign_field_values_are_refused(13)},
	    Case{.name = "garbage is refused without a crash (seed 14)", .holds = garbage_is_refused(14)},
	};
	return engine::testing::report(cases);
}

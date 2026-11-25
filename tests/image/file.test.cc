// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.file;
import engine.image;
import engine.image_synthetic;
import engine.sha256;
import engine.testing;

namespace {

namespace image = engine::image;
namespace synthetic = engine::image_synthetic;

using synthetic::SpecimenCatalog;

using Image = image::Image<SpecimenCatalog>;

// The specimen image, written by `image.emit` into this test's working directory.
constexpr auto specimen_path = std::string_view{"specimen.image"};

/** The mapped image hands out the payload and every extent exactly as the baker wrote them. */
[[nodiscard]] auto mapped_image_reads_back() -> bool {
	auto const opened = Image::open(specimen_path, synthetic::synthetic_checkpoint);
	if (!opened) {
		return false;
	}
	auto const baked = synthetic::specimen_image();
	if (!std::ranges::equal(opened->payload(), std::span{baked.file}.subspan(image::payload_start<SpecimenCatalog>))) {
		return false;
	}
	if (opened->payload_digest() != baked.finished.payload || opened->payload_digest() != engine::sha256::digest(opened->payload())) {
		return false;
	}
	auto const fixed = opened->extent<6>();
	static_assert(decltype(fixed)::extent == 4098);
	auto every = true;
	for (auto index = std::size_t{0}; index < std::size(SpecimenCatalog::extents); ++index) {
		every = every && std::ranges::equal(opened->bytes(index), synthetic::content<SpecimenCatalog>(index, synthetic::specimen_seed));
	}
	return every && std::ranges::equal(fixed, synthetic::content<SpecimenCatalog>(6, synthetic::specimen_seed));
}

/** The image survives a move of its owner: the spans come from the mapping, not from where the owner sat. */
[[nodiscard]] auto moved_image_still_reads() -> bool {
	auto opened = Image::open(specimen_path, synthetic::synthetic_checkpoint);
	if (!opened) {
		return false;
	}
	auto const moved = std::move(*opened);
	return std::ranges::equal(moved.bytes(0), synthetic::content<SpecimenCatalog>(0, synthetic::specimen_seed));
}

[[nodiscard]] auto refusals_reach_the_caller() -> bool {
	static_assert(SpecimenCatalog::target == image::Target::CardOne);
	auto const as_host = image::Image<synthetic::SyntheticCatalog<image::Target::Host>>::open(specimen_path, synthetic::synthetic_checkpoint);
	auto const missing = Image::open("no-such.image", synthetic::synthetic_checkpoint);
	auto const directory = Image::open(".", synthetic::synthetic_checkpoint);
	auto const is = [](auto const& result, image::ImageError const& error) { return !result && result.error() == error; };
	return is(as_host, image::ImageError{.cause = image::Rejection{.failure = image::Failure::WrongTarget}}) &&
	       is(missing, image::ImageError{.cause = engine::file::FileError{.failure = engine::file::Failure::NotFound}}) &&
	       is(directory, image::ImageError{.cause = engine::file::FileError{.failure = engine::file::Failure::NotRegular}});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a mapped image reads back its payload, its recorded payload digest and every extent", .holds = mapped_image_reads_back()},
	    engine::testing::Case{.name = "a moved image still reads its extents", .holds = moved_image_still_reads()},
	    engine::testing::Case{.name = "mapping failures and refusals reach the caller as image errors", .holds = refusals_reach_the_caller()},
	};
	return engine::testing::report(cases);
}

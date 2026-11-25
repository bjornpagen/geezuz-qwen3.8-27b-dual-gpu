// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.file;
import engine.image;
import engine.image_synthetic;
import engine.testing;

namespace {

namespace image = engine::image;
namespace synthetic = engine::image_synthetic;

using synthetic::SyntheticCatalog;

using Image = image::Image<SyntheticCatalog>;

// The specimen image, written by `image.emit` beside this test's working directory.
constexpr auto specimen_path = std::string_view{"specimen.image"};

[[nodiscard]] auto expect_for(std::span<image::Key const> keys, image::Card card) -> image::Expect<SyntheticCatalog> {
	return {.card = card, .checkpoint = synthetic::synthetic_checkpoint, .order = image::Order<SyntheticCatalog>{keys}};
}

/** The mapped image hands out the payload and every extent exactly as the baker wrote them. */
[[nodiscard]] auto mapped_image_reads_back() -> bool {
	auto const keys = synthetic::specimen_keys();
	auto const opened = Image::open(specimen_path, expect_for(keys, synthetic::specimen_stamp.card));
	if (!opened) {
		return false;
	}
	auto const baked = synthetic::specimen_image();
	auto const payload = std::span{baked.file}.subspan(image::payload_offset(keys.size()));
	if (!std::ranges::equal(opened->payload(), payload) || !std::ranges::equal(opened->sections(), image::place(image::Order<SyntheticCatalog>{keys}))) {
		return false;
	}
	auto const fixed = opened->extent<image::Key{.index = 6}>();
	static_assert(decltype(fixed)::extent == 4098);
	return std::ranges::equal(fixed, synthetic::content<SyntheticCatalog>(image::Key{.index = 6}, synthetic::specimen_seed)) &&
	       std::ranges::all_of(keys, [&opened](image::Key key) {
		       return std::ranges::equal(opened->bytes(key), synthetic::content<SyntheticCatalog>(key, synthetic::specimen_seed));
	       });
}

/** The image survives a move of its owner: the spans come from the mapping, not from where the owner sat. */
[[nodiscard]] auto moved_image_still_reads() -> bool {
	auto const keys = synthetic::specimen_keys();
	auto opened = Image::open(specimen_path, expect_for(keys, synthetic::specimen_stamp.card));
	if (!opened) {
		return false;
	}
	auto const moved = std::move(*opened);
	return std::ranges::equal(moved.bytes(keys.front()), synthetic::content<SyntheticCatalog>(keys.front(), synthetic::specimen_seed));
}

[[nodiscard]] auto refusals_reach_the_caller() -> bool {
	auto const keys = synthetic::specimen_keys();
	static_assert(synthetic::specimen_stamp.card == image::Card::One);
	auto const wrong_card = Image::open(specimen_path, expect_for(keys, image::Card::Zero));
	auto const missing = Image::open("no-such.image", expect_for(keys, synthetic::specimen_stamp.card));
	auto const directory = Image::open(".", expect_for(keys, synthetic::specimen_stamp.card));
	auto const is = [](auto const& result, image::ImageError const& error) { return !result && result.error() == error; };
	return is(wrong_card, image::ImageError{.cause = image::Rejection{.failure = image::Failure::WrongCard}}) &&
	       is(missing, image::ImageError{.cause = engine::file::FileError{.failure = engine::file::Failure::NotFound}}) &&
	       is(directory, image::ImageError{.cause = engine::file::FileError{.failure = engine::file::Failure::NotRegular}});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a mapped image reads back the payload and every extent", .holds = mapped_image_reads_back()},
	    engine::testing::Case{.name = "a moved image still reads its extents", .holds = moved_image_still_reads()},
	    engine::testing::Case{.name = "mapping failures and refusals reach the caller as image errors", .holds = refusals_reach_the_caller()},
	};
	return engine::testing::report(cases);
}

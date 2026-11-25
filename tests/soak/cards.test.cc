// SPDX-License-Identifier: AGPL-3.0-only
//
// The soak on the CUDA driver and NVML test double, over two small test-double card images baked with the image
// writer. The double runs no kernels, so the digest state stays as each pass starts it and the first finished pass is
// a mismatch: which makes the double the place to check the mismatch report end to end (the card, the pass, the baked
// digest, what the card left, both cards' readings), the launches the soak issues and their parameters, the images it
// refuses, and that every failing vendor call while setting up is reported and leaks nothing.
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.file_write;
import engine.image;
import engine.model;
import engine.platform;
import engine.sha256;
import engine.soak;
import engine.testing;

namespace {

namespace platform = engine::platform;
namespace device = engine::device;
namespace fake = engine::driver_double;
namespace image = engine::image;
namespace model = engine::model;
namespace sha256 = engine::sha256;
namespace soak = engine::soak;

using model::Card;

inline constexpr auto test_description = sha256::digest("soak test card images: a test double");

/** Card zero's test image: 257 whole blocks and a 9-word tail. */
struct TestCardZero {
	static constexpr image::Target target = image::Target::CardZero;
	static constexpr std::array<image::Extent, 2> extents = {{
	    {.identity = 1, .bytes = image::alignment(image::Part::Tiles), .part = image::Part::Tiles},
	    {.identity = 2, .bytes = 100, .part = image::Part::Whole},
	}};
	static constexpr sha256::Digest description = test_description;
};

/** Card one's test image: 558 whole blocks and a 14-word tail, whose padding takes a block of its own. */
struct TestCardOne {
	static constexpr image::Target target = image::Target::CardOne;
	static constexpr std::array<image::Extent, 2> extents = {{
	    {.identity = 3, .bytes = 2 * image::alignment(image::Part::Tiles), .part = image::Part::Tiles},
	    {.identity = 4, .bytes = 3000, .part = image::Part::Whole},
	}};
	static constexpr sha256::Digest description = test_description;
};

struct TestCards {
	using CardZero = TestCardZero;
	using CardOne = TestCardOne;
};

static_assert(platform::CardCatalogs<TestCards>);
static_assert(soak::pass_of(image::payload_bytes<TestCardZero>).blocks == 257 && soak::pass_of(image::payload_bytes<TestCardZero>).tail_words == 9);
static_assert(soak::pass_of(image::payload_bytes<TestCardOne>).blocks == 558 && soak::pass_of(image::payload_bytes<TestCardOne>).tail_words == 14);

inline constexpr image::Checkpoint test_checkpoint = {
    .commit = image::checkpoint_commit,
    .manifest = sha256::digest("soak test checkpoint: a test double"),
};

/** A baked test image: its file's bytes and its payload's. */
struct Baked {
	std::vector<std::byte> file;
	std::vector<std::byte> payload;
};

template<image::Catalog C>
[[nodiscard]] auto bake(std::uint64_t seed) -> Baked {
	auto random = engine::testing::generator(seed);
	auto writer = image::Writer<C>{test_checkpoint};
	auto file = std::vector<std::byte>(image::file_bytes<C>);
	for (auto index = std::size_t{0}; index < std::size(C::extents); ++index) {
		auto bytes = std::vector<std::byte>(std::span{C::extents}[index].bytes);
		std::ranges::generate(bytes, [&] { return std::byte{static_cast<std::uint8_t>(random() & 0xffU)}; });
		std::ranges::copy(bytes, std::span{file}.subspan(writer.write(index, bytes)).begin());
	}
	auto const finished = std::move(writer).finish();
	std::ranges::copy(finished.header, file.begin());
	auto payload = std::vector<std::byte>{file.begin() + static_cast<std::ptrdiff_t>(image::payload_start<C>), file.end()};
	return {.file = std::move(file), .payload = std::move(payload)};
}

auto put(std::filesystem::path const& path, std::span<std::byte const> bytes) -> void {
	auto output = engine::file_write::Output::create(path, bytes.size());
	engine::core::invariant(output.has_value() && output->write(0, bytes).has_value());
	engine::core::invariant(std::move(*output).commit().has_value());
}

/** A scratch directory holding both baked images, as geezuz-bake leaves them. */
struct Images {
	std::filesystem::path directory;
	Baked zero;
	Baked one;

	[[nodiscard]] auto paths() const -> platform::ImagePaths {
		return {.card_zero = directory / "card0.image", .card_one = directory / "card1.image"};
	}
};

[[nodiscard]] auto images(std::string_view name) -> Images {
	auto const directory = std::filesystem::current_path() / "soak.scratch" / name;
	std::filesystem::remove_all(directory);
	std::filesystem::create_directories(directory);
	auto baked = Images{.directory = directory, .zero = bake<TestCardZero>(11), .one = bake<TestCardOne>(12)};
	put(baked.paths().card_zero, baked.zero.file);
	put(baked.paths().card_one, baked.one.file);
	return baked;
}

/** A clock whose time moves on by a fixed tick each time it is read, and to the deadline when waited on. */
struct TickingClock {
	std::chrono::steady_clock::time_point steady = std::chrono::steady_clock::time_point{} + std::chrono::hours{1};

	[[nodiscard]] auto now() -> platform::Moment {
		auto const moment = platform::Moment{.wall = std::chrono::system_clock::time_point{}, .steady = steady};
		steady += std::chrono::milliseconds{1};
		return moment;
	}

	auto sleep_until(std::chrono::steady_clock::time_point until) -> void {
		steady = std::max(steady, until);
	}
};

inline auto const far = std::chrono::steady_clock::time_point::max();

[[nodiscard]] auto open(fake::Fake const& driver, platform::ImagePaths const& paths) -> engine::core::Result<platform::Cards<TestCards>, platform::SetupError> {
	auto devices = device::Devices::open(driver.driver());
	engine::core::invariant(devices.has_value());
	return platform::Cards<TestCards>::open(std::move(*devices), paths, test_checkpoint, far);
}

/** The digest the card leaves when no kernel ran: the initial state's words, each most significant byte first. */
[[nodiscard]] auto untouched_digest() -> sha256::Digest {
	auto digest = sha256::Digest{};
	auto const words = std::array<std::uint32_t, 8>{0x6a09'e667, 0xbb67'ae85, 0x3c6e'f372, 0xa54f'f53a, 0x510e'527f, 0x9b05'688c, 0x1f83'd9ab, 0x5be0'cd19};
	for (auto index = std::size_t{0}; index < digest.bytes.size(); ++index) {
		digest.bytes.at(index) = std::byte{static_cast<std::uint8_t>(words.at(index / 4) >> (24 - (8 * (index % 4))))};
	}
	return digest;
}

/** The modelled card's readings, as the soak's telemetry must report them. */
[[nodiscard]] auto modelled(fake::CardModel const& card) -> platform::Telemetry {
	return {
	    .memory = {.megahertz = card.memory_clock, .state = static_cast<device::PerformanceState>(card.performance_state)},
	    .sm = {.megahertz = card.sm_clock},
	    .power = {.draw_milliwatts = card.power_draw_milliwatts, .limit_milliwatts = card.power_limit_milliwatts},
	    .temperature = {.celsius = card.gpu_celsius},
	    .reasons = {.bits = card.clocks_event_reasons},
	    .counters = card.clocks_event_counters,
	};
}

// The typed launch passes exactly its kernel's parameter struct, so reading the bytes back as that struct cannot fail.
/* PIN(clang-tidy-optional-merged-record): launch checks read device-header structs, so they hold no optional */
template<class Params>
[[nodiscard]] auto params_of(fake::Launch const& launch) -> Params {
	engine::core::invariant(launch.parameters.size() == sizeof(Params));
	auto bytes = std::array<std::byte, sizeof(Params)>{};
	std::ranges::copy(launch.parameters, bytes.begin());
	return std::bit_cast<Params>(bytes);
}

[[nodiscard]] auto launches_of(fake::Fake const& driver, int ordinal, std::string_view entry) -> std::vector<fake::Launch> {
	auto found = std::vector<fake::Launch>{};
	std::ranges::copy_if(driver.launches(), std::back_inserter(found), [&](fake::Launch const& launch) { return launch.ordinal == ordinal && launch.entry == entry; });
	return found;
}

/** Each card's one pass: one absorbing launch over all its blocks, then the finish over its tail, on one state. */
[[nodiscard]] auto image_launches_hold(fake::Fake const& driver, int ordinal, std::uint64_t bytes) -> bool {
	auto const pass = soak::pass_of(bytes);
	auto const absorbs = launches_of(driver, ordinal, "soak_absorb");
	auto const finishes = launches_of(driver, ordinal, "soak_finish");
	if (absorbs.size() != 1 || finishes.size() != 1) {
		return false;
	}
	auto const absorb = params_of<soak::AbsorbParams>(absorbs.front());
	auto const finish = params_of<soak::FinishParams>(finishes.front());
	return absorb.blocks == pass.blocks && finish.state == absorb.state && finish.tail == absorb.source + pass.blocks * soak::block_bytes &&
	       finish.tail_words == pass.tail_words && finish.total_bytes == bytes &&
	       absorbs.front().grid.count() == 1 && absorbs.front().block.x == static_cast<std::uint32_t>(soak::hash_budget.threads);
}

/** Each card's pattern lane: round 0 written over the region, then checked over the same region into its tallies. */
[[nodiscard]] auto pattern_launches_hold(fake::Fake const& driver, int ordinal) -> bool {
	auto const writes = launches_of(driver, ordinal, "soak_write");
	auto const checks = launches_of(driver, ordinal, "soak_check");
	if (writes.size() != 1 || checks.size() != 1) {
		return false;
	}
	auto const write = params_of<soak::WriteParams>(writes.front());
	auto const check = params_of<soak::CheckParams>(checks.front());
	return write.round == 0 && check.round == 0 && check.region == write.region && writes.front().grid.x == soak::pattern_blocks;
}

[[nodiscard]] auto mismatch_is_reported_end_to_end() -> bool {
	auto const baked = images("report");
	auto const driver = fake::Fake{fake::Machine{}};
	auto holds = false;
	{
		auto cards = open(driver, baked.paths());
		if (!cards) {
			return false;
		}
		auto clock = TickingClock{};
		auto const outcome = platform::run(*cards, clock, std::chrono::minutes{1});
		auto const* const found = outcome ? std::get_if<platform::Found>(&*outcome) : nullptr;
		auto const* const mismatch = found == nullptr ? nullptr : std::get_if<platform::ImageMismatch>(&found->sighting.finding);
		holds = mismatch != nullptr && mismatch->card == Card::Zero && mismatch->pass == 1 && mismatch->expected == sha256::digest(baked.zero.payload) &&
		        mismatch->found == untouched_digest() && found->cards.at(0) == platform::Progress{.passes = 0, .rounds = 0} &&
		        std::ranges::all_of(device::card_pair,
		                            [&](Card card) {
			                            auto const& reading = found->sighting.cards.at(std::to_underlying(card));
			                            return reading && *reading == modelled(fake::Machine{}.cards.at(std::to_underlying(card)));
		                            }) &&
		        image_launches_hold(driver, 0, image::payload_bytes<TestCardZero>) && image_launches_hold(driver, 1, image::payload_bytes<TestCardOne>) &&
		        pattern_launches_hold(driver, 0) && pattern_launches_hold(driver, 1);
	}
	return holds && driver.nothing_live() && driver.violations().empty();
}

[[nodiscard]] auto rejection_of(engine::core::Result<platform::Cards<TestCards>, platform::SetupError> const& opened) -> std::optional<image::Failure> {
	if (opened) {
		return std::nullopt;
	}
	auto const* const refused = std::get_if<image::ImageError>(&opened.error().reason);
	auto const* const rejection = refused == nullptr ? nullptr : std::get_if<image::Rejection>(&refused->cause);
	return rejection == nullptr ? std::nullopt : std::optional{rejection->failure};
}

/** A corrupted payload, the other card's image, and a missing image are each refused before the soak starts. */
[[nodiscard]] auto bad_images_are_refused() -> bool {
	auto const baked = images("refused");
	auto corrupt = baked.zero.file;
	corrupt.at(corrupt.size() - 7) ^= std::byte{0x10};
	put(baked.directory / "corrupt.image", corrupt);
	auto const driver = fake::Fake{fake::Machine{}};
	auto holds = true;
	{
		holds = holds &&
		        rejection_of(open(driver, {.card_zero = baked.directory / "corrupt.image", .card_one = baked.paths().card_one})) == image::Failure::PayloadMismatch;
		holds = holds && rejection_of(open(driver, {.card_zero = baked.paths().card_one, .card_one = baked.paths().card_one})) == image::Failure::WrongTarget;
		auto const missing = open(driver, {.card_zero = baked.paths().card_zero, .card_one = baked.directory / "absent.image"});
		holds = holds && !missing && std::holds_alternative<image::ImageError>(missing.error().reason) && !rejection_of(missing);
	}
	return holds && driver.nothing_live();
}

/**
 * Every vendor call while setting up, failing, is reported and leaks nothing; except the releases setting up makes
 * along the way (each card's copy event and its image's pin, once the copy is done), whose failure has no remedy and
 * leaves the set-up whole.
 */
[[nodiscard]] auto every_failing_setup_call_is_reported() -> bool {
	constexpr auto releases = std::size_t{2} * model::cards;
	auto const baked = images("failing");
	auto const counted = [&] {
		auto const driver = fake::Fake{fake::Machine{}};
		auto devices = device::Devices::open(driver.driver());
		engine::core::invariant(devices.has_value());
		auto const before = driver.calls();
		auto const opened = platform::Cards<TestCards>::open(std::move(*devices), baked.paths(), test_checkpoint, far);
		return std::pair{before, opened ? driver.calls() : before};
	}();
	auto unreported = std::size_t{0};
	auto holds = counted.second > counted.first + releases;
	for (auto call = counted.first; call < counted.second && holds; ++call) {
		auto const driver = fake::Fake{fake::Machine{.failing_call = call}};
		auto reported = false;
		{
			auto devices = device::Devices::open(driver.driver());
			engine::core::invariant(devices.has_value());
			auto const opened = platform::Cards<TestCards>::open(std::move(*devices), baked.paths(), test_checkpoint, far);
			reported = !opened && std::holds_alternative<device::DeviceError>(opened.error().reason);
			holds = reported || opened.has_value();
		}
		unreported += reported ? 0 : 1;
		holds = holds && (!reported || driver.nothing_live());
	}
	return holds && unreported <= releases;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "on the double, card zero's first pass is a mismatch reported with its baked digest, what the card left, both cards' "
	                "readings, and exactly the launches a pass and a round take",
	        .holds = mismatch_is_reported_end_to_end(),
	    },
	    engine::testing::Case{
	        .name = "a corrupted payload, the other card's image and a missing image are refused before the soak",
	        .holds = bad_images_are_refused(),
	    },
	    engine::testing::Case{
	        .name = "every vendor call of the set-up, failing, is reported and leaks nothing, but for the releases along the way",
	        .holds = every_failing_setup_call_is_reported(),
	    },
	};
	return engine::testing::report(cases);
}

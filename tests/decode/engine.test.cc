// SPDX-License-Identifier: AGPL-3.0-only
//
// Bringing the decode engine up on the driver double, with small synthetic images standing in for the baked ones: the
// images are opened and verified, each card's goes to its card and the host's to mapped memory, the loops launch once
// per card, co-resident, with exactly the parameters the engine states (each card's own memories and its peer's
// workspace and control block, cross-wired), and stop ends both loops. Every failure the driver can return during the
// bring-up is reached by a search over the failing call: each fails the open with an error and leaves nothing behind.
import std;
import engine.core;
import engine.decode;
import engine.decode_device;
import engine.device;
import engine.driver;
import engine.driver_double;
import engine.file_write;
import engine.image;
import engine.image_synthetic;
import engine.model;
import engine.testing;

namespace {

namespace decode = engine::decode;
namespace dd = engine::decode_device;
namespace device = engine::device;
namespace fake = engine::driver_double;
namespace image = engine::image;
namespace synthetic = engine::image_synthetic;

using engine::model::Card;

struct SyntheticCatalogs {
	using CardZero = synthetic::SyntheticCatalog<image::Target::CardZero>;
	using CardOne = synthetic::SyntheticCatalog<image::Target::CardOne>;
	using Host = synthetic::SyntheticCatalog<image::Target::Host>;
};

using Engine = decode::Engine<SyntheticCatalogs>;

auto const deadline_after = [] {
	return std::chrono::steady_clock::now() + std::chrono::seconds{30};
};

/** Polls `poll` until it yields an outcome, as the pipeline's run loop does; a poll that never does fails as abandoned. */
template<class Poll>
[[nodiscard]] auto settled(Poll poll) -> std::remove_cvref_t<decltype(*poll())> {
	auto const deadline = deadline_after();
	for (;;) {
		if (auto outcome = poll()) {
			return std::move(*outcome);
		}
		if (std::chrono::steady_clock::now() >= deadline) {
			return std::unexpected(decode::DecodeError{.reason = decode::Abandoned{.ticket = std::nullopt, .kernels = decode::Kernels::MayRun}});
		}
		std::this_thread::yield();
	}
}

template<image::Catalog C>
[[nodiscard]] auto write_image(std::filesystem::path const& path, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const baked = synthetic::bake<C>(synthetic::synthetic_checkpoint, seed, random);
	auto output = engine::file_write::Output::create(path, baked.file.size());
	return output && output->write(0, baked.file).has_value() && std::move(*output).commit().has_value();
}

/** A directory of this run's own under the system's temporary directory, removed with everything in it when it dies. */
class Scratch {
public:
	[[nodiscard]] static auto create() -> std::optional<Scratch> {
		auto failed = std::error_code{};
		auto const base = std::filesystem::temp_directory_path(failed);
		if (failed) {
			return std::nullopt;
		}
		auto random = std::random_device{};
		for (auto attempt = 0; attempt < 16; ++attempt) {
			auto directory = base / std::format("decode_engine.{:08x}{:08x}", random(), random());
			// A directory that already exists belongs to someone else: draw another name.
			if (std::filesystem::create_directory(directory, failed) && !failed) {
				return Scratch{std::move(directory)};
			}
		}
		return std::nullopt;
	}

	Scratch(Scratch&& other) noexcept : directory_{std::exchange(other.directory_, {})} {}
	auto operator=(Scratch&&) -> Scratch& = delete;
	Scratch(Scratch const&) = delete;
	auto operator=(Scratch const&) -> Scratch& = delete;

	~Scratch() {
		if (!directory_.empty()) {
			auto failed = std::error_code{};
			std::filesystem::remove_all(directory_, failed);
		}
	}

	[[nodiscard]] auto directory() const -> std::filesystem::path const& {
		return directory_;
	}

private:
	explicit Scratch(std::filesystem::path directory) : directory_{std::move(directory)} {}

	std::filesystem::path directory_;
};

/** The three images, written once into the run's scratch directory. */
[[nodiscard]] auto images(Scratch const& scratch) -> std::optional<decode::Paths> {
	auto const& directory = scratch.directory();
	auto const paths = decode::Paths{.card_zero = directory / "card0.image", .card_one = directory / "card1.image", .host = directory / "host.image"};
	auto const written = write_image<SyntheticCatalogs::CardZero>(paths.card_zero, 1) && write_image<SyntheticCatalogs::CardOne>(paths.card_one, 2) &&
	                     write_image<SyntheticCatalogs::Host>(paths.host, 3);
	return written ? std::optional{paths} : std::nullopt;
}

[[nodiscard]] auto open(fake::Fake const& driver, decode::Paths const& paths) -> engine::core::Result<Engine, decode::DecodeError> {
	auto devices = device::Devices::open(driver.driver());
	if (!devices) {
		return std::unexpected(decode::DecodeError{.reason = devices.error()});
	}
	return Engine::open(std::move(*devices), paths, synthetic::synthetic_checkpoint, deadline_after());
}

[[nodiscard]] auto bytes_of(dd::Params const& params) -> std::vector<std::byte> {
	auto const raw = std::bit_cast<std::array<std::byte, sizeof(dd::Params)>>(params);
	return {raw.begin(), raw.end()};
}

/** The decode loop's launches, apart from the rope table's fill. */
[[nodiscard]] auto loops(fake::Fake const& driver) -> std::vector<fake::Launch> {
	auto found = std::vector<fake::Launch>{};
	std::ranges::copy_if(driver.launches(), std::back_inserter(found), [](fake::Launch const& launch) {
		return launch.entry == "decode_loop";
	});
	return found;
}

/** A fake loop that only answers the exit ticket. */
struct Exits {
	[[nodiscard]] auto operator()(dd::Descriptor const& /*descriptor*/) const -> dd::Result {
		return {};
	}
};

using Kernel = fake::RingKernel<dd::Descriptor, dd::Result, dd::ring_slots, Exits>;

/** The engine opens, launches each card's loop with its stated parameters, cross-wired to the peer, and stops. */
[[nodiscard]] auto brings_up_and_stops(decode::Paths const& paths) -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto held = false;
	{
		auto engine = open(driver, paths);
		if (!engine) {
			return false;
		}
		auto const zero = engine->parameters(Card::Zero);
		auto const one = engine->parameters(Card::One);
		auto const wired = zero.peer_workspace == one.workspace && one.peer_workspace == zero.workspace && zero.peer_control == one.control &&
		                   one.peer_control == zero.control && zero.card == 0 && one.card == 1 && zero.embedding == one.embedding &&
		                   zero.image != one.image && zero.pool != one.pool && zero.states != one.states && zero.ring != one.ring &&
		                   zero.masks != one.masks && zero.logits != one.logits;
		auto const memories = engine->memories(Card::One);
		auto const lent = memories.image == one.image && memories.pool == one.pool && memories.states == one.states &&
		                  memories.rotations == one.workspace + dd::workspace.exchange.rope_table && memories.tables == one.workspace + dd::workspace.tables &&
		                  memories.prompt_row == one.workspace + dd::workspace.prompt;
		auto const borrowed = engine->card(Card::One);
		auto const access = borrowed.context.ordinal() == 1 && borrowed.stream.ordinal() == 1 && engine->card(Card::Zero).stream.ordinal() == 0;
		if (!wired || !lent || !access || !engine->start()) {
			return false;
		}
		auto const launches = loops(driver);
		auto const launched = launches.size() == 2 && std::ranges::all_of(std::views::iota(std::size_t{0}, std::size_t{2}), [&](std::size_t index) {
			auto const& launch = launches[index];
			auto const card = index == 0 ? Card::Zero : Card::One;
			return std::cmp_equal(launch.ordinal, index) && launch.entry == "decode_loop" && launch.grid.x == dd::sms && launch.grid.count() == dd::sms &&
			       launch.block.x == static_cast<std::uint32_t>(dd::decode_budget.threads) &&
			       launch.shared_bytes == static_cast<std::uint32_t>(dd::decode_budget.shared_bytes) && launch.residency == engine::driver::Residency::CoResident &&
			       launch.parameters == bytes_of(engine->parameters(card));
		});
		auto const stopped = [&] {
			auto const loop_zero = Kernel{engine->parameters(Card::Zero).ring, Exits{}};
			auto const loop_one = Kernel{engine->parameters(Card::One).ring, Exits{}};
			return settled([&] { return engine->try_stop(); }).has_value();
		};
		auto const first = stopped();
		// A prefill chunk runs here, the loops gone; then they come back on fresh rings with the same memories.
		auto const old_ring = engine->parameters(Card::Zero).ring;
		auto const restarted = engine->start().has_value() && engine->parameters(Card::Zero).ring != old_ring &&
		                       loops(driver).size() == 4 && engine->parameters(Card::Zero).workspace == zero.workspace;
		held = launched && first && restarted && stopped();
	}
	return held && driver.violations().empty() && driver.nothing_live();
}

/**
 * A shutdown ends the loops from every state: unlaunched and halted loops at once, running ones once both have answered
 * their exits and their kernels have finished; loops that never answer fail at the deadline, and the failure is kept.
 */
[[nodiscard]] auto shuts_down_from_every_state(decode::Paths const& paths) -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto held = false;
	{
		auto engine = open(driver, paths);
		if (!engine) {
			return false;
		}
		auto const now = [] {
			return std::chrono::steady_clock::now();
		};
		auto const unlaunched = engine->try_shutdown(now()) == std::optional{engine::core::Result<void, decode::DecodeError>{}};
		auto const running = engine->start().has_value() && [&] {
			auto const loop_zero = Kernel{engine->parameters(Card::Zero).ring, Exits{}};
			auto const loop_one = Kernel{engine->parameters(Card::One).ring, Exits{}};
			return settled([&] { return engine->try_shutdown(deadline_after()); }).has_value();
		}();
		auto const halted = engine->try_shutdown(now()) == std::optional{engine::core::Result<void, decode::DecodeError>{}};
		// Relaunched with nothing serving the rings: the exits go unanswered until the deadline.
		auto const silent = engine->start().has_value() ? engine->try_shutdown(now()) : std::nullopt;
		auto const abandoned = silent && !*silent && std::holds_alternative<decode::Abandoned>(silent->error().reason);
		auto const kept = engine->try_shutdown(now()) == silent;
		held = unlaunched && running && halted && abandoned && kept;
	}
	return held && driver.violations().empty() && driver.nothing_live();
}

/**
 * Fail closed: loops abandoned while a kernel may still run keep every allocation the kernels could write. The engine
 * dies without a single driver call (no free, no context destroyed), and its memory stays live on the cards.
 */
[[nodiscard]] auto abandoned_engines_free_nothing(decode::Paths const& paths) -> bool {
	// Every recorded event answers "not ready" this many times: the launched loops still run at the deadline.
	auto const driver = fake::Fake{fake::Machine{.pending_queries = 64}};
	auto calls = std::size_t{0};
	auto live = std::size_t{0};
	auto abandoned = false;
	{
		auto engine = open(driver, paths);
		if (!engine || !engine->start()) {
			return false;
		}
		auto const ended = engine->try_shutdown(std::chrono::steady_clock::now());
		abandoned = ended && !*ended &&
		            ended->error() == decode::DecodeError{.reason = decode::Abandoned{.ticket = 1, .kernels = decode::Kernels::MayRun}} &&
		            decode::describe(ended->error()).contains("never freed");
		calls = driver.calls();
		live = driver.live(fake::Kind::DeviceMemory);
	}
	return abandoned && live > 0 && driver.calls() == calls && driver.live(fake::Kind::DeviceMemory) == live && driver.violations().empty();
}

/** An image for the wrong memory is refused at open, before anything reaches a card. */
[[nodiscard]] auto wrong_image_is_refused(decode::Paths const& paths) -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto swapped = paths;
	std::swap(swapped.card_zero, swapped.card_one);
	auto const engine = open(driver, swapped);
	auto const unloadable = engine ? std::nullopt : engine::core::alternative<decode::Unloadable>(engine.error().reason);
	auto const refused = unloadable && unloadable->image == decode::ImageOf::CardZero && std::holds_alternative<image::Rejection>(unloadable->error.cause);
	return refused && driver.nothing_live() && driver.violations().empty();
}

/**
 * Every driver call of the bring-up, failed in turn: the open fails with an error and nothing it took is left. A call
 * that tears something down cannot be retried (the owner has no remedy), so where the failed call was one of those the
 * resource it held outlives its context, which the driver reports as just that and nothing else.
 */
[[nodiscard]] auto every_failure_is_clean(decode::Paths const& paths) -> bool {
	auto calls = std::size_t{0};
	{
		auto const driver = fake::Fake{fake::Machine{}};
		auto const engine = open(driver, paths);
		if (!engine) {
			return false;
		}
		calls = driver.calls();
	}
	auto torn_down = std::size_t{0};
	for (auto failing = std::size_t{0}; failing < calls; ++failing) {
		auto const driver = fake::Fake{fake::Machine{.failing_call = failing}};
		auto failed = false;
		{
			auto const engine = open(driver, paths);
			failed = !engine.has_value();
		}
		auto const orphaned = std::ranges::all_of(driver.violations(), [](fake::Violation violation) {
			return violation == fake::Violation::OutlivedContext;
		});
		torn_down += driver.violations().empty() ? 0 : 1;
		if (!driver.nothing_live() || !orphaned || (driver.violations().empty() && !failed)) {
			return false;
		}
	}
	// Teardown calls are a minority of the bring-up's calls; most failures must fail the open cleanly.
	return calls > 0 && torn_down * 4 < calls;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const scratch = Scratch::create();
	auto const paths = scratch ? images(*scratch) : std::nullopt;
	if (!paths) {
		std::println("FAIL: the synthetic images could not be written");
		return 1;
	}
	auto const cases = std::array{
	    Case{.name = "the engine launches each card's loop with its cross-wired parameters and stops", .holds = brings_up_and_stops(*paths)},
	    Case{.name = "a shutdown ends the loops from every state, and keeps a failure", .holds = shuts_down_from_every_state(*paths)},
	    Case{.name = "an abandoned engine dies without a driver call, its memory kept", .holds = abandoned_engines_free_nothing(*paths)},
	    Case{.name = "an image for the wrong memory is refused at open", .holds = wrong_image_is_refused(*paths)},
	    Case{.name = "every driver failure during bring-up fails cleanly", .holds = every_failure_is_clean(*paths)},
	};
	return engine::testing::report(cases);
}

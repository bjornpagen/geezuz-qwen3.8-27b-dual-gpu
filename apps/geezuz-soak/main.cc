// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz-soak: the bring-up's memory acceptance, run on the cards.
//
//   geezuz-soak clocks                          both cards' memory clocks under load, against the accepted clock
//   geezuz-soak integrity <image-dir> <minutes> the acceptance soak: each card's baked weight image hashed where it
//                                               lies, pass after pass, beside a pattern over the memory the engine
//                                               writes, for <minutes>
//
// `clocks` runs the very check the engine runs when it starts, and prints what each card reads even when it refuses,
// so the operator sees the clock a soak would vouch for. `integrity` loads the images geezuz-bake wrote into
// <image-dir> and stops at the first mismatch, printing it with its time and both cards' clocks, power and slowdown
// counters. Exit status 0 when the clocks are accepted or the soak ran clean; 1 on any refusal, mismatch, failure or a
// soak too short to vouch for anything; 2 for a command line that is not one of the above.
import std;
import engine.bake;
import engine.core;
import engine.device;
import engine.image;
import engine.model;
import engine.platform;
import engine.soak;

namespace {

namespace platform = engine::platform;
namespace device = engine::device;
namespace model = engine::model;

/** The subcommands; nothing else selects what the program does. */
enum class Command : std::uint8_t {
	Clocks,
	Integrity,
};

/** The machine's clocks: steady time for deadlines, wall time for the report, and real waits. */
struct MachineClock {
	[[nodiscard]] auto now() const -> platform::Moment {
		return {.wall = std::chrono::system_clock::now(), .steady = std::chrono::steady_clock::now()};
	}

	auto sleep_until(std::chrono::steady_clock::time_point until) const -> void {
		std::this_thread::sleep_until(until);
	}
};

static_assert(platform::Clock<MachineClock>);

/** How long both card images may take to copy to their cards: 14 GB each over pcie takes seconds. */
constexpr auto upload_patience = std::chrono::minutes{10};

[[nodiscard]] auto run_clocks() -> int {
	auto const devices = device::start();
	if (!devices) {
		std::println("geezuz-soak: {}", platform::describe(platform::ClockError{.reason = devices.error()}));
		return 1;
	}
	auto clock = MachineClock{};
	auto const found = platform::loaded_memory_clocks(*devices, clock);
	if (!found) {
		std::println("geezuz-soak: {}", platform::describe(found.error()));
		return 1;
	}
	auto accepted = true;
	for (auto const card : device::card_pair) {
		auto const reading = found->at(std::to_underlying(card));
		std::println("card {}: memory {} MHz in P{} under load", std::to_underlying(card), reading.megahertz,
		             reading.state.transform([](device::PerformanceState state) { return std::to_underlying(state); }).value_or(255));
		if (auto const judged = platform::judge(platform::accepted_memory_clock, card, reading); !judged) {
			std::println("refused: {}", platform::describe(platform::ClockError{.reason = judged.error()}));
			accepted = false;
		}
	}
	if (accepted) {
		std::println("accepted: both cards run at the accepted memory clock");
	}
	return accepted ? 0 : 1;
}

[[nodiscard]] auto minutes_of(std::string_view text) -> std::optional<std::chrono::minutes> {
	auto value = std::int64_t{0};
	auto const read = std::from_chars(text.data(), std::to_address(text.end()), value);
	if (text.empty() || std::make_error_code(read.ec) || read.ptr != std::to_address(text.end()) || value <= 0) {
		return std::nullopt;
	}
	return std::chrono::minutes{value};
}

/** What one pass covers on a card: the operator sizes the soak by it, at roughly a second and a half per slice. */
template<engine::image::Catalog C>
auto print_plan(model::Card card) -> void {
	constexpr auto bytes = engine::image::payload_bytes<C>;
	constexpr auto pass = engine::soak::pass_of(bytes);
	std::println("card {}: weight image {} bytes, {} slices of {} blocks per pass; pattern region {} bytes", std::to_underlying(card), bytes,
	             engine::soak::slices(pass, engine::soak::slice_blocks), engine::soak::slice_blocks, engine::soak::region_bytes);
}

[[nodiscard]] auto run_integrity(std::filesystem::path const& directory, std::chrono::minutes duration) -> int {
	print_plan<platform::ModelCards::CardZero>(model::Card::Zero);
	print_plan<platform::ModelCards::CardOne>(model::Card::One);
	auto devices = device::start();
	if (!devices) {
		std::println("geezuz-soak: {}", platform::describe(platform::SetupError{.reason = devices.error()}));
		return 1;
	}
	auto const paths = platform::ImagePaths{
	    .card_zero = directory / engine::bake::image_name(engine::image::Target::CardZero),
	    .card_one = directory / engine::bake::image_name(engine::image::Target::CardOne),
	};
	auto cards = platform::Cards<platform::ModelCards>::open(std::move(*devices), paths, engine::bake::checkpoint,
	                                                         std::chrono::steady_clock::now() + upload_patience);
	if (!cards) {
		std::println("geezuz-soak: {}", platform::describe(cards.error()));
		return 1;
	}
	std::println("both images are on their cards; soaking for {} minutes", duration.count());
	auto clock = MachineClock{};
	auto const verdict = platform::run(*cards, clock, duration);
	if (!verdict) {
		std::println("geezuz-soak: {}", platform::describe(verdict.error()));
		return 1;
	}
	std::print("{}", platform::render(*verdict));
	return platform::accepts(*verdict) ? 0 : 1;
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	auto words = std::vector<std::string_view>{};
	for (auto const* argument : arguments.subspan(1)) {
		words.emplace_back(argument);
	}
	auto const command = [&] -> std::optional<Command> {
		if (words.size() == 1 && words[0] == "clocks") {
			return Command::Clocks;
		}
		if (words.size() == 3 && words[0] == "integrity" && minutes_of(words[2])) {
			return Command::Integrity;
		}
		return std::nullopt;
	}();
	if (!command) {
		std::println("usage: geezuz-soak clocks | geezuz-soak integrity <image-dir> <minutes>");
		return 2;
	}
	switch (*command) {
	case Command::Clocks:
		return run_clocks();
	case Command::Integrity:
		return run_integrity(std::filesystem::path{words[1]}, minutes_of(words[2]).value_or(std::chrono::minutes{0}));
	}
	std::unreachable();
}

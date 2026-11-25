// SPDX-License-Identifier: AGPL-3.0-only
//
// The soak's loop control against a rig: a test double that is both the bench and the clock, simulating each launch's
// time on a steady clock it owns, serving scripted faults (a wrong digest, a wrong pattern round, readings that drift,
// a launch that never completes, a launch that fails), and recording every launch so the tests can hold the loop to its
// protocol: never a launch on a lane with work in flight, slices in order, a finish only after the last slice, nothing
// launched after a finding but the check of a round already written, and every lane drained when it returns.
import std;
import engine.core;
import engine.device;
import engine.model;
import engine.platform;
import engine.sha256;
import engine.soak;
import engine.testing;

namespace {

namespace platform = engine::platform;
namespace device = engine::device;
namespace model = engine::model;
namespace soak = engine::soak;
namespace sha256 = engine::sha256;

using model::Card;
using platform::Lane;
using platform::Telemetry;
using Steady = std::chrono::steady_clock;
using std::chrono::milliseconds;

/** How long each kind of launch runs on the rig's clock. */
struct Timing {
	milliseconds absorb;
	milliseconds finish;
	milliseconds write;
	milliseconds check;
};

inline constexpr Timing card_like = {.absorb = milliseconds{1500}, .finish = milliseconds{2}, .write = milliseconds{13}, .check = milliseconds{13}};

/** A reading that takes effect on one card from a time into the run. */
struct Change {
	Card card;
	Steady::duration at;
	Telemetry reading;
};

/** What the rig gets wrong on purpose. */
struct Faults {
	std::optional<std::pair<Card, std::uint64_t>> bad_pass;
	std::optional<std::pair<Card, std::uint32_t>> bad_round;
	std::optional<Change> change;
	std::optional<std::pair<Card, Lane>> hang;
	std::optional<std::size_t> failing_launch;
};

inline auto const no_faults = Faults{.bad_pass = std::nullopt, .bad_round = std::nullopt, .change = std::nullopt, .hang = std::nullopt, .failing_launch = std::nullopt};

[[nodiscard]] auto with_bad_pass(Card card, std::uint64_t pass) -> Faults {
	auto faults = no_faults;
	faults.bad_pass = std::pair{card, pass};
	return faults;
}

[[nodiscard]] auto with_bad_round(Card card, std::uint32_t round) -> Faults {
	auto faults = no_faults;
	faults.bad_round = std::pair{card, round};
	return faults;
}

[[nodiscard]] auto with_change(Change change) -> Faults {
	auto faults = no_faults;
	faults.change = std::move(change);
	return faults;
}

[[nodiscard]] auto with_hang(Card card, Lane lane) -> Faults {
	auto faults = no_faults;
	faults.hang = std::pair{card, lane};
	return faults;
}

[[nodiscard]] auto with_failing_launch(std::size_t launch) -> Faults {
	auto faults = no_faults;
	faults.failing_launch = launch;
	return faults;
}

[[nodiscard]] auto counters(std::chrono::nanoseconds power_cap) -> device::ClocksEventCounters {
	return {
	    .sw_power_cap = power_cap,
	    .sync_boost = std::chrono::nanoseconds{0},
	    .sw_thermal_slowdown = std::chrono::nanoseconds{0},
	    .hw_thermal_slowdown = std::chrono::nanoseconds{0},
	    .hw_power_brake_slowdown = std::nullopt,
	};
}

/** An idle card: its lowest performance state's memory clock, which the soak must not hold the card to. */
inline auto const idle_reading = Telemetry{
    .memory = {.megahertz = 405, .state = device::PerformanceState::P8},
    .sm = {.megahertz = 210},
    .power = {.draw_milliwatts = 30'000, .limit_milliwatts = 400'000},
    .temperature = {.celsius = 31},
    .reasons = {.bits = 0x1},
    .counters = counters(std::chrono::nanoseconds{1'000'000}),
};

/** A working card, from one second into the run. */
inline auto const working_reading = Telemetry{
    .memory = {.megahertz = 10'001, .state = device::PerformanceState::P2},
    .sm = {.megahertz = 1'875},
    .power = {.draw_milliwatts = 351'000, .limit_milliwatts = 400'000},
    .temperature = {.celsius = 58},
    .reasons = {.bits = 0},
    .counters = counters(std::chrono::nanoseconds{1'000'000}),
};

inline constexpr auto warm_up = std::chrono::seconds{1};

[[nodiscard]] auto other_digest(sha256::Digest digest) -> sha256::Digest {
	digest.bytes.at(7) ^= std::byte{0x40};
	return digest;
}

/** What the rig saw of one card. */
struct CardRig {
	std::optional<Steady::time_point> image_until;
	std::optional<Steady::time_point> pattern_until;
	std::uint64_t pass = 0;
	std::uint64_t absorbed = 0;
	std::optional<std::uint32_t> written;
	std::uint32_t checked = 0;
	std::uint64_t good_finishes = 0;
	std::uint32_t good_checks = 0;
	std::array<soak::Tally, soak::tallies> tallies = {};
};

/** One launch as the rig saw it. */
struct Launched {
	Card card;
	Lane lane;
	Steady::time_point at;
};

class Rig {
public:
	Rig(soak::Pass pass, Timing timing, Faults faults) : pass_{pass}, timing_{timing}, faults_{std::move(faults)} {}

	// The clock.

	[[nodiscard]] auto now() const -> platform::Moment {
		return {.wall = std::chrono::system_clock::time_point{std::chrono::duration_cast<std::chrono::system_clock::duration>(now_ - Steady::time_point{})}, .steady = now_};
	}

	auto sleep_until(Steady::time_point until) -> void {
		now_ = std::max(now_, until);
	}

	// The bench.

	auto begin_pass(Card card) -> void {
		auto& rig = of(card);
		if (busy(rig.image_until)) {
			++violations_;
		}
		++rig.pass;
		rig.absorbed = 0;
	}

	[[nodiscard]] auto absorb(Card card, soak::Slice const& slice) -> engine::core::Result<void, device::DeviceError> {
		auto& rig = of(card);
		if (slice.first != rig.absorbed || slice.count == 0 || slice.first + slice.count > pass_.blocks) {
			++violations_;
		}
		rig.absorbed += slice.count;
		return launch(card, Lane::Image, rig.image_until, timing_.absorb);
	}

	[[nodiscard]] auto finish(Card card) -> engine::core::Result<void, device::DeviceError> {
		auto& rig = of(card);
		if (rig.absorbed != pass_.blocks) {
			++violations_;
		}
		return launch(card, Lane::Image, rig.image_until, timing_.finish);
	}

	[[nodiscard]] auto write(Card card, std::uint32_t round) -> engine::core::Result<void, device::DeviceError> {
		auto& rig = of(card);
		if (rig.written && round != *rig.written + 1) {
			++violations_;
		}
		rig.written = round;
		return launch(card, Lane::Pattern, rig.pattern_until, timing_.write);
	}

	[[nodiscard]] auto check(Card card, std::uint32_t round) -> engine::core::Result<void, device::DeviceError> {
		auto& rig = of(card);
		if (rig.written != round) {
			++violations_;
		}
		rig.checked = round;
		auto const bad = faults_.bad_round == std::pair{card, round};
		for (auto index = std::size_t{0}; index < rig.tallies.size(); ++index) {
			// A bad round's tallies: some warps report words wrong at chunks whose least is 4242.
			auto const wrong = bad && index % 97 == 3;
			rig.tallies.at(index) = {.mismatched_words = wrong ? 2U : 0U, .first_chunk = wrong ? 4242U + static_cast<std::uint32_t>(index) : 0U};
		}
		return launch(card, Lane::Pattern, rig.pattern_until, timing_.check);
	}

	[[nodiscard]] auto state(Card card, Lane lane) const -> engine::core::Result<device::EventState, device::DeviceError> {
		auto const& rig = of(card);
		auto const& until = lane == Lane::Image ? rig.image_until : rig.pattern_until;
		return busy(until) ? device::EventState::Pending : device::EventState::Complete;
	}

	[[nodiscard]] auto pass(Card /*card*/) const -> soak::Pass {
		return pass_;
	}

	[[nodiscard]] auto expected(Card card) const -> sha256::Digest {
		return sha256::digest(std::format("the image of card {}", std::to_underlying(card)));
	}

	[[nodiscard]] auto digest(Card card) const -> sha256::Digest {
		auto const& rig = of(card);
		if (faults_.bad_pass == std::pair{card, rig.pass}) {
			served();
			return other_digest(expected(card));
		}
		return expected(card);
	}

	[[nodiscard]] auto tallies(Card card) const -> std::span<soak::Tally const> {
		auto const& rig = of(card);
		if (faults_.bad_round == std::pair{card, rig.checked}) {
			served();
		}
		return rig.tallies;
	}

	[[nodiscard]] auto telemetry(Card card) const -> engine::core::Result<Telemetry, device::DeviceError> {
		auto const into = now_ - start_;
		if (faults_.change && faults_.change->card == card && into >= faults_.change->at) {
			served();
			return faults_.change->reading;
		}
		return into < warm_up ? idle_reading : working_reading;
	}

	// What the rig saw.

	/** Counts what the soak read back as good: called by the tests after the run, from the launch log's completions. */
	[[nodiscard]] auto violations() const -> std::size_t {
		return violations_;
	}

	[[nodiscard]] auto launches() const -> std::span<Launched const> {
		return launches_;
	}

	/** Whether every lane's last launch has completed by now. */
	[[nodiscard]] auto drained() const -> bool {
		return std::ranges::none_of(cards_, [&](CardRig const& rig) { return busy(rig.image_until) || busy(rig.pattern_until); });
	}

	[[nodiscard]] auto elapsed() const -> Steady::duration {
		return now_ - start_;
	}

	/** When the rig first handed the soak something wrong, if it has. */
	[[nodiscard]] auto first_served() const -> std::optional<Steady::time_point> {
		return served_;
	}

private:
	// The rig serves its faults through the bench's reads, which the soak makes on a const view.
	auto served() const -> void {
		served_ = served_.value_or(now_);
	}

	[[nodiscard]] auto of(Card card) -> CardRig& {
		return cards_.at(std::to_underlying(card));
	}

	[[nodiscard]] auto of(Card card) const -> CardRig const& {
		return cards_.at(std::to_underlying(card));
	}

	[[nodiscard]] auto busy(std::optional<Steady::time_point> const& until) const -> bool {
		return until.has_value() && *until > now_;
	}

	[[nodiscard]] auto launch(Card card, Lane lane, std::optional<Steady::time_point>& until, milliseconds takes)
	    -> engine::core::Result<void, device::DeviceError> {
		if (busy(until)) {
			++violations_;
		}
		auto const number = launches_.size();
		launches_.push_back({.card = card, .lane = lane, .at = now_});
		if (faults_.failing_launch == number) {
			return std::unexpected(device::DeviceError{.reason = device::Failure{.symbol = "cuLaunchKernel", .status = device::Status::Unknown}});
		}
		auto const hangs = faults_.hang == std::pair{card, lane};
		until = hangs ? Steady::time_point::max() : now_ + takes;
		return {};
	}

	soak::Pass pass_;
	Timing timing_;
	Faults faults_;
	Steady::time_point start_ = Steady::time_point{} + std::chrono::hours{1};
	Steady::time_point now_ = start_;
	std::array<CardRig, model::cards> cards_ = {};
	std::vector<Launched> launches_;
	std::size_t violations_ = 0;
	mutable std::optional<Steady::time_point> served_;
};

static_assert(platform::Bench<Rig> && platform::Clock<Rig>);

/** A pass of three slices and a tail: 4.5 s of absorbing per pass on the rig. */
inline constexpr auto three_slices = soak::pass_of((std::uint64_t{2} * soak::slice_blocks + 5) * soak::block_bytes + 28);

using Outcome = engine::core::Result<platform::Verdict, platform::SoakError>;

[[nodiscard]] auto soak_on(Rig& rig, std::chrono::minutes duration) -> Outcome {
	return platform::run(rig, rig, duration);
}

template<class T>
[[nodiscard]] auto verdict_as(Outcome const& outcome) -> T const* {
	return outcome ? std::get_if<T>(&*outcome) : nullptr;
}

template<class T>
[[nodiscard]] auto finding_as(Outcome const& outcome) -> T const* {
	auto const* const found = verdict_as<platform::Found>(outcome);
	return found == nullptr ? nullptr : std::get_if<T>(&found->sighting.finding);
}

/** After a finding at `at`, nothing is launched but the check of a round already written. */
[[nodiscard]] auto nothing_new_after(Rig const& rig, Steady::time_point at) -> bool {
	auto pattern_launches = std::array<int, model::cards>{};
	for (auto const& launch : rig.launches()) {
		if (launch.at <= at) {
			continue;
		}
		if (launch.lane == Lane::Image) {
			return false;
		}
		++pattern_launches.at(std::to_underlying(launch.card));
	}
	return std::ranges::all_of(pattern_launches, [](int count) { return count <= 1; });
}

[[nodiscard]] auto clean_soak_runs_its_time() -> bool {
	auto rig = Rig{three_slices, card_like, no_faults};
	auto const outcome = soak_on(rig, std::chrono::minutes{10});
	auto const* const clean = verdict_as<platform::Clean>(outcome);
	if (clean == nullptr) {
		return false;
	}
	// 4.5 s per pass and 26 ms per round, for ten minutes, less the last unfinished pass or round.
	auto const passes_expected = std::uint64_t{600'000 / 4'502};
	return rig.violations() == 0 && rig.drained() && rig.elapsed() >= std::chrono::minutes{10} && rig.elapsed() < std::chrono::minutes{10} + std::chrono::seconds{2} &&
	       std::ranges::all_of(clean->cards,
	                           [&](platform::Progress const& progress) {
		                           return progress.passes + 1 >= passes_expected && progress.passes <= passes_expected &&
		                                  progress.rounds + 1 >= 600'000U / 26U && progress.rounds <= 600'000U / 26U + 1;
	                           }) &&
	       std::ranges::all_of(clean->clocks, [](device::MemoryClock const& clock) { return clock == working_reading.memory; });
}

[[nodiscard]] auto wrong_digest_is_found_and_drained() -> bool {
	auto rig = Rig{three_slices, card_like, with_bad_pass(Card::One, 3)};
	auto const outcome = soak_on(rig, std::chrono::minutes{30});
	auto const* const mismatch = finding_as<platform::ImageMismatch>(outcome);
	auto const* const found = verdict_as<platform::Found>(outcome);
	return mismatch != nullptr && mismatch->card == Card::One && mismatch->pass == 3 && mismatch->expected == rig.expected(Card::One) &&
	       mismatch->found == other_digest(rig.expected(Card::One)) && found->cards.at(1).passes == 2 &&
	       found->sighting.at.steady - found->started.steady >= std::chrono::milliseconds{3 * 4'502} &&
	       found->sighting.at.steady - found->started.steady < std::chrono::milliseconds{3 * 4'502 + 10} &&
	       std::ranges::all_of(found->sighting.cards, [](auto const& reading) { return reading == working_reading; }) &&
	       nothing_new_after(rig, found->sighting.at.steady) && rig.drained() && rig.violations() == 0;
}

[[nodiscard]] auto wrong_round_is_found() -> bool {
	auto rig = Rig{three_slices, card_like, with_bad_round(Card::Zero, 40)};
	auto const outcome = soak_on(rig, std::chrono::minutes{30});
	auto const* const mismatch = finding_as<platform::PatternMismatch>(outcome);
	auto const* const found = verdict_as<platform::Found>(outcome);
	auto const wrong_warps = (soak::tallies + 93) / 97;
	return mismatch != nullptr && mismatch->card == Card::Zero && mismatch->round == 40 && mismatch->words == std::uint64_t{2} * wrong_warps &&
	       mismatch->first_chunk == 4242 + 3 && found->cards.at(0).rounds == 40 && nothing_new_after(rig, found->sighting.at.steady) && rig.drained() &&
	       rig.violations() == 0;
}

/** What a soak finds when card one's readings change to `changed` at `at`, provided it keeps its protocol doing so. */
[[nodiscard]] auto drift_found(Telemetry const& changed, std::chrono::seconds at) -> std::optional<platform::Finding> {
	auto rig = Rig{three_slices, card_like, with_change(Change{.card = Card::One, .at = at, .reading = changed})};
	auto const outcome = soak_on(rig, std::chrono::minutes{30});
	auto const* const found = verdict_as<platform::Found>(outcome);
	if (found == nullptr || rig.violations() != 0 || !rig.drained() || !nothing_new_after(rig, found->sighting.at.steady) ||
	    found->sighting.at.steady - found->started.steady < at) {
		return std::nullopt;
	}
	return found->sighting.finding;
}

[[nodiscard]] auto moved_clock_is_found() -> bool {
	auto changed = working_reading;
	changed.memory = {.megahertz = 9'501, .state = device::PerformanceState::P2};
	auto const finding = drift_found(changed, std::chrono::seconds{100});
	auto const* const moved = finding ? std::get_if<platform::ClockMoved>(&*finding) : nullptr;
	return moved != nullptr && moved->card == Card::One && moved->started == working_reading.memory && moved->now == changed.memory;
}

[[nodiscard]] auto grown_counter_is_found() -> bool {
	auto changed = working_reading;
	changed.counters.hw_thermal_slowdown = std::chrono::nanoseconds{250'000};
	auto const finding = drift_found(changed, std::chrono::seconds{100});
	auto const* const throttled = finding ? std::get_if<platform::Throttled>(&*finding) : nullptr;
	return throttled != nullptr && throttled->card == Card::One && throttled->counter == platform::Counter::HwThermalSlowdown &&
	       throttled->grew == std::chrono::nanoseconds{250'000};
}

[[nodiscard]] auto vanished_counter_is_found() -> bool {
	auto changed = working_reading;
	changed.counters.sw_power_cap = std::nullopt;
	auto const finding = drift_found(changed, std::chrono::seconds{100});
	auto const* const throttled = finding ? std::get_if<platform::Throttled>(&*finding) : nullptr;
	return throttled != nullptr && throttled->counter == platform::Counter::SwPowerCap && !throttled->grew.has_value();
}

[[nodiscard]] auto hung_lane_is_an_error() -> bool {
	auto rig = Rig{three_slices, card_like, with_hang(Card::Zero, Lane::Pattern)};
	auto const outcome = soak_on(rig, std::chrono::minutes{30});
	return !outcome && outcome.error() == platform::SoakError{.reason = platform::Hung{.card = Card::Zero, .lane = Lane::Pattern}} &&
	       rig.elapsed() > std::chrono::seconds{10} && rig.elapsed() < std::chrono::seconds{11};
}

[[nodiscard]] auto every_failing_launch_is_an_error() -> bool {
	auto holds = true;
	for (auto failing = std::size_t{0}; failing < 40; ++failing) {
		auto rig = Rig{three_slices, card_like, with_failing_launch(failing)};
		auto const outcome = soak_on(rig, std::chrono::minutes{1});
		holds = holds && !outcome && std::holds_alternative<device::DeviceError>(outcome.error().reason) && rig.launches().size() == failing + 1;
	}
	return holds;
}

[[nodiscard]] auto short_soak_vouches_for_nothing() -> bool {
	// A pass of 200 slices takes five minutes on the rig: a two-minute soak finishes none.
	auto rig = Rig{soak::pass_of(std::uint64_t{200} * soak::slice_blocks * soak::block_bytes), card_like, no_faults};
	auto const outcome = soak_on(rig, std::chrono::minutes{2});
	auto const* const too_short = verdict_as<platform::TooShort>(outcome);
	return too_short != nullptr && too_short->cards.at(0).passes == 0 && too_short->cards.at(0).rounds > 0 && rig.drained() && rig.violations() == 0;
}

[[nodiscard]] auto unkept_counter_is_refused() -> bool {
	auto changed = working_reading;
	changed.counters.sw_thermal_slowdown = std::nullopt;
	auto rig = Rig{three_slices, card_like, with_change(Change{.card = Card::Zero, .at = std::chrono::seconds{0}, .reading = changed})};
	auto const outcome = soak_on(rig, std::chrono::minutes{5});
	return !outcome && outcome.error() == platform::SoakError{.reason = platform::Unwatchable{.card = Card::Zero, .counter = platform::Counter::SwThermalSlowdown}};
}

/** A tiny image, under one block, goes straight to its finish every pass. */
[[nodiscard]] auto image_under_a_block_still_passes() -> bool {
	auto rig = Rig{soak::pass_of(40), card_like, no_faults};
	auto const outcome = soak_on(rig, std::chrono::minutes{1});
	auto const* const clean = verdict_as<platform::Clean>(outcome);
	return clean != nullptr && clean->cards.at(0).passes > 1000 && rig.violations() == 0 && rig.drained();
}

/**
 * Random scripts: random image sizes, timings, durations and faults. Whatever happens, the loop keeps its protocol,
 * drains, and its verdict is the first fault the rig served, or clean when it served none in time.
 */
[[nodiscard]] auto random_scripts_keep_the_protocol(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	auto found_trials = 0;
	auto clean_trials = 0;
	for (auto trial = 0; trial < 40 && holds; ++trial) {
		auto const pass = soak::pass_of(std::uniform_int_distribution<std::uint64_t>{1, std::uint64_t{4} * soak::slice_blocks * soak::block_bytes / 4}(random) * 4);
		auto const timing = Timing{
		    .absorb = milliseconds{std::uniform_int_distribution<int>{5, 2000}(random)},
		    .finish = milliseconds{std::uniform_int_distribution<int>{1, 20}(random)},
		    .write = milliseconds{std::uniform_int_distribution<int>{1, 50}(random)},
		    .check = milliseconds{std::uniform_int_distribution<int>{1, 50}(random)},
		};
		auto const card = std::uniform_int_distribution<int>{0, 1}(random) == 0 ? Card::Zero : Card::One;
		auto faults = no_faults;
		switch (std::uniform_int_distribution<int>{0, 3}(random)) {
		case 0:
			break;
		case 1:
			faults.bad_pass = std::pair{card, std::uniform_int_distribution<std::uint64_t>{1, 30}(random)};
			break;
		case 2:
			faults.bad_round = std::pair{card, std::uniform_int_distribution<std::uint32_t>{0, 3000}(random)};
			break;
		default:
			faults.change = Change{
			    .card = card,
			    .at = std::chrono::seconds{std::uniform_int_distribution<int>{10, 600}(random)},
			    .reading = Telemetry{.memory = {.megahertz = 9'751, .state = device::PerformanceState::P0},
			                         .sm = working_reading.sm,
			                         .power = working_reading.power,
			                         .temperature = working_reading.temperature,
			                         .reasons = working_reading.reasons,
			                         .counters = working_reading.counters},
			};
			break;
		}
		auto rig = Rig{pass, timing, faults};
		auto const outcome = soak_on(rig, std::chrono::minutes{std::uniform_int_distribution<int>{1, 8}(random)});
		auto const* const found = verdict_as<platform::Found>(outcome);
		holds = outcome.has_value() && rig.violations() == 0 && rig.drained() && rig.first_served().has_value() == (found != nullptr);
		found_trials += found != nullptr ? 1 : 0;
		clean_trials += verdict_as<platform::Clean>(outcome) != nullptr ? 1 : 0;
		if (found != nullptr) {
			holds = holds && found->sighting.at.steady == rig.first_served();
			holds = holds && nothing_new_after(rig, found->sighting.at.steady) && faults.bad_pass.has_value() == std::holds_alternative<platform::ImageMismatch>(found->sighting.finding) &&
			        faults.bad_round.has_value() == std::holds_alternative<platform::PatternMismatch>(found->sighting.finding) &&
			        faults.change.has_value() == std::holds_alternative<platform::ClockMoved>(found->sighting.finding);
		}
	}
	// Neither outcome may be vacuous: the trials must both find faults and run clean.
	return holds && found_trials >= 10 && clean_trials >= 5;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "a clean ten-minute soak runs back-to-back passes and rounds on both cards for its time, holding the working clock",
	        .holds = clean_soak_runs_its_time(),
	    },
	    engine::testing::Case{
	        .name = "a wrong digest in card one's third pass is found when that pass finishes, with both cards' readings, and drains",
	        .holds = wrong_digest_is_found_and_drained(),
	    },
	    engine::testing::Case{
	        .name = "a wrong pattern round on card zero is found with the summed words and the lowest chunk",
	        .holds = wrong_round_is_found(),
	    },
	    engine::testing::Case{
	        .name = "a memory clock that moves after the cards settled is found",
	        .holds = moved_clock_is_found(),
	    },
	    engine::testing::Case{
	        .name = "a thermal slowdown counter that grows is found, with how much it grew",
	        .holds = grown_counter_is_found(),
	    },
	    engine::testing::Case{
	        .name = "a watched counter NVML stops keeping is found",
	        .holds = vanished_counter_is_found(),
	    },
	    engine::testing::Case{
	        .name = "a launch that never completes is a hung card once its lane's patience runs out",
	        .holds = hung_lane_is_an_error(),
	    },
	    engine::testing::Case{
	        .name = "whichever of the first 40 launches fails, the soak stops with that failure",
	        .holds = every_failing_launch_is_an_error(),
	    },
	    engine::testing::Case{
	        .name = "a soak shorter than one image pass vouches for nothing",
	        .holds = short_soak_vouches_for_nothing(),
	    },
	    engine::testing::Case{
	        .name = "a card whose thermal counter NVML does not keep is refused once it settles",
	        .holds = unkept_counter_is_refused(),
	    },
	    engine::testing::Case{
	        .name = "an image under one block is finished straight away, pass after pass",
	        .holds = image_under_a_block_still_passes(),
	    },
	    engine::testing::Case{
	        .name = "random images, timings, durations and faults: the protocol holds, the lanes drain, the finding is the fault served (seed 77)",
	        .holds = random_scripts_keep_the_protocol(77),
	    },
	};
	return engine::testing::report(cases);
}

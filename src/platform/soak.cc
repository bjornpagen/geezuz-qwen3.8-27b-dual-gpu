// SPDX-License-Identifier: AGPL-3.0-only
//
// The acceptance soak's loop. Per card two lanes run side by side: the image lane hashes the card's whole weight image
// where it lies, slice by slice, and holds each pass's digest against the baked one; the pattern lane writes a pattern
// over the memory the engine writes, reads it back and counts what came back wrong. Every completed hash step samples
// both clocks and the slowdown counters, which must not move. The first finding stops every lane from launching more;
// in-flight work drains, and the soak reports the finding with its time and both cards' readings. Generic over the
// bench and the clock, so its control is checked against test doubles and runs unchanged on the cards.
export module engine.platform:soak;

import std;
import engine.core;
import engine.device;
import engine.model;
import engine.sha256;
import engine.soak;
import :telemetry;

namespace engine::platform {

/** A card's image, hashed where it lies, does not give the digest the baker stamped in it. */
export struct ImageMismatch {
	model::Card card;
	std::uint64_t pass;
	sha256::Digest expected;
	sha256::Digest found;

	[[nodiscard]] constexpr auto operator==(ImageMismatch const&) const -> bool = default;
};

/** Words of a card's pattern region read back other than written: how many, and the lowest chunk holding one. */
export struct PatternMismatch {
	model::Card card;
	std::uint32_t round;
	std::uint64_t words;
	std::uint32_t first_chunk;

	[[nodiscard]] constexpr auto operator==(PatternMismatch const&) const -> bool = default;
};

/** What the soak can find wrong. */
export using Finding = std::variant<ImageMismatch, PatternMismatch, ClockMoved, Throttled>;

/** A moment of the soak: wall time, to match against other logs (the junction temperature's), and steady time. */
export struct Moment {
	std::chrono::system_clock::time_point wall;
	std::chrono::steady_clock::time_point steady;

	[[nodiscard]] constexpr auto operator==(Moment const&) const -> bool = default;
};

using Reading = core::Result<Telemetry, device::DeviceError>;

/** A finding as the soak reports it: what, when, and both cards' readings taken right after. */
export struct Sighting {
	Finding finding;
	Moment at;
	std::array<Reading, model::cards> cards;
};

/** What a card got through: whole image passes that matched, and pattern rounds that read back clean. */
export struct Progress {
	std::uint64_t passes;
	std::uint32_t rounds;

	[[nodiscard]] constexpr auto operator==(Progress const&) const -> bool = default;
};

/**
 * The soak ran its time and found nothing, with at least one whole image pass on each card, every reading held to the
 * memory clock in `clocks`: the clock this soak vouches for.
 */
export struct Clean {
	Moment started;
	Moment ended;
	std::array<Progress, model::cards> cards;
	std::array<device::MemoryClock, model::cards> clocks;
};

/** The soak found something: the first finding, and what each card got through before it. */
export struct Found {
	Moment started;
	Sighting sighting;
	std::array<Progress, model::cards> cards;
};

/** The soak found nothing but ended before a card finished one whole image pass or settled: it vouches for nothing. */
export struct TooShort {
	Moment started;
	Moment ended;
	std::array<Progress, model::cards> cards;
};

export using Verdict = std::variant<Clean, Found, TooShort>;

/** The two kinds of work the soak keeps in flight on each card. */
export enum class Lane : std::uint8_t {
	Image,
	Pattern,
};

/** A launch did not complete within its lane's patience: the card is hung. */
export struct Hung {
	model::Card card;
	Lane lane;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Hung const&) const -> bool = default;
};

/** Why the soak could not run to a verdict. */
export struct [[nodiscard]] SoakError {
	std::variant<device::DeviceError, Hung, Unwatchable> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(SoakError const&) const -> bool = default;
};

/**
 * What the soak runs on: per card, the image digest's launches over the card's weight image and its state cell, the
 * pattern's launches over the card's region, each lane's completion, the readings, and what the lanes left behind. A
 * launch on a lane records that lane's completion; the soak never launches on a lane whose last launch is in flight.
 */
export template<class B>
concept Bench = requires(B& bench, B const& view, model::Card card, soak::Slice slice, std::uint32_t round, Lane lane) {
	{ bench.begin_pass(card) } -> std::same_as<void>;
	{ bench.absorb(card, slice) } -> std::same_as<core::Result<void, device::DeviceError>>;
	{ bench.finish(card) } -> std::same_as<core::Result<void, device::DeviceError>>;
	{ bench.write(card, round) } -> std::same_as<core::Result<void, device::DeviceError>>;
	{ bench.check(card, round) } -> std::same_as<core::Result<void, device::DeviceError>>;
	{ view.state(card, lane) } -> std::same_as<core::Result<device::EventState, device::DeviceError>>;
	{ view.pass(card) } -> std::same_as<soak::Pass>;
	{ view.expected(card) } -> std::same_as<sha256::Digest>;
	{ view.digest(card) } -> std::same_as<sha256::Digest>;
	{ view.tallies(card) } -> std::same_as<std::span<soak::Tally const>>;
	{ view.telemetry(card) } -> std::same_as<Reading>;
};

/** The soak's time: the moment now, and a wait until a steady deadline. */
export template<class C>
concept Clock = requires(C& clock, std::chrono::steady_clock::time_point until) {
	{ clock.now() } -> std::same_as<Moment>;
	{ clock.sleep_until(until) } -> std::same_as<void>;
};

/** A pattern check's tallies summed: how many words came back wrong, and the lowest chunk holding one. */
export struct PatternTally {
	std::uint64_t words;
	std::uint32_t first_chunk;

	[[nodiscard]] constexpr auto operator==(PatternTally const&) const -> bool = default;
};

/** The tallies' sum, or nothing when every word read back as written. */
export [[nodiscard]] constexpr auto summarize(std::span<soak::Tally const> tallies) -> std::optional<PatternTally> {
	auto total = PatternTally{.words = 0, .first_chunk = 0};
	for (auto const& tally : tallies) {
		if (tally.mismatched_words == 0) {
			continue;
		}
		total.first_chunk = total.words == 0 ? tally.first_chunk : std::min(total.first_chunk, tally.first_chunk);
		total.words += tally.mismatched_words;
	}
	return total.words == 0 ? std::nullopt : std::optional{total};
}

/**
 * How long one launch may run before its card counts as hung: a 64 MiB slice of one SHA-256 chain takes about a second
 * and a half, a pattern write or check over the region tens of milliseconds; each patience is tens of times that.
 */
inline constexpr std::chrono::seconds image_patience{60};
inline constexpr std::chrono::seconds pattern_patience{10};

/** How often the soak looks at its lanes: far below any launch's time, far above an event query's cost. */
inline constexpr std::chrono::milliseconds poll_interval{1};

/**
 * How long the cards run before their readings are taken as what the soak holds them to: an idle card reads its lowest
 * performance state's clocks until the driver raises it for the work that arrived. Two seconds is a margin chosen,
 * not measured on these cards; the readings the soak prints show whether it holds.
 */
export inline constexpr std::chrono::seconds settle{2};

struct Absorbing {
	std::uint64_t slice;
};

struct Finishing {};

struct Writing {
	std::uint32_t round;
};

struct Checking {
	std::uint32_t round;
};

/** A lane with nothing in flight that launches nothing more. */
struct Idle {};

using ImagePhase = std::variant<Absorbing, Finishing, Idle>;
using PatternPhase = std::variant<Writing, Checking, Idle>;

/** A card still coming up to its working clocks, until `until`; nothing is held against its readings yet. */
struct Settling {
	std::chrono::steady_clock::time_point until;
};

/** A card whose readings are held to `baseline`. */
struct Watching {
	Baseline baseline;
};

/** One card's lanes: what each has in flight and by when it must complete, what the card got through, its watch. */
struct CardLoop {
	model::Card card;
	std::variant<Settling, Watching> watch;
	ImagePhase image;
	std::chrono::steady_clock::time_point image_due;
	PatternPhase pattern;
	std::chrono::steady_clock::time_point pattern_due;
	Progress progress;
};

template<class T>
[[nodiscard]] auto fail(T reason) -> std::unexpected<SoakError> {
	return std::unexpected(SoakError{.reason = std::move(reason)});
}

/** The soak's own state: when it ends, and the first finding, which stops every lane from launching more. */
struct Course {
	std::chrono::steady_clock::time_point end;
	std::optional<Sighting> first;

	[[nodiscard]] auto stopping(Moment const& now) const -> bool {
		return first.has_value() || now.steady >= end;
	}
};

/** Keeps the first finding only: what follows it may be its consequence. */
template<Bench B>
auto sighted(B const& bench, Course& course, Finding finding, Moment const& at) -> void {
	if (course.first) {
		return;
	}
	course.first = Sighting{.finding = std::move(finding), .at = at, .cards = {bench.telemetry(model::Card::Zero), bench.telemetry(model::Card::One)}};
}

/** Moves a lane to `next` with its patience from `now`, once its launch went in. */
template<class Phase>
[[nodiscard]] auto launched(core::Result<void, device::DeviceError> launch, Phase& phase, Phase next, std::chrono::steady_clock::time_point& due,
                            std::chrono::steady_clock::time_point by) -> core::Result<void, SoakError> {
	if (!launch) {
		return fail(launch.error());
	}
	phase = std::move(next);
	due = by;
	return {};
}

/** Starts an image pass: from the initial state, the first slice, or straight to the finish for an image under a block. */
template<Bench B>
[[nodiscard]] auto begin(B& bench, CardLoop& loop, std::chrono::steady_clock::time_point now) -> core::Result<void, SoakError> {
	bench.begin_pass(loop.card);
	auto const pass = bench.pass(loop.card);
	if (soak::slices(pass, soak::slice_blocks) == 0) {
		return launched(bench.finish(loop.card), loop.image, ImagePhase{Finishing{}}, loop.image_due, now + image_patience);
	}
	return launched(bench.absorb(loop.card, soak::slice(pass, soak::slice_blocks, 0)), loop.image, ImagePhase{Absorbing{.slice = 0}}, loop.image_due,
	                now + image_patience);
}

template<Bench B>
[[nodiscard]] auto write(B& bench, CardLoop& loop, std::uint32_t round, std::chrono::steady_clock::time_point now) -> core::Result<void, SoakError> {
	return launched(bench.write(loop.card, round), loop.pattern, PatternPhase{Writing{.round = round}}, loop.pattern_due, now + pattern_patience);
}

/** Whether `lane`'s launch has completed; one past its patience is a hung card. */
template<Bench B>
[[nodiscard]] auto completed(B const& bench, model::Card card, Lane lane, std::chrono::steady_clock::time_point due, Moment const& now)
    -> core::Result<device::EventState, SoakError> {
	auto const state = bench.state(card, lane);
	if (!state) {
		return fail(state.error());
	}
	if (*state == device::EventState::Pending && now.steady > due) {
		return fail(Hung{.card = card, .lane = lane});
	}
	return *state;
}

/**
 * Reads the card: once it has settled, the readings become its baseline (refused when a counter acceptance needs is not
 * kept); from then on any drift from the baseline is a finding.
 */
template<Bench B>
[[nodiscard]] auto watch(B const& bench, Course& course, CardLoop& loop, Moment const& now) -> core::Result<void, SoakError> {
	auto const* const settling = std::get_if<Settling>(&loop.watch);
	if (settling != nullptr && now.steady < settling->until) {
		return {};
	}
	auto const reading = bench.telemetry(loop.card);
	if (!reading) {
		return fail(reading.error());
	}
	if (settling != nullptr) {
		auto const held = baseline(loop.card, *reading);
		if (!held) {
			return fail(held.error());
		}
		loop.watch = Watching{.baseline = *held};
		return {};
	}
	if (auto const drifted = drift(loop.card, std::get<Watching>(loop.watch).baseline, *reading)) {
		sighted(bench, course, std::visit([](auto const& kind) { return Finding{kind}; }, *drifted), now);
	}
	return {};
}

template<Bench B>
[[nodiscard]] auto step_image(B& bench, Course& course, CardLoop& loop, Moment const& now) -> core::Result<void, SoakError> {
	if (std::holds_alternative<Idle>(loop.image)) {
		return {};
	}
	auto const state = completed(bench, loop.card, Lane::Image, loop.image_due, now);
	if (!state || *state == device::EventState::Pending) {
		return state.transform([](device::EventState) {});
	}
	if (auto watched = watch(bench, course, loop, now); !watched) {
		return std::unexpected(watched.error());
	}
	if (auto const* const absorbing = std::get_if<Absorbing>(&loop.image)) {
		auto const pass = bench.pass(loop.card);
		auto const next = absorbing->slice + 1;
		if (course.stopping(now)) {
			loop.image = Idle{};
			return {};
		}
		if (next < soak::slices(pass, soak::slice_blocks)) {
			return launched(bench.absorb(loop.card, soak::slice(pass, soak::slice_blocks, next)), loop.image, ImagePhase{Absorbing{.slice = next}},
			                loop.image_due, now.steady + image_patience);
		}
		return launched(bench.finish(loop.card), loop.image, ImagePhase{Finishing{}}, loop.image_due, now.steady + image_patience);
	}
	auto const found = bench.digest(loop.card);
	auto const expected = bench.expected(loop.card);
	if (found != expected) {
		sighted(bench, course, ImageMismatch{.card = loop.card, .pass = loop.progress.passes + 1, .expected = expected, .found = found}, now);
	} else if (!course.first) {
		++loop.progress.passes;
	}
	if (course.stopping(now)) {
		loop.image = Idle{};
		return {};
	}
	return begin(bench, loop, now.steady);
}

template<Bench B>
[[nodiscard]] auto step_pattern(B& bench, Course& course, CardLoop& loop, Moment const& now) -> core::Result<void, SoakError> {
	if (std::holds_alternative<Idle>(loop.pattern)) {
		return {};
	}
	auto const state = completed(bench, loop.card, Lane::Pattern, loop.pattern_due, now);
	if (!state || *state == device::EventState::Pending) {
		return state.transform([](device::EventState) {});
	}
	if (auto const* const writing = std::get_if<Writing>(&loop.pattern)) {
		// A round written is a round checked, even once stopping: its check takes milliseconds.
		auto const round = writing->round;
		return launched(bench.check(loop.card, round), loop.pattern, PatternPhase{Checking{.round = round}}, loop.pattern_due,
		                now.steady + pattern_patience);
	}
	auto const round = std::get<Checking>(loop.pattern).round;
	if (auto const wrong = summarize(bench.tallies(loop.card))) {
		sighted(bench, course, PatternMismatch{.card = loop.card, .round = round, .words = wrong->words, .first_chunk = wrong->first_chunk}, now);
	} else if (!course.first) {
		++loop.progress.rounds;
	}
	if (course.stopping(now)) {
		loop.pattern = Idle{};
		return {};
	}
	return write(bench, loop, round + 1, now.steady);
}

[[nodiscard]] auto idle(CardLoop const& loop) -> bool {
	return std::holds_alternative<Idle>(loop.image) && std::holds_alternative<Idle>(loop.pattern);
}

/**
 * Soaks both cards for `duration`: whole image passes and pattern rounds back to back on each, every image step's
 * readings held to the start's. Returns at the first finding once in-flight work has drained, or when the time is up.
 */
export template<Bench B, Clock C>
[[nodiscard]] auto run(B& bench, C& clock, std::chrono::minutes duration) -> core::Result<Verdict, SoakError> {
	auto const started = clock.now();
	auto course = Course{.end = started.steady + duration, .first = std::nullopt};
	auto loops = std::vector<CardLoop>{};
	for (auto const card : device::card_pair) {
		loops.push_back({
		    .card = card,
		    .watch = Settling{.until = started.steady + settle},
		    .image = Idle{},
		    .image_due = started.steady,
		    .pattern = Idle{},
		    .pattern_due = started.steady,
		    .progress = {.passes = 0, .rounds = 0},
		});
	}
	for (auto& loop : loops) {
		if (auto begun = begin(bench, loop, started.steady).and_then([&] { return write(bench, loop, 0, started.steady); }); !begun) {
			return std::unexpected(begun.error());
		}
	}
	while (!std::ranges::all_of(loops, idle)) {
		auto const now = clock.now();
		for (auto& loop : loops) {
			if (auto stepped = step_image(bench, course, loop, now).and_then([&] { return step_pattern(bench, course, loop, now); }); !stepped) {
				return std::unexpected(stepped.error());
			}
		}
		clock.sleep_until(now.steady + poll_interval);
	}
	auto const cards = std::array{loops[0].progress, loops[1].progress};
	if (course.first) {
		return Verdict{Found{.started = started, .sighting = std::move(*course.first), .cards = cards}};
	}
	auto const ended = clock.now();
	auto const* const zero = std::get_if<Watching>(&loops[0].watch);
	auto const* const one = std::get_if<Watching>(&loops[1].watch);
	if (zero == nullptr || one == nullptr || std::ranges::any_of(cards, [](Progress const& progress) { return progress.passes == 0; })) {
		return Verdict{TooShort{.started = started, .ended = ended, .cards = cards}};
	}
	return Verdict{Clean{.started = started, .ended = ended, .cards = cards, .clocks = {zero->baseline.memory, one->baseline.memory}}};
}

}

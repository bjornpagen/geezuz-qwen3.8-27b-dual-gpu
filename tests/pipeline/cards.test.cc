// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.card_scheduler;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.exec;
import engine.model;
import engine.net;
import engine.testing;

namespace {

namespace device = engine::device;
namespace ex = engine::exec;
namespace fake = engine::driver_double;
namespace net = engine::net;
using engine::card_scheduler::finished;

using Words = device::DeviceArray<std::uint32_t, 1024>;
using Staging = device::MappedArray<std::uint32_t, 1024>;

/** Both cards' contexts and a stream on each, over a fake driver. */
struct Pair {
	device::Devices devices;
	std::vector<device::Stream> streams;

	[[nodiscard]] auto card(engine::model::Card which) const -> engine::card_scheduler::Card {
		return engine::card_scheduler::Card{devices.context(which), streams[std::to_underlying(which)]};
	}
};

[[nodiscard]] auto pair(fake::Fake const& driver) -> std::optional<Pair> {
	auto devices = device::Devices::open(driver.driver());
	if (!devices) {
		return std::nullopt;
	}
	auto streams = std::vector<device::Stream>{};
	for (auto const card : device::card_pair) {
		auto stream = devices->context(card).create_stream();
		if (!stream) {
			return std::nullopt;
		}
		streams.push_back(std::move(*stream));
	}
	return Pair{.devices = std::move(*devices), .streams = std::move(streams)};
}

/** A scheduled sender completes only once its event has fired, however long the card keeps it pending. */
[[nodiscard]] auto schedule_completes_when_the_event_fires() -> bool {
	return std::ranges::all_of(std::array<std::size_t, 4>{0, 1, 7, 100}, [](std::size_t pending) {
		auto const driver = fake::Fake{fake::Machine{.pending_queries = pending}};
		auto cards = pair(driver);
		if (!cards) {
			return false;
		}
		auto const before = driver.calls();
		auto const waited = ex::wait<device::DeviceError>(cards->card(engine::model::Card::Zero).schedule() | ex::then([] {
			                                                  return 7;
		                                                  }));
		// Creating and recording the event, then one query per pending answer and the one that finds it fired.
		auto const queries = driver.calls() - before;
		return waited && *waited && std::get<0>(**waited) == 7 && queries >= pending + 1 && driver.violations().empty();
	});
}

/** `when_all` over both cards' schedulers joins both, whichever fires first. */
[[nodiscard]] auto when_all_joins_both_cards() -> bool {
	auto const driver = fake::Fake{fake::Machine{.pending_queries = 3}};
	auto cards = pair(driver);
	if (!cards) {
		return false;
	}
	auto joined = ex::when_all(cards->card(engine::model::Card::Zero).schedule(), cards->card(engine::model::Card::One).schedule()) |
	              ex::then([] {
		              return std::string{"both"};
	              });
	auto const waited = ex::wait<device::DeviceError>(std::move(joined));
	return waited && *waited && std::get<0>(**waited) == "both" && driver.violations().empty();
}

/** An upload on each card, joined over both, then a download of each: the words come back exactly as sent. */
[[nodiscard]] auto uploads_on_both_cards_round_trip(std::uint64_t seed) -> bool {
	auto const driver = fake::Fake{fake::Machine{.pending_queries = 2}};
	auto cards = pair(driver);
	if (!cards) {
		return false;
	}
	auto random = engine::testing::generator(seed);
	auto sent = std::array<std::vector<std::uint32_t>, 2>{};
	auto ups = std::vector<device::Transfer<Words, Staging>>{};
	auto backs = std::vector<Staging>{};
	for (auto const card : device::card_pair) {
		auto const& context = cards->devices.context(card);
		auto staging = context.allocate_mapped<std::uint32_t, 1024>();
		auto back = context.allocate_mapped<std::uint32_t, 1024>();
		auto array = context.allocate<std::uint32_t, 1024>();
		if (!staging || !back || !array) {
			return false;
		}
		std::ranges::generate(staging->host(), [&] {
			return static_cast<std::uint32_t>(random());
		});
		sent[std::to_underlying(card)].assign(staging->host().begin(), staging->host().end());
		ups.push_back(context.upload(cards->streams[std::to_underlying(card)], std::move(*array), std::move(*staging), 0));
		backs.push_back(std::move(*back));
	}
	auto uploaded = ex::wait<device::DeviceError>(ex::when_all(finished(std::move(ups[0])), finished(std::move(ups[1]))));
	if (!uploaded || !*uploaded) {
		return false;
	}
	auto& [zero_up, one_up] = **uploaded;
	if (!zero_up.copied || !one_up.copied) {
		return false;
	}
	auto const& zero = cards->devices.context(engine::model::Card::Zero);
	auto const& one = cards->devices.context(engine::model::Card::One);
	auto downloaded = ex::wait<device::DeviceError>(
	    ex::when_all(finished(zero.download(cards->streams[0], std::move(backs[0]), std::move(zero_up.destination))),
	                 finished(one.download(cards->streams[1], std::move(backs[1]), std::move(one_up.destination)))));
	if (!downloaded || !*downloaded) {
		return false;
	}
	auto const& [zero_down, one_down] = **downloaded;
	return zero_down.copied && one_down.copied && std::ranges::equal(zero_down.destination.host(), sent[0]) &&
	       std::ranges::equal(one_down.destination.host(), sent[1]) && driver.violations().empty();
}

/**
 * A driver call that fails while a scheduled sender records or polls its event reaches the waiter as the sender's
 * typed error, never as a hang or a misuse of the driver: searched over every call the wait makes.
 */
[[nodiscard]] auto every_failing_call_is_the_senders_error() -> bool {
	auto const machine = fake::Machine{.pending_queries = 3};
	auto span = std::pair<std::size_t, std::size_t>{};
	{
		auto const driver = fake::Fake{machine};
		auto cards = pair(driver);
		if (!cards) {
			return false;
		}
		span.first = driver.calls();
		auto const waited = ex::wait<device::DeviceError>(cards->card(engine::model::Card::One).schedule());
		span.second = driver.calls();
		if (!waited || !*waited) {
			return false;
		}
	}
	auto errors = std::size_t{0};
	for (auto call = span.first; call < span.second; ++call) {
		auto failing = machine;
		failing.failing_call = call;
		auto const driver = fake::Fake{failing};
		auto cards = pair(driver);
		if (!cards) {
			return false;
		}
		auto const waited = ex::wait<device::DeviceError>(cards->card(engine::model::Card::One).schedule());
		if (!waited || !driver.violations().empty()) {
			return false;
		}
		errors += *waited ? 0 : 1;
	}
	// Every call the wait makes but the last is the event's creation, its record or a query, and fails the wait; the
	// last destroys the event once the wait is over, where a failure has no remedy and changes nothing.
	return errors + 1 == span.second - span.first;
}

struct Note {
	std::uint64_t value;
};

/** The sum of every message the worker took, and how many times it woke. */
struct Tally {
	std::uint64_t sum;
	std::uint64_t messages;

	[[nodiscard]] constexpr auto operator==(Tally const&) const -> bool = default;
};

using Worker = engine::card_scheduler::Worker<Note, 64, Tally>;

/** A worker's thread takes every message posted, in order, and hands back its body's outcome when stopped. */
[[nodiscard]] auto a_worker_takes_every_message_and_returns_its_outcome(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto started = Worker::start([](engine::card_scheduler::Inbox<Note, 64>& inbox) {
		auto tally = Tally{.sum = 0, .messages = 0};
		auto last = std::uint64_t{0};
		auto ordered = true;
		while (!inbox.stopping()) {
			inbox.drain([&](Note const& note) {
				ordered = ordered && note.value > last;
				last = note.value;
				tally.sum += note.value;
				tally.messages += 1;
			});
			engine::core::invariant(inbox.wait(std::nullopt).has_value());
		}
		inbox.drain([&](Note const& note) {
			tally.sum += note.value;
			tally.messages += 1;
		});
		return ordered ? tally : Tally{.sum = 0, .messages = 0};
	});
	if (!started) {
		return false;
	}
	auto& worker = **started;
	auto expected = Tally{.sum = 0, .messages = 0};
	auto value = std::uint64_t{0};
	for (auto round = 0; round < 50; ++round) {
		auto const burst = random() % 40;
		for (auto index = std::uint64_t{0}; index < burst; ++index) {
			value += 1 + (random() % 5);
			// The worker drains on every wake, but a burst of 40 fits even if it has not yet.
			if (worker.post(Note{.value = value}) != net::Posted::Accepted) {
				return false;
			}
			expected.sum += value;
			expected.messages += 1;
		}
		std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::microseconds{200});
	}
	return worker.stop() == expected;
}

}

auto main() -> int {
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "a scheduled sender completes when its event fires", .holds = schedule_completes_when_the_event_fires()},
	    {.name = "when_all joins both cards", .holds = when_all_joins_both_cards()},
	    {.name = "uploads on both cards round trip (seed 1)", .holds = uploads_on_both_cards_round_trip(1)},
	    {.name = "uploads on both cards round trip (seed 2)", .holds = uploads_on_both_cards_round_trip(2)},
	    {.name = "every failing driver call is the sender's error", .holds = every_failing_call_is_the_senders_error()},
	    {.name = "a worker takes every message and returns its outcome (seed 3)", .holds = a_worker_takes_every_message_and_returns_its_outcome(3)},
	    {.name = "a worker takes every message and returns its outcome (seed 4)", .holds = a_worker_takes_every_message_and_returns_its_outcome(4)},
	});
	return engine::testing::report(cases);
}

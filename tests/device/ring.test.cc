// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.testing;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;

using device::Devices;
using device::Rank;
using device::Ticket;
using fake::Fake;
using fake::Machine;
using fake::Served;

/** A descriptor of several words, so that a torn read (some words old, some new) shows in the answer. */
struct Work {
	std::uint64_t seed;
	std::array<std::uint64_t, 7> words;

	[[nodiscard]] constexpr auto operator==(Work const&) const -> bool = default;
};

struct Answer {
	std::uint64_t seed;
	std::array<std::uint64_t, 7> mixed;

	[[nodiscard]] constexpr auto operator==(Answer const&) const -> bool = default;
};

/** The work the modelled kernel does, and the test's oracle for its answers. */
[[nodiscard]] constexpr auto serve(Work const& work) -> Answer {
	auto answer = Answer{.seed = work.seed, .mixed = {}};
	for (auto index = std::size_t{0}; index < work.words.size(); ++index) {
		answer.mixed[index] = (work.words[index] * 0x9e37'79b9'7f4a'7c15U) ^ (work.seed + index);
	}
	return answer;
}

[[nodiscard]] auto work_from(std::mt19937_64& random) -> Work {
	auto work = Work{.seed = random(), .words = {}};
	std::ranges::generate(work.words, [&] {
		return random();
	});
	return work;
}

constexpr auto capacity = std::size_t{3};

using Ring = device::Ring<Work, Answer, capacity>;

/**
 * The protocol, searched over interleavings in one thread: host submissions, host collections, kernel steps and one
 * exit, in random order, against a model of the exchange. The model: tickets are handed out in order while fewer than
 * `capacity` are in flight; the kernel serves exactly the published, unserved tickets, in order; the host collects
 * exactly the served, uncollected tickets, in order, each with the kernel's answer to its own descriptor; the exit
 * ticket is served and collected like any other and ends the kernel's loop.
 */
[[nodiscard]] auto the_protocol_matches_its_model(std::uint64_t seed) -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto ring = Ring::create(opened->context(Rank::Zero));
	if (!ring) {
		return false;
	}
	auto kernel = fake::RingDevice<Work, Answer, capacity>{ring->device_address()};
	auto body = [](Work const& work) {
		return serve(work);
	};
	auto random = engine::testing::generator(seed);
	auto submitted = std::deque<std::pair<Ticket, std::optional<Work>>>{};
	auto next_ticket = std::uint64_t{1};
	auto served = std::uint64_t{0};
	auto exit = std::optional<Ticket>{};
	auto exited = false;
	auto collected_exit = false;
	auto holds = true;
	for (auto step = 0; step < 3000 && holds && !collected_exit; ++step) {
		auto const choice = std::uniform_int_distribution{0, 99}(random);
		auto const room = submitted.size() < capacity;
		if (choice < 35 && !exit) {
			auto const work = work_from(random);
			auto const ticket = ring->try_submit(work);
			holds = room ? ticket == Ticket{.value = next_ticket} : !ticket.has_value();
			if (ticket) {
				submitted.emplace_back(*ticket, work);
				++next_ticket;
			}
		} else if (choice < 37 && !exit && step > 1500) {
			auto const ticket = ring->try_exit();
			holds = room ? ticket == Ticket{.value = next_ticket} : !ticket.has_value();
			if (ticket) {
				submitted.emplace_back(*ticket, std::nullopt);
				exit = ticket;
				++next_ticket;
			}
		} else if (choice < 70) {
			auto const unserved = served + 1 < next_ticket;
			auto const outcome = exited ? Served::Idle : kernel.step(body);
			auto const expected = !unserved || exited ? Served::Idle : exit && exit->value == served + 1 ? Served::Exited : Served::Ran;
			holds = outcome == expected;
			served += outcome == Served::Idle ? 0 : 1;
			exited = exited || outcome == Served::Exited;
		} else {
			auto const completion = ring->try_collect();
			auto const ready = !submitted.empty() && submitted.front().first.value <= served;
			if (!ready) {
				holds = !completion.has_value();
				continue;
			}
			auto const [ticket, work] = submitted.front();
			submitted.pop_front();
			auto const expected = work ? device::Completion<Answer>{device::Ran<Answer>{.ticket = ticket, .result = serve(*work)}}
			                           : device::Completion<Answer>{device::Exited{.ticket = ticket}};
			holds = completion == expected;
			collected_exit = !work.has_value();
		}
		holds = holds && ring->in_flight() == submitted.size();
	}
	return holds;
}

/**
 * The protocol under real concurrency: the modelled kernel on its own thread, the host here, many thousands of
 * tickets through a few slots. Every answer arrives once, in order, and is the answer to its own descriptor, whole.
 */
[[nodiscard]] auto concurrent_exchange_is_exact(std::uint64_t seed) -> bool {
	constexpr auto tickets = std::uint64_t{20'000};
	constexpr auto slots = std::size_t{4};
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto ring = device::Ring<Work, Answer, slots>::create(opened->context(Rank::One));
	if (!ring) {
		return false;
	}
	auto random = engine::testing::generator(seed);
	auto expected = std::deque<std::pair<Ticket, Answer>>{};
	auto submitted = std::uint64_t{0};
	auto exit = std::optional<Ticket>{};
	auto exited = false;
	auto holds = true;
	auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
	auto const body = [](Work const& work) {
		return serve(work);
	};
	{
		auto const kernel = fake::RingKernel<Work, Answer, slots, decltype(body)>{ring->device_address(), body};
		while (holds && !exited && std::chrono::steady_clock::now() < deadline) {
			if (submitted < tickets) {
				auto const work = work_from(random);
				if (auto const ticket = ring->try_submit(work)) {
					expected.emplace_back(*ticket, serve(work));
					++submitted;
				}
			} else if (expected.empty() && !exit) {
				exit = ring->try_exit();
				holds = exit == Ticket{.value = tickets + 1};
			}
			if (auto const completion = ring->try_collect()) {
				if (expected.empty()) {
					exited = *completion == device::Completion<Answer>{device::Exited{.ticket = {.value = tickets + 1}}};
					holds = exited;
				} else {
					auto const [ticket, answer] = expected.front();
					expected.pop_front();
					holds = *completion == device::Completion<Answer>{device::Ran<Answer>{.ticket = ticket, .result = answer}};
				}
			}
		}
	}
	return holds && exited && ring->in_flight() == 0;
}

[[nodiscard]] auto the_ring_is_mapped_pinned_memory_released_with_its_owner() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto holds = true;
	{
		auto const ring = Ring::create(opened->context(Rank::Zero));
		holds = ring.has_value() && ring->in_flight() == 0 && driver.live(fake::Kind::HostMemory) == 1;
	}
	return holds && driver.live(fake::Kind::HostMemory) == 0 && driver.violations().empty();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "random interleavings of submit, serve, collect and exit match the protocol's model (seed 31)",
	        .holds = the_protocol_matches_its_model(31),
	    },
	    engine::testing::Case{
	        .name = "random interleavings of submit, serve, collect and exit match the protocol's model (seed 32)",
	        .holds = the_protocol_matches_its_model(32),
	    },
	    engine::testing::Case{
	        .name = "random interleavings of submit, serve, collect and exit match the protocol's model (seed 33)",
	        .holds = the_protocol_matches_its_model(33),
	    },
	    engine::testing::Case{
	        .name = "a kernel thread and the host exchange 20000 tickets through 4 slots exactly, in order (seed 34)",
	        .holds = concurrent_exchange_is_exact(34),
	    },
	    engine::testing::Case{
	        .name = "the ring is one mapped pinned allocation, released with its owner",
	        .holds = the_ring_is_mapped_pinned_memory_released_with_its_owner(),
	    },
	};
	return engine::testing::report(cases);
}

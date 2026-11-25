// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.model;
import engine.testing;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;

using device::Devices;
using engine::model::Card;
using fake::Fake;
using fake::Kind;
using fake::Machine;

constexpr auto gib = std::size_t{1} << 30U;

using OneGib = device::DeviceArray<std::byte, gib>;
using FourGib = device::DeviceArray<std::byte, 4 * gib>;
using EightGib = device::DeviceArray<std::byte, 8 * gib>;
using Owned = std::variant<OneGib, FourGib, EightGib>;

/**
 * Allocation against a model of the card: an allocation succeeds exactly when it fits beside what is live, a refusal
 * names cuMemAlloc_v2 running out of memory and is classified transient, and the driver's live count always equals
 * the model's. Allocations die in a random order.
 */
[[nodiscard]] auto allocation_follows_the_card_model(std::uint64_t seed) -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Card::Zero);
	auto const capacity = Machine{}.cards[0].memory_bytes;
	auto random = engine::testing::generator(seed);
	auto owned = std::vector<Owned>{};
	auto in_use = std::uint64_t{0};
	auto holds = true;
	auto const attempt = [&](auto allocation, std::uint64_t bytes, bool fits) {
		if (allocation) {
			owned.emplace_back(std::move(*allocation));
			in_use += bytes;
		}
		auto const refused_as_oom =
		    !allocation &&
		    allocation.error() ==
		        device::DeviceError{.reason = device::Failure{.symbol = "cuMemAlloc_v2", .status = device::Status::OutOfMemory}} &&
		    allocation.error().retry() == engine::core::Retry::Transient;
		holds = holds && (fits ? allocation.has_value() : refused_as_oom);
	};
	for (auto step = 0; step < 300; ++step) {
		switch (std::uniform_int_distribution{0, 3}(random)) {
		case 0:
			attempt(context.allocate<std::byte, gib>(), gib, in_use + gib <= capacity);
			break;
		case 1:
			attempt(context.allocate<std::byte, 4 * gib>(), 4 * gib, in_use + 4 * gib <= capacity);
			break;
		case 2:
			attempt(context.allocate<std::byte, 8 * gib>(), 8 * gib, in_use + 8 * gib <= capacity);
			break;
		default:
			if (!owned.empty()) {
				auto const victim = std::uniform_int_distribution<std::size_t>{0, owned.size() - 1}(random);
				in_use -= std::visit(
				    [](auto const& array) -> std::uint64_t {
					    return array.bytes;
				    },
				    owned[victim]);
				owned.erase(owned.begin() + static_cast<std::ptrdiff_t>(victim));
			}
			break;
		}
		holds = holds && driver.live(Kind::DeviceMemory) == owned.size();
	}
	return holds && driver.violations().empty();
}

[[nodiscard]] auto one_card_running_out_leaves_the_other_alone() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const first = opened->context(Card::Zero).allocate<std::byte, 8 * gib>();
	auto const second = opened->context(Card::Zero).allocate<std::byte, 8 * gib>();
	auto const third = opened->context(Card::Zero).allocate<std::byte, 8 * gib>();
	auto const elsewhere = opened->context(Card::One).allocate<std::byte, 8 * gib>();
	return first && second && !third && elsewhere && driver.live(Kind::DeviceMemory) == 3 && first->address() != second->address();
}

struct Pair {
	std::uint64_t key;
	std::uint32_t value;
	std::uint32_t tag;

	[[nodiscard]] constexpr auto operator==(Pair const&) const -> bool = default;
};

[[nodiscard]] auto mapped_memory_starts_zeroed_and_maps_into_the_card() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto mapped = opened->context(Card::One).allocate_mapped<Pair, 1000>();
	return mapped.has_value() && mapped->device_address() != 0 &&
	       std::ranges::all_of(mapped->host(),
	                           [](Pair const& pair) {
		                           return pair == Pair{};
	                           }) &&
	       driver.live(Kind::HostMemory) == 1;
}

/** The engine's own shape: the machine first, then everything created from it, so everything dies before it. */
struct Workspace {
	Devices devices;
	device::DeviceArray<float, 4096> partials;
	device::MappedArray<Pair, 64> staging;
	device::Stream stream;
	device::Event done;
};

[[nodiscard]] auto build(Devices devices) -> std::optional<Workspace> {
	auto const& context = devices.context(Card::One);
	auto partials = context.allocate<float, 4096>();
	auto staging = context.allocate_mapped<Pair, 64>();
	auto stream = context.create_stream();
	auto done = context.create_event();
	if (!partials || !staging || !stream || !done) {
		return std::nullopt;
	}
	return Workspace{
	    .devices = std::move(devices),
	    .partials = std::move(*partials),
	    .staging = std::move(*staging),
	    .stream = std::move(*stream),
	    .done = std::move(*done),
	};
}

[[nodiscard]] auto owners_declared_after_the_machine_release_before_it() -> bool {
	auto const driver = Fake{Machine{}};
	auto opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto built = std::optional<Workspace>{};
	built = build(std::move(*opened));
	auto const acquired = built.has_value() && driver.live(Kind::DeviceMemory) == 1 && driver.live(Kind::HostMemory) == 1 &&
	                      driver.live(Kind::Stream) == 1 && driver.live(Kind::Event) == 1;
	built.reset();
	return acquired && driver.violations().empty() && driver.nothing_live();
}

/** The double sees the wrong order: memory outliving its card's context is destroyed with it and freed after it. */
[[nodiscard]] auto an_owner_outliving_the_machine_is_caught() -> bool {
	auto const driver = Fake{Machine{}};
	auto opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto machine = std::optional<Devices>{std::move(*opened)};
	auto allocated = false;
	{
		auto const stray = machine->context(Card::Zero).allocate<float, 16>();
		allocated = stray.has_value();
		machine.reset();
	}
	auto const violations = driver.violations();
	return allocated && std::ranges::contains(violations, fake::Violation::OutlivedContext) &&
	       std::ranges::contains(violations, fake::Violation::ReleasedAfterContext);
}

[[nodiscard]] auto events_order_one_card_after_the_other() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const producer = opened->context(Card::Zero).create_stream();
	auto const consumer = opened->context(Card::One).create_stream();
	auto const done = opened->context(Card::Zero).create_event();
	if (!producer || !consumer || !done) {
		return false;
	}
	auto const recorded = done->record(*producer);
	auto const ordered = consumer->wait(*done);
	auto const state = done->wait_until(std::chrono::steady_clock::now() + std::chrono::seconds{1});
	return recorded && ordered && state == device::EventState::Complete && driver.violations().empty();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "allocations succeed exactly when they fit, and refusals are transient out-of-memory (seed 11)",
	        .holds = allocation_follows_the_card_model(11),
	    },
	    engine::testing::Case{
	        .name = "allocations succeed exactly when they fit, and refusals are transient out-of-memory (seed 12)",
	        .holds = allocation_follows_the_card_model(12),
	    },
	    engine::testing::Case{
	        .name = "one card running out of memory leaves the other card's allocations alone",
	        .holds = one_card_running_out_leaves_the_other_alone(),
	    },
	    engine::testing::Case{
	        .name = "mapped memory starts value-initialized and has a device address",
	        .holds = mapped_memory_starts_zeroed_and_maps_into_the_card(),
	    },
	    engine::testing::Case{
	        .name = "owners declared after the machine release before its contexts, leaving nothing",
	        .holds = owners_declared_after_the_machine_release_before_it(),
	    },
	    engine::testing::Case{
	        .name = "an owner that outlives its card's context is caught by the double",
	        .holds = an_owner_outliving_the_machine_is_caught(),
	    },
	    engine::testing::Case{
	        .name = "an event recorded on one card orders a stream of the other",
	        .holds = events_order_one_card_after_the_other(),
	    },
	};
	return engine::testing::report(cases);
}

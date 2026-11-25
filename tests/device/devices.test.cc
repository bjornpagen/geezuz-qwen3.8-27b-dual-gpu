// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.testing;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;

using device::DeviceError;
using device::Devices;
using fake::CardModel;
using fake::Fake;
using fake::LinkModel;
using fake::Machine;

/** What the fake saw is clean: no misuse of the driver, and nothing the engine took is left. */
[[nodiscard]] auto clean(Fake const& driver) -> bool {
	return driver.violations().empty() && driver.nothing_live();
}

/** Opens the machine and drops it again; the error when it refuses, after checking that refusing left nothing. */
[[nodiscard]] auto open_and_close(Machine machine) -> std::optional<std::optional<DeviceError>> {
	auto const driver = Fake{std::move(machine)};
	auto outcome = std::optional<DeviceError>{};
	{
		auto opened = Devices::open(driver.driver());
		if (!opened) {
			outcome = opened.error();
		}
	}
	if (!clean(driver)) {
		return std::nullopt;
	}
	return outcome;
}

[[nodiscard]] auto refused_with(Machine machine, DeviceError const& expected) -> bool {
	auto const outcome = open_and_close(std::move(machine));
	return outcome.has_value() && *outcome == expected;
}

[[nodiscard]] auto opens(Machine machine) -> bool {
	auto const outcome = open_and_close(std::move(machine));
	return outcome.has_value() && !outcome->has_value();
}

[[nodiscard]] auto error(auto reason) -> DeviceError {
	return DeviceError{.reason = reason};
}

[[nodiscard]] auto the_machine_opens_peered_and_releases_everything() -> bool {
	auto const driver = Fake{Machine{}};
	auto holds = true;
	{
		auto const opened = Devices::open(driver.driver());
		holds = opened.has_value() && driver.peer_access(0, 1) && driver.peer_access(1, 0) && driver.live(fake::Kind::Context) == 2 &&
		        driver.live(fake::Kind::ManagementSession) == 1 && opened->context(device::Rank::Zero).ordinal() == 0 &&
		        opened->context(device::Rank::One).ordinal() == 1;
	}
	return holds && clean(driver);
}

[[nodiscard]] auto exactly_two_cards_open() -> bool {
	auto holds = true;
	for (auto count = std::size_t{0}; count <= 6; ++count) {
		auto machine = Machine{.cards = std::vector<CardModel>(count)};
		holds = holds && (count == 2 ? opens(std::move(machine))
		                             : refused_with(std::move(machine), error(device::CardCount{.found = static_cast<int>(count)})));
	}
	return holds;
}

[[nodiscard]] auto a_driver_without_cards_is_a_census_of_zero() -> bool {
	return refused_with(Machine{.init = device::Status::NoDevice}, error(device::CardCount{.found = 0})) &&
	       refused_with(Machine{.init = device::Status::InsufficientDriver},
	                    error(device::Failure{.symbol = "cuInit", .status = device::Status::InsufficientDriver}));
}

[[nodiscard]] auto drivers_older_than_the_isa_needs_are_refused() -> bool {
	auto holds = true;
	for (auto const version : {12'090, 13'000, device::required_driver_version - 10, device::required_driver_version,
	                           device::required_driver_version + 10, 14'000}) {
		holds = holds && (version >= device::required_driver_version
		                      ? opens(Machine{.driver_version = version})
		                      : refused_with(Machine{.driver_version = version},
		                                     error(device::DriverTooOld{.found = version, .required = device::required_driver_version})));
	}
	return holds;
}

/** The oracle for card identity: the first card, in ordinal order, whose first fact is wrong. */
[[nodiscard]] auto expected_card_refusal(std::span<CardModel const> cards) -> std::optional<device::WrongCard> {
	for (auto ordinal = 0; std::cmp_less(ordinal, cards.size()); ++ordinal) {
		auto const& card = cards[static_cast<std::size_t>(ordinal)];
		auto const capability = card.compute_major * 10 + card.compute_minor;
		if (capability != device::compute_capability) {
			return device::WrongCard{.ordinal = ordinal, .fact = device::CardFact::ComputeCapability, .found = capability};
		}
		if (card.sm_count != device::sm_count) {
			return device::WrongCard{.ordinal = ordinal, .fact = device::CardFact::SmCount, .found = card.sm_count};
		}
		if (card.memory_bytes < device::memory_floor || card.memory_bytes > device::memory_ceiling) {
			return device::WrongCard{
			    .ordinal = ordinal,
			    .fact = device::CardFact::MemorySize,
			    .found = static_cast<std::int64_t>(card.memory_bytes),
			};
		}
	}
	return std::nullopt;
}

/** Draws from `choices`, the first one (the engine's own card) about half the time. */
template<class T, std::size_t N>
[[nodiscard]] auto draw(std::mt19937_64& random, std::array<T, N> const& choices) -> T {
	auto const index = std::uniform_int_distribution<std::size_t>{0, 2 * N - 3}(random);
	return choices[index < N - 1 ? 0 : index - (N - 2)];
}

[[nodiscard]] auto only_3090s_open(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	constexpr auto capabilities = std::array{std::pair{8, 6}, std::pair{7, 5}, std::pair{8, 0}, std::pair{8, 9}, std::pair{9, 0}};
	constexpr auto sms = std::array{82, 68, 80, 84};
	constexpr auto memories = std::array{
	    std::uint64_t{25'447'170'048}, std::uint64_t{12} << 30U,       (std::uint64_t{22} << 30U) - 1, std::uint64_t{22} << 30U,
	    std::uint64_t{24} << 30U,      (std::uint64_t{24} << 30U) + 1, std::uint64_t{48} << 30U,
	};
	auto holds = true;
	for (auto round = 0; round < 400; ++round) {
		auto machine = Machine{};
		for (auto& card : machine.cards) {
			std::tie(card.compute_major, card.compute_minor) = draw(random, capabilities);
			card.sm_count = draw(random, sms);
			card.memory_bytes = draw(random, memories);
		}
		auto const expected = expected_card_refusal(machine.cards);
		holds = holds && (expected ? refused_with(std::move(machine), error(*expected)) : opens(std::move(machine)));
	}
	return holds;
}

/** The oracle for the peer path: peer access both ways, then every link of card 0, then of card 1, to the peer. */
[[nodiscard]] auto expected_link_refusal(Machine const& machine) -> std::optional<DeviceError> {
	for (auto ordinal = 0; ordinal < 2; ++ordinal) {
		if (!machine.cards[static_cast<std::size_t>(ordinal)].reaches_peer) {
			return error(device::PeerUnreachable{.from = ordinal, .to = 1 - ordinal});
		}
	}
	for (auto ordinal = 0; ordinal < 2; ++ordinal) {
		auto const& links = machine.cards[static_cast<std::size_t>(ordinal)].links;
		for (auto link = std::uint32_t{0}; link < links.size(); ++link) {
			switch (links[link]) {
			case LinkModel::ToPeer:
				break;
			case LinkModel::Elsewhere:
				return error(device::NotNvlink{.ordinal = ordinal, .link = link, .fault = device::LinkFault::Elsewhere});
			case LinkModel::Down:
				return error(device::NotNvlink{.ordinal = ordinal, .link = link, .fault = device::LinkFault::Down});
			case LinkModel::Missing:
				return error(device::NotNvlink{.ordinal = ordinal, .link = link, .fault = device::LinkFault::Missing});
			}
		}
	}
	return std::nullopt;
}

[[nodiscard]] auto only_a_full_nvlink_peer_path_opens(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	constexpr auto models = std::array{LinkModel::ToPeer, LinkModel::Elsewhere, LinkModel::Down, LinkModel::Missing};
	auto holds = true;
	for (auto round = 0; round < 600; ++round) {
		auto machine = Machine{};
		for (auto& card : machine.cards) {
			card.reaches_peer = std::uniform_int_distribution{0, 7}(random) != 0;
			for (auto& link : card.links) {
				link = draw(random, models);
			}
		}
		auto const expected = expected_link_refusal(machine);
		holds = holds && (expected ? refused_with(std::move(machine), *expected) : opens(std::move(machine)));
	}
	return holds;
}

[[nodiscard]] auto plain_pcie_peer_access_is_refused() -> bool {
	auto machine = Machine{};
	for (auto& card : machine.cards) {
		card.links.fill(LinkModel::Missing);
	}
	return refused_with(std::move(machine), error(device::NotNvlink{.ordinal = 0, .link = 0, .fault = device::LinkFault::Missing}));
}

/**
 * Every vendor call opening makes can fail; whichever one does, opening fails with that call's failure and leaves
 * nothing behind. The count of calls comes from a successful open, so the search covers every call it makes.
 */
[[nodiscard]] auto every_failing_call_is_reported_and_leaves_nothing() -> bool {
	auto calls = std::size_t{0};
	{
		auto const driver = Fake{Machine{}};
		auto const opened = Devices::open(driver.driver());
		calls = driver.calls();
		if (!opened) {
			return false;
		}
	}
	auto holds = calls > 0;
	for (auto failing = std::size_t{0}; failing < calls; ++failing) {
		auto const outcome = open_and_close(Machine{.failing_call = failing});
		if (!outcome.has_value() || !outcome->has_value()) {
			return false;
		}
		auto const& reason = (*outcome)->reason;
		auto const cuda =
		    std::holds_alternative<device::Failure>(reason) && std::get<device::Failure>(reason).status == device::Status::Unknown;
		auto const management = std::holds_alternative<device::ManagementFailure>(reason) &&
		                        std::get<device::ManagementFailure>(reason).status == device::ManagementStatus::Unknown;
		holds = holds && (cuda || management);
	}
	return holds;
}

[[nodiscard]] auto memory_clock_reads_each_card() -> bool {
	auto machine = Machine{};
	machine.cards[0].memory_clock = 9501;
	machine.cards[0].performance_state = 2;
	machine.cards[1].memory_clock = 9751;
	machine.cards[1].performance_state = 0;
	auto const driver = Fake{std::move(machine)};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const zero = opened->memory_clock(device::Rank::Zero);
	auto const one = opened->memory_clock(device::Rank::One);
	return zero == device::MemoryClock{.megahertz = 9501, .state = device::PerformanceState::P2} &&
	       one == device::MemoryClock{.megahertz = 9751, .state = device::PerformanceState::P0};
}

[[nodiscard]] auto an_unknown_performance_state_is_absent() -> bool {
	auto machine = Machine{};
	machine.cards[1].performance_state = 32;
	auto const driver = Fake{std::move(machine)};
	auto const opened = Devices::open(driver.driver());
	return opened.has_value() && opened->memory_clock(device::Rank::One) == device::MemoryClock{.megahertz = 9501, .state = std::nullopt};
}

[[nodiscard]] auto only_running_out_of_memory_is_worth_retrying() -> bool {
	using engine::core::Retry;
	auto const permanent = std::array{
	    error(device::DriverUnavailable{.library = device::Library::Cuda, .stage = device::LoadStage::Open, .name = "x"}),
	    error(device::DriverTooOld{.found = 1, .required = 2}),
	    error(device::CardCount{.found = 1}),
	    error(device::WrongCard{.ordinal = 0, .fact = device::CardFact::SmCount, .found = 1}),
	    error(device::PeerUnreachable{.from = 0, .to = 1}),
	    error(device::NotNvlink{.ordinal = 0, .link = 0, .fault = device::LinkFault::Down}),
	    error(device::Failure{.symbol = "cuMemAlloc_v2", .status = device::Status::InvalidValue}),
	    error(device::ManagementFailure{.symbol = "nvmlInit_v2", .status = {}}),
	    error(device::ModuleRejected{.status = device::Status::InvalidPtx, .log = "x"}),
	    error(device::KernelMissing{.entry = "x"}),
	    error(device::KernelSignature{.entry = "x", .expected = 1, .first = std::nullopt, .second = std::nullopt}),
	};
	return std::ranges::all_of(permanent,
	                           [](DeviceError const& e) {
		                           return e.retry() == Retry::Permanent;
	                           }) &&
	       error(device::Failure{.symbol = "cuMemAlloc_v2", .status = device::Status::OutOfMemory}).retry() == Retry::Transient;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "two bridged 3090s open with peer access both ways, and closing releases everything",
	        .holds = the_machine_opens_peered_and_releases_everything(),
	    },
	    engine::testing::Case{
	        .name = "every card count but two is refused with the count found (0 to 6)",
	        .holds = exactly_two_cards_open(),
	    },
	    engine::testing::Case{
	        .name = "a driver that finds no card reports a census of zero; other init failures name cuInit",
	        .holds = a_driver_without_cards_is_a_census_of_zero(),
	    },
	    engine::testing::Case{
	        .name = "a driver older than the ptx isa needs is refused with both versions",
	        .holds = drivers_older_than_the_isa_needs_are_refused(),
	    },
	    engine::testing::Case{
	        .name = "random card facts: only 3090s open, else the first wrong fact is named (seed 1)",
	        .holds = only_3090s_open(1),
	    },
	    engine::testing::Case{
	        .name = "random card facts: only 3090s open, else the first wrong fact is named (seed 2)",
	        .holds = only_3090s_open(2),
	    },
	    engine::testing::Case{
	        .name = "random peer paths: only peer access over all four links to the peer opens (seed 3)",
	        .holds = only_a_full_nvlink_peer_path_opens(3),
	    },
	    engine::testing::Case{
	        .name = "random peer paths: only peer access over all four links to the peer opens (seed 4)",
	        .holds = only_a_full_nvlink_peer_path_opens(4),
	    },
	    engine::testing::Case{
	        .name = "peer access without nvlink links is refused as not nvlink",
	        .holds = plain_pcie_peer_access_is_refused(),
	    },
	    engine::testing::Case{
	        .name = "every vendor call opening makes, failing, is reported and leaves nothing taken",
	        .holds = every_failing_call_is_reported_and_leaves_nothing(),
	    },
	    engine::testing::Case{
	        .name = "the memory clock and performance state are read per card",
	        .holds = memory_clock_reads_each_card(),
	    },
	    engine::testing::Case{
	        .name = "a performance state nvml does not know reads as absent",
	        .holds = an_unknown_performance_state_is_absent(),
	    },
	    engine::testing::Case{
	        .name = "only running out of device memory is classified transient",
	        .holds = only_running_out_of_memory_is_worth_retrying(),
	    },
	};
	return engine::testing::report(cases);
}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.device:devices;

import std;
import engine.core;
import engine.driver;
import engine.model;
import engine.prims;
import engine.ptx;

namespace engine::device {

/** Both cards, in rank order: what a loop over the pair walks. */
export inline constexpr std::array card_pair = {model::Card::Zero, model::Card::One};

static_assert(card_pair.size() == model::cards);

export [[nodiscard]] constexpr auto peer(model::Card card) -> model::Card {
	return card == model::Card::Zero ? model::Card::One : model::Card::Zero;
}

/** A card's CUDA ordinal is its tensor-parallel rank. */
[[nodiscard]] constexpr auto ordinal(model::Card card) -> driver::CUdevice {
	return std::to_underlying(card);
}

[[nodiscard]] constexpr auto index(model::Card card) -> std::size_t {
	return std::to_underlying(card);
}

/**
 * The machine: two RTX 3090s, bridged by all four nvlink links. Each card's compute capability is the target the
 * embedded images declare, its SM count the sm_86 machine the kernels' budgets are checked against, its memory the
 * model's card memory; this module adds only the link count, which nothing else knows.
 */
export inline constexpr std::uint32_t nvlink_links = 4;

/**
 * The memory a card may report: at most the card's whole memory, and at least what the memory plan needs, the whole
 * less the driver's reserve. The driver reports the whole less a slice it keeps, of a size that varies by release; a
 * card reporting less than the plan needs is refused here rather than failing an allocation later.
 */
[[nodiscard]] constexpr auto memory_fits_the_plan(std::uint64_t reported) -> bool {
	return reported <= model::card_memory_bytes && reported >= model::card_memory_bytes - model::driver_reserve_bytes;
}

/**
 * The oldest driver whose jit reads the engine's ptx, as the driver reports its CUDA version (major * 1000 + minor *
 * 10). Derived from the isa the embedded images declare: ptx isa 9.x first shipped with CUDA 13.x.
 */
export inline constexpr int required_driver_version = [] {
	constexpr auto isa = ptx::abi.header.isa;
	static_assert(isa / 10 == 9, "the isa-to-CUDA mapping holds for ptx 9.x; revisit it with the isa");
	return 13'000 + (isa % 10) * 10;
}();

[[nodiscard]] auto refusal(driver::CardFacts const& card) -> std::optional<driver::WrongCard> {
	auto const capability = card.compute_major * 10 + card.compute_minor;
	if (capability != ptx::abi.header.sm) {
		return driver::WrongCard{.ordinal = card.ordinal, .fact = driver::CardFact::ComputeCapability, .found = capability};
	}
	if (card.sm_count != prims::sm86.sms) {
		return driver::WrongCard{.ordinal = card.ordinal, .fact = driver::CardFact::SmCount, .found = card.sm_count};
	}
	if (!memory_fits_the_plan(card.memory_bytes)) {
		return driver::WrongCard{
		    .ordinal = card.ordinal,
		    .fact = driver::CardFact::MemorySize,
		    .found = core::narrow<std::int64_t>(card.memory_bytes),
		};
	}
	return std::nullopt;
}

[[nodiscard]] auto refuse(auto reason) -> std::unexpected<driver::DeviceError> {
	return std::unexpected(driver::DeviceError{.reason = std::move(reason)});
}

/** How one link reading fails to join a card to its peer, if it does. */
struct LinkVerdict {
	driver::PciAddress peer;

	[[nodiscard]] auto operator()(driver::LinkMissing /*reading*/) const -> std::optional<driver::LinkFault> {
		return driver::LinkFault::Missing;
	}

	[[nodiscard]] auto operator()(driver::LinkDown /*reading*/) const -> std::optional<driver::LinkFault> {
		return driver::LinkFault::Down;
	}

	[[nodiscard]] auto operator()(driver::LinkUp const& reading) const -> std::optional<driver::LinkFault> {
		return reading.remote == peer ? std::nullopt : std::optional{driver::LinkFault::Elsewhere};
	}
};

/** Every link of `card` joins it to `peer_pci`, or the first that does not. */
[[nodiscard]] auto check_links(driver::ManagementSession const& management, model::Card card, driver::PciAddress address,
                               driver::PciAddress peer_pci) -> core::Result<void, driver::DeviceError> {
	for (auto link = std::uint32_t{0}; link < nvlink_links; ++link) {
		auto const reading = management.link(address, link);
		if (!reading) {
			return std::unexpected(reading.error());
		}
		if (auto const fault = std::visit(LinkVerdict{.peer = peer_pci}, *reading)) {
			return refuse(driver::NotNvlink{.ordinal = ordinal(card), .link = link, .fault = *fault});
		}
	}
	return {};
}

/**
 * The two cards, checked to be the engine's machine, each with its primary context retained and the other card's
 * memory mapped into it over nvlink. Everything created from a card's context must die before this does.
 */
export class Devices {
public:
	/**
	 * Builds the machine over the given vendor tables. Every fact is checked before any resource is taken: the driver
	 * version, exactly two cards, each an RTX 3090, peer access both ways, and every nvlink link of each card leading
	 * to the other. Peer access over pcie alone is refused: on GeForce the stock driver only grants it over nvlink, and
	 * the link check proves the path rather than trusting that.
	 */
	[[nodiscard]] static auto open(driver::Driver const& driver) -> core::Result<Devices, driver::DeviceError> {
		auto const census = driver::census(driver.cuda);
		if (!census) {
			return std::unexpected(census.error());
		}
		if (census->driver_version < required_driver_version) {
			return refuse(driver::DriverTooOld{.found = census->driver_version, .required = required_driver_version});
		}
		if (census->cards.size() != model::cards) {
			return refuse(driver::CardCount{.found = core::narrow<int>(census->cards.size())});
		}
		for (auto const& card : census->cards) {
			if (auto const wrong = refusal(card)) {
				return refuse(*wrong);
			}
		}
		auto const pci = [&](model::Card card) {
			return census->cards[index(card)].pci;
		};
		for (auto const card : card_pair) {
			auto const reaches = driver::can_access_peer(driver.cuda, ordinal(card), ordinal(peer(card)));
			if (!reaches) {
				return std::unexpected(reaches.error());
			}
			if (!*reaches) {
				return refuse(driver::PeerUnreachable{.from = ordinal(card), .to = ordinal(peer(card))});
			}
		}
		auto management = driver::ManagementSession::open(driver.management);
		if (!management) {
			return std::unexpected(management.error());
		}
		for (auto const card : card_pair) {
			if (auto linked = check_links(*management, card, pci(card), pci(peer(card))); !linked) {
				return std::unexpected(linked.error());
			}
		}
		auto zero = driver::Context::retain(driver.cuda, ordinal(model::Card::Zero));
		auto one = zero.and_then([&](auto const&) {
			return driver::Context::retain(driver.cuda, ordinal(model::Card::One));
		});
		if (!one) {
			return std::unexpected(one.error());
		}
		if (auto mapped = zero->enable_peer_access(*one).and_then([&] {
			    return one->enable_peer_access(*zero);
		    });
		    !mapped) {
			return std::unexpected(mapped.error());
		}
		return Devices{
		    driver.cuda,
		    std::move(*management),
		    {
		        Card{.context = std::move(*zero), .pci = pci(model::Card::Zero)},
		        Card{.context = std::move(*one), .pci = pci(model::Card::One)},
		    },
		};
	}

	[[nodiscard]] auto context(model::Card card) const -> driver::Context const& {
		return cards_[index(card)].context;
	}

	/**
	 * The card's memory clock and performance state now. Bring-up sets the clocks; the engine reads them to refuse a
	 * card that came up in an unexpected state (GeForce Ampere runs CUDA in P2, below its P0 memory clock).
	 */
	[[nodiscard]] auto memory_clock(model::Card card) const -> core::Result<driver::MemoryClock, driver::DeviceError> {
		return management_.memory_clock(cards_[index(card)].pci);
	}

	// The reads below only observe: nothing in the engine sets a clock or a limit; bring-up does.

	[[nodiscard]] auto sm_clock(model::Card card) const -> core::Result<driver::SmClock, driver::DeviceError> {
		return management_.sm_clock(cards_[index(card)].pci);
	}

	/** The card's power draw now and the limit the driver enforces, in milliwatts. */
	[[nodiscard]] auto power(model::Card card) const -> core::Result<driver::Power, driver::DeviceError> {
		return management_.power(cards_[index(card)].pci);
	}

	/** The card's GPU die temperature now; the memory junction temperature is not readable through NVML on a 3090. */
	[[nodiscard]] auto gpu_temperature(model::Card card) const -> core::Result<driver::GpuTemperature, driver::DeviceError> {
		return management_.gpu_temperature(cards_[index(card)].pci);
	}

	/** The reasons holding the card's clocks down now. */
	[[nodiscard]] auto clocks_event_reasons(model::Card card) const -> core::Result<driver::ClocksEventReasons, driver::DeviceError> {
		return management_.clocks_event_reasons(cards_[index(card)].pci);
	}

	/** How long each reason has held the card's clocks down since the driver loaded. */
	[[nodiscard]] auto clocks_event_counters(model::Card card) const -> core::Result<driver::ClocksEventCounters, driver::DeviceError> {
		return management_.clocks_event_counters(cards_[index(card)].pci);
	}

	/** The version and speed of each of the card's nvlink links, in link order. */
	[[nodiscard]] auto nvlinks(model::Card card) const
	    -> core::Result<std::array<driver::NvlinkLinkReading, nvlink_links>, driver::DeviceError> {
		auto readings = std::array<driver::NvlinkLinkReading, nvlink_links>{};
		for (auto link = std::uint32_t{0}; link < nvlink_links; ++link) {
			auto const reading = management_.nvlink(cards_[index(card)].pci, link);
			if (!reading) {
				return std::unexpected(reading.error());
			}
			readings[link] = *reading;
		}
		return readings;
	}

	/** What the driver says about the path from `card` to its peer: performance rank and native atomics. */
	[[nodiscard]] auto peer_facts(model::Card card) const -> core::Result<driver::PeerFacts, driver::DeviceError> {
		return driver::peer_facts(cuda_, ordinal(card), ordinal(peer(card)));
	}

	/** The card's limits for persisting L2 accesses. */
	[[nodiscard]] auto l2_persistence(model::Card card) const -> core::Result<driver::L2Persistence, driver::DeviceError> {
		return driver::l2_persistence(cuda_, ordinal(card));
	}

private:
	struct Card {
		driver::Context context;
		driver::PciAddress pci;
	};

	Devices(driver::Api const& cuda, driver::ManagementSession management, std::array<Card, model::cards> cards)
	    : cuda_{cuda}, management_{std::move(management)}, cards_{std::move(cards)} {}

	driver::Api cuda_;
	driver::ManagementSession management_;
	std::array<Card, model::cards> cards_;
};

/** Loads the vendor libraries and opens the machine: the product's one way to the cards. */
export [[nodiscard]] auto start() -> core::Result<Devices, driver::DeviceError> {
	return driver::load().and_then([](driver::Driver const& driver) {
		return Devices::open(driver);
	});
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.device:devices;

import std;
import engine.core;
import engine.driver;
import engine.prims;
import engine.ptx;

namespace engine::device {

/** A card's tensor-parallel rank, which is also its CUDA ordinal. */
export enum class Rank : std::uint8_t {
	Zero,
	One,
};

export inline constexpr std::array<Rank, 2> ranks = {Rank::Zero, Rank::One};

export [[nodiscard]] constexpr auto peer(Rank rank) -> Rank {
	return rank == Rank::Zero ? Rank::One : Rank::Zero;
}

[[nodiscard]] constexpr auto ordinal(Rank rank) -> driver::CUdevice {
	return std::to_underlying(rank);
}

/**
 * The machine: two RTX 3090s, bridged by all four nvlink links. Each card's compute capability is the target the
 * embedded images declare and its SM count is the sm_86 machine the kernels' budgets are checked against; this
 * module adds only what nothing else knows: how many cards, how many links, how much memory.
 */
export inline constexpr std::size_t card_count = ranks.size();
export inline constexpr std::uint32_t nvlink_links = 4;

/**
 * The band a 3090's reported memory falls in. The driver reports the 24 GiB less a slice it keeps for itself, whose size
 * varies by driver release; a total outside the band is another card.
 */
export inline constexpr std::uint64_t memory_floor = std::uint64_t{22} << 30U;
export inline constexpr std::uint64_t memory_ceiling = std::uint64_t{24} << 30U;

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
	if (card.memory_bytes < memory_floor || card.memory_bytes > memory_ceiling) {
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
[[nodiscard]] auto check_links(driver::ManagementSession const& management, Rank rank, driver::PciAddress card, driver::PciAddress peer_pci)
    -> core::Result<void, driver::DeviceError> {
	for (auto link = std::uint32_t{0}; link < nvlink_links; ++link) {
		auto const reading = management.link(card, link);
		if (!reading) {
			return std::unexpected(reading.error());
		}
		if (auto const fault = std::visit(LinkVerdict{.peer = peer_pci}, *reading)) {
			return refuse(driver::NotNvlink{.ordinal = ordinal(rank), .link = link, .fault = *fault});
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
		if (census->cards.size() != card_count) {
			return refuse(driver::CardCount{.found = core::narrow<int>(census->cards.size())});
		}
		for (auto const& card : census->cards) {
			if (auto const wrong = refusal(card)) {
				return refuse(*wrong);
			}
		}
		auto const pci = [&](Rank rank) {
			return census->cards[std::to_underlying(rank)].pci;
		};
		for (auto const rank : ranks) {
			auto const reaches = driver::can_access_peer(driver.cuda, ordinal(rank), ordinal(peer(rank)));
			if (!reaches) {
				return std::unexpected(reaches.error());
			}
			if (!*reaches) {
				return refuse(driver::PeerUnreachable{.from = ordinal(rank), .to = ordinal(peer(rank))});
			}
		}
		auto management = driver::ManagementSession::open(driver.management);
		if (!management) {
			return std::unexpected(management.error());
		}
		for (auto const rank : ranks) {
			if (auto linked = check_links(*management, rank, pci(rank), pci(peer(rank))); !linked) {
				return std::unexpected(linked.error());
			}
		}
		auto zero = driver::Context::retain(driver.cuda, ordinal(Rank::Zero));
		auto one = zero.and_then([&](auto const&) {
			return driver::Context::retain(driver.cuda, ordinal(Rank::One));
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
		    std::move(*management),
		    {
		        Card{.context = std::move(*zero), .pci = pci(Rank::Zero)},
		        Card{.context = std::move(*one), .pci = pci(Rank::One)},
		    },
		};
	}

	[[nodiscard]] auto context(Rank rank) const -> driver::Context const& {
		return cards_[std::to_underlying(rank)].context;
	}

	/**
	 * The card's memory clock and performance state now. Bring-up sets the clocks; the engine reads them to refuse a
	 * card that came up in an unexpected state (GeForce Ampere runs CUDA in P2, below its P0 memory clock).
	 */
	[[nodiscard]] auto memory_clock(Rank rank) const -> core::Result<driver::MemoryClock, driver::DeviceError> {
		return management_.memory_clock(cards_[std::to_underlying(rank)].pci);
	}

private:
	struct Card {
		driver::Context context;
		driver::PciAddress pci;
	};

	Devices(driver::ManagementSession management, std::array<Card, card_count> cards)
	    : management_{std::move(management)}, cards_{std::move(cards)} {}

	driver::ManagementSession management_;
	std::array<Card, card_count> cards_;
};

/** Loads the vendor libraries and opens the machine: the product's one way to the cards. */
export [[nodiscard]] auto start() -> core::Result<Devices, driver::DeviceError> {
	return driver::load().and_then([](driver::Driver const& driver) {
		return Devices::open(driver);
	});
}

}

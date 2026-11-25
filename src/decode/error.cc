// SPDX-License-Identifier: AGPL-3.0-only
export module engine.decode:error;

import std;
import engine.core;
import engine.device;
import engine.image;

namespace engine::decode {

/** Why a batch or its masks are refused before anything reaches the cards. */
export enum class Refusal : std::uint8_t {
	/** A batch with no rows: there is no cycle to run. */
	NoRows,
	/** More rows than the row budget holds at the batch's depth. */
	OverBudget,
	/** Rows not in ascending slot order, or a slot twice. */
	SlotOrder,
	/** A row whose cycle would write kv past the longest sequence. */
	PastContext,
	/** Masks for a different number of verify rows than the batch has. */
	MaskCount,
	/** A verify row whose mask allows no token: nothing could be sampled. */
	EmptyMask,
	/** Masks published for a ticket other than the one begun. */
	WrongTicket,
};

export struct [[nodiscard]] Refused {
	Refusal reason;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (reason) {
		case Refusal::NoRows:
		case Refusal::OverBudget:
		case Refusal::SlotOrder:
		case Refusal::PastContext:
		case Refusal::MaskCount:
		case Refusal::EmptyMask:
		case Refusal::WrongTicket:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(Refused const&) const -> bool = default;
};

/** What the host was waiting for when its deadline passed. */
export enum class Waiting : std::uint8_t {
	/** A free ring slot to submit into. */
	Submit,
	/** The cards' answer to a ticket. */
	Answer,
};

/** The deadline passed first; the ticket may still be answered later. */
export struct [[nodiscard]] TimedOut {
	Waiting waiting;
	std::uint64_t ticket;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator==(TimedOut const&) const -> bool = default;
};

/**
 * The two cards answered one ticket differently. Both compute the same verdict from the same merged candidates, so
 * this is a broken exchange or a broken card, never a race the engine allows.
 */
export struct [[nodiscard]] CardsDisagree {
	std::uint64_t ticket;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(CardsDisagree const&) const -> bool = default;
};

/** A card answered a cycle with the exit acknowledgement, or the loop was asked to exit twice: the loop is gone. */
export struct [[nodiscard]] LoopGone {
	std::uint64_t ticket;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(LoopGone const&) const -> bool = default;
};

/** Every way decoding on the cards fails. */
export struct [[nodiscard]] DecodeError {
	std::variant<device::DeviceError, image::ImageError, Refused, TimedOut, CardsDisagree, LoopGone> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(DecodeError const&) const -> bool = default;
};

[[nodiscard]] auto fail(auto reason) -> std::unexpected<DecodeError> {
	return std::unexpected(DecodeError{.reason = std::move(reason)});
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.decode:error;

import std;
import engine.core;
import engine.device;
import engine.image;
import engine.model;

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
	/** The ticket whose masks were refused; none for a batch refused before it was numbered. */
	std::optional<std::uint64_t> ticket;

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

/** What the bring-up was waiting for when its deadline passed. */
export enum class Waiting : std::uint8_t {
	/** The copy of the card's image to the card. */
	Weights,
	/** The copy of the card's lm_head row-to-token table. */
	RowTokens,
	/** The zeroing of the card's control block. */
	Control,
	/** The rope table's fill. */
	Rotations,
};

/** The bring-up's deadline passed first, waiting on `card` for `waiting`. */
export struct [[nodiscard]] TimedOut {
	model::Card card;
	Waiting waiting;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator==(TimedOut const&) const -> bool = default;
};

/** Which of the three images an image failure is about. */
export enum class ImageOf : std::uint8_t {
	Host,
	CardZero,
	CardOne,
};

/** An image could not be opened: its file could not be mapped, or its bytes were refused. */
export struct [[nodiscard]] Unloadable {
	ImageOf image;
	image::ImageError error;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return error.retry();
	}

	[[nodiscard]] constexpr auto operator==(Unloadable const&) const -> bool = default;
};

/**
 * A shutdown gave up on the loops at its deadline: `ticket` (the begun one or an exit) was never answered, or, with
 * none, every ticket was answered but a kernel never finished. The cards are in no known state, and nothing more is sent
 * to them.
 */
export struct [[nodiscard]] Abandoned {
	std::optional<std::uint64_t> ticket;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Abandoned const&) const -> bool = default;
};

/** What of a slot's outcome the cards answered differently. */
export enum class Disagreement : std::uint8_t {
	/** How many drafts the slot accepted. */
	Accepted,
	/** A token the slot committed. */
	Committed,
	/** A draft for the slot's next cycle. */
	Drafted,
};

/**
 * The two cards answered one ticket differently: for `slot`, its `what` at index `at` (0 for `Accepted`), card zero
 * saying `zero` and card one `one`; the first difference, in row order. Both cards compute the same verdict from the
 * same merged candidates, so this is a broken exchange or a broken card, never a race the engine allows.
 */
export struct [[nodiscard]] CardsDisagree {
	std::uint64_t ticket;
	std::uint32_t slot;
	Disagreement what;
	std::uint32_t at;
	std::uint32_t zero;
	std::uint32_t one;

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
	std::variant<device::DeviceError, Unloadable, Refused, TimedOut, Abandoned, CardsDisagree, LoopGone> reason;

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

[[nodiscard]] constexpr auto name(Refusal refusal) -> std::string_view {
	switch (refusal) {
	case Refusal::NoRows:
		return "it has no rows";
	case Refusal::OverBudget:
		return "it has more rows than the row budget holds at its depth";
	case Refusal::SlotOrder:
		return "its rows are not in ascending slot order, or name a slot twice";
	case Refusal::PastContext:
		return "a row would write kv past the longest sequence";
	case Refusal::MaskCount:
		return "the masks are not one per verify row";
	case Refusal::EmptyMask:
		return "a verify row's mask allows no token";
	case Refusal::WrongTicket:
		return "the masks are for a ticket other than the one begun";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto name(Waiting waiting) -> std::string_view {
	switch (waiting) {
	case Waiting::Weights:
		return "the copy of its image";
	case Waiting::RowTokens:
		return "the copy of its lm_head row-to-token table";
	case Waiting::Control:
		return "the zeroing of its control block";
	case Waiting::Rotations:
		return "the fill of the rope table";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto name(ImageOf image) -> std::string_view {
	switch (image) {
	case ImageOf::Host:
		return "the host image";
	case ImageOf::CardZero:
		return "card 0's image";
	case ImageOf::CardOne:
		return "card 1's image";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto name(Disagreement what) -> std::string_view {
	switch (what) {
	case Disagreement::Accepted:
		return "drafts accepted";
	case Disagreement::Committed:
		return "committed token";
	case Disagreement::Drafted:
		return "next draft";
	}
	std::unreachable();
}

struct Describe {
	[[nodiscard]] auto operator()(device::DeviceError const& error) const -> std::string {
		return std::format("a card failed: {}", device::describe(error));
	}

	[[nodiscard]] auto operator()(Unloadable const& unloadable) const -> std::string {
		return std::format("{} {}", name(unloadable.image), image::describe(unloadable.error));
	}

	[[nodiscard]] auto operator()(Refused const& refused) const -> std::string {
		if (refused.ticket) {
			return std::format("the masks for ticket {} are refused: {}", *refused.ticket, name(refused.reason));
		}
		return std::format("a batch is refused before it reached the cards: {}", name(refused.reason));
	}

	[[nodiscard]] auto operator()(TimedOut const& timed_out) const -> std::string {
		return std::format("the bring-up's deadline passed on card {} waiting for {}", std::to_underlying(timed_out.card), name(timed_out.waiting));
	}

	[[nodiscard]] auto operator()(Abandoned const& abandoned) const -> std::string {
		if (abandoned.ticket) {
			return std::format("the decode loops were abandoned at the shutdown's deadline: ticket {} was never answered", *abandoned.ticket);
		}
		return "the decode loops were abandoned at the shutdown's deadline: every ticket was answered but a kernel never finished";
	}

	[[nodiscard]] auto operator()(CardsDisagree const& disagree) const -> std::string {
		return std::format("the cards disagree on ticket {}: slot {}'s {} {} is {} on card 0 and {} on card 1", disagree.ticket, disagree.slot, name(disagree.what),
		                   disagree.at, disagree.zero, disagree.one);
	}

	[[nodiscard]] auto operator()(LoopGone const& gone) const -> std::string {
		return std::format("a decode loop is gone: ticket {} was answered by an exit, or an exit by a result", gone.ticket);
	}
};

/** One sentence for an operator naming what failed and every fact it failed on: the one description of a decode error. */
export [[nodiscard]] auto describe(DecodeError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

}

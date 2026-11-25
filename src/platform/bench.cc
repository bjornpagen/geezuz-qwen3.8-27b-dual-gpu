// SPDX-License-Identifier: AGPL-3.0-only
//
// The soak on the cards: each card's baked image opened, verified and copied whole into its memory, the region the
// engine writes allocated beside it, the integrity kernels loaded, and two streams per card, one per lane, so the image
// hash and the pattern run at once and the memory stays hot under both. The catalogs are a parameter only so the tests
// can stand small images in for the real ones.
export module engine.platform:bench;

import std;
import engine.core;
import engine.device;
import engine.image;
import engine.model;
import engine.sha256;
import engine.soak;
import :soak;
import :telemetry;

namespace engine::platform {

/** The two card images a soak loads, by catalog. */
export template<class C>
concept CardCatalogs = image::Catalog<typename C::CardZero> && image::Catalog<typename C::CardOne>;

/** The model's card images, as the baker writes them. */
export struct ModelCards {
	using CardZero = image::CardCatalog<model::Card::Zero>;
	using CardOne = image::CardCatalog<model::Card::One>;
};

/** Where the two card images are. */
export struct ImagePaths {
	std::filesystem::path card_zero;
	std::filesystem::path card_one;
};

/** A card image's copy to its card did not finish by the deadline. */
export struct UploadStalled {
	model::Card card;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(UploadStalled const&) const -> bool = default;
};

/** Why the soak's cards could not be set up. */
export struct [[nodiscard]] SetupError {
	std::variant<device::DeviceError, image::ImageError, UploadStalled> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(SetupError const&) const -> bool = default;
};

template<class T>
[[nodiscard]] auto refuse(T reason) -> std::unexpected<SetupError> {
	return std::unexpected(SetupError{.reason = std::move(reason)});
}

/** An opened image as the host bytes a copy reads: its payload, which is copied to its card whole. */
template<image::Catalog C>
class Payload {
public:
	explicit Payload(image::Image<C> opened) : image_{std::move(opened)} {}

	[[nodiscard]] auto bytes() const -> std::span<std::byte const> {
		return image_.payload();
	}

private:
	image::Image<C> image_;
};

/**
 * The kernels' bounds hold for an image only when its payload is whole 32-bit words: the finishing launch reads the
 * tail in words, and a part word would read past the payload's end.
 */
template<image::Catalog C>
inline constexpr bool hashable = image::payload_bytes<C> > 0 && image::payload_bytes<C> % 4 == 0;

/** The integrity kernels, loaded on one card. */
[[nodiscard]] auto load_program(device::Context const& context) -> core::Result<soak::Program, device::DeviceError> {
	return context.load<soak::AbsorbKernel, soak::FinishKernel, soak::WriteKernel, soak::CheckKernel>();
}

/** What the pattern needs on one card: its region, its tallies, the kernels, and a stream and an event of its own. */
struct PatternCard {
	device::Stream stream;
	device::Event done;
	soak::Region region;
	soak::Tallies tallies;
	soak::Program program;
};

[[nodiscard]] auto pattern_card(device::Context const& context) -> core::Result<PatternCard, device::DeviceError> {
	auto stream = context.create_stream();
	auto done = stream.and_then([&](auto const&) { return context.create_event(); });
	auto region = done.and_then([&](auto const&) { return context.allocate<soak::Chunk, soak::region_chunks>(); });
	auto tallies = region.and_then([&](auto const&) { return context.allocate_mapped<soak::Tally, soak::tallies>(); });
	auto program = tallies.and_then([&](auto const&) { return load_program(context); });
	if (!program) {
		return std::unexpected(program.error());
	}
	return PatternCard{
	    .stream = std::move(*stream),
	    .done = std::move(*done),
	    .region = std::move(*region),
	    .tallies = std::move(*tallies),
	    .program = std::move(*program),
	};
}

/** Launches round `round`'s pattern write on one card and records its completion. */
[[nodiscard]] auto write_round(device::Context const& context, PatternCard const& card, std::uint32_t round) -> core::Result<void, device::DeviceError> {
	return context.launch<soak::WriteKernel>(card.program, card.stream, soak::WriteKernel::params(card.region, round)).and_then([&] {
		return card.done.record(card.stream);
	});
}

/** Launches round `round`'s pattern check on one card and records its completion. */
[[nodiscard]] auto check_round(device::Context const& context, PatternCard const& card, std::uint32_t round) -> core::Result<void, device::DeviceError> {
	return context.launch<soak::CheckKernel>(card.program, card.stream, soak::CheckKernel::params(card.region, card.tallies, round)).and_then([&] {
		return card.done.record(card.stream);
	});
}

/** One card's soak: its weight image in its memory and the image lane's stream, event and state, beside its pattern. */
template<image::Catalog C>
struct Card {
	static_assert(hashable<C>, "a card image's payload is whole 32-bit words");

	device::Stream stream;
	device::Event done;
	soak::Weights<image::payload_bytes<C>> weights;
	soak::StateCell state;
	sha256::Digest expected;
	PatternCard pattern;
};

/** Opens and verifies card `card`'s image, takes its digest, copies it to the card, and sets up both lanes. */
template<image::Catalog C>
[[nodiscard]] auto load_card(device::Context const& context, std::filesystem::path const& path, image::Checkpoint const& checkpoint, model::Card card,
                             std::chrono::steady_clock::time_point deadline) -> core::Result<Card<C>, SetupError> {
	auto opened = image::Image<C>::open(path, checkpoint);
	if (!opened) {
		return refuse(opened.error());
	}
	// Opening checked the payload against the digest the header records, so this is that digest.
	auto const expected = sha256::digest(opened->payload());
	auto stream = context.create_stream();
	auto done = stream.and_then([&](auto const&) { return context.create_event(); });
	auto state = done.and_then([&](auto const&) { return context.allocate_mapped<soak::HashState, 1>(); });
	auto pinned = state.and_then([&](auto const&) { return context.pin(Payload<C>{std::move(*opened)}); });
	auto weights = pinned.and_then([&](auto const&) { return context.allocate<std::byte, image::payload_bytes<C>>(); });
	if (!weights) {
		return refuse(weights.error());
	}
	auto transfer = context.upload(*stream, std::move(*weights), std::move(*pinned), 0);
	if (transfer.wait_until(deadline) == device::EventState::Pending) {
		// The copy still reads its source, so the transfer may not be dropped before it ends.
		core::invariant(transfer.wait_until(std::chrono::steady_clock::time_point::max()) == device::EventState::Complete);
		return refuse(UploadStalled{.card = card});
	}
	auto delivered = transfer.try_finish();
	core::invariant(delivered.has_value());
	if (!delivered->copied) {
		return refuse(delivered->copied.error());
	}
	auto pattern = pattern_card(context);
	if (!pattern) {
		return refuse(pattern.error());
	}
	return Card<C>{
	    .stream = std::move(*stream),
	    .done = std::move(*done),
	    .weights = std::move(delivered->destination),
	    .state = std::move(*state),
	    .expected = expected,
	    .pattern = std::move(*pattern),
	};
}

/**
 * Both cards set up for the soak, and the bench the soak's loop runs on: it owns the machine, so everything it
 * allocated dies before the contexts it came from.
 */
export template<CardCatalogs C>
class Cards {
public:
	[[nodiscard]] static auto open(device::Devices devices, ImagePaths const& paths, image::Checkpoint const& checkpoint,
	                               std::chrono::steady_clock::time_point deadline) -> core::Result<Cards, SetupError> {
		auto zero = load_card<typename C::CardZero>(devices.context(model::Card::Zero), paths.card_zero, checkpoint, model::Card::Zero, deadline);
		if (!zero) {
			return std::unexpected(zero.error());
		}
		auto one = load_card<typename C::CardOne>(devices.context(model::Card::One), paths.card_one, checkpoint, model::Card::One, deadline);
		if (!one) {
			return std::unexpected(one.error());
		}
		return Cards{std::move(devices), std::move(*zero), std::move(*one)};
	}

	auto begin_pass(model::Card card) -> void {
		on(card, [](auto& held) { soak::begin_pass(held.state); });
	}

	[[nodiscard]] auto absorb(model::Card card, soak::Slice const& slice) -> core::Result<void, device::DeviceError> {
		auto const& context = devices_.context(card);
		return seen(card, [&](auto const& held) {
			return context.template launch<soak::AbsorbKernel>(held.pattern.program, held.stream, soak::AbsorbKernel::params(held.state, held.weights, slice))
			    .and_then([&] { return held.done.record(held.stream); });
		});
	}

	[[nodiscard]] auto finish(model::Card card) -> core::Result<void, device::DeviceError> {
		auto const& context = devices_.context(card);
		return seen(card, [&](auto const& held) {
			return context.template launch<soak::FinishKernel>(held.pattern.program, held.stream, soak::FinishKernel::params(held.state, held.weights))
			    .and_then([&] { return held.done.record(held.stream); });
		});
	}

	[[nodiscard]] auto write(model::Card card, std::uint32_t round) -> core::Result<void, device::DeviceError> {
		return seen(card, [&](auto const& held) { return write_round(devices_.context(card), held.pattern, round); });
	}

	[[nodiscard]] auto check(model::Card card, std::uint32_t round) -> core::Result<void, device::DeviceError> {
		return seen(card, [&](auto const& held) { return check_round(devices_.context(card), held.pattern, round); });
	}

	[[nodiscard]] auto state(model::Card card, Lane lane) const -> core::Result<device::EventState, device::DeviceError> {
		return seen(card, [&](auto const& held) {
			switch (lane) {
			case Lane::Image:
				return held.done.query();
			case Lane::Pattern:
				return held.pattern.done.query();
			}
			std::unreachable();
		});
	}

	[[nodiscard]] auto pass(model::Card card) const -> soak::Pass {
		return card == model::Card::Zero ? soak::pass_of(image::payload_bytes<typename C::CardZero>)
		                                 : soak::pass_of(image::payload_bytes<typename C::CardOne>);
	}

	[[nodiscard]] auto expected(model::Card card) const -> sha256::Digest {
		return seen(card, [](auto const& held) { return held.expected; });
	}

	[[nodiscard]] auto digest(model::Card card) const -> sha256::Digest {
		return seen(card, [](auto const& held) { return soak::digest_of(held.state); });
	}

	[[nodiscard]] auto tallies(model::Card card) const -> std::span<soak::Tally const> {
		return seen(card, [](auto const& held) { return std::span<soak::Tally const>{held.pattern.tallies.host()}; });
	}

	[[nodiscard]] auto telemetry(model::Card card) const -> core::Result<Telemetry, device::DeviceError> {
		return read(devices_, card);
	}

private:
	Cards(device::Devices devices, Card<typename C::CardZero> zero, Card<typename C::CardOne> one)
	    : devices_{std::move(devices)}, zero_{std::move(zero)}, one_{std::move(one)} {}

	/** Calls `act` with card `card`'s set-up, whichever catalog it is of, to read it. */
	template<class Act>
	[[nodiscard]] auto seen(model::Card card, Act act) const -> decltype(act(std::declval<Card<typename C::CardZero> const&>())) {
		return card == model::Card::Zero ? act(zero_) : act(one_);
	}

	/** Calls `act` with card `card`'s set-up, whichever catalog it is of, to change it. */
	template<class Act>
	auto on(model::Card card, Act act) -> decltype(act(std::declval<Card<typename C::CardZero>&>())) {
		return card == model::Card::Zero ? act(zero_) : act(one_);
	}

	device::Devices devices_;
	Card<typename C::CardZero> zero_;
	Card<typename C::CardOne> one_;
};

static_assert(Bench<Cards<ModelCards>>, "the cards are a bench the soak runs on");

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.core;

import std;
import engine.fail;

namespace engine::core {

/**
 * The decision every caller of a fallible operation makes: could the same operation, retried later with nothing
 * changed by the caller, plausibly succeed? Contention, peer disconnects and resource pressure are transient;
 * protocol violations, misuse, corruption and configuration errors are permanent. When unsure, permanent: giving up
 * too early is a bug report, retrying a permanent failure forever is an outage.
 */
export enum class Retry : std::uint8_t {
	Permanent,
	Transient,
};

/** An error type a caller can decide on: it classifies retryability. */
export template<class E>
concept Classified = std::regular<E> && requires(E const& error) {
	{ error.retry() } -> std::same_as<Retry>;
};

/** Recoverable failure. Only a classified error can be the failure of a result. */
export template<class T, Classified E>
using Result = std::expected<T, E>;

/** Impossible programmer state: reports the violated site and aborts, in every build. */
export using fail::invariant;

/** The value does not fit the target integer type. */
export struct [[nodiscard]] OutOfRange {
	[[nodiscard]] constexpr auto retry() const -> Retry {
		return Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(OutOfRange const&) const -> bool = default;
};

/** Converts an integer whose range the program guarantees; a value that does not fit is an invariant violation. */
export template<std::integral To, std::integral From>
[[nodiscard]] constexpr auto narrow(From value) -> To {
	invariant(std::in_range<To>(value));
	return static_cast<To>(value);
}

/** Converts an integer that came from outside the program; a value that does not fit is the caller's decision. */
export template<std::integral To, std::integral From>
[[nodiscard]] constexpr auto try_narrow(From value) -> Result<To, OutOfRange> {
	if (!std::in_range<To>(value)) {
		return std::unexpected(OutOfRange{});
	}
	return static_cast<To>(value);
}

/** The variant's `T` alternative as a borrow, or nullopt when it holds another: `std::get_if` without a pointer. */
export template<class T, class... Alternatives>
[[nodiscard]] constexpr auto alternative(std::variant<Alternatives...>& value) -> std::optional<T&> {
	if (!std::holds_alternative<T>(value)) {
		return std::nullopt;
	}
	return std::get<T>(value);
}

export template<class T, class... Alternatives>
[[nodiscard]] constexpr auto alternative(std::variant<Alternatives...> const& value) -> std::optional<T const&> {
	if (!std::holds_alternative<T>(value)) {
		return std::nullopt;
	}
	return std::get<T>(value);
}

}

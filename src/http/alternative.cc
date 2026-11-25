// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:alternative;

import std;

namespace engine::http {

/** The variant's `T` alternative as a borrow, or nullopt: `get_if` without a pointer. */
template<class T, class... Alternatives>
[[nodiscard]] constexpr auto alternative(std::variant<Alternatives...>& value) -> std::optional<T&> {
	if (!std::holds_alternative<T>(value)) {
		return std::nullopt;
	}
	return std::get<T>(value);
}

template<class T, class... Alternatives>
[[nodiscard]] constexpr auto alternative(std::variant<Alternatives...> const& value) -> std::optional<T const&> {
	if (!std::holds_alternative<T>(value)) {
		return std::nullopt;
	}
	return std::get<T>(value);
}

}

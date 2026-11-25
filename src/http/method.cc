// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:method;

import std;

namespace engine::http {

/** The request methods some route takes; every other method is one no route takes. */
export enum class Method : std::uint8_t {
	Get,
	Post,
	Delete,
};

/** Every method, in the order an Allow field lists them. */
constexpr auto methods = std::array{Method::Get, Method::Post, Method::Delete};

/** The method's token on the wire: methods are case-sensitive. */
[[nodiscard]] constexpr auto method_name(Method method) -> std::string_view {
	switch (method) {
	case Method::Get:
		return "GET";
	case Method::Post:
		return "POST";
	case Method::Delete:
		return "DELETE";
	}
	return "GET";
}

/** The method a request line names; nullopt for a token no route takes. */
[[nodiscard]] constexpr auto method_of(std::string_view token) -> std::optional<Method> {
	auto const found = std::ranges::find(methods, token, method_name);
	if (found == methods.end()) {
		return std::nullopt;
	}
	return *found;
}

/** A set of methods: those one path takes, which a 405's Allow field lists. */
export struct Methods {
	std::bitset<methods.size()> members;

	[[nodiscard]] constexpr auto contains(Method method) const -> bool {
		return members.test(std::to_underlying(method));
	}

	[[nodiscard]] constexpr auto with(Method method) const -> Methods {
		auto out = *this;
		out.members.set(std::to_underlying(method));
		return out;
	}

	[[nodiscard]] constexpr auto operator==(Methods const&) const -> bool = default;
};

/** An Allow field's value: the set's methods, comma-separated. */
[[nodiscard]] auto allow_value(Methods allowed) -> std::string {
	auto text = std::string{};
	for (auto const method : methods | std::views::filter([allowed](Method method) {
		                         return allowed.contains(method);
	                         })) {
		text += text.empty() ? "" : ", ";
		text += method_name(method);
	}
	return text;
}

}

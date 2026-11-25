// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:target;

import std;
import engine.core;
import :error;
import :limits;
import :method;
import :request;

namespace engine::http {

/** The client sent `Expect: 100-continue` and waits for an interim response before its body. */
enum class Expectation : std::uint8_t {
	None,
	Continue,
};

/** POST /v1/responses before its body is read: how long the body is, and whether the client waits to send it. */
struct Creating {
	std::size_t body_length;
	Expectation expectation;
};

/** A route that takes no body: whole once its head is read. */
using Bodiless = std::variant<RetrieveResponse, DeleteResponse, ListInputItems, CancelResponse, ListModels, Health, ServerStatus>;

/** What a request head routes to: a response to create once its body arrives, or a route already whole. */
using Routed = std::variant<Creating, Bodiless>;

enum class Target : std::uint8_t {
	CreateResponse,
	RetrieveResponse,
	DeleteResponse,
	ListInputItems,
	CancelResponse,
	ListModels,
	Health,
	Status,
};

/** What a route reads besides its path; it refuses the rest. */
enum class Reads : std::uint8_t {
	Nothing,
	Query,
	Body,
};

/** One route: a path pattern, where `{id}` stands for one segment naming a resource, and a method. */
struct Mapping {
	std::string_view path;
	Method method;
	Target target;
	Reads reads;
};

constexpr std::string_view id_segment = "{id}";

constexpr auto mappings = std::array{
    Mapping{.path = "/v1/responses", .method = Method::Post, .target = Target::CreateResponse, .reads = Reads::Body},
    Mapping{.path = "/v1/responses/{id}", .method = Method::Get, .target = Target::RetrieveResponse, .reads = Reads::Query},
    Mapping{.path = "/v1/responses/{id}", .method = Method::Delete, .target = Target::DeleteResponse, .reads = Reads::Nothing},
    Mapping{.path = "/v1/responses/{id}/input_items", .method = Method::Get, .target = Target::ListInputItems, .reads = Reads::Query},
    Mapping{.path = "/v1/responses/{id}/cancel", .method = Method::Post, .target = Target::CancelResponse, .reads = Reads::Nothing},
    Mapping{.path = "/v1/models", .method = Method::Get, .target = Target::ListModels, .reads = Reads::Nothing},
    Mapping{.path = "/health", .method = Method::Get, .target = Target::Health, .reads = Reads::Nothing},
    Mapping{.path = "/status", .method = Method::Get, .target = Target::Status, .reads = Reads::Nothing},
};

/** A path a pattern matches, with the segment its `{id}` stood for when it has one. */
struct Match {
	std::optional<std::string_view> id;
};

[[nodiscard]] auto match(std::string_view pattern, std::string_view path) -> std::optional<Match> {
	auto wanted = std::views::split(pattern, '/');
	auto segments = std::views::split(path, '/');
	if (std::ranges::distance(wanted) != std::ranges::distance(segments)) {
		return std::nullopt;
	}
	auto found = Match{.id = std::nullopt};
	for (auto const [want, segment] : std::views::zip(wanted, segments)) {
		if (std::string_view{want} == id_segment) {
			found.id = std::string_view{segment};
		} else if (std::string_view{want} != std::string_view{segment}) {
			return std::nullopt;
		}
	}
	return found;
}

[[nodiscard]] constexpr auto is_resource_id_char(char c) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

[[nodiscard]] auto resource_id(std::string_view segment) -> core::Result<ResourceId, HttpError> {
	if (segment.empty() || segment.size() > max_resource_id_bytes || !std::ranges::all_of(segment, is_resource_id_char)) {
		return std::unexpected(HttpError{HttpErrorKind::InvalidResourceId});
	}
	return ResourceId{.text = std::string{segment}};
}

/** The route a method and path name, and the id the path names when its pattern has one, already checked. */
struct Found {
	Target target;
	Reads reads;
	std::optional<ResourceId> id;
};

/**
 * The route for a method and path. A path some route has, named with a method none of its routes takes, is a 405
 * whose Allow lists the methods they take; `method` is nullopt for a token no route takes.
 */
[[nodiscard]] auto find_route(std::optional<Method> method, std::string_view path) -> core::Result<Found, HttpError> {
	auto allowed = Methods{};
	for (auto const& mapping : mappings) {
		auto const matched = match(mapping.path, path);
		if (!matched) {
			continue;
		}
		if (method != mapping.method) {
			allowed = allowed.with(mapping.method);
			continue;
		}
		if (!matched->id) {
			return Found{.target = mapping.target, .reads = mapping.reads, .id = std::nullopt};
		}
		auto id = resource_id(*matched->id);
		if (!id) {
			return std::unexpected(id.error());
		}
		return Found{.target = mapping.target, .reads = mapping.reads, .id = std::move(*id)};
	}
	// A path no route has takes no method.
	if (allowed.members.none()) {
		return std::unexpected(HttpError{HttpErrorKind::NotFound});
	}
	return std::unexpected(HttpError{WrongMethod{.allowed = allowed}});
}

[[nodiscard]] constexpr auto hex_digit(char c) -> std::optional<std::uint8_t> {
	if (c >= '0' && c <= '9') {
		return core::narrow<std::uint8_t>(c - '0');
	}
	if (c >= 'a' && c <= 'f') {
		return core::narrow<std::uint8_t>(c - 'a' + 10);
	}
	if (c >= 'A' && c <= 'F') {
		return core::narrow<std::uint8_t>(c - 'A' + 10);
	}
	return std::nullopt;
}

/** A query name or value as application/x-www-form-urlencoded spells it: '+' is a space, '%' and two hex digits a byte. */
[[nodiscard]] auto form_decoded(std::string_view text) -> core::Result<std::string, HttpError> {
	auto out = std::string{};
	out.reserve(text.size());
	while (!text.empty()) {
		if (text.front() != '%') {
			out += text.front() == '+' ? ' ' : text.front();
			text.remove_prefix(1);
			continue;
		}
		auto const high = text.size() >= 3 ? hex_digit(text[1]) : std::nullopt;
		auto const low = text.size() >= 3 ? hex_digit(text[2]) : std::nullopt;
		if (!high || !low) {
			return std::unexpected(HttpError{HttpErrorKind::Malformed});
		}
		out += std::bit_cast<char>(core::narrow<std::uint8_t>((unsigned{*high} << 4U) | unsigned{*low}));
		text.remove_prefix(3);
	}
	return out;
}

/** The query after a target's '?': '&'-separated parameters, each a name with an optional '=' and value. */
[[nodiscard]] auto parse_query(std::string_view text) -> core::Result<Query, HttpError> {
	auto query = Query{.parameters = {}};
	if (text.empty()) {
		return query;
	}
	query.parameters.reserve(max_query_parameters);
	for (auto const piece : std::views::split(text, '&')) {
		if (query.parameters.size() == max_query_parameters) {
			return std::unexpected(HttpError{HttpErrorKind::TooManyQueryParameters});
		}
		auto const parameter = std::string_view{piece};
		auto const equals = parameter.find('=');
		auto name = form_decoded(parameter.substr(0, equals));
		auto value = form_decoded(equals == std::string_view::npos ? std::string_view{} : parameter.substr(equals + 1));
		// An empty name (an empty piece between two '&' among them) names nothing a route could read.
		if (!name || !value || name->empty()) {
			return std::unexpected(HttpError{HttpErrorKind::Malformed});
		}
		query.parameters.push_back(QueryParameter{.name = std::move(*name), .value = std::move(*value)});
	}
	return query;
}

/** The route a request line names once the head's framing is known; MaxBody bounds the body of a response to create. */
template<std::size_t MaxBody>
[[nodiscard]] auto route(std::string_view method, std::string_view target, std::optional<std::size_t> length, Expectation expectation)
    -> core::Result<Routed, HttpError> {
	auto const query_start = target.find('?');
	auto found = find_route(method_of(method), target.substr(0, query_start));
	if (!found) {
		return std::unexpected(found.error());
	}
	auto query = parse_query(query_start == std::string_view::npos ? std::string_view{} : target.substr(query_start + 1));
	if (!query) {
		return std::unexpected(query.error());
	}
	if (found->reads != Reads::Query && !query->parameters.empty()) {
		return std::unexpected(HttpError{HttpErrorKind::QueryNotAllowed});
	}
	if (found->reads == Reads::Body && !length) {
		return std::unexpected(HttpError{HttpErrorKind::LengthRequired});
	}
	auto const body_length = length.value_or(0);
	if (found->reads != Reads::Body && body_length != 0) {
		return std::unexpected(HttpError{HttpErrorKind::BodyNotAllowed});
	}
	if (body_length > MaxBody) {
		return std::unexpected(HttpError{HttpErrorKind::BodyTooLarge});
	}

	auto const id = [&found] {
		// Every route whose type holds an id has `{id}` in its pattern, so the match captured one.
		core::invariant(found->id.has_value());
		return std::move(*found->id);
	};
	switch (found->target) {
	case Target::CreateResponse:
		return Creating{.body_length = body_length, .expectation = body_length != 0 ? expectation : Expectation::None};
	case Target::RetrieveResponse:
		return Bodiless{RetrieveResponse{.id = id(), .query = std::move(*query)}};
	case Target::DeleteResponse:
		return Bodiless{DeleteResponse{.id = id()}};
	case Target::ListInputItems:
		return Bodiless{ListInputItems{.id = id(), .query = std::move(*query)}};
	case Target::CancelResponse:
		return Bodiless{CancelResponse{.id = id()}};
	case Target::ListModels:
		return Bodiless{ListModels{}};
	case Target::Health:
		return Bodiless{Health{}};
	case Target::Status:
		return Bodiless{ServerStatus{}};
	}
	return Bodiless{Health{}};
}

}

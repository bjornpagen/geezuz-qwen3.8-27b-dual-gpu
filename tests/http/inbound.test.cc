// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.http;
import engine.http_test_client;
import engine.testing;

namespace http = engine::http;
namespace client = engine::http_test_client;

namespace {

/** The body bounds these tests instantiate the server with; the application uses its json parser's own. */
constexpr http::BodyBounds bounds{.bytes = std::size_t{1} << 20U, .padding = 64};

using Rng = std::mt19937_64;
using Route = http::Route<bounds>;

/** A request as lines, before rendering: defects are applied to the lines, so each stays one named change. */
struct Draft {
	std::string request_line;
	std::vector<std::string> fields;
	std::string body;
};

[[nodiscard]] auto render(Draft const& draft) -> std::string {
	auto text = draft.request_line + "\r\n";
	for (auto const& field : draft.fields) {
		text += field + "\r\n";
	}
	text += "\r\n";
	text += draft.body;
	return text;
}

/** What the assembler must produce for a draft: the oracle, known by construction. */
using Expected = std::variant<http::Assembled<bounds>, http::HttpError>;

struct Generated {
	Draft draft;
	Expected expected;
	/** The draft asks for 100 Continue on a nonempty body. */
	bool continues;
};

[[nodiscard]] auto pick(Rng& rng, std::size_t count) -> std::size_t {
	return static_cast<std::size_t>(rng() % count);
}

[[nodiscard]] auto chance(Rng& rng, std::size_t in) -> bool {
	return pick(rng, in) == 0;
}

[[nodiscard]] auto spell(Rng& rng, std::string_view name) -> std::string {
	auto spelled = std::string{name};
	switch (pick(rng, 3)) {
	case 0:
		std::ranges::transform(spelled, spelled.begin(), [](char c) {
			return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c;
		});
		break;
	case 1:
		std::ranges::transform(spelled, spelled.begin(), [](char c) {
			return c >= 'a' && c <= 'z' ? static_cast<char>(c - 32) : c;
		});
		break;
	default:
		break;
	}
	return spelled;
}

[[nodiscard]] auto ows(Rng& rng) -> std::string_view {
	constexpr auto spaces = std::array<std::string_view, 4>{"", " ", "  ", "\t "};
	return spaces[pick(rng, spaces.size())];
}

[[nodiscard]] auto field(Rng& rng, std::string_view name, std::string_view value) -> std::string {
	return std::format("{}:{}{}{}", spell(rng, name), ows(rng), value, ows(rng));
}

[[nodiscard]] auto text(Rng& rng, std::size_t length, std::string_view alphabet) -> std::string {
	auto out = std::string(length, ' ');
	std::ranges::generate(out, [&] {
		return alphabet[pick(rng, alphabet.size())];
	});
	return out;
}

[[nodiscard]] auto bytes(Rng& rng, std::size_t length) -> std::string {
	auto out = std::string(length, '\0');
	std::ranges::generate(out, [&rng] {
		return static_cast<char>(rng() & 0xffU);
	});
	return out;
}

constexpr std::string_view id_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:";
constexpr std::string_view resource_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";
constexpr std::string_view value_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 !\"#$%&'()*+,-./;<=>?@[\\]^_`{|}~";

/** The paths the server routes, as the protocol names them; `{id}` is where a response id goes. */
enum class Shape : std::uint8_t {
	Responses,
	Response,
	InputItems,
	Cancel,
	Models,
	Health,
	Status,
};

constexpr auto shapes =
    std::array{Shape::Responses, Shape::Response, Shape::InputItems, Shape::Cancel, Shape::Models, Shape::Health, Shape::Status};

[[nodiscard]] auto path_of(Shape shape, std::string_view id) -> std::string {
	switch (shape) {
	case Shape::Responses:
		return "/v1/responses";
	case Shape::Response:
		return std::format("/v1/responses/{}", id);
	case Shape::InputItems:
		return std::format("/v1/responses/{}/input_items", id);
	case Shape::Cancel:
		return std::format("/v1/responses/{}/cancel", id);
	case Shape::Models:
		return "/v1/models";
	case Shape::Health:
		return "/health";
	case Shape::Status:
		return "/status";
	}
	return "/";
}

/** The methods each path takes, written from the api's route list: the oracle for every 405's Allow. */
[[nodiscard]] auto methods_of(Shape shape) -> http::Methods {
	using enum http::Method;
	auto const none = http::Methods{};
	switch (shape) {
	case Shape::Responses:
	case Shape::Cancel:
		return none.with(Post);
	case Shape::Response:
		return none.with(Get).with(Delete);
	case Shape::InputItems:
	case Shape::Models:
	case Shape::Health:
	case Shape::Status:
		return none.with(Get);
	}
	return none;
}

[[nodiscard]] auto token(http::Method method) -> std::string_view {
	switch (method) {
	case http::Method::Get:
		return "GET";
	case http::Method::Post:
		return "POST";
	case http::Method::Delete:
		return "DELETE";
	}
	return "GET";
}

[[nodiscard]] auto reads_query(Shape shape) -> bool {
	return shape == Shape::Response || shape == Shape::InputItems;
}

/** The route a request for a shape and an allowed method must assemble to. */
[[nodiscard]] auto route_for(Shape shape, http::Method method, std::string_view id, http::Query query, std::string_view body) -> Route {
	auto const resource = http::ResourceId{.text = std::string{id}};
	switch (shape) {
	case Shape::Responses:
		return http::CreateResponse<bounds>{.body = http::Body<bounds>::copy_of(body)};
	case Shape::Response:
		if (method == http::Method::Delete) {
			return http::DeleteResponse{.id = resource};
		}
		return http::RetrieveResponse{.id = resource, .query = std::move(query)};
	case Shape::InputItems:
		return http::ListInputItems{.id = resource, .query = std::move(query)};
	case Shape::Cancel:
		return http::CancelResponse{.id = resource};
	case Shape::Models:
		return http::ListModels{};
	case Shape::Health:
		return http::Health{};
	case Shape::Status:
		return http::ServerStatus{};
	}
	return http::Health{};
}

/** Random parameters: names of any bytes but never empty, values of any bytes. */
[[nodiscard]] auto parameters(Rng& rng, std::size_t count) -> std::vector<client::Parameter> {
	auto out = std::vector<client::Parameter>{};
	for (auto index = std::size_t{0}; index < count; ++index) {
		out.push_back(client::Parameter{.name = bytes(rng, 1 + pick(rng, 8)), .value = bytes(rng, pick(rng, 12))});
	}
	return out;
}

[[nodiscard]] auto query_of(std::span<client::Parameter const> sent) -> http::Query {
	return http::Query{.parameters = sent | std::views::transform([](client::Parameter const& parameter) {
		                                 return http::QueryParameter{.name = parameter.name, .value = parameter.value};
	                                 }) |
	                                 std::ranges::to<std::vector>()};
}

/** The target's query part: nothing or a bare '?' for no parameters, else '?' and their spelling. */
[[nodiscard]] auto suffix(Rng& rng, std::span<client::Parameter const> sent) -> std::string {
	if (sent.empty()) {
		return chance(rng, 3) ? "?" : "";
	}
	return "?" + client::query_string(sent, rng);
}

/** A valid request of a random route, with random ids, queries, spellings, persistence, fillers and bodies. */
[[nodiscard]] auto valid(Rng& rng) -> Generated {
	auto const shape = shapes[pick(rng, shapes.size())];
	auto const allowed = methods_of(shape);
	auto method = http::Method::Get;
	do {
		method = std::array{http::Method::Get, http::Method::Post, http::Method::Delete}[pick(rng, 3)];
	} while (!allowed.contains(method));
	auto const id = text(rng, 1 + pick(rng, http::max_resource_id_bytes), resource_alphabet);
	auto const sent = reads_query(shape) && method == http::Method::Get ? parameters(rng, pick(rng, http::max_query_parameters + 1))
	                                                                    : std::vector<client::Parameter>{};
	auto draft = Draft{.request_line = std::format("{} {}{} HTTP/1.1", token(method), path_of(shape, id), suffix(rng, sent)),
	                   .fields = {field(rng, "Host", "localhost:8080")},
	                   .body = ""};
	auto continues = false;
	if (shape == Shape::Responses) {
		draft.body = bytes(rng, chance(rng, 3) ? 0 : pick(rng, 3000));
		draft.fields.push_back(field(rng, "Content-Length", std::format("{}{}", chance(rng, 5) ? "00" : "", draft.body.size())));
		draft.fields.push_back(field(rng, "Content-Type", "application/json"));
		if (chance(rng, 3)) {
			draft.fields.push_back(field(rng, "Expect", spell(rng, "100-continue")));
			continues = !draft.body.empty();
		}
	} else if (chance(rng, 4)) {
		draft.fields.push_back(field(rng, "Content-Length", "0"));
	}
	auto expected = http::Assembled<bounds>{.route = route_for(shape, method, id, query_of(sent), draft.body),
	                                        .client_id = std::nullopt,
	                                        .persistence = http::Persistence::KeepAlive};
	switch (pick(rng, 4)) {
	case 0: {
		auto const client_id = text(rng, 1 + pick(rng, http::max_request_id_bytes), id_alphabet);
		draft.fields.push_back(field(rng, "X-Request-Id", client_id));
		expected.client_id = http::RequestId::parse(client_id);
		break;
	}
	case 1: {
		// An insane id (too long, or a character outside the alphabet) is replaced, not refused.
		auto const client_id = chance(rng, 2) ? text(rng, http::max_request_id_bytes + 1, id_alphabet) : std::string{"has/slash"};
		draft.fields.push_back(field(rng, "X-Request-Id", client_id));
		break;
	}
	default:
		break;
	}
	switch (pick(rng, 4)) {
	case 0:
		draft.fields.push_back(field(rng, "Connection", spell(rng, "close")));
		expected.persistence = http::Persistence::Close;
		break;
	case 1:
		draft.fields.push_back(field(rng, "Connection", "keep-alive, " + spell(rng, "close")));
		expected.persistence = http::Persistence::Close;
		break;
	case 2:
		draft.fields.push_back(field(rng, "Connection", spell(rng, "keep-alive")));
		break;
	default:
		break;
	}
	for (auto filler = pick(rng, 6); filler > 0; --filler) {
		draft.fields.push_back(field(rng, std::format("X-Filler-{}", filler), text(rng, pick(rng, 40), value_alphabet)));
	}
	std::ranges::shuffle(draft.fields, rng);
	return Generated{.draft = std::move(draft), .expected = std::move(expected), .continues = continues};
}

[[nodiscard]] auto get_draft(Rng& rng, std::string_view target) -> Draft {
	return Draft{.request_line = std::format("GET {} HTTP/1.1", target), .fields = {field(rng, "Host", "h")}, .body = ""};
}

[[nodiscard]] auto create_draft(Rng& rng) -> Draft {
	return Draft{.request_line = "POST /v1/responses HTTP/1.1",
	             .fields = {field(rng, "Host", "h"), field(rng, "Content-Length", "2")},
	             .body = "{}"};
}

/** A query of `count` parameters that are valid on their own. */
[[nodiscard]] auto valid_query(Rng& rng, std::size_t count) -> std::string {
	return client::query_string(parameters(rng, count), rng);
}

/** Query text holding one malformed piece among valid ones: a broken escape, or a parameter without a name. */
[[nodiscard]] auto malformed_query(Rng& rng) -> std::string {
	constexpr auto broken = std::array<std::string_view, 9>{"a=%", "a=%4", "a=%G1", "a=%1g", "%=1", "=1", "", "=", "b%zz=1"};
	auto pieces = std::vector<std::string>{std::string{broken[pick(rng, broken.size())]}};
	for (auto extra = pick(rng, 3); extra > 0; --extra) {
		pieces.push_back(valid_query(rng, 1));
	}
	std::ranges::shuffle(pieces, rng);
	auto out = std::string{};
	for (auto const& piece : pieces) {
		out += out.empty() ? "" : "&";
		out += piece;
	}
	// A lone empty piece is an empty query, which is valid: give it a neighbour.
	return out.empty() ? "&" : out;
}

/** A request with exactly one named defect, and the refusal it must draw. */
[[nodiscard]] auto defective(Rng& rng) -> Generated {
	using enum http::HttpErrorKind;
	auto draft = get_draft(rng, "/health");
	auto error = http::HttpError{Malformed};
	auto const id = text(rng, 1 + pick(rng, 40), resource_alphabet);
	switch (pick(rng, 36)) {
	case 0:
		draft.request_line = "G@T /health HTTP/1.1";
		break;
	case 1:
		draft.request_line = "GET  /health HTTP/1.1";
		break;
	case 2:
		draft.request_line = "GET /health";
		break;
	case 3:
		draft.fields.emplace_back("NoColonHere");
		break;
	case 4:
		draft.fields.emplace_back("X-Bad : value");
		break;
	case 5:
		draft.fields.emplace_back(" folded: continuation");
		break;
	case 6:
		draft.fields.emplace_back("X-Bad: a\x01z");
		break;
	case 7:
		draft.fields.emplace_back("X-Bad: a\nInjected: yes");
		break;
	case 8:
		draft.request_line = chance(rng, 2) ? "GET /health HTTP/1.0" : "GET /health HTTP/2.0";
		error = http::HttpError{UnsupportedVersion};
		break;
	case 9:
		draft.fields.clear();
		error = http::HttpError{HostRequired};
		break;
	case 10:
		draft.fields.push_back(field(rng, "Host", "second"));
		error = http::HttpError{HostRequired};
		break;
	case 11:
		draft = create_draft(rng);
		draft.fields.push_back(field(rng, "Transfer-Encoding", "chunked"));
		error = http::HttpError{AmbiguousFraming};
		break;
	case 12:
		draft = create_draft(rng);
		draft.fields.pop_back();
		draft.fields.push_back(field(rng, "Transfer-Encoding", spell(rng, "chunked")));
		error = http::HttpError{ChunkedUpload};
		break;
	case 13:
		draft = create_draft(rng);
		draft.fields.pop_back();
		draft.fields.push_back(field(rng, "Transfer-Encoding", "chunked, gzip"));
		error = http::HttpError{UnsupportedTransferCoding};
		break;
	case 14: {
		constexpr auto lengths = std::array<std::string_view, 5>{"+2", "2, 2", "two", "99999999999999999999999", "-1"};
		draft = create_draft(rng);
		draft.fields.back() = field(rng, "Content-Length", lengths[pick(rng, lengths.size())]);
		error = http::HttpError{BadContentLength};
		break;
	}
	case 15:
		draft = create_draft(rng);
		draft.fields.push_back(field(rng, "Content-Length", "2"));
		error = http::HttpError{BadContentLength};
		break;
	case 16:
		draft = create_draft(rng);
		draft.fields.pop_back();
		error = http::HttpError{LengthRequired};
		break;
	case 17:
		draft = create_draft(rng);
		draft.fields.back() = field(rng, "Content-Length", std::format("{}", bounds.bytes + 1));
		draft.body.clear();
		error = http::HttpError{BodyTooLarge};
		break;
	case 18: {
		// A body on every route that takes none: the GET routes, DELETE, and cancel.
		constexpr auto bodiless = std::array<std::string_view, 4>{"GET /health", "GET /v1/responses/{}", "DELETE /v1/responses/{}",
		                                                          "POST /v1/responses/{}/cancel"};
		draft.request_line = std::vformat(bodiless[pick(rng, bodiless.size())], std::make_format_args(id)) + " HTTP/1.1";
		draft.fields.push_back(field(rng, "Content-Length", "3"));
		draft.body = "abc";
		error = http::HttpError{BodyNotAllowed};
		break;
	}
	case 19:
		draft.fields.push_back(field(rng, "Expect", "200-ok"));
		error = http::HttpError{UnsupportedExpectation};
		break;
	case 20: {
		constexpr auto missing = std::array<std::string_view, 9>{"/v1/missing",
		                                                         "/v1/chat/completions",
		                                                         "/v1/responses/abc/",
		                                                         "/v1/Responses",
		                                                         "/v1/responses/abc/input_items/x",
		                                                         "/v1/responses/abc/cancel/",
		                                                         "/health/",
		                                                         "http://h/health",
		                                                         "*"};
		draft.request_line = std::format("{} {} HTTP/1.1", chance(rng, 2) ? "GET" : "POST", missing[pick(rng, missing.size())]);
		error = http::HttpError{NotFound};
		break;
	}
	case 21: {
		// A method no route on the path takes, including tokens no route anywhere takes.
		constexpr auto tokens = std::array<std::string_view, 8>{"GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS", "get"};
		auto const shape = shapes[pick(rng, shapes.size())];
		auto method = tokens[0];
		do {
			method = tokens[pick(rng, tokens.size())];
		} while (std::ranges::any_of(std::array{http::Method::Get, http::Method::Post, http::Method::Delete}, [&](http::Method known) {
			return token(known) == method && methods_of(shape).contains(known);
		}));
		draft.request_line = std::format("{} {} HTTP/1.1", method, path_of(shape, id));
		error = http::HttpError{http::WrongMethod{.allowed = methods_of(shape)}};
		break;
	}
	case 22:
		for (auto index = std::size_t{0}; index < http::max_header_count; ++index) {
			draft.fields.push_back(std::format("X-Filler-{}: {}", index, index));
		}
		error = http::HttpError{TooManyHeaders};
		break;
	case 23:
		draft.fields.push_back("X-Huge: " + std::string(http::max_head_bytes, 'a'));
		error = http::HttpError{HeadTooLarge};
		break;
	case 24:
		draft.request_line = "GET /health HTTP/11";
		break;
	case 25: {
		// An id that is empty, too long, or holds a character outside the alphabet.
		constexpr std::string_view outside = ".~%!*+:@$'(),;=#[]";
		auto bad = std::string{};
		switch (pick(rng, 3)) {
		case 0:
			break;
		case 1:
			bad = text(rng, http::max_resource_id_bytes + 1 + pick(rng, 8), resource_alphabet);
			break;
		default:
			bad = id;
			bad.insert(pick(rng, bad.size() + 1), 1, outside[pick(rng, outside.size())]);
			break;
		}
		constexpr auto with_ids = std::array{Shape::Response, Shape::InputItems, Shape::Cancel};
		auto const shape = with_ids[pick(rng, with_ids.size())];
		auto const method = shape == Shape::Cancel ? "POST" : (shape == Shape::Response && chance(rng, 2) ? "DELETE" : "GET");
		draft.request_line = std::format("{} {} HTTP/1.1", method, path_of(shape, bad));
		error = http::HttpError{InvalidResourceId};
		break;
	}
	case 26:
	case 27:
		draft =
		    get_draft(rng, std::format("{}?{}", path_of(chance(rng, 2) ? Shape::Response : Shape::InputItems, id), malformed_query(rng)));
		break;
	case 28:
		draft = get_draft(rng, std::format("{}?{}", path_of(chance(rng, 2) ? Shape::Response : Shape::InputItems, id),
		                                   valid_query(rng, http::max_query_parameters + 1 + pick(rng, 4))));
		error = http::HttpError{TooManyQueryParameters};
		break;
	case 29:
	case 30:
	case 31: {
		// A query on every route that reads none.
		constexpr auto without = std::array{Shape::Responses, Shape::Response, Shape::Cancel, Shape::Models, Shape::Health, Shape::Status};
		auto const shape = without[pick(rng, without.size())];
		auto const method = shape == Shape::Response                              ? http::Method::Delete
		                    : shape == Shape::Responses || shape == Shape::Cancel ? http::Method::Post
		                                                                          : http::Method::Get;
		draft.request_line = std::format("{} {}?{} HTTP/1.1", token(method), path_of(shape, id),
		                                 valid_query(rng, 1 + pick(rng, http::max_query_parameters)));
		if (shape == Shape::Responses) {
			draft.fields.push_back(field(rng, "Content-Length", "2"));
			draft.body = "{}";
		}
		error = http::HttpError{QueryNotAllowed};
		break;
	}
	default:
		draft.request_line = "GET /health\x80 HTTP/1.1";
		break;
	}
	std::ranges::shuffle(draft.fields, rng);
	return Generated{.draft = std::move(draft), .expected = std::move(error), .continues = false};
}

/** Pipelined requests on one connection: valid ones, then possibly one defective one, which ends the stream. */
struct Stream {
	std::string bytes;
	std::vector<Generated> requests;
};

[[nodiscard]] auto stream(Rng& rng) -> Stream {
	auto out = Stream{.bytes = "", .requests = {}};
	for (auto count = 1 + pick(rng, 5); count > 0; --count) {
		out.requests.push_back(valid(rng));
	}
	if (chance(rng, 2)) {
		out.requests.push_back(defective(rng));
	}
	for (auto const& request : out.requests) {
		out.bytes += render(request.draft);
	}
	return out;
}

using Outcome = std::variant<http::Assembled<bounds>, http::HttpError>;

struct Observed {
	std::vector<Outcome> outcomes;
	std::size_t continues;
};

/**
 * Delivers `bytes` the way the server's reads would: each read is at most `piece(rng)` bytes and at most the space
 * the assembler offers, and everything assemblable is assembled after each read.
 */
template<class Piece>
[[nodiscard]] auto deliver(std::string_view bytes, Piece piece) -> Observed {
	auto inbound = std::make_unique<http::Inbound<bounds>>();
	auto observed = Observed{.outcomes = {}, .continues = 0};
	auto offset = std::size_t{0};
	for (;;) {
		for (;;) {
			auto assembly = inbound->assemble();
			if (std::holds_alternative<http::Partial>(assembly)) {
				break;
			}
			if (std::holds_alternative<http::Continue>(assembly)) {
				++observed.continues;
				continue;
			}
			if (auto* assembled = std::get_if<http::Assembled<bounds>>(&assembly)) {
				observed.outcomes.emplace_back(std::move(*assembled));
				continue;
			}
			observed.outcomes.emplace_back(std::get<http::Refused>(assembly).error);
			return observed;
		}
		if (offset == bytes.size()) {
			return observed;
		}
		auto const space = inbound->space();
		auto const count = std::min({space.size(), piece(), bytes.size() - offset});
		std::ranges::copy(bytes.substr(offset, count), space.begin());
		inbound->commit(count);
		offset += count;
	}
}

[[nodiscard]] auto whole(std::string_view bytes) -> std::vector<Outcome> {
	return deliver(bytes,
	               [] {
		               return std::size_t{4096};
	               })
	    .outcomes;
}

[[nodiscard]] auto expected_outcomes(Stream const& generated) -> std::vector<Outcome> {
	auto out = std::vector<Outcome>{};
	for (auto const& request : generated.requests) {
		std::visit(
		    [&out](auto const& expected) {
			    out.emplace_back(expected);
		    },
		    request.expected);
	}
	return out;
}

/**
 * The search: random pipelined streams, each delivered whole, in random pieces, and one byte at a time. Every
 * delivery must agree with the generator's oracle; 100 Continue may differ (it depends on whether the body came
 * with the head) but never exceeds the requests that asked for it.
 */
[[nodiscard]] auto fragmentation_never_changes_the_parse(std::uint64_t seed, std::size_t streams) -> bool {
	auto rng = engine::testing::generator(seed);
	for (auto index = std::size_t{0}; index < streams; ++index) {
		auto const generated = stream(rng);
		auto const expected = expected_outcomes(generated);
		auto const asked = static_cast<std::size_t>(std::ranges::count_if(generated.requests, &Generated::continues));
		auto const at_once = deliver(generated.bytes, [] {
			return std::numeric_limits<std::size_t>::max();
		});
		auto const pieces = deliver(generated.bytes, [&rng] {
			return 1 + pick(rng, 64);
		});
		auto const single = deliver(generated.bytes, [] {
			return std::size_t{1};
		});
		if (at_once.outcomes != expected || pieces.outcomes != expected || single.outcomes != expected || at_once.continues > asked ||
		    pieces.continues > asked || single.continues != asked) {
			std::println("seed {} stream {}: mismatch over {} requests", seed, index, generated.requests.size());
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto request(std::string_view method, std::string_view target) -> std::string {
	return std::format("{} {} HTTP/1.1\r\nHost: x\r\n\r\n", method, target);
}

[[nodiscard]] auto assembled(Route route) -> std::vector<Outcome> {
	return {http::Assembled<bounds>{.route = std::move(route), .client_id = std::nullopt, .persistence = http::Persistence::KeepAlive}};
}

[[nodiscard]] auto refused(http::HttpError error) -> std::vector<Outcome> {
	return {std::move(error)};
}

[[nodiscard]] auto is_hex(char c) -> bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

[[nodiscard]] auto hex_value(char c) -> unsigned {
	constexpr std::string_view digits = "0123456789abcdef";
	return static_cast<unsigned>(digits.find(c >= 'A' && c <= 'F' ? static_cast<char>(c + 32) : c));
}

/**
 * Every escape of two visible characters, as a parameter's name and as its value: two hex digits of either case
 * decode to their byte, anything else refuses the request as malformed.
 */
[[nodiscard]] auto every_escape_decodes_or_is_refused() -> bool {
	for (auto high = '!'; high <= '~'; ++high) {
		for (auto low = '!'; low <= '~'; ++low) {
			auto const escape = std::format("%{}{}", high, low);
			auto const as_value = whole(request("GET", "/v1/responses/r?v=" + escape));
			auto const as_name = whole(request("GET", "/v1/responses/r/input_items?" + escape + "=v"));
			if (!is_hex(high) || !is_hex(low)) {
				if (as_value != refused(http::HttpError{http::HttpErrorKind::Malformed}) ||
				    as_name != refused(http::HttpError{http::HttpErrorKind::Malformed})) {
					std::println("escape {} was not refused", escape);
					return false;
				}
				continue;
			}
			auto const byte = std::string(1, std::bit_cast<char>(static_cast<std::uint8_t>((hex_value(high) << 4U) | hex_value(low))));
			auto const id = http::ResourceId{.text = "r"};
			if (as_value != assembled(http::RetrieveResponse{.id = id, .query = {.parameters = {{.name = "v", .value = byte}}}}) ||
			    as_name != assembled(http::ListInputItems{.id = id, .query = {.parameters = {{.name = byte, .value = "v"}}}})) {
				std::println("escape {} did not decode", escape);
				return false;
			}
		}
	}
	return true;
}

/** '+' is a space, an empty value may drop its '=', and names repeat in the order sent. */
[[nodiscard]] auto plus_is_a_space_and_order_is_kept() -> bool {
	auto const id = http::ResourceId{.text = "resp_1"};
	return whole(request("GET", "/v1/responses/resp_1?a+b=c+%2B+d&x&x=&a+b=2")) ==
	       assembled(http::RetrieveResponse{.id = id,
	                                        .query = {.parameters = {{.name = "a b", .value = "c + d"},
	                                                                 {.name = "x", .value = ""},
	                                                                 {.name = "x", .value = ""},
	                                                                 {.name = "a b", .value = "2"}}}});
}

[[nodiscard]] auto exactly_max_query_parameters_are_read() -> bool {
	auto const with = [](std::size_t count) {
		auto query = std::string{};
		auto expected = http::Query{.parameters = {}};
		for (auto index = std::size_t{0}; index < count; ++index) {
			query += std::format("{}p{}={}", index == 0 ? "" : "&", index, index);
			expected.parameters.push_back({.name = std::format("p{}", index), .value = std::format("{}", index)});
		}
		return std::pair{whole(request("GET", "/v1/responses/r/input_items?" + query)), std::move(expected)};
	};
	auto const [at, query] = with(http::max_query_parameters);
	return at == assembled(http::ListInputItems{.id = {.text = "r"}, .query = query}) &&
	       with(http::max_query_parameters + 1).first == refused(http::HttpError{http::HttpErrorKind::TooManyQueryParameters});
}

[[nodiscard]] auto is_resource_char(char c) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

[[nodiscard]] auto is_visible(char c) -> bool {
	return c > ' ' && c < '\x7f';
}

/**
 * Every one-byte id: an id character is the id; '/' makes another path; '?' leaves the id empty; any other visible
 * character is an invalid id; anything else breaks the request line.
 */
[[nodiscard]] auto every_one_byte_id_is_read_or_refused() -> bool {
	for (auto value = 0; value < 256; ++value) {
		auto const c = static_cast<char>(value);
		auto const outcome = whole(request("DELETE", std::string{"/v1/responses/"} + c));
		auto const expected = is_resource_char(c) ? assembled(http::DeleteResponse{.id = {.text = std::string(1, c)}})
		                      : c == '/'          ? refused(http::HttpError{http::HttpErrorKind::NotFound})
		                      : is_visible(c)     ? refused(http::HttpError{http::HttpErrorKind::InvalidResourceId})
		                                          : refused(http::HttpError{http::HttpErrorKind::Malformed});
		if (outcome != expected) {
			std::println("one-byte id {:#04x} misrouted", value);
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto an_id_of_max_resource_id_bytes_is_read_and_one_more_is_not() -> bool {
	auto const at = std::string(http::max_resource_id_bytes, 'r');
	auto const over = at + "r";
	return whole(request("POST", "/v1/responses/" + at + "/cancel")) == assembled(http::CancelResponse{.id = {.text = at}}) &&
	       whole(request("POST", "/v1/responses/" + over + "/cancel")) == refused(http::HttpError{http::HttpErrorKind::InvalidResourceId});
}

/**
 * Every path with every method token: an allowed method assembles the path's route, any other is a 405 whose Allow
 * is the path's methods, from the oracle's table.
 */
[[nodiscard]] auto every_method_on_every_path() -> bool {
	constexpr auto tokens = std::array<std::string_view, 9>{"GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS", "get", "Delete"};
	constexpr auto known = std::array{http::Method::Get, http::Method::Post, http::Method::Delete};
	for (auto const shape : shapes) {
		for (auto const method : tokens) {
			auto const target = path_of(shape, "resp_0");
			auto const outcome =
			    whole(shape == Shape::Responses ? std::format("{} {} HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n", method, target)
				                                : request(method, target));
			auto const allowed = std::ranges::find_if(known, [&](http::Method candidate) {
				return token(candidate) == method && methods_of(shape).contains(candidate);
			});
			auto const expected = allowed == known.end()
			                          ? refused(http::HttpError{http::WrongMethod{.allowed = methods_of(shape)}})
			                          : assembled(route_for(shape, *allowed, "resp_0", http::Query{.parameters = {}}, ""));
			if (outcome != expected) {
				std::println("{} {} misrouted", method, target);
				return false;
			}
		}
	}
	return true;
}

[[nodiscard]] auto every_torn_prefix_is_partial() -> bool {
	constexpr std::string_view whole_request = "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\nhello";
	for (auto length = std::size_t{0}; length < whole_request.size(); ++length) {
		auto inbound = std::make_unique<http::Inbound<bounds>>();
		std::ranges::copy(whole_request.substr(0, length), inbound->space().begin());
		inbound->commit(length);
		if (!std::holds_alternative<http::Partial>(inbound->assemble())) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto a_head_that_fills_the_buffer_is_refused_and_one_that_fits_is_not() -> bool {
	auto const outcome = [](std::size_t length) {
		auto head = std::string{"GET /health HTTP/1.1\r\nHost: x\r\nX-Pad: "};
		head += std::string(length - head.size() - 4, 'a');
		head += "\r\n\r\n";
		return whole(head);
	};
	auto const fits = outcome(http::max_head_bytes);
	auto const over = outcome(http::max_head_bytes + 1);
	return fits == assembled(http::Health{}) && over == refused(http::HttpError{http::HttpErrorKind::HeadTooLarge});
}

[[nodiscard]] auto a_body_at_the_limit_is_read() -> bool {
	auto body = std::string(bounds.bytes, 'b');
	auto const sent = std::format("POST /v1/responses HTTP/1.1\r\nHost: h\r\nContent-Length: {}\r\n\r\n", body.size()) + body;
	auto const observed = deliver(sent, [] {
		return std::size_t{1} << 16U;
	});
	if (observed.outcomes.size() != 1 || !std::holds_alternative<http::Assembled<bounds>>(observed.outcomes.front())) {
		return false;
	}
	auto const& route = std::get<http::Assembled<bounds>>(observed.outcomes.front()).route;
	if (!std::holds_alternative<http::CreateResponse<bounds>>(route)) {
		return false;
	}
	auto const& read = std::get<http::CreateResponse<bounds>>(route).body;
	// The parser reads past the body in place: the padding follows the last byte, inside the same storage.
	return read.text() == body && read.padded().size() == body.size() + bounds.padding &&
	       std::ranges::equal(read.padded().first(body.size()), body);
}

[[nodiscard]] auto exactly_max_header_count_fields_are_accepted() -> bool {
	auto const with = [](std::size_t count) {
		auto head = std::string{"GET /health HTTP/1.1\r\nHost: x\r\n"};
		for (auto index = std::size_t{1}; index < count; ++index) {
			head += std::format("X-F{}: v\r\n", index);
		}
		return whole(head + "\r\n");
	};
	return with(http::max_header_count) == assembled(http::Health{}) &&
	       with(http::max_header_count + 1) == refused(http::HttpError{http::HttpErrorKind::TooManyHeaders});
}

[[nodiscard]] auto nothing_after_a_refusal_is_parsed() -> bool {
	return whole(request("GET", "/nope") + request("GET", "/health")) == refused(http::HttpError{http::HttpErrorKind::NotFound});
}

[[nodiscard]] auto status_of_is_the_named_response() -> bool {
	using Kind = http::HttpErrorKind;
	using Status = http::StatusCode;
	auto const status = [](Kind kind) {
		return http::status_of(http::HttpError{kind});
	};
	return status(Kind::ChunkedUpload) == Status::LengthRequired && status(Kind::LengthRequired) == Status::LengthRequired &&
	       status(Kind::BodyTooLarge) == Status::ContentTooLarge && status(Kind::AmbiguousFraming) == Status::BadRequest &&
	       status(Kind::UnsupportedTransferCoding) == Status::BadRequest && status(Kind::HeadTooLarge) == Status::HeaderFieldsTooLarge &&
	       status(Kind::UnsupportedVersion) == Status::VersionNotSupported && status(Kind::NotFound) == Status::NotFound &&
	       status(Kind::InvalidResourceId) == Status::BadRequest && status(Kind::QueryNotAllowed) == Status::BadRequest &&
	       status(Kind::TooManyQueryParameters) == Status::BadRequest &&
	       http::status_of(http::HttpError{http::WrongMethod{.allowed = methods_of(Shape::Response)}}) == Status::MethodNotAllowed;
}

/** Every refusal has its own error code: a client or a log can tell any two refusals apart by it. */
[[nodiscard]] auto error_codes_are_distinct() -> bool {
	using enum http::HttpErrorKind;
	constexpr auto kinds = std::array{Malformed,
	                                  HeadTooLarge,
	                                  TooManyHeaders,
	                                  UnsupportedVersion,
	                                  HostRequired,
	                                  AmbiguousFraming,
	                                  UnsupportedTransferCoding,
	                                  ChunkedUpload,
	                                  BadContentLength,
	                                  LengthRequired,
	                                  BodyTooLarge,
	                                  BodyNotAllowed,
	                                  UnsupportedExpectation,
	                                  NotFound,
	                                  InvalidResourceId,
	                                  TooManyQueryParameters,
	                                  QueryNotAllowed};
	auto codes = kinds | std::views::transform([](http::HttpErrorKind kind) {
		             return std::string{http::error_code(http::HttpError{kind})};
	             }) |
	             std::ranges::to<std::set>();
	codes.insert(std::string{http::error_code(http::HttpError{http::WrongMethod{.allowed = methods_of(Shape::Health)}})});
	return codes.size() == kinds.size() + 1;
}

[[nodiscard]] auto request_ids_are_sane_or_absent() -> bool {
	return http::RequestId::parse("req_01HZ-x.y:z").has_value() && !http::RequestId::parse("").has_value() &&
	       !http::RequestId::parse("a b").has_value() && !http::RequestId::parse(std::string(http::max_request_id_bytes + 1, 'a')) &&
	       http::RequestId::parse(std::string(http::max_request_id_bytes, 'a')).has_value();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "pipelined streams parse as generated, whole, in pieces and byte by byte (seed 1)",
		                      .holds = fragmentation_never_changes_the_parse(1, 300)},
	    engine::testing::Case{.name = "pipelined streams parse as generated, whole, in pieces and byte by byte (seed 2)",
		                      .holds = fragmentation_never_changes_the_parse(2, 300)},
	    engine::testing::Case{.name = "pipelined streams parse as generated, whole, in pieces and byte by byte (seed 3)",
		                      .holds = fragmentation_never_changes_the_parse(3, 300)},
	    engine::testing::Case{.name = "every escape of two visible characters decodes to its byte or is refused",
		                      .holds = every_escape_decodes_or_is_refused()},
	    engine::testing::Case{.name = "'+' is a space, an empty value may drop its '=', and parameters keep their order",
		                      .holds = plus_is_a_space_and_order_is_kept()},
	    engine::testing::Case{.name = "max_query_parameters parameters are read; one more is refused",
		                      .holds = exactly_max_query_parameters_are_read()},
	    engine::testing::Case{.name = "every one-byte id is read, or refused by name", .holds = every_one_byte_id_is_read_or_refused()},
	    engine::testing::Case{.name = "an id of max_resource_id_bytes is read; one byte more is refused",
		                      .holds = an_id_of_max_resource_id_bytes_is_read_and_one_more_is_not()},
	    engine::testing::Case{.name = "every method on every path is routed, or a 405 allowing the path's methods",
		                      .holds = every_method_on_every_path()},
	    engine::testing::Case{.name = "every torn prefix of a request is partial, never a refusal",
		                      .holds = every_torn_prefix_is_partial()},
	    engine::testing::Case{.name = "a head of max_head_bytes is read; one byte more is 431",
		                      .holds = a_head_that_fills_the_buffer_is_refused_and_one_that_fits_is_not()},
	    engine::testing::Case{.name = "a body at the route's bound is read whole, followed by its padding",
		                      .holds = a_body_at_the_limit_is_read()},
	    engine::testing::Case{.name = "max_header_count fields are read; one more is refused",
		                      .holds = exactly_max_header_count_fields_are_accepted()},
	    engine::testing::Case{.name = "nothing after a refused request is parsed", .holds = nothing_after_a_refusal_is_parsed()},
	    engine::testing::Case{.name = "refusals are the named 411, 413, 400, 404 and 405 responses",
		                      .holds = status_of_is_the_named_response()},
	    engine::testing::Case{.name = "every refusal has its own error code", .holds = error_codes_are_distinct()},
	    engine::testing::Case{.name = "a request id is 1 to 64 id characters", .holds = request_ids_are_sane_or_absent()},
	};
	return engine::testing::report(cases);
}

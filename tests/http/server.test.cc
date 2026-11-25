// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.http;
import engine.http_test_client;
import engine.json;
import engine.net;
import engine.testing;

namespace http = engine::http;
namespace net = engine::net;
namespace client = engine::http_test_client;
using namespace std::chrono_literals;

namespace {

constexpr http::BodyBounds bounds{.bytes = std::size_t{1} << 20U, .padding = 64};

/** The pipeline's half of an exchange, as a test drives it: messages posted to the service. */
struct Reply {
	http::Exchange exchange;
	http::Response response;
};

struct Open {
	http::Exchange exchange;
};

struct Emit {
	http::Exchange exchange;
	std::string data;
};

struct End {
	http::Exchange exchange;
};

using Message = std::variant<Reply, Open, Emit, End>;

/** How the test service answers a request to create a response. */
enum class Mode : std::uint8_t {
	Echo,
	Stream,
	Defer,
	Oversized,
};

/** A test double for the openai service: it answers by route and records everything the server tells it. */
struct Recorder {
	Mode mode;
	std::vector<http::Exchange> exchanges;
	std::vector<http::Request<bounds>> requests;
	std::vector<http::Exchange> gone;
	std::vector<http::Exchange> resumed;
	std::vector<http::Flow> flows;
	std::vector<http::Delivery> deliveries;

	[[nodiscard]] auto serve(http::Exchange exchange, http::Request<bounds> request) -> http::Answer {
		exchanges.push_back(exchange);
		requests.push_back(request);
		auto const id = std::string{request.id.text()};
		if (std::holds_alternative<http::Health>(request.route)) {
			return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Text, .body = "health " + id};
		}
		if (std::holds_alternative<http::ListModels>(request.route)) {
			return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = "models " + id};
		}
		if (std::holds_alternative<http::ServerStatus>(request.route)) {
			return http::Deferred{};
		}
		if (!std::holds_alternative<http::CreateResponse<bounds>>(request.route)) {
			return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = "answered " + id};
		}
		switch (mode) {
		case Mode::Echo:
			return http::Response{.status = http::StatusCode::Ok,
			                      .content_type = http::ContentType::Json,
			                      .body = std::string{std::get<http::CreateResponse<bounds>>(request.route).body.text()}};
		case Mode::Stream:
			return http::OpenStream{};
		case Mode::Defer:
			return http::Deferred{};
		case Mode::Oversized:
			return http::Response{.status = http::StatusCode::Ok,
			                      .content_type = http::ContentType::Text,
			                      .body = std::string(http::max_response_body_bytes + 1, 'x')};
		}
		return http::Deferred{};
	}

	auto deliver(Message message, http::Port<bounds>& port) -> void {
		if (std::holds_alternative<Reply>(message)) {
			auto const& reply = std::get<Reply>(message);
			deliveries.push_back(port.respond(reply.exchange, reply.response));
		} else if (std::holds_alternative<Open>(message)) {
			deliveries.push_back(port.open_stream(std::get<Open>(message).exchange));
		} else if (std::holds_alternative<Emit>(message)) {
			auto const& emit = std::get<Emit>(message);
			flows.push_back(port.send(emit.exchange, http::Event{.data = emit.data}));
		} else {
			deliveries.push_back(port.finish(std::get<End>(message).exchange));
		}
	}

	auto drained(http::Exchange exchange, http::Port<bounds>&) -> void {
		resumed.push_back(exchange);
	}

	auto abandoned(http::Exchange exchange) -> void {
		gone.push_back(exchange);
	}
};

static_assert(http::Service<Recorder, Message, bounds>);

struct Timeouts {
	std::chrono::milliseconds read;
	std::chrono::milliseconds write;
};

constexpr auto quick = Timeouts{.read = 2000ms, .write = 2000ms};

/** One server, its service and its poster, driven by the test's thread between client operations. */
struct Harness {
	http::Server<Message, bounds> server;
	Recorder service;
	http::Poster<Message> poster;

	/** Turns the server until `done` holds after the client has pulled what arrived, or until `budget` passes. */
	template<class Done>
	[[nodiscard]] auto pump(client::Client& peer, Done done, std::chrono::milliseconds budget) -> bool {
		auto const deadline = net::Clock::now() + budget;
		for (;;) {
			peer.pull();
			if (done()) {
				return true;
			}
			if (net::Clock::now() >= deadline) {
				return false;
			}
			auto const turned = server.turn(service, net::Clock::now() + 2ms);
			if (!turned) {
				return false;
			}
		}
	}

	/** Turns the server for `span` with no client to watch. */
	auto idle(std::chrono::milliseconds span) -> void {
		auto const deadline = net::Clock::now() + span;
		while (net::Clock::now() < deadline) {
			std::ignore = server.turn(service, net::Clock::now() + 2ms);
		}
	}

	/** Sends every byte, turning the server whenever the client's kernel buffer is full. */
	[[nodiscard]] auto push(client::Client& peer, std::string_view bytes) -> bool {
		auto sent = std::size_t{0};
		auto const deadline = net::Clock::now() + 5000ms;
		while (sent < bytes.size() && net::Clock::now() < deadline) {
			sent += peer.push(bytes.substr(sent));
			std::ignore = server.turn(service, net::Clock::now() + 1ms);
		}
		return sent == bytes.size();
	}

	[[nodiscard]] auto post(Message message) const -> bool {
		return poster.post(std::move(message)) == net::Posted::Accepted;
	}
};

[[nodiscard]] auto harness(Mode mode, Timeouts timeouts) -> std::unique_ptr<Harness> {
	auto server = http::Server<Message, bounds>::open(http::ServerConfig{
	    .endpoint = net::Endpoint{.address = net::loopback, .port = 0},
	    .read_timeout = timeouts.read,
	    .write_timeout = timeouts.write,
	    .id_seed = 7,
	});
	if (!server) {
		std::println("server open failed: errno {}", server.error().code);
		return nullptr;
	}
	auto const poster = server->poster();
	return std::make_unique<Harness>(Harness{
	    .server = std::move(*server),
	    .service = Recorder{.mode = mode, .exchanges = {}, .requests = {}, .gone = {}, .resumed = {}, .flows = {}, .deliveries = {}},
	    .poster = poster,
	});
}

[[nodiscard]] auto bodiless(std::string_view method, std::string_view target, std::string_view extra) -> std::string {
	return std::format("{} {} HTTP/1.1\r\nHost: test\r\n{}\r\n", method, target, extra);
}

[[nodiscard]] auto get(std::string_view target, std::string_view extra) -> std::string {
	return bodiless("GET", target, extra);
}

[[nodiscard]] auto create(std::string_view body, std::string_view extra) -> std::string {
	return std::format("POST /v1/responses HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n{}",
	                   body.size(), extra, body);
}

[[nodiscard]] auto responses(client::Client const& peer) -> client::Responses {
	return client::parse_responses(peer.received);
}

[[nodiscard]] auto has_responses(client::Client const& peer, std::size_t count) {
	return [&peer, count] {
		return responses(peer).complete.size() >= count;
	};
}

[[nodiscard]] auto closed(client::Client const& peer) {
	return [&peer] {
		return peer.peer == client::Peer::Closed;
	};
}

[[nodiscard]] auto field(client::Parsed const& response, std::string const& name) -> std::string {
	auto const found = response.fields.find(name);
	return found == response.fields.end() ? std::string{} : found->second;
}

[[nodiscard]] auto is_issued_id(std::string_view id) -> bool {
	return id.size() == 16 && std::ranges::all_of(id, [](char c) {
		       return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
	       });
}

[[nodiscard]] auto keep_alive_serves_requests_in_turn() -> bool {
	auto h = harness(Mode::Echo, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, get("/health", "")) || !h->pump(*peer, has_responses(*peer, 1), 2000ms) ||
	    !h->push(*peer, get("/v1/models", "X-Request-Id: client-7\r\n")) || !h->pump(*peer, has_responses(*peer, 2), 2000ms)) {
		return false;
	}
	auto const parsed = responses(*peer);
	auto const& first = parsed.complete[0];
	auto const& second = parsed.complete[1];
	auto const first_id = field(first, "x-request-id");
	return !parsed.malformed && parsed.rest == 0 && first.status == 200 && first.body == "health " + first_id && is_issued_id(first_id) &&
	       field(first, "connection") == "keep-alive" && field(first, "content-type") == "text/plain; charset=utf-8" &&
	       second.status == 200 && field(second, "x-request-id") == "client-7" && second.body == "models client-7" &&
	       field(second, "content-type") == "application/json" && peer->peer == client::Peer::Open && h->service.exchanges.size() == 2 &&
	       h->service.exchanges[0].connection == h->service.exchanges[1].connection &&
	       h->service.exchanges[0].sequence != h->service.exchanges[1].sequence;
}

[[nodiscard]] auto an_insane_client_id_is_replaced() -> bool {
	auto h = harness(Mode::Echo, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto const request = get("/health", "X-Request-Id: has spaces and/slash\r\n") + get("/health", "") + get("/health", "");
	if (!peer || !h->push(*peer, request) || !h->pump(*peer, has_responses(*peer, 3), 2000ms)) {
		return false;
	}
	auto ids = std::set<std::string>{};
	for (auto const& response : responses(*peer).complete) {
		auto const id = field(response, "x-request-id");
		if (!is_issued_id(id) || response.body != "health " + id) {
			return false;
		}
		ids.insert(id);
	}
	return ids.size() == 3;
}

[[nodiscard]] auto connection_close_is_honoured() -> bool {
	auto h = harness(Mode::Echo, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, get("/health", "Connection: close\r\n") + get("/health", "")) || !h->pump(*peer, closed(*peer), 4000ms)) {
		return false;
	}
	auto const parsed = responses(*peer);
	return parsed.complete.size() == 1 && field(parsed.complete[0], "connection") == "close" && h->service.exchanges.size() == 1;
}

/** An OpenAI error object's members, as the engine's own JSON reader reads them: the oracle reads JSON, not text. */
struct ErrorObject {
	std::string message;
	std::string type;
	engine::json::Kind param;
	std::string code;
};

using ErrorReader = engine::json::Reader<engine::json::request_bounds>;

/** Reads `{"error": {...}}` member by member: exactly the four members, nothing else, and nothing after the root. */
[[nodiscard]] auto read_error(ErrorReader& reader, std::string_view body) -> std::optional<ErrorObject> {
	auto root = reader.read_copy(body);
	auto outer = root ? root->as_object() : std::unexpected(root.error());
	auto first = outer ? outer->next() : std::unexpected(outer.error());
	if (!first || !*first || (**first).name != "error") {
		return std::nullopt;
	}
	auto inner = (**first).value.as_object();
	if (!inner) {
		return std::nullopt;
	}
	auto out = ErrorObject{.message = "", .type = "", .param = engine::json::Kind::Object, .code = ""};
	auto names = std::set<std::string>{};
	for (;;) {
		auto member = inner->next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			break;
		}
		auto& [name, value] = **member;
		names.emplace(name);
		if (name == "param") {
			auto const kind = value.kind();
			if (!kind) {
				return std::nullopt;
			}
			out.param = *kind;
			continue;
		}
		auto const text = value.as_string();
		if (!text) {
			return std::nullopt;
		}
		if (name == "message") {
			out.message = *text;
		} else if (name == "type") {
			out.type = *text;
		} else if (name == "code") {
			out.code = *text;
		}
	}
	auto const after = outer->next();
	if (names != std::set<std::string>{"code", "message", "param", "type"} || !after || *after || !reader.finish()) {
		return std::nullopt;
	}
	return out;
}

/** The response is OpenAI's error object as valid JSON, with `type`, a null `param` and the stable `code`. */
[[nodiscard]] auto is_error_object(ErrorReader& reader, client::Parsed const& response, std::string_view type, std::string_view code)
    -> bool {
	auto const error = read_error(reader, response.body);
	return field(response, "content-type") == "application/json" && is_issued_id(field(response, "x-request-id")) && error &&
	       !error->message.empty() && error->type == type && error->param == engine::json::Kind::Null && error->code == code;
}

struct Refusal {
	std::string request;
	int status;
	http::HttpError error;
	/** The Allow field a 405 carries, written from the api's route list. */
	std::optional<std::string_view> allow;
};

[[nodiscard]] auto wrong(std::initializer_list<http::Method> allowed) -> http::HttpError {
	auto methods = http::Methods{};
	for (auto const method : allowed) {
		methods = methods.with(method);
	}
	return http::HttpError{http::WrongMethod{.allowed = methods}};
}

[[nodiscard]] auto refusals_are_named_and_close_the_connection() -> bool {
	using enum http::HttpErrorKind;
	using enum http::Method;
	auto reader = ErrorReader::make();
	if (!reader) {
		return false;
	}
	auto const too_many = std::views::iota(std::size_t{0}, http::max_query_parameters + 1) | std::views::transform([](std::size_t index) {
		                      return std::format("p{}=1", index);
	                      }) |
	                      std::views::join_with('&') | std::ranges::to<std::string>();
	auto const refusals = std::array{
	    Refusal{.request = get("/nowhere", ""), .status = 404, .error = {NotFound}, .allow = std::nullopt},
	    Refusal{.request = "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\n\r\n{}",
		        .status = 404,
		        .error = {NotFound},
		        .allow = std::nullopt},
	    Refusal{.request = get("/v1/responses", ""), .status = 405, .error = wrong({Post}), .allow = "POST"},
	    Refusal{
	        .request = bodiless("POST", "/v1/responses/resp_1", ""), .status = 405, .error = wrong({Get, Delete}), .allow = "GET, DELETE"},
	    Refusal{
	        .request = bodiless("PUT", "/v1/responses/resp_1", ""), .status = 405, .error = wrong({Get, Delete}), .allow = "GET, DELETE"},
	    Refusal{.request = bodiless("DELETE", "/v1/responses/r/input_items", ""), .status = 405, .error = wrong({Get}), .allow = "GET"},
	    Refusal{.request = get("/v1/responses/r/cancel", ""), .status = 405, .error = wrong({Post}), .allow = "POST"},
	    Refusal{.request = bodiless("DELETE", "/health", ""), .status = 405, .error = wrong({Get}), .allow = "GET"},
	    Refusal{.request = get("/v1/responses/resp.1", ""), .status = 400, .error = {InvalidResourceId}, .allow = std::nullopt},
	    Refusal{.request = get("/v1/responses/r?after=%zz", ""), .status = 400, .error = {Malformed}, .allow = std::nullopt},
	    Refusal{.request = get("/v1/responses/r/input_items?" + too_many, ""),
		        .status = 400,
		        .error = {TooManyQueryParameters},
		        .allow = std::nullopt},
	    Refusal{.request = get("/v1/models?limit=1", ""), .status = 400, .error = {QueryNotAllowed}, .allow = std::nullopt},
	    Refusal{.request = bodiless("DELETE", "/v1/responses/r?x", ""), .status = 400, .error = {QueryNotAllowed}, .allow = std::nullopt},
	    Refusal{.request = bodiless("DELETE", "/v1/responses/r", "Content-Length: 2\r\n") + "{}",
		        .status = 400,
		        .error = {BodyNotAllowed},
		        .allow = std::nullopt},
	    Refusal{.request = bodiless("POST", "/v1/responses/r/cancel", "Content-Length: 2\r\n") + "{}",
		        .status = 400,
		        .error = {BodyNotAllowed},
		        .allow = std::nullopt},
	    Refusal{.request = "POST /v1/responses HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n",
		        .status = 411,
		        .error = {ChunkedUpload},
		        .allow = std::nullopt},
	    Refusal{.request = bodiless("POST", "/v1/responses", ""), .status = 411, .error = {LengthRequired}, .allow = std::nullopt},
	    Refusal{.request = std::format("POST /v1/responses HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n{}", bounds.bytes + 1,
		                               std::string(4096, 'z')),
		        .status = 413,
		        .error = {BodyTooLarge},
		        .allow = std::nullopt},
	    Refusal{.request = "POST /v1/responses HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}",
		        .status = 400,
		        .error = {AmbiguousFraming},
		        .allow = std::nullopt},
	    Refusal{.request = "GET /health HTTP/1.0\r\nHost: t\r\n\r\n", .status = 505, .error = {UnsupportedVersion}, .allow = std::nullopt},
	    Refusal{.request = get("/health", "X-Big: " + std::string(http::max_head_bytes, 'a') + "\r\n"),
		        .status = 431,
		        .error = {HeadTooLarge},
		        .allow = std::nullopt},
	    Refusal{.request = "GET /health HTTP/1.1\r\n\r\n", .status = 400, .error = {HostRequired}, .allow = std::nullopt},
	    Refusal{.request = "G@T /health HTTP/1.1\r\nHost: t\r\n\r\n", .status = 400, .error = {Malformed}, .allow = std::nullopt},
	};
	for (auto const& refusal : refusals) {
		auto h = harness(Mode::Echo, quick);
		auto peer = h ? client::connect(h->server.port()) : std::nullopt;
		if (!peer) {
			return false;
		}
		std::ignore = h->push(*peer, refusal.request);
		if (!h->pump(*peer, closed(*peer), 4000ms)) {
			return false;
		}
		auto const parsed = responses(*peer);
		auto const allow = parsed.complete.empty()                       ? std::nullopt
		                   : parsed.complete[0].fields.contains("allow") ? std::optional<std::string>{field(parsed.complete[0], "allow")}
		                                                                 : std::nullopt;
		if (parsed.complete.size() != 1 || parsed.complete[0].status != refusal.status ||
		    field(parsed.complete[0], "connection") != "close" || !h->service.exchanges.empty() || allow != refusal.allow ||
		    !is_error_object(*reader, parsed.complete[0], "invalid_request_error", http::error_code(refusal.error))) {
			std::println("refusal {} answered {}", http::error_code(refusal.error),
			             parsed.complete.empty() ? 0 : parsed.complete[0].status);
			return false;
		}
	}
	return true;
}

/** A pipelined stream of valid requests over every route, and the routes the service must receive, in order. */
struct Pipelined {
	std::string bytes;
	std::vector<http::Route<bounds>> routes;
};

/** Each request carries its own X-Request-Id, so the answers are fully determined. */
[[nodiscard]] auto pipelined(std::uint64_t seed, std::size_t count) -> Pipelined {
	auto rng = engine::testing::generator(seed);
	auto out = Pipelined{.bytes = "", .routes = {}};
	auto const random_bytes = [&rng](std::size_t length) {
		auto text = std::string(length, '\0');
		std::ranges::generate(text, [&rng] {
			return static_cast<char>(rng() & 0xffU);
		});
		return text;
	};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const extra = std::format("X-Request-Id: r{}-{}\r\n", seed, index);
		auto const id = http::ResourceId{.text = std::format("resp_{:x}", rng())};
		auto sent = std::vector<client::Parameter>{};
		for (auto parameter = rng() % 4; parameter > 0; --parameter) {
			sent.push_back(client::Parameter{.name = random_bytes(1 + (rng() % 6)), .value = random_bytes(rng() % 9)});
		}
		auto const query = http::Query{.parameters = sent | std::views::transform([](client::Parameter const& parameter) {
			                                             return http::QueryParameter{.name = parameter.name, .value = parameter.value};
		                                             }) |
		                                             std::ranges::to<std::vector>()};
		auto const target = sent.empty() ? std::string{} : "?" + client::query_string(sent, rng);
		switch (rng() % 7) {
		case 0:
			out.bytes += get("/health", extra);
			out.routes.emplace_back(http::Health{});
			break;
		case 1:
			out.bytes += get("/v1/models", extra);
			out.routes.emplace_back(http::ListModels{});
			break;
		case 2:
			out.bytes += get("/v1/responses/" + id.text + target, extra);
			out.routes.emplace_back(http::RetrieveResponse{.id = id, .query = query});
			break;
		case 3:
			out.bytes += bodiless("DELETE", "/v1/responses/" + id.text, extra);
			out.routes.emplace_back(http::DeleteResponse{.id = id});
			break;
		case 4:
			out.bytes += get("/v1/responses/" + id.text + "/input_items" + target, extra);
			out.routes.emplace_back(http::ListInputItems{.id = id, .query = query});
			break;
		case 5:
			out.bytes += bodiless("POST", "/v1/responses/" + id.text + "/cancel", extra);
			out.routes.emplace_back(http::CancelResponse{.id = id});
			break;
		default: {
			auto const body = random_bytes(rng() % 300);
			out.bytes += create(body, extra);
			out.routes.emplace_back(http::CreateResponse<bounds>{.body = http::Body<bounds>::copy_of(body)});
			break;
		}
		}
	}
	return out;
}

struct Answered {
	std::vector<client::Parsed> responses;
	std::vector<http::Route<bounds>> routes;
};

[[nodiscard]] auto answered(std::string_view bytes, std::size_t count, std::size_t piece) -> std::optional<Answered> {
	auto h = harness(Mode::Echo, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer) {
		return std::nullopt;
	}
	for (auto offset = std::size_t{0}; offset < bytes.size(); offset += piece) {
		if (!h->push(*peer, bytes.substr(offset, piece))) {
			return std::nullopt;
		}
		std::ignore = h->server.turn(h->service, net::Clock::now());
	}
	if (!h->pump(*peer, has_responses(*peer, count), 5000ms)) {
		return std::nullopt;
	}
	return Answered{.responses = responses(*peer).complete,
	                .routes = h->service.requests | std::views::transform(&http::Request<bounds>::route) | std::ranges::to<std::vector>()};
}

/**
 * The same pipelined bytes, sent whole, one byte per server turn and in 7-byte pieces, draw byte-identical answers,
 * and the service receives exactly the routes generated, ids and decoded queries included.
 */
[[nodiscard]] auto socket_fragmentation_never_changes_the_answers(std::uint64_t seed) -> bool {
	constexpr auto count = std::size_t{8};
	auto const generated = pipelined(seed, count);
	auto const whole = answered(generated.bytes, count, generated.bytes.size());
	auto const single = answered(generated.bytes, count, 1);
	auto const odd = answered(generated.bytes, count, 7);
	auto const same = [](client::Parsed const& left, client::Parsed const& right) {
		return left.status == right.status && left.body == right.body && left.fields == right.fields;
	};
	return whole && single && odd && whole->responses.size() == count && std::ranges::equal(whole->responses, single->responses, same) &&
	       std::ranges::equal(whole->responses, odd->responses, same) && whole->routes == generated.routes &&
	       single->routes == generated.routes && odd->routes == generated.routes;
}

[[nodiscard]] auto a_deferred_answer_arrives_through_the_mailbox() -> bool {
	auto h = harness(Mode::Defer, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, get("/status", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	auto const response = http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = R"({"live":1})"};
	if (!h->post(Reply{.exchange = exchange, .response = response}) || !h->pump(*peer, has_responses(*peer, 1), 2000ms)) {
		return false;
	}
	// The exchange is over: a second answer from a straggling pipeline message finds nothing to answer.
	auto const parsed = responses(*peer);
	return parsed.complete.size() == 1 && parsed.complete[0].body == R"({"live":1})" &&
	       h->service.deliveries == std::vector{http::Delivery::Accepted};
}

[[nodiscard]] auto random_data(std::mt19937_64& rng) -> std::string {
	constexpr std::string_view alphabet = "abc xyz{}\":,\r\n\r\n\xc3\xa9\xe2\x82\xac[DONE]";
	auto data = std::string(rng() % 200, ' ');
	std::ranges::generate(data, [&] {
		return alphabet[rng() % alphabet.size()];
	});
	return data;
}

/** Events of random data, line breaks included, decode to exactly that data, and the connection then serves on. */
[[nodiscard]] auto events_round_trip(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	auto h = harness(Mode::Stream, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, create("{}", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	auto sent = std::vector<std::string>{};
	for (auto count = 1 + (rng() % 40); count > 0; --count) {
		sent.push_back(random_data(rng));
		if (!h->post(Emit{.exchange = exchange, .data = sent.back()})) {
			return false;
		}
	}
	if (!h->post(End{.exchange = exchange}) || !h->pump(*peer, has_responses(*peer, 1), 2000ms) || !h->push(*peer, get("/health", "")) ||
	    !h->pump(*peer, has_responses(*peer, 2), 2000ms)) {
		return false;
	}
	auto const parsed = responses(*peer);
	auto const& stream = parsed.complete[0];
	auto const decoded = client::decode_events(stream.body);
	auto const expected = sent | std::views::transform([](std::string const& data) {
		                      return client::normalized(data);
	                      });
	return stream.status == 200 && field(stream, "content-type") == "text/event-stream" &&
	       field(stream, "transfer-encoding") == "chunked" && field(stream, "cache-control") == "no-cache" &&
	       std::ranges::equal(decoded, expected) && parsed.complete[1].status == 200 &&
	       std::ranges::all_of(h->service.flows, [](http::Flow flow) {
		       return flow == http::Flow::Accepted;
	       });
}

/** An ordinary event's size: a delta the service coalesces when the client falls behind. */
constexpr std::size_t delta_bytes = 4096;

[[nodiscard]] auto event_data(std::size_t index) -> std::string {
	return std::format("{:05}:{}", index, std::string(delta_bytes - 6, 'e'));
}

/**
 * Posts ordinary events to a stream whose client does not read, until the server refuses one and no
 * drain follows: the window and the kernel's buffers are full. Returns the accepted events' data, or nullopt if
 * the server took 80 MiB without refusing (an unbounded buffer).
 */
[[nodiscard]] auto fill_until_stuck(Harness& h, http::Exchange exchange) -> std::optional<std::vector<std::string>> {
	auto accepted = std::vector<std::string>{};
	auto index = std::size_t{0};
	while (index < 20000) {
		auto const first = h.service.flows.size();
		auto const resumed = h.service.resumed.size();
		for (auto batch = 0; batch < 32; ++batch, ++index) {
			if (!h.post(Emit{.exchange = exchange, .data = event_data(index)})) {
				return std::nullopt;
			}
		}
		std::ignore = h.server.turn(h.service, net::Clock::now() + 1ms);
		for (auto flow = first; flow < h.service.flows.size(); ++flow) {
			if (h.service.flows[flow] == http::Flow::Accepted) {
				accepted.push_back(event_data(flow));
			}
		}
		if (std::ranges::contains(std::span{h.service.flows}.subspan(first), http::Flow::Full)) {
			h.idle(50ms);
			if (h.service.resumed.size() == resumed) {
				return accepted;
			}
		}
	}
	return std::nullopt;
}

/**
 * A client that does not read: the server refuses events once its window is full instead of growing, tells the
 * service when the client has caught up, and every accepted event arrives in order with nothing lost or repeated.
 */
[[nodiscard]] auto a_slow_client_is_held_to_the_window() -> bool {
	auto h = harness(Mode::Stream, Timeouts{.read = 2000ms, .write = 5000ms});
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, create("{}", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	auto accepted = fill_until_stuck(*h, exchange);
	if (!accepted) {
		return false;
	}
	auto const resumed = h->service.resumed.size();
	// Reading lets the server drain; it tells the service, and the stream resumes and ends.
	if (!h->pump(
	        *peer,
	        [&] {
		        return h->service.resumed.size() > resumed;
	        },
	        5000ms) ||
	    h->service.resumed.back() != exchange) {
		return false;
	}
	if (!h->post(Emit{.exchange = exchange, .data = "after"}) || !h->post(End{.exchange = exchange}) ||
	    !h->pump(*peer, has_responses(*peer, 1), 5000ms)) {
		return false;
	}
	accepted->emplace_back("after");
	auto const decoded = client::decode_events(responses(*peer).complete.front().body);
	return decoded == *accepted && h->service.flows.back() == http::Flow::Accepted;
}

[[nodiscard]] auto a_stalled_client_is_abandoned_at_the_write_deadline() -> bool {
	auto h = harness(Mode::Stream, Timeouts{.read = 2000ms, .write = 300ms});
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, create("{}", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	auto const started = net::Clock::now();
	if (!fill_until_stuck(*h, exchange)) {
		return false;
	}
	auto const deadline = net::Clock::now() + 2000ms;
	while (h->service.gone.empty() && net::Clock::now() < deadline) {
		std::ignore = h->server.turn(h->service, net::Clock::now() + 2ms);
	}
	if (h->service.gone != std::vector{exchange} || net::Clock::now() - started < 300ms) {
		return false;
	}
	h->service.flows.clear();
	std::ignore = h->post(Emit{.exchange = exchange, .data = "late"});
	h->idle(20ms);
	return h->service.flows == std::vector{http::Flow::Gone};
}

enum class Departure : std::uint8_t {
	Close,
	Reset,
};

/** A client that leaves mid-stream is reported once, as the event the pipeline turns into cancellation. */
[[nodiscard]] auto a_departing_client_abandons_its_stream(Departure departure) -> bool {
	auto h = harness(Mode::Stream, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, create("{}", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	if (!h->post(Emit{.exchange = exchange, .data = "first"}) || !h->pump(
	                                                                 *peer,
	                                                                 [&] {
		                                                                 return peer->received.contains("first");
	                                                                 },
	                                                                 2000ms)) {
		return false;
	}
	if (departure == Departure::Reset) {
		std::move(peer->socket).abort();
	}
	peer.reset();
	auto const deadline = net::Clock::now() + 2000ms;
	while (h->service.gone.empty() && net::Clock::now() < deadline) {
		std::ignore = h->server.turn(h->service, net::Clock::now() + 2ms);
	}
	std::ignore = h->post(Emit{.exchange = exchange, .data = "late"});
	std::ignore = h->post(End{.exchange = exchange});
	h->idle(20ms);
	return h->service.gone == std::vector{exchange} && h->service.flows == std::vector{http::Flow::Accepted, http::Flow::Gone} &&
	       h->service.deliveries.back() == http::Delivery::Gone;
}

[[nodiscard]] auto a_client_that_leaves_before_the_answer_abandons_it() -> bool {
	auto h = harness(Mode::Defer, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, create("{\"slow\":true}", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	peer.reset();
	auto const deadline = net::Clock::now() + 2000ms;
	while (h->service.gone.empty() && net::Clock::now() < deadline) {
		std::ignore = h->server.turn(h->service, net::Clock::now() + 2ms);
	}
	std::ignore = h->post(
	    Reply{.exchange = exchange, .response = {.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = "{}"}});
	h->idle(20ms);
	return h->service.gone == std::vector{exchange} && h->service.deliveries == std::vector{http::Delivery::Gone};
}

[[nodiscard]] auto an_idle_connection_is_closed_at_the_read_deadline() -> bool {
	auto h = harness(Mode::Echo, Timeouts{.read = 200ms, .write = 2000ms});
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer || !h->push(*peer, "GET /health HTTP/1.1\r\nHo")) {
		return false;
	}
	auto const started = net::Clock::now();
	return h->pump(*peer, closed(*peer), 2000ms) && net::Clock::now() - started >= 150ms && peer->received.empty() &&
	       h->service.exchanges.empty() && h->service.gone.empty();
}

[[nodiscard]] auto continue_is_sent_before_the_body() -> bool {
	auto h = harness(Mode::Echo, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	constexpr std::string_view body = R"({"messages":[]})";
	auto const head =
	    std::format("POST /v1/responses HTTP/1.1\r\nHost: t\r\nExpect: 100-continue\r\nContent-Length: {}\r\n\r\n", body.size());
	if (!peer || !h->push(*peer, head) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return responses(*peer).interim == 1;
	        },
	        2000ms) ||
	    !h->push(*peer, body) || !h->pump(*peer, has_responses(*peer, 1), 2000ms)) {
		return false;
	}
	auto const parsed = responses(*peer);
	return parsed.interim == 1 && parsed.complete[0].status == 200 && parsed.complete[0].body == body;
}

[[nodiscard]] auto an_oversized_answer_is_a_500() -> bool {
	auto h = harness(Mode::Oversized, quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reader = ErrorReader::make();
	if (!peer || !reader || !h->push(*peer, create("{}", "")) || !h->pump(*peer, has_responses(*peer, 1), 2000ms)) {
		return false;
	}
	auto const parsed = responses(*peer);
	auto const& response = parsed.complete[0];
	return response.status == 500 && is_error_object(*reader, response, "server_error", "response_too_large");
}

/** Stopping refuses new work, lets the open stream finish, closes its connection after it, and then stops. */
[[nodiscard]] auto stop_drains_in_flight_streams() -> bool {
	auto h = harness(Mode::Stream, quick);
	auto streaming = h ? client::connect(h->server.port()) : std::nullopt;
	auto idle = h ? client::connect(h->server.port()) : std::nullopt;
	if (!streaming || !idle || !h->push(*streaming, create("{}", "")) ||
	    !h->pump(
	        *streaming,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	auto const exchange = h->service.exchanges.front();
	h->server.stop();
	if (!h->pump(*idle, closed(*idle), 2000ms)) {
		return false;
	}
	auto const late = client::connect(h->server.port());
	if (!h->post(Emit{.exchange = exchange, .data = "last"}) || !h->post(End{.exchange = exchange}) ||
	    !h->pump(*streaming, closed(*streaming), 4000ms)) {
		return false;
	}
	auto const parsed = responses(*streaming);
	// The server has shut its side and lingers until the client closes; then nothing is open and it has stopped.
	streaming.reset();
	auto stopped = false;
	auto const deadline = net::Clock::now() + 3000ms;
	while (!stopped && net::Clock::now() < deadline) {
		auto const turned = h->server.turn(h->service, net::Clock::now() + 2ms);
		stopped = turned && *turned == http::Turn::Stopped;
	}
	return !late && stopped && parsed.complete.size() == 1 &&
	       client::decode_events(parsed.complete[0].body) == std::vector<std::string>{"last"} && h->service.gone.empty();
}

/** A second stop signal closes everything at once, abandoning what is in flight. */
[[nodiscard]] auto a_second_signal_stops_hard() -> bool {
	auto h = harness(Mode::Defer, quick);
	if (!h || !h->server.stop_on_signals()) {
		return false;
	}
	auto peer = client::connect(h->server.port());
	if (!peer || !h->push(*peer, create("{}", "")) ||
	    !h->pump(
	        *peer,
	        [&] {
		        return !h->service.exchanges.empty();
	        },
	        2000ms)) {
		return false;
	}
	net::raise_stop(net::StopSignal::Terminate);
	auto first = h->server.turn(h->service, net::Clock::now() + 200ms);
	net::raise_stop(net::StopSignal::Interrupt);
	auto second = h->server.turn(h->service, net::Clock::now() + 200ms);
	return first && *first == http::Turn::Running && second && *second == http::Turn::Stopped &&
	       h->service.gone == std::vector{h->service.exchanges.front()};
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "keep-alive: two requests on one connection, ids issued or echoed",
		                      .holds = keep_alive_serves_requests_in_turn()},
	    engine::testing::Case{.name = "an insane X-Request-Id is replaced, and issued ids differ",
		                      .holds = an_insane_client_id_is_replaced()},
	    engine::testing::Case{.name = "Connection: close is answered and the next pipelined request is not read",
		                      .holds = connection_close_is_honoured()},
	    engine::testing::Case{.name = "refusals, 405s with their Allow among them, are OpenAI error objects with stable codes, then the "
		                              "connection closes",
		                      .holds = refusals_are_named_and_close_the_connection()},
	    engine::testing::Case{.name = "pipelined requests on every route, sent whole, byte by byte and in 7-byte pieces, draw the same "
		                              "answers and reach the service as sent (seed 1)",
		                      .holds = socket_fragmentation_never_changes_the_answers(1)},
	    engine::testing::Case{.name = "pipelined requests on every route, sent whole, byte by byte and in 7-byte pieces, draw the same "
		                              "answers and reach the service as sent (seed 2)",
		                      .holds = socket_fragmentation_never_changes_the_answers(2)},
	    engine::testing::Case{.name = "a deferred exchange is answered through the mailbox",
		                      .holds = a_deferred_answer_arrives_through_the_mailbox()},
	    engine::testing::Case{.name = "server-sent events round-trip random data, then keep-alive serves on (seed 1)",
		                      .holds = events_round_trip(1)},
	    engine::testing::Case{.name = "server-sent events round-trip random data, then keep-alive serves on (seed 2)",
		                      .holds = events_round_trip(2)},
	    engine::testing::Case{.name = "a client that does not read is held to the window, then drained in order",
		                      .holds = a_slow_client_is_held_to_the_window()},
	    engine::testing::Case{.name = "a stalled client is abandoned at the write deadline",
		                      .holds = a_stalled_client_is_abandoned_at_the_write_deadline()},
	    engine::testing::Case{.name = "a client closing mid-stream abandons the stream",
		                      .holds = a_departing_client_abandons_its_stream(Departure::Close)},
	    engine::testing::Case{.name = "a client resetting mid-stream abandons the stream",
		                      .holds = a_departing_client_abandons_its_stream(Departure::Reset)},
	    engine::testing::Case{.name = "a client leaving before a deferred answer abandons it",
		                      .holds = a_client_that_leaves_before_the_answer_abandons_it()},
	    engine::testing::Case{.name = "a partial request is closed at the read deadline",
		                      .holds = an_idle_connection_is_closed_at_the_read_deadline()},
	    engine::testing::Case{.name = "Expect: 100-continue draws the interim response before the body",
		                      .holds = continue_is_sent_before_the_body()},
	    engine::testing::Case{.name = "a whole response over max_response_body_bytes is answered 500 as an OpenAI error object",
		                      .holds = an_oversized_answer_is_a_500()},
	    engine::testing::Case{.name = "stop closes idle connections, refuses new ones, and drains the open stream",
		                      .holds = stop_drains_in_flight_streams()},
	    engine::testing::Case{.name = "a second stop signal stops hard and abandons the exchange in flight",
		                      .holds = a_second_signal_stops_hard()},
	};
	return engine::testing::report(cases);
}

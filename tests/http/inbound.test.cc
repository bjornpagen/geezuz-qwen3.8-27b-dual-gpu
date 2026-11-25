// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.http;
import engine.testing;

namespace http = engine::http;

namespace {

/** The body bounds these tests instantiate the server with; the application uses its json parser's own. */
constexpr http::BodyBounds bounds{.bytes = std::size_t{1} << 20U, .padding = 64};

using Rng = std::mt19937_64;

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
using Expected = std::variant<http::Assembled<bounds>, http::HttpErrorKind>;

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

constexpr std::string_view id_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:";
constexpr std::string_view value_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 !\"#$%&'()*+,-./;<=>?@[\\]^_`{|}~";

/** A valid request of a random route, with random spellings, ids, persistence, fillers and (for chat) body. */
[[nodiscard]] auto valid(Rng& rng) -> Generated {
	auto const route = pick(rng, 4);
	auto const query = chance(rng, 4) ? std::string{"?x=1"} : std::string{};
	auto draft = Draft{.request_line = "", .fields = {}, .body = ""};
	auto expected =
	    http::Assembled<bounds>{.route = http::Health{}, .client_id = std::nullopt, .persistence = http::Persistence::KeepAlive};
	auto continues = false;
	draft.fields.push_back(field(rng, "Host", "localhost:8080"));
	switch (route) {
	case 0: {
		draft.request_line = "POST /v1/chat/completions" + query + " HTTP/1.1";
		auto body = std::string(pick(rng, 3) == 0 ? 0 : pick(rng, 3000), '\0');
		std::ranges::generate(body, [&] {
			return static_cast<char>(rng() & 0xffU);
		});
		draft.fields.push_back(field(rng, "Content-Length", std::format("{}{}", chance(rng, 5) ? "00" : "", body.size())));
		draft.fields.push_back(field(rng, "Content-Type", "application/json"));
		if (chance(rng, 3)) {
			draft.fields.push_back(field(rng, "Expect", spell(rng, "100-continue")));
			continues = !body.empty();
		}
		expected.route = http::ChatCompletions<bounds>{.body = http::Body<bounds>::copy_of(body)};
		draft.body = std::move(body);
		break;
	}
	case 1:
		draft.request_line = "GET /v1/models" + query + " HTTP/1.1";
		expected.route = http::ListModels{};
		break;
	case 2:
		draft.request_line = "GET /health" + query + " HTTP/1.1";
		expected.route = http::Health{};
		if (chance(rng, 4)) {
			draft.fields.push_back(field(rng, "Content-Length", "0"));
		}
		break;
	default:
		draft.request_line = "GET /status" + query + " HTTP/1.1";
		expected.route = http::ServerStatus{};
		break;
	}
	switch (pick(rng, 4)) {
	case 0: {
		auto const id = text(rng, 1 + pick(rng, http::max_request_id_bytes), id_alphabet);
		draft.fields.push_back(field(rng, "X-Request-Id", id));
		expected.client_id = http::RequestId::parse(id);
		break;
	}
	case 1: {
		// An insane id (too long, or a character outside the alphabet) is replaced, not refused.
		auto const id = chance(rng, 2) ? text(rng, http::max_request_id_bytes + 1, id_alphabet) : std::string{"has/slash"};
		draft.fields.push_back(field(rng, "X-Request-Id", id));
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

[[nodiscard]] auto get_draft(Rng& rng) -> Draft {
	return Draft{.request_line = "GET /health HTTP/1.1", .fields = {field(rng, "Host", "h")}, .body = ""};
}

[[nodiscard]] auto chat_draft(Rng& rng) -> Draft {
	return Draft{.request_line = "POST /v1/chat/completions HTTP/1.1",
	             .fields = {field(rng, "Host", "h"), field(rng, "Content-Length", "2")},
	             .body = "{}"};
}

/** A request with exactly one named defect, and the refusal it must draw. */
[[nodiscard]] auto defective(Rng& rng) -> Generated {
	using enum http::HttpErrorKind;
	auto draft = get_draft(rng);
	auto kind = Malformed;
	switch (pick(rng, 26)) {
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
		kind = UnsupportedVersion;
		break;
	case 9:
		draft.fields.clear();
		kind = HostRequired;
		break;
	case 10:
		draft.fields.push_back(field(rng, "Host", "second"));
		kind = HostRequired;
		break;
	case 11:
		draft = chat_draft(rng);
		draft.fields.push_back(field(rng, "Transfer-Encoding", "chunked"));
		kind = AmbiguousFraming;
		break;
	case 12:
		draft = chat_draft(rng);
		draft.fields.pop_back();
		draft.fields.push_back(field(rng, "Transfer-Encoding", spell(rng, "chunked")));
		kind = ChunkedUpload;
		break;
	case 13:
		draft = chat_draft(rng);
		draft.fields.pop_back();
		draft.fields.push_back(field(rng, "Transfer-Encoding", "chunked, gzip"));
		kind = UnsupportedTransferCoding;
		break;
	case 14: {
		constexpr auto lengths = std::array<std::string_view, 5>{"+2", "2, 2", "two", "99999999999999999999999", "-1"};
		draft = chat_draft(rng);
		draft.fields.back() = field(rng, "Content-Length", lengths[pick(rng, lengths.size())]);
		kind = BadContentLength;
		break;
	}
	case 15:
		draft = chat_draft(rng);
		draft.fields.push_back(field(rng, "Content-Length", "2"));
		kind = BadContentLength;
		break;
	case 16:
		draft = chat_draft(rng);
		draft.fields.pop_back();
		kind = LengthRequired;
		break;
	case 17:
		draft = chat_draft(rng);
		draft.fields.back() = field(rng, "Content-Length", std::format("{}", bounds.bytes + 1));
		draft.body.clear();
		kind = BodyTooLarge;
		break;
	case 18:
		draft.fields.push_back(field(rng, "Content-Length", "3"));
		draft.body = "abc";
		kind = BodyNotAllowed;
		break;
	case 19:
		draft.fields.push_back(field(rng, "Expect", "200-ok"));
		kind = UnsupportedExpectation;
		break;
	case 20:
		draft.request_line = "GET /v1/missing HTTP/1.1";
		kind = NotFound;
		break;
	case 21:
		draft.request_line = "GET /v1/chat/completions HTTP/1.1";
		kind = ExpectedPost;
		break;
	case 22:
		draft.request_line = chance(rng, 2) ? "POST /health HTTP/1.1" : "DELETE /v1/models HTTP/1.1";
		kind = ExpectedGet;
		break;
	case 23:
		for (auto index = std::size_t{0}; index < http::max_header_count; ++index) {
			draft.fields.push_back(std::format("X-Filler-{}: {}", index, index));
		}
		kind = TooManyHeaders;
		break;
	case 24:
		draft.fields.push_back("X-Huge: " + std::string(http::max_head_bytes, 'a'));
		kind = HeadTooLarge;
		break;
	default:
		draft.request_line = "GET /health HTTP/11";
		break;
	}
	std::ranges::shuffle(draft.fields, rng);
	return Generated{.draft = std::move(draft), .expected = kind, .continues = false};
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

using Outcome = std::variant<http::Assembled<bounds>, http::HttpErrorKind>;

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
			observed.outcomes.emplace_back(std::get<http::Refused>(assembly).error.kind);
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
		auto const whole = deliver(generated.bytes, [] {
			return std::numeric_limits<std::size_t>::max();
		});
		auto const pieces = deliver(generated.bytes, [&rng] {
			return 1 + pick(rng, 64);
		});
		auto const bytes = deliver(generated.bytes, [] {
			return std::size_t{1};
		});
		if (whole.outcomes != expected || pieces.outcomes != expected || bytes.outcomes != expected || whole.continues > asked ||
		    pieces.continues > asked || bytes.continues != asked) {
			std::println("seed {} stream {}: mismatch over {} requests", seed, index, generated.requests.size());
			return false;
		}
	}
	return true;
}

constexpr std::string_view simple = "GET /hello HTTP/1.1\r\nHost: x\r\n\r\n";

[[nodiscard]] auto every_torn_prefix_is_partial() -> bool {
	constexpr std::string_view request = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\nhello";
	for (auto length = std::size_t{0}; length < request.size(); ++length) {
		auto inbound = std::make_unique<http::Inbound<bounds>>();
		std::ranges::copy(request.substr(0, length), inbound->space().begin());
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
		return deliver(head,
		               [] {
			               return std::size_t{4096};
		               })
		    .outcomes;
	};
	auto const fits = outcome(http::max_head_bytes);
	auto const over = outcome(http::max_head_bytes + 1);
	return fits.size() == 1 && std::holds_alternative<http::Assembled<bounds>>(fits.front()) &&
	       over == std::vector<Outcome>{http::HttpErrorKind::HeadTooLarge};
}

[[nodiscard]] auto a_body_at_the_limit_is_read() -> bool {
	auto body = std::string(bounds.bytes, 'b');
	auto const request = std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\nContent-Length: {}\r\n\r\n", body.size()) + body;
	auto const observed = deliver(request, [] {
		return std::size_t{1} << 16U;
	});
	if (observed.outcomes.size() != 1 || !std::holds_alternative<http::Assembled<bounds>>(observed.outcomes.front())) {
		return false;
	}
	auto const& route = std::get<http::Assembled<bounds>>(observed.outcomes.front()).route;
	if (!std::holds_alternative<http::ChatCompletions<bounds>>(route)) {
		return false;
	}
	auto const& read = std::get<http::ChatCompletions<bounds>>(route).body;
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
		return deliver(head + "\r\n",
		               [] {
			               return std::size_t{4096};
		               })
		    .outcomes;
	};
	auto const at = with(http::max_header_count);
	return at.size() == 1 && std::holds_alternative<http::Assembled<bounds>>(at.front()) &&
	       with(http::max_header_count + 1) == std::vector<Outcome>{http::HttpErrorKind::TooManyHeaders};
}

[[nodiscard]] auto nothing_after_a_refusal_is_parsed() -> bool {
	auto const observed = deliver(std::string{"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n"} + std::string{simple}, [] {
		return std::size_t{4096};
	});
	return observed.outcomes == std::vector<Outcome>{http::HttpErrorKind::NotFound};
}

[[nodiscard]] auto status_of_is_the_named_response() -> bool {
	using Kind = http::HttpErrorKind;
	using Status = http::StatusCode;
	return http::status_of(Kind::ChunkedUpload) == Status::LengthRequired &&
	       http::status_of(Kind::LengthRequired) == Status::LengthRequired &&
	       http::status_of(Kind::BodyTooLarge) == Status::ContentTooLarge &&
	       http::status_of(Kind::AmbiguousFraming) == Status::BadRequest &&
	       http::status_of(Kind::UnsupportedTransferCoding) == Status::BadRequest &&
	       http::status_of(Kind::HeadTooLarge) == Status::HeaderFieldsTooLarge &&
	       http::status_of(Kind::UnsupportedVersion) == Status::VersionNotSupported &&
	       http::status_of(Kind::ExpectedGet) == Status::MethodNotAllowed && http::status_of(Kind::NotFound) == Status::NotFound;
}

/** Every refusal kind has its own error code: a client or a log can tell any two refusals apart by it. */
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
	                                  ExpectedPost,
	                                  ExpectedGet};
	auto const codes = kinds | std::views::transform(http::error_code) | std::ranges::to<std::set>();
	return codes.size() == kinds.size();
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
	    engine::testing::Case{.name = "every torn prefix of a request is partial, never a refusal",
		                      .holds = every_torn_prefix_is_partial()},
	    engine::testing::Case{.name = "a head of max_head_bytes is read; one byte more is 431",
		                      .holds = a_head_that_fills_the_buffer_is_refused_and_one_that_fits_is_not()},
	    engine::testing::Case{.name = "a body at the route's bound is read whole, followed by its padding",
		                      .holds = a_body_at_the_limit_is_read()},
	    engine::testing::Case{.name = "max_header_count fields are read; one more is refused",
		                      .holds = exactly_max_header_count_fields_are_accepted()},
	    engine::testing::Case{.name = "nothing after a refused request is parsed", .holds = nothing_after_a_refusal_is_parsed()},
	    engine::testing::Case{.name = "framing refusals are the named 411, 413 and 400 responses",
		                      .holds = status_of_is_the_named_response()},
	    engine::testing::Case{.name = "every refusal kind has its own error code", .holds = error_codes_are_distinct()},
	    engine::testing::Case{.name = "a request id is 1 to 64 id characters", .holds = request_ids_are_sane_or_absent()},
	};
	return engine::testing::report(cases);
}

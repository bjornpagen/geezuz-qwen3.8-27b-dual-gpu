// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.grammar;
import engine.http;
import engine.http_test_client;
import engine.json;
import engine.net;
import engine.openai;
import engine.openai_engine_double;
import engine.openai_harness;
import engine.pipeline;
import engine.sched;
import engine.testing;
import engine.tokenizer;

namespace {

namespace http = engine::http;
namespace json = engine::json;
namespace openai = engine::openai;
namespace pipeline = engine::pipeline;
namespace sched = engine::sched;
namespace tok = engine::tokenizer;
namespace client = engine::http_test_client;
using engine::openai_engine_double::Cancels;
using engine::openai_engine_double::EngineDouble;
using engine::openai_engine_double::Script;
using engine::openai_harness::create;
using engine::openai_harness::echoes;
using engine::openai_harness::FixedSources;
using engine::openai_harness::get;
using engine::openai_harness::Json;
using engine::openai_harness::quick;
using engine::openai_harness::Timeouts;
using namespace std::chrono_literals;

/** Bounds small enough to reach in a test: three requests at once, and a 64 KiB backlog. */
constexpr auto small = openai::Limits{
    .requests = 3,
    .backlog_bytes = std::size_t{64} << 10U,
    .store = {.responses = 16, .bytes = std::size_t{1} << 24U},
};

/** The server's own backlog bound, for a client that reads late but reads. */
constexpr auto roomy =
    openai::Limits{.requests = small.requests, .backlog_bytes = openai::server_limits.backlog_bytes, .store = small.store};

template<openai::Limits L>
using Harness = engine::openai_harness::Harness<openai::BasicService<EngineDouble, FixedSources, L>>;

template<openai::Limits L>
[[nodiscard]] auto loading(Timeouts timeouts, Cancels cancels) -> std::unique_ptr<Harness<L>> {
	return engine::openai_harness::harness<openai::BasicService<EngineDouble, FixedSources, L>>(
	    timeouts,
	    [cancels](http::Poster<openai::Update> poster) {
		    return EngineDouble{poster, cancels};
	    },
	    FixedSources{.state = 7});
}

template<openai::Limits L>
[[nodiscard]] auto serving(Timeouts timeouts, Cancels cancels) -> std::unique_ptr<Harness<L>> {
	auto h = loading<L>(timeouts, cancels);
	if (!h || !h->service.engine().post(openai::Update{openai::Ready{}})) {
		return nullptr;
	}
	h->idle(5ms);
	return h;
}

[[nodiscard]] auto finished() -> openai::Output {
	return openai::Finished{.ending = sched::Stopped{.sequence = {1, tok::id(tok::Added::im_end)}},
	                        .usage = json::Usage{.input_tokens = 1, .cached_tokens = 0, .output_tokens = 1, .reasoning_tokens = 0}};
}

constexpr std::string_view hello = R"({"input":"hi"})";
constexpr std::string_view hello_streamed = R"({"input":"hi","stream":true})";

[[nodiscard]] auto code_of(Json& reading, client::Parsed const& answered) -> std::string {
	return reading.text(answered.body, {"error", "code"}).value_or("?");
}

/**
 * Readiness is held once: while loading, /health is 503 with the reason and every other route is refused 503
 * `engine_loading`; once the engine posts that it serves, all are served.
 */
[[nodiscard]] auto readiness_gates_every_route() -> bool {
	auto h = loading<small>(quick, Cancels::Answered);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto const health = h->ask(*peer, get("/health", "health-1"));
	auto const refused = h->ask(*peer, create(hello, "create-1"));
	auto const models = h->ask(*peer, get("/v1/models", ""));
	auto const status = h->ask(*peer, get("/status", ""));
	auto const stored = h->ask(*peer, get("/v1/responses/resp_00000000000000000000000000000000", ""));
	if (!health || !refused || !models || !status || !stored || !h->service.engine().post(openai::Update{openai::Ready{}})) {
		return false;
	}
	h->idle(5ms);
	auto const ready = h->ask(*peer, get("/health", "health-2"));
	auto const listed = h->ask(*peer, get("/v1/models", ""));
	auto const snapshot = h->ask(*peer, get("/status", ""));
	return health->status == 503 && reading->text(health->body, {"status"}) == "loading" &&
	       reading->text(health->body, {"reason"}).has_value() && echoes(*health, "health-1") && refused->status == 503 &&
	       code_of(*reading, *refused) == "engine_loading" && reading->text(refused->body, {"error", "type"}) == "server_error" &&
	       echoes(*refused, "create-1") && models->status == 503 && status->status == 503 && stored->status == 503 &&
	       code_of(*reading, *stored) == "engine_loading" && ready && ready->status == 200 &&
	       reading->text(ready->body, {"status"}) == "ok" && listed && listed->status == 200 &&
	       reading->text(listed->body, {"data", std::size_t{0}, "id"}) == openai::model_id && snapshot && snapshot->status == 200 &&
	       reading->kind(snapshot->body, {"engine"}) == json::Kind::Null && h->service.engine().submissions().empty();
}

/** An engine that could not start is reported as such: /health 503 with its reason, every create 503 `engine_failed`. */
[[nodiscard]] auto a_failed_start_fails_closed() -> bool {
	auto h = loading<small>(quick, Cancels::Answered);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading || !h->service.engine().post(openai::Update{openai::EngineFailed{.reason = "no CUDA driver on this machine"}})) {
		return false;
	}
	h->idle(5ms);
	auto const health = h->ask(*peer, get("/health", ""));
	auto const refused = h->ask(*peer, create(hello, ""));
	return health && health->status == 503 && reading->text(health->body, {"status"}) == "failed" &&
	       reading->text(health->body, {"reason"}) == "no CUDA driver on this machine" && refused && refused->status == 503 &&
	       code_of(*reading, *refused) == "engine_failed" &&
	       reading->text(refused->body, {"error", "message"}).value_or("").contains("no CUDA driver") &&
	       h->service.engine().submissions().empty();
}

/** /status writes the engine's latest snapshot: every slot, the queue, the pool and the counters. */
[[nodiscard]] auto status_writes_the_latest_snapshot() -> bool {
	auto h = serving<small>(quick, Cancels::Answered);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto snapshot =
	    openai::Snapshot{.slots = {}, .queued = 2, .pool = {.blocks = 100, .used_blocks = 30, .cached_blocks = 20}, .counters = {}};
	snapshot.slots[0] =
	    pipeline::Prefilling<std::string>{.tag = "req-a", .prompt_tokens = 100, .cached_tokens = 40, .prefilled_tokens = 64};
	snapshot.slots[3] = pipeline::Decoding<std::string>{.tag = "req-b", .prompt_tokens = 10, .cached_tokens = 0, .generated_tokens = 7};
	snapshot.counters.cycles = 50;
	snapshot.counters.preemptions = 1;
	if (!h->service.engine().post(openai::Update{snapshot})) {
		return false;
	}
	h->idle(5ms);
	auto const answered = h->ask(*peer, get("/status", "status-1"));
	if (!answered) {
		return false;
	}
	auto const& body = answered->body;
	return answered->status == 200 && echoes(*answered, "status-1") && reading->count(body, {"engine", "slots"}) == 8 &&
	       reading->text(body, {"engine", "slots", std::size_t{0}, "phase"}) == "prefill" &&
	       reading->text(body, {"engine", "slots", std::size_t{0}, "request"}) == "req-a" &&
	       reading->integer(body, {"engine", "slots", std::size_t{0}, "prefilled_tokens"}) == 64 &&
	       reading->text(body, {"engine", "slots", std::size_t{1}, "phase"}) == "vacant" &&
	       reading->integer(body, {"engine", "slots", std::size_t{3}, "generated_tokens"}) == 7 &&
	       reading->integer(body, {"engine", "queued"}) == 2 && reading->integer(body, {"engine", "kv_pool", "used_blocks"}) == 30 &&
	       reading->integer(body, {"engine", "counters", "cycles"}) == 50 &&
	       reading->integer(body, {"engine", "counters", "preemptions"}) == 1;
}

/** The exchange of the engine's latest submission. */
template<class H>
[[nodiscard]] auto latest(H& h) -> http::Exchange {
	return h.service.engine().submissions().back().first;
}

/** Sends a request on `peer` and waits until the engine has it; its exchange. */
template<class H>
[[nodiscard]] auto submitted(H& h, client::Client& peer, std::string_view body) -> std::optional<http::Exchange> {
	auto const before = h.service.engine().submissions().size();
	if (!h.push(peer, create(body, "")) || !h.wait(
	                                           [&] {
		                                           return h.service.engine().submissions().size() > before;
	                                           },
	                                           2000ms)) {
		return std::nullopt;
	}
	return latest(h);
}

/**
 * An engine that fails while it serves fails closed: a stream under way ends with `response.failed` carrying the
 * error, a whole response and a request not yet accepted are answered 500, and new requests are refused 503.
 */
[[nodiscard]] auto an_engine_failure_ends_every_request() -> bool {
	auto h = serving<small>(quick, Cancels::Answered);
	auto reading = Json::make();
	auto streaming = h ? client::connect(h->server.port()) : std::nullopt;
	auto gathering = h ? client::connect(h->server.port()) : std::nullopt;
	auto waiting = h ? client::connect(h->server.port()) : std::nullopt;
	if (!reading || !streaming || !gathering || !waiting) {
		return false;
	}
	auto const stream = submitted(*h, *streaming, hello_streamed);
	auto const whole = submitted(*h, *gathering, hello);
	auto const pending = submitted(*h, *waiting, hello);
	if (!stream || !whole || !pending || !h->service.engine().post(*stream, {openai::Accepted{}, openai::Content{.text = "partial"}}) ||
	    !h->service.engine().post(*whole, {openai::Accepted{}, openai::Content{.text = "partial"}}) ||
	    !h->service.engine().post(openai::Update{openai::EngineFailed{.reason = "card 1 lost its link"}})) {
		return false;
	}
	auto const streamed = h->ask(*streaming, "");
	auto const gathered = h->ask(*gathering, "");
	auto const pended = h->ask(*waiting, "");
	auto fresh = client::connect(h->server.port());
	auto const refused = fresh ? h->ask(*fresh, create(hello, "")) : std::nullopt;
	auto const health = fresh ? h->ask(*fresh, get("/health", "")) : std::nullopt;
	if (!streamed || !gathered || !pended || !refused || !health) {
		return false;
	}
	auto const events = client::decode_events(streamed->body);
	auto const& last = events.back();
	return reading->text(last, {"type"}) == "response.failed" && reading->text(last, {"response", "status"}) == "failed" &&
	       reading->text(last, {"response", "error", "code"}) == "server_error" &&
	       reading->text(last, {"response", "error", "message"}) == "card 1 lost its link" &&
	       reading->integer(last, {"sequence_number"}) == std::ssize(events) - 1 && gathered->status == 500 &&
	       code_of(*reading, *gathered) == "server_error" && pended->status == 500 && refused->status == 503 &&
	       code_of(*reading, *refused) == "engine_failed" && health->status == 503 &&
	       reading->text(health->body, {"reason"}) == "card 1 lost its link" && h->service.engine().cancellations().empty();
}

/**
 * The intake bound counts every engine request without its terminal output: a request whose client left still
 * counts until the engine answers its cancel, so creates past the bound are refused 503 `engine_busy` until then.
 */
[[nodiscard]] auto the_intake_bound_counts_unanswered_requests() -> bool {
	auto h = serving<small>(quick, Cancels::Held);
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto peers = std::vector<std::optional<client::Client>>{};
	auto exchanges = std::vector<http::Exchange>{};
	for (auto index = std::size_t{0}; index < small.requests; ++index) {
		auto peer = client::connect(h->server.port());
		auto const exchange = peer ? submitted(*h, *peer, hello) : std::nullopt;
		if (!exchange) {
			return false;
		}
		peers.push_back(std::move(peer));
		exchanges.push_back(*exchange);
	}
	auto extra = client::connect(h->server.port());
	auto const busy = extra ? h->ask(*extra, create(hello, "busy")) : std::nullopt;
	// The first client leaves: its request is cancelled, but the engine has not let go of it yet.
	peers.front().reset();
	if (!h->wait(
	        [&] {
		        return h->service.engine().cancellations().size() == 1;
	        },
	        2000ms)) {
		return false;
	}
	auto again = client::connect(h->server.port());
	auto const still = again ? h->ask(*again, create(hello, "still")) : std::nullopt;
	if (!h->service.engine().post(exchanges.front(), {openai::Cancelled{}})) {
		return false;
	}
	h->idle(5ms);
	auto const before = h->service.engine().submissions().size();
	auto freed = client::connect(h->server.port());
	auto const taken = freed ? submitted(*h, *freed, hello) : std::nullopt;
	// A finished request frees its place as well.
	h->service.engine().queue(Script{});
	auto const answered = h->service.engine().post(exchanges[1], {openai::Accepted{}, finished()});
	auto const done = answered ? h->ask(peers[1].value(), "") : std::nullopt;
	auto last = client::connect(h->server.port());
	auto const after_finish = last ? submitted(*h, *last, hello) : std::nullopt;
	return busy && busy->status == 503 && code_of(*reading, *busy) == "engine_busy" && echoes(*busy, "busy") && still &&
	       still->status == 503 && h->service.engine().cancellations().front() == exchanges.front() && taken &&
	       h->service.engine().submissions().size() == before + 2 && done && done->status == 200 && after_finish;
}

[[nodiscard]] auto content_of(Json& reading, std::vector<std::string> const& events) -> std::string {
	auto text = std::string{};
	for (auto const& event : events) {
		if (reading.text(event, {"type"}) == "response.output_text.delta") {
			text += reading.text(event, {"delta"}).value_or("");
		}
	}
	return text;
}

/**
 * A client that reads nothing while the engine writes: once the server refuses events, deltas merge instead of
 * queueing, and when the client reads, every delta arrives in order, in fewer events than the engine wrote.
 */
[[nodiscard]] auto a_slow_client_receives_merged_deltas() -> bool {
	auto h = serving<roomy>(Timeouts{.read = 5000ms, .write = 10000ms}, Cancels::Answered);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto const exchange = peer ? submitted(*h, *peer, hello_streamed) : std::nullopt;
	if (!exchange || !reading || !h->service.engine().post(*exchange, {openai::Accepted{}})) {
		return false;
	}
	auto written = std::string{};
	constexpr auto pieces = 3000;
	for (auto index = 0; index < pieces; ++index) {
		auto text = std::format("{:05}:{}|", index, std::string(250, static_cast<char>('a' + (index % 26))));
		written += text;
		if (!h->service.engine().post(*exchange, {openai::Content{.text = std::move(text)}})) {
			return false;
		}
		if (index % 32 == 0) {
			h->turn();
		}
	}
	if (!h->service.engine().post(*exchange, {finished()})) {
		return false;
	}
	h->idle(20ms);
	auto const answered = h->ask(*peer, "");
	if (!answered) {
		return false;
	}
	auto const events = client::decode_events(answered->body);
	auto const deltas = std::ranges::count_if(events, [&](std::string const& event) {
		return reading->text(event, {"type"}) == "response.output_text.delta";
	});
	auto const numbered = std::ranges::all_of(std::views::enumerate(events), [&](auto const& entry) {
		auto const& [position, event] = entry;
		return reading->integer(event, {"sequence_number"}) == position;
	});
	return content_of(*reading, events) == written && deltas < pieces && numbered &&
	       reading->text(events.back(), {"type"}) == "response.completed" && h->service.engine().cancellations().empty();
}

/**
 * A client that never reads cannot make the server hold more than the backlog bound: past it the request is
 * cancelled, and the stream ends with a `stream_backlog` error after a prefix of what the engine wrote.
 */
[[nodiscard]] auto a_stalled_stream_is_bounded() -> bool {
	auto h = serving<small>(Timeouts{.read = 5000ms, .write = 20000ms}, Cancels::Answered);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto const exchange = peer ? submitted(*h, *peer, hello_streamed) : std::nullopt;
	if (!exchange || !reading || !h->service.engine().post(*exchange, {openai::Accepted{}})) {
		return false;
	}
	auto written = std::string{};
	// 16 MiB without a cancellation would be an unbounded buffer.
	for (auto index = 0; index < 1024 && h->service.engine().cancellations().empty(); ++index) {
		auto text = std::format("{:05}{}", index, std::string(16'384 - 5, static_cast<char>('a' + (index % 26))));
		written += text;
		if (!h->service.engine().post(*exchange, {openai::Content{.text = std::move(text)}})) {
			return false;
		}
		h->turn();
		h->turn();
	}
	if (h->service.engine().cancellations().size() != 1 || h->service.engine().cancellations().front() != *exchange) {
		return false;
	}
	auto const answered = h->ask(*peer, "");
	if (!answered) {
		return false;
	}
	auto const events = client::decode_events(answered->body);
	auto const received = content_of(*reading, events);
	return reading->text(events.back(), {"type"}) == "error" && reading->text(events.back(), {"code"}) == "stream_backlog" &&
	       written.starts_with(received) && received.size() < written.size();
}

enum class Departure : std::uint8_t {
	MidStream,
	WhileGathering,
	BeforeAcceptance,
};

/** A client that leaves cancels its request in the engine, once, and what the engine still posts for it is dropped. */
[[nodiscard]] auto a_departing_client_cancels_its_request(Departure departure) -> bool {
	auto h = serving<small>(quick, Cancels::Held);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer) {
		return false;
	}
	auto const body = departure == Departure::MidStream ? hello_streamed : hello;
	auto const exchange = submitted(*h, *peer, body);
	if (!exchange) {
		return false;
	}
	if (departure != Departure::BeforeAcceptance &&
	    !h->service.engine().post(*exchange, {openai::Accepted{}, openai::Content{.text = "first"}})) {
		return false;
	}
	h->idle(5ms);
	peer.reset();
	if (!h->wait(
	        [&] {
		        return !h->service.engine().cancellations().empty();
	        },
	        2000ms)) {
		return false;
	}
	// What the engine posted before it saw the cancel, then its answer to the cancel, are dropped without a trace.
	if (!h->service.engine().post(*exchange, {openai::Content{.text = "late"}, openai::Cancelled{}})) {
		return false;
	}
	h->idle(20ms);
	return h->service.engine().cancellations().size() == 1 && h->service.engine().cancellations().front() == *exchange;
}

/**
 * A request whose grammar outgrew its bounds while it decoded ends there by name: a whole response is refused 400
 * `grammar_overrun`, a stream ends with that `error` event after what it said; neither is stored, nothing is cancelled.
 */
[[nodiscard]] auto an_overrun_grammar_ends_its_request() -> bool {
	auto h = serving<small>(quick, Cancels::Answered);
	auto reading = Json::make();
	auto whole_peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto stream_peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!reading || !whole_peer || !stream_peer) {
		return false;
	}
	auto const fault = openai::Overran{.fault = engine::grammar::MatchError{.fault = engine::grammar::MatchFault::LexerTooLarge}};
	auto const whole = submitted(*h, *whole_peer, hello);
	auto const stream = submitted(*h, *stream_peer, hello_streamed);
	if (!whole || !stream || !h->service.engine().post(*whole, {openai::Accepted{}, openai::Content{.text = "so far"}, fault}) ||
	    !h->service.engine().post(*stream, {openai::Accepted{}, openai::Content{.text = "so far"}, fault})) {
		return false;
	}
	auto const refused = h->ask(*whole_peer, "");
	auto const streamed = h->ask(*stream_peer, "");
	if (!refused || !streamed) {
		return false;
	}
	auto const events = engine::http_test_client::decode_events(streamed->body);
	return refused->status == 400 && code_of(*reading, *refused) == "grammar_overrun" && content_of(*reading, events) == "so far" &&
	       reading->text(events.back(), {"type"}) == "error" && reading->text(events.back(), {"code"}) == "grammar_overrun" &&
	       h->service.engine().cancellations().empty();
}

/** A whole response that outgrows one body is cancelled in the engine and answered 500 `response_too_large`. */
[[nodiscard]] auto an_outgrown_whole_response_is_refused() -> bool {
	auto h = serving<small>(quick, Cancels::Answered);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto const exchange = peer ? submitted(*h, *peer, hello) : std::nullopt;
	if (!exchange || !reading || !h->service.engine().post(*exchange, {openai::Accepted{}})) {
		return false;
	}
	auto const piece = std::string(std::size_t{1} << 20U, 'x');
	for (auto index = std::size_t{0}; index * piece.size() <= http::max_response_body_bytes; ++index) {
		if (!h->service.engine().post(*exchange, {openai::Content{.text = piece}})) {
			return false;
		}
		h->turn();
	}
	auto const answered = h->ask(*peer, "");
	return answered && answered->status == 500 && code_of(*reading, *answered) == "response_too_large" &&
	       h->service.engine().cancellations().size() == 1;
}

}

auto main() -> int {
	auto const checks = std::array{
	    engine::testing::Case{.name = "readiness gates every route but /health until the engine serves",
		                      .holds = readiness_gates_every_route()},
	    engine::testing::Case{.name = "an engine that could not start fails closed", .holds = a_failed_start_fails_closed()},
	    engine::testing::Case{.name = "/status writes the engine's latest snapshot", .holds = status_writes_the_latest_snapshot()},
	    engine::testing::Case{.name = "an engine failure ends every request it owed and refuses new ones",
		                      .holds = an_engine_failure_ends_every_request()},
	    engine::testing::Case{.name = "the intake bound counts requests until the engine answers them",
		                      .holds = the_intake_bound_counts_unanswered_requests()},
	    engine::testing::Case{.name = "a slow client receives every delta, merged", .holds = a_slow_client_receives_merged_deltas()},
	    engine::testing::Case{.name = "a client that never reads is cut off at the backlog bound", .holds = a_stalled_stream_is_bounded()},
	    engine::testing::Case{.name = "a client leaving mid-stream cancels its request",
		                      .holds = a_departing_client_cancels_its_request(Departure::MidStream)},
	    engine::testing::Case{.name = "a client leaving a whole response cancels its request",
		                      .holds = a_departing_client_cancels_its_request(Departure::WhileGathering)},
	    engine::testing::Case{.name = "a client leaving before acceptance cancels its request",
		                      .holds = a_departing_client_cancels_its_request(Departure::BeforeAcceptance)},
	    engine::testing::Case{.name = "a grammar that outgrew its bounds ends its request by name",
		                      .holds = an_overrun_grammar_ends_its_request()},
	    engine::testing::Case{.name = "a whole response that outgrows one body is refused 500",
		                      .holds = an_outgrown_whole_response_is_refused()},
	};
	return engine::testing::report(checks);
}

// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.http;
import engine.http_test_client;
import engine.json;
import engine.net;
import engine.openai;
import engine.openai_engine_double;
import engine.sampling;
import engine.testing;

namespace http = engine::http;
namespace net = engine::net;
namespace json = engine::json;
namespace openai = engine::openai;
namespace client = engine::http_test_client;
using engine::openai_engine_double::EngineDouble;
using engine::openai_engine_double::Script;
using namespace std::chrono_literals;

namespace {

constexpr std::int64_t now = 1'700'000'000;

/** A fixed clock and a splitmix64 stream: every id and seed the service makes replays. */
struct FixedSources {
	std::uint64_t state;

	[[nodiscard]] auto unix_seconds() const -> std::int64_t {
		return now;
	}

	[[nodiscard]] auto draw() -> std::uint64_t {
		state += 0x9e37'79b9'7f4a'7c15ULL;
		auto value = state;
		value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
		value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
		return value ^ (value >> 31U);
	}
};

using Service = openai::Service<EngineDouble, FixedSources>;

static_assert(http::Service<Service, openai::Update, openai::body_bounds>);

struct Timeouts {
	std::chrono::milliseconds read;
	std::chrono::milliseconds write;
};

constexpr auto quick = Timeouts{.read = 5000ms, .write = 5000ms};

/** One server and its service, turned by the test's thread between client operations. */
struct Harness {
	openai::Server server;
	Service service;

	[[nodiscard]] auto engine() -> EngineDouble& {
		return service.engine();
	}

	auto turn() -> void {
		std::ignore = server.turn(service, net::Clock::now() + 1ms);
	}

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
			turn();
		}
	}

	/** Turns the server until `done` holds, reading nothing. */
	template<class Done>
	[[nodiscard]] auto wait(Done done, std::chrono::milliseconds budget) -> bool {
		auto const deadline = net::Clock::now() + budget;
		while (!done()) {
			if (net::Clock::now() >= deadline) {
				return false;
			}
			turn();
		}
		return true;
	}

	auto idle(std::chrono::milliseconds span) -> void {
		auto const deadline = net::Clock::now() + span;
		while (net::Clock::now() < deadline) {
			turn();
		}
	}

	[[nodiscard]] auto push(client::Client& peer, std::string_view bytes) -> bool {
		auto sent = std::size_t{0};
		auto const deadline = net::Clock::now() + 5000ms;
		while (sent < bytes.size() && net::Clock::now() < deadline) {
			sent += peer.push(bytes.substr(sent));
			turn();
		}
		return sent == bytes.size();
	}

	/** Sends one request on `peer` and waits for its whole response. */
	[[nodiscard]] auto ask(client::Client& peer, std::string_view request) -> std::optional<client::Parsed> {
		auto const before = client::parse_responses(peer.received).complete.size();
		if (!push(peer, request) || !pump(
		                                peer,
		                                [&] {
			                                return client::parse_responses(peer.received).complete.size() > before;
		                                },
		                                10000ms)) {
			return std::nullopt;
		}
		return client::parse_responses(peer.received).complete[before];
	}

	/** Posts the engine's status with every slot vacant. */
	[[nodiscard]] auto serving() -> bool {
		auto status = openai::loading_status();
		status.readiness = openai::Readiness::Serving;
		if (!engine().post_status(status)) {
			return false;
		}
		idle(5ms);
		return true;
	}
};

[[nodiscard]] auto harness(Timeouts timeouts) -> std::unique_ptr<Harness> {
	auto server = openai::Server::open(http::ServerConfig{
	    .endpoint = net::Endpoint{.address = net::loopback, .port = 0},
	    .read_timeout = timeouts.read,
	    .write_timeout = timeouts.write,
	    .id_seed = 11,
	});
	if (!server) {
		std::println("server open failed: errno {}", server.error().code);
		return nullptr;
	}
	auto service = Service::open(EngineDouble{server->poster()}, FixedSources{.state = 5});
	if (!service) {
		return nullptr;
	}
	return std::make_unique<Harness>(Harness{.server = std::move(*server), .service = std::move(*service)});
}

[[nodiscard]] auto serving_harness(Timeouts timeouts) -> std::unique_ptr<Harness> {
	auto h = harness(timeouts);
	if (!h || !h->serving()) {
		return nullptr;
	}
	return h;
}

[[nodiscard]] auto id_field(std::string_view id) -> std::string {
	return id.empty() ? std::string{} : std::format("X-Request-Id: {}\r\n", id);
}

[[nodiscard]] auto get(std::string_view path, std::string_view id) -> std::string {
	return std::format("GET {} HTTP/1.1\r\nHost: test\r\n{}\r\n", path, id_field(id));
}

[[nodiscard]] auto chat(std::string_view body, std::string_view id) -> std::string {
	return std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n{}",
	                   body.size(), id_field(id), body);
}

[[nodiscard]] auto echoes(client::Parsed const& response, std::string_view id) -> bool {
	auto const field = response.fields.find("x-request-id");
	return field != response.fields.end() && field->second == id;
}

using Step = std::variant<std::string_view, std::size_t>;

/** Reads members of a response body by path: the oracle reads JSON with the engine's reader, not by text. */
class Json {
public:
	[[nodiscard]] static auto make() -> std::optional<Json> {
		auto reader = json::Reader<json::request_bounds>::make();
		if (!reader) {
			return std::nullopt;
		}
		return Json{*std::move(reader)};
	}

	[[nodiscard]] auto text(std::string_view document, std::initializer_list<Step> path) -> std::optional<std::string> {
		return read<std::string>(document, path, [](auto& value) -> std::optional<std::string> {
			auto const text = value.as_string();
			return text ? std::optional{std::string{*text}} : std::nullopt;
		});
	}

	[[nodiscard]] auto integer(std::string_view document, std::initializer_list<Step> path) -> std::optional<std::int64_t> {
		return read<std::int64_t>(document, path, [](auto& value) -> std::optional<std::int64_t> {
			auto const number = value.as_integer();
			return number ? std::optional{*number} : std::nullopt;
		});
	}

	[[nodiscard]] auto real(std::string_view document, std::initializer_list<Step> path) -> std::optional<double> {
		return read<double>(document, path, [](auto& value) -> std::optional<double> {
			auto const number = value.as_real();
			return number ? std::optional{*number} : std::nullopt;
		});
	}

	[[nodiscard]] auto kind(std::string_view document, std::initializer_list<Step> path) -> std::optional<json::Kind> {
		return read<json::Kind>(document, path, [](auto& value) -> std::optional<json::Kind> {
			auto const kind = value.kind();
			return kind ? std::optional{*kind} : std::nullopt;
		});
	}

	[[nodiscard]] auto count(std::string_view document, std::initializer_list<Step> path) -> std::optional<std::size_t> {
		return read<std::size_t>(document, path, [](auto& value) -> std::optional<std::size_t> {
			auto array = value.as_array();
			if (!array) {
				return std::nullopt;
			}
			auto const size = array->size();
			return size ? std::optional{*size} : std::nullopt;
		});
	}

private:
	explicit Json(json::Reader<json::request_bounds> reader) : reader_{std::move(reader)} {}

	template<class T, class Value, class F>
	[[nodiscard]] static auto at(Value& value, std::span<Step const> path, F const& read) -> std::optional<T> {
		if (path.empty()) {
			return read(value);
		}
		if (auto const name = engine::core::alternative<std::string_view>(path.front())) {
			auto object = value.as_object();
			if (!object) {
				return std::nullopt;
			}
			auto member = object->find(*name);
			if (!member || !*member) {
				return std::nullopt;
			}
			return at<T>(**member, path.subspan(1), read);
		}
		auto array = value.as_array();
		if (!array) {
			return std::nullopt;
		}
		for (auto index = std::size_t{0};; ++index) {
			auto element = array->next();
			if (!element || !*element) {
				return std::nullopt;
			}
			if (index == std::get<std::size_t>(path.front())) {
				return at<T>(**element, path.subspan(1), read);
			}
		}
	}

	template<class T, class F>
	[[nodiscard]] auto read(std::string_view document, std::initializer_list<Step> path, F const& read) -> std::optional<T> {
		auto root = reader_.read_copy(document);
		if (!root) {
			return std::nullopt;
		}
		return at<T>(*root, std::span{path.begin(), path.size()}, read);
	}

	json::Reader<json::request_bounds> reader_;
};

constexpr std::string_view hello = R"({"messages":[{"role":"user","content":"hi"}]})";

[[nodiscard]] auto finished(std::uint64_t prompt, std::uint64_t completion, std::uint64_t cached) -> openai::Output {
	return openai::Finished{.ending = openai::Ending::Stop,
	                        .usage = json::Usage{.prompt_tokens = prompt, .completion_tokens = completion, .cached_tokens = cached}};
}

/** /health follows the engine's readiness, chat waits for it, and every answer carries the client's request id. */
[[nodiscard]] auto health_follows_readiness() -> bool {
	auto h = harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto const loading = h->ask(*peer, get("/health", "health-1"));
	auto const refused = h->ask(*peer, chat(hello, "chat-1"));
	if (!loading || !refused || !h->serving()) {
		return false;
	}
	auto const ready = h->ask(*peer, get("/health", "health-2"));
	return ready && loading->status == 503 && reading->text(loading->body, {"status"}) == "loading" && echoes(*loading, "health-1") &&
	       refused->status == 503 && reading->text(refused->body, {"error", "code"}) == "engine_loading" &&
	       reading->text(refused->body, {"error", "type"}) == "server_error" && echoes(*refused, "chat-1") && ready->status == 200 &&
	       reading->text(ready->body, {"status"}) == "ok" && echoes(*ready, "health-2") && h->engine().submissions().empty();
}

[[nodiscard]] auto models_lists_the_one_model() -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto const models = peer ? h->ask(*peer, get("/v1/models", "models-1")) : std::nullopt;
	if (!models || !reading) {
		return false;
	}
	auto const& body = models->body;
	return models->status == 200 && echoes(*models, "models-1") && models->fields.at("content-type") == "application/json" &&
	       reading->text(body, {"object"}) == "list" && reading->count(body, {"data"}) == 1 &&
	       reading->text(body, {"data", std::size_t{0}, "id"}) == openai::model_id &&
	       reading->text(body, {"data", std::size_t{0}, "object"}) == "model" &&
	       reading->integer(body, {"data", std::size_t{0}, "created"}) == now;
}

/** /status writes the engine's latest snapshot: every slot, the queue, the pool and the draft counters. */
[[nodiscard]] auto status_writes_the_latest_snapshot() -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto prefilling = http::RequestId::parse("req-a");
	auto decoding = http::RequestId::parse("req-b");
	if (!peer || !reading || !prefilling || !decoding) {
		return false;
	}
	auto status = openai::loading_status();
	status.readiness = openai::Readiness::Serving;
	status.slots[0] = openai::Prefilling{.id = *prefilling, .prompt_tokens = 100, .cached_tokens = 40, .prefilled_tokens = 64};
	status.slots[3] = openai::Decoding{.id = *decoding, .prompt_tokens = 10, .cached_tokens = 0, .generated_tokens = 7};
	status.queued = 2;
	status.preemptions = 1;
	status.pool = openai::Pool{.blocks = 100, .used_blocks = 30, .cached_blocks = 20};
	status.cycles = openai::Cycles{.cycles = 50, .drafted_tokens = 80, .accepted_tokens = 60};
	if (!h->engine().post_status(status)) {
		return false;
	}
	h->idle(5ms);
	auto const answered = h->ask(*peer, get("/status", "status-1"));
	if (!answered) {
		return false;
	}
	auto const& body = answered->body;
	return answered->status == 200 && echoes(*answered, "status-1") && reading->text(body, {"readiness"}) == "serving" &&
	       reading->integer(body, {"busy"}) == 2 && reading->integer(body, {"queued"}) == 2 && reading->integer(body, {"preemptions"}) == 1 &&
	       reading->count(body, {"slots"}) == 8 && reading->text(body, {"slots", std::size_t{0}, "phase"}) == "prefill" &&
	       reading->text(body, {"slots", std::size_t{0}, "id"}) == "req-a" &&
	       reading->integer(body, {"slots", std::size_t{0}, "prefilled_tokens"}) == 64 &&
	       reading->integer(body, {"slots", std::size_t{0}, "cached_tokens"}) == 40 &&
	       reading->text(body, {"slots", std::size_t{1}, "phase"}) == "vacant" &&
	       reading->integer(body, {"slots", std::size_t{3}, "slot"}) == 3 &&
	       reading->text(body, {"slots", std::size_t{3}, "phase"}) == "decode" &&
	       reading->integer(body, {"slots", std::size_t{3}, "generated_tokens"}) == 7 &&
	       reading->integer(body, {"kv_pool", "used_blocks"}) == 30 && reading->integer(body, {"kv_pool", "cached_blocks"}) == 20 &&
	       reading->real(body, {"draft", "accept_rate"}) == 0.75;
}

/** A whole completion is the engine's outputs gathered: reasoning and content concatenated, the usage reported. */
[[nodiscard]] auto a_whole_completion_gathers_the_outputs() -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	h->engine().queue(Script{
	    {openai::Accepted{}},
	    {openai::ReasoningText{.text = "Let me think."}, openai::ContentText{.text = "Hello, "}},
	    {openai::ContentText{.text = "wörld"}},
	    {finished(12, 5, 4)},
	});
	auto const answered = h->ask(*peer, chat(hello, "whole-1"));
	if (!answered || h->engine().submissions().size() != 1) {
		return false;
	}
	auto const& body = answered->body;
	auto const& submission = h->engine().submissions().front().second;
	auto const id = reading->text(body, {"id"});
	return answered->status == 200 && echoes(*answered, "whole-1") && id && id->starts_with("chatcmpl-") &&
	       reading->text(body, {"object"}) == "chat.completion" && reading->integer(body, {"created"}) == now &&
	       reading->text(body, {"model"}) == openai::model_id &&
	       reading->text(body, {"choices", std::size_t{0}, "message", "role"}) == "assistant" &&
	       reading->text(body, {"choices", std::size_t{0}, "message", "content"}) == "Hello, wörld" &&
	       reading->text(body, {"choices", std::size_t{0}, "message", "reasoning_content"}) == "Let me think." &&
	       !reading->kind(body, {"choices", std::size_t{0}, "message", "tool_calls"}) &&
	       reading->text(body, {"choices", std::size_t{0}, "finish_reason"}) == "stop" &&
	       reading->integer(body, {"usage", "prompt_tokens"}) == 12 && reading->integer(body, {"usage", "completion_tokens"}) == 5 &&
	       reading->integer(body, {"usage", "total_tokens"}) == 17 &&
	       reading->integer(body, {"usage", "prompt_tokens_details", "cached_tokens"}) == 4 && submission.id.text() == "whole-1" &&
	       submission.messages == std::vector<json::Message>{json::UserMessage{.content = "hi"}} &&
	       submission.tool_choice == json::ToolChoice{json::ToolChoiceMode::None} &&
	       submission.budget == openai::Budget{openai::UntilContextFull{}} && submission.stop.sequences().empty();
}

/** The sampling a request gets: its own fields, else its thinking mode's defaults; its seed, else a fresh one. */
[[nodiscard]] auto sampling_follows_the_thinking_mode() -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer) {
		return false;
	}
	auto const bodies = std::array<std::string_view, 4>{
	    hello,
	    R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}})",
	    R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false},"top_p":0.5,"seed":-3,"max_tokens":9})",
	    R"({"messages":[{"role":"user","content":"hi"}],"temperature":0})",
	};
	for (auto const body : bodies) {
		h->engine().queue(Script{{openai::Accepted{}, finished(1, 1, 0)}});
		auto const answered = h->ask(*peer, chat(body, ""));
		if (!answered || answered->status != 200) {
			return false;
		}
	}
	auto const submissions = h->engine().submissions();
	auto const sample = [&](std::size_t index) {
		return std::get<engine::sampling::Sample>(submissions[index].second.sampling);
	};
	auto const thinking = sample(0);
	auto const instruct = sample(1);
	auto const named = sample(2);
	return submissions.size() == 4 && thinking.temperature() == engine::sampling::thinking_defaults.temperature &&
	       thinking.top_p() == engine::sampling::thinking_defaults.top_p && thinking.top_k() == engine::sampling::thinking_defaults.top_k &&
	       instruct.temperature() == engine::sampling::instruct_defaults.temperature &&
	       instruct.top_p() == engine::sampling::instruct_defaults.top_p && named.top_p() == 0.5 &&
	       named.seed() == std::bit_cast<std::uint64_t>(std::int64_t{-3}) && thinking.seed() != instruct.seed() &&
	       submissions[2].second.budget == openai::Budget{openai::AtMost{.tokens = 9}} &&
	       std::holds_alternative<engine::sampling::Greedy>(submissions[3].second.sampling);
}

[[nodiscard]] auto chunk(std::string_view id, std::string_view choice) -> std::string {
	return std::format(R"(data: {{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{}]}})"
	                   "\n\n",
	                   id, now, openai::model_id, choice);
}

/**
 * A stream's body, byte for byte, against the OpenAI chunk format written out by hand: the role opening, each
 * delta in its own chunk, the finish, the usage chunk exactly when asked for, and [DONE].
 */
[[nodiscard]] auto a_stream_is_framed_exactly(json::Delivery delivery) -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	h->engine().queue(Script{
	    {openai::Accepted{}},
	    {openai::ReasoningText{.text = "r1"}},
	    {openai::ContentText{.text = "c\"1\né"}},
	    {openai::Finished{.ending = openai::Ending::Length,
	                      .usage = json::Usage{.prompt_tokens = 30, .completion_tokens = 2, .cached_tokens = 16}}},
	});
	auto const usage = delivery == json::Delivery::StreamWithUsage;
	auto const body = usage ? R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}})"
	                        : R"({"messages":[{"role":"user","content":"hi"}],"stream":true})";
	auto const answered = h->ask(*peer, chat(body, "stream-1"));
	if (!answered) {
		return false;
	}
	auto const events = client::decode_events(answered->body);
	auto const id = events.empty() ? std::nullopt : reading->text(events.front(), {"id"});
	if (!id) {
		return false;
	}
	auto expected = chunk(*id, R"({"index":0,"delta":{"role":"assistant"},"finish_reason":null})") +
	                chunk(*id, R"({"index":0,"delta":{"reasoning_content":"r1"},"finish_reason":null})") +
	                chunk(*id, "{\"index\":0,\"delta\":{\"content\":\"c\\\"1\\né\"},\"finish_reason\":null}") +
	                chunk(*id, R"({"index":0,"delta":{},"finish_reason":"length"})");
	if (usage) {
		expected += std::format(R"(data: {{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[],)"
		                        R"("usage":{{"prompt_tokens":30,"completion_tokens":2,"total_tokens":32,)"
		                        R"("prompt_tokens_details":{{"cached_tokens":16}}}}}})"
		                        "\n\n",
		                        *id, now, openai::model_id);
	}
	expected += "data: [DONE]\n\n";
	return answered->status == 200 && echoes(*answered, "stream-1") && answered->fields.at("content-type") == "text/event-stream" &&
	       id->starts_with("chatcmpl-") && answered->body == expected;
}

/**
 * A JSON string of random characters, with quotes, escapes, controls and multibyte text, long enough that its call's
 * arguments cannot travel in one event.
 */
template<class G>
[[nodiscard]] auto long_note(G& generator) -> std::string {
	constexpr auto pieces = std::array<std::string_view, 7>{"a", "\\\"", "\\n", "\\u0001", "\u00e9", "\u65e5", " "};
	auto note = std::string{};
	for (auto count = 0; count < 3000; ++count) {
		note += pieces[std::uniform_int_distribution<std::size_t>{0, pieces.size() - 1}(generator)];
	}
	return note;
}

struct StreamedCall {
	std::string id;
	std::string name;
	std::string arguments;
};

/**
 * The same tool calls streamed and whole: the streamed argument pieces of each call (split where one event cannot
 * hold them) concatenate to exactly the whole response's arguments and to what the engine wrote, names and order
 * agree, and both finish `tool_calls`.
 */
[[nodiscard]] auto streamed_arguments_concatenate_to_the_whole(std::uint64_t seed) -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto generator = engine::testing::generator(seed);
	auto const weather = std::format(R"({{"city": "Paris", "units": ["c", "f"], "note": "{}"}})", long_note(generator));
	auto const calls = std::array<std::pair<std::string_view, std::string_view>, 2>{{
	    {"get_weather", weather},
	    {"lookup", R"({"query": "日本", "limit": 3})"},
	}};
	auto const script = [&] {
		auto outputs = std::vector<openai::Output>{openai::Accepted{}, openai::ContentText{.text = "Checking."}};
		for (auto index = std::uint32_t{0}; index < calls.size(); ++index) {
			outputs.emplace_back(openai::CallBegin{.index = index, .name = std::string{calls[index].first}});
			outputs.emplace_back(openai::CallEnd{.index = index, .arguments = std::string{calls[index].second}});
		}
		outputs.push_back(finished(40, 30, 0));
		// One output per batch, as a cycle per token would post them.
		return outputs | std::views::transform([](openai::Output const& output) { return std::vector{output}; }) |
		       std::ranges::to<Script>();
	};
	constexpr auto tools = R"("tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}},)"
	                       R"({"type":"function","function":{"name":"lookup"}}])";
	h->engine().queue(script());
	h->engine().queue(script());
	auto const streamed = h->ask(*peer, chat(std::format(R"({{"messages":[{{"role":"user","content":"weather?"}}],"stream":true,{}}})", tools), ""));
	auto const whole = h->ask(*peer, chat(std::format(R"({{"messages":[{{"role":"user","content":"weather?"}}],{}}})", tools), ""));
	if (!streamed || !whole || streamed->status != 200 || whole->status != 200) {
		return false;
	}
	auto received = std::vector<StreamedCall>{};
	auto finish = std::optional<std::string>{};
	for (auto const& event : client::decode_events(streamed->body)) {
		if (event == "[DONE]") {
			continue;
		}
		if (auto const reason = reading->text(event, {"choices", std::size_t{0}, "finish_reason"})) {
			finish = reason;
		}
		auto const index = reading->integer(event, {"choices", std::size_t{0}, "delta", "tool_calls", std::size_t{0}, "index"});
		if (!index) {
			continue;
		}
		if (auto const id = reading->text(event, {"choices", std::size_t{0}, "delta", "tool_calls", std::size_t{0}, "id"})) {
			if (std::cmp_not_equal(*index, received.size())) {
				return false;
			}
			received.push_back(StreamedCall{
			    .id = *id,
			    .name = reading->text(event, {"choices", std::size_t{0}, "delta", "tool_calls", std::size_t{0}, "function", "name"})
			                .value_or(""),
			    .arguments = {},
			});
		}
		if (std::cmp_greater_equal(*index, received.size())) {
			return false;
		}
		received[engine::core::narrow<std::size_t>(*index)].arguments +=
		    reading->text(event, {"choices", std::size_t{0}, "delta", "tool_calls", std::size_t{0}, "function", "arguments"}).value_or("");
	}
	auto const& body = whole->body;
	if (received.size() != calls.size() || reading->count(body, {"choices", std::size_t{0}, "message", "tool_calls"}) != calls.size()) {
		return false;
	}
	for (auto index = std::size_t{0}; index < calls.size(); ++index) {
		auto const whole_arguments = reading->text(body, {"choices", std::size_t{0}, "message", "tool_calls", index, "function", "arguments"});
		auto const whole_name = reading->text(body, {"choices", std::size_t{0}, "message", "tool_calls", index, "function", "name"});
		auto const whole_id = reading->text(body, {"choices", std::size_t{0}, "message", "tool_calls", index, "id"});
		if (received[index].arguments != whole_arguments || whole_arguments != calls[index].second || received[index].name != whole_name ||
		    whole_name != calls[index].first || !whole_id || !whole_id->starts_with("call_") || !received[index].id.starts_with("call_")) {
			return false;
		}
	}
	return weather.size() > http::max_event_bytes && received[0].id != received[1].id && finish == "tool_calls" &&
	       reading->text(body, {"choices", std::size_t{0}, "finish_reason"}) == "tool_calls" &&
	       reading->text(body, {"choices", std::size_t{0}, "message", "content"}) == "Checking.";
}

/** A call cut off by the budget is no call: the whole response leaves it out and finishes `length`. */
[[nodiscard]] auto a_cut_off_call_is_left_out() -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	h->engine().queue(Script{{
	    openai::Accepted{},
	    openai::CallBegin{.index = 0, .name = "f"},
	    openai::Finished{.ending = openai::Ending::Length, .usage = json::Usage{.prompt_tokens = 3, .completion_tokens = 4, .cached_tokens = 0}},
	}});
	auto const answered = h->ask(*peer, chat(hello, ""));
	return answered && answered->status == 200 &&
	       !reading->kind(answered->body, {"choices", std::size_t{0}, "message", "tool_calls"}) &&
	       reading->text(answered->body, {"choices", std::size_t{0}, "finish_reason"}) == "length";
}

/** What a refused request must draw: its status, code and param (absent: `param` is null). */
struct Refusal {
	int status;
	std::string_view code;
	std::optional<std::string_view> param;
};

[[nodiscard]] auto refused_as(Json& reading, client::Parsed const& answered, Refusal const& expected) -> bool {
	auto const& body = answered.body;
	auto const param = expected.param ? reading.text(body, {"error", "param"}) == std::string{*expected.param}
	                                  : reading.kind(body, {"error", "param"}) == json::Kind::Null;
	auto const type = expected.status == 400 ? "invalid_request_error" : "server_error";
	auto const matches = answered.status == expected.status && reading.text(body, {"error", "code"}) == expected.code && param &&
	                     reading.text(body, {"error", "type"}) == type && reading.kind(body, {"error", "message"}) == json::Kind::String &&
	                     answered.fields.at("content-type") == "application/json";
	if (!matches) {
		std::println("expected {} {} {}, got {} {}", expected.status, expected.code, expected.param.value_or("null"), answered.status, body);
	}
	return matches;
}

/** A valid request with one member replaced: `member` is the member's JSON text, spliced in after `messages`. */
[[nodiscard]] auto with(std::string_view member) -> std::string {
	return std::format(R"({{"messages":[{{"role":"user","content":"hi"}}],{}}})", member);
}

[[nodiscard]] auto with_messages(std::string_view messages) -> std::string {
	return std::format(R"({{"messages":{}}})", messages);
}

struct Mutation {
	std::string_view name;
	std::string body;
	Refusal refusal;
};

[[nodiscard]] auto mutation(std::string_view name, std::string body, std::string_view code, std::optional<std::string_view> param)
    -> Mutation {
	return Mutation{.name = name, .body = std::move(body), .refusal = Refusal{.status = 400, .code = code, .param = param}};
}

/**
 * Every class of refusal, each reached by mutating the valid request in one place: decode faults, the template's
 * rules, tools and tool choice, the budget, stop sequences and every sampling bound. None reaches the engine.
 */
[[nodiscard]] auto every_refusal_is_named() -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto const long_name = std::string(openai::max_tool_name_bytes + 1, 'f');
	auto const long_stop = std::string(openai::max_stop_bytes + 1, 's');
	auto const forced = [](std::string_view name) {
		return std::format(R"("tool_choice":{{"type":"function","function":{{"name":"{}"}}}})", name);
	};
	auto const tool = [](std::string_view name) { return std::format(R"({{"type":"function","function":{{"name":"{}"}}}})", name); };
	auto const mutations = std::vector<Mutation>{
	    mutation("not json", R"({"messages":)", "json_truncated", std::nullopt),
	    mutation("no messages member", "{}", "missing_required_parameter", "messages"),
	    mutation("unknown member", with(R"("frobnicate":1)"), "unknown_parameter", "frobnicate"),
	    mutation("unsupported n", with(R"("n":2)"), "unsupported_parameter", "n"),
	    mutation("wrong type", with_messages("3"), "invalid_type", "messages"),
	    mutation("unknown role", with_messages(R"([{"role":"wizard","content":"hi"}])"), "invalid_value", "messages[0].role"),
	    mutation("both budgets", with(R"("max_tokens":3,"max_completion_tokens":4)"), "conflicting_parameters", "max_completion_tokens"),
	    mutation("seventeen stops", with(R"("stop":["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q"])"),
	             "too_many_items", "stop"),
	    mutation("no messages", with_messages("[]"), "messages_empty", "messages"),
	    mutation("late system", with_messages(R"([{"role":"user","content":"hi"},{"role":"system","content":"s"}])"),
	             "system_message_not_first", "messages[1]"),
	    mutation("late developer", with_messages(R"([{"role":"user","content":"hi"},{"role":"developer","content":"s"}])"),
	             "system_message_not_first", "messages[1]"),
	    mutation("only tool responses", with_messages(R"([{"role":"user","content":" <tool_response>x</tool_response>\n"}])"),
	             "no_user_query", "messages"),
	    mutation("only a system message", with_messages(R"([{"role":"system","content":"s"}])"), "no_user_query", "messages"),
	    mutation("duplicate tool", with(std::format(R"("tools":[{},{}])", tool("f"), tool("f"))), "duplicate_tool_name",
	             "tools[1].function.name"),
	    mutation("long tool name", with(std::format(R"("tools":[{}])", tool(long_name))), "invalid_tool_name", "tools[0].function.name"),
	    mutation("required without tools", with(R"("tool_choice":"required")"), "tool_choice_without_tools", "tool_choice"),
	    mutation("forced without tools", with(forced("f")), "tool_choice_without_tools", "tool_choice"),
	    mutation("forced unknown tool", with(std::format(R"("tools":[{}],{})", tool("f"), forced("g"))), "unknown_tool",
	             "tool_choice.function.name"),
	    mutation("zero max_tokens", with(R"("max_tokens":0)"), "max_tokens_below_one", "max_tokens"),
	    mutation("max_tokens past the context", with(R"("max_tokens":262145)"), "max_tokens_above_context", "max_tokens"),
	    mutation("empty stop", with(R"("stop":["a",""])"), "stop_sequence_empty", "stop[1]"),
	    mutation("long stop", with(std::format(R"("stop":"{}")", long_stop)), "stop_sequence_too_long", "stop[0]"),
	    mutation("hot", with(R"("temperature":2.5)"), "temperature_out_of_range", "temperature"),
	    mutation("negative temperature", with(R"("temperature":-0.5)"), "temperature_out_of_range", "temperature"),
	    mutation("top_k zero", with(R"("top_k":0)"), "top_k_below_one", "top_k"),
	    mutation("top_k past the bound", with(std::format(R"("top_k":{})", engine::sampling::top_k_bound + 1)), "top_k_above_bound",
	             "top_k"),
	    mutation("top_p above one", with(R"("top_p":1.5)"), "top_p_out_of_range", "top_p"),
	    mutation("negative min_p", with(R"("min_p":-0.1)"), "min_p_out_of_range", "min_p"),
	};
	for (auto const& mutation : mutations) {
		auto const answered = h->ask(*peer, chat(mutation.body, "refused"));
		if (!answered || !refused_as(*reading, *answered, mutation.refusal) || !echoes(*answered, "refused")) {
			std::println("mutation '{}' was not refused as expected", mutation.name);
			return false;
		}
	}
	return h->engine().submissions().empty();
}

/** The engine's refusal of a prompt too long for the context is a 400 before any stream opens. */
[[nodiscard]] auto a_prompt_too_long_is_a_400(json::Delivery delivery) -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	h->engine().queue(Script{{openai::PromptTooLong{.prompt_tokens = 300'000}}});
	auto const body = delivery == json::Delivery::Whole ? with(R"("max_tokens":100)") : with(R"("max_tokens":100,"stream":true)");
	auto const answered = h->ask(*peer, chat(body, "long-1"));
	return answered && refused_as(*reading, *answered, Refusal{.status = 400, .code = "context_length_exceeded", .param = "messages"}) &&
	       echoes(*answered, "long-1") && h->engine().cancellations().empty();
}

enum class Role : std::uint8_t {
	System,
	User,
	ToolResponse,
	Assistant,
	Tool,
};

constexpr auto every_role = std::array{Role::System, Role::User, Role::ToolResponse, Role::Assistant, Role::Tool};

/** The oracle: the chat template's rules over a role sequence, written from the template, not from the service. */
[[nodiscard]] auto template_verdict(std::span<Role const> roles) -> std::optional<Refusal> {
	if (roles.empty()) {
		return Refusal{.status = 400, .code = "messages_empty", .param = "messages"};
	}
	for (auto index = std::size_t{1}; index < roles.size(); ++index) {
		if (roles[index] == Role::System) {
			return Refusal{.status = 400, .code = "system_message_not_first", .param = std::nullopt};
		}
	}
	if (!std::ranges::contains(roles, Role::User)) {
		return Refusal{.status = 400, .code = "no_user_query", .param = "messages"};
	}
	return std::nullopt;
}

[[nodiscard]] auto message_text(Role role) -> std::string_view {
	switch (role) {
	case Role::System:
		return R"({"role":"system","content":"be brief"})";
	case Role::User:
		return R"({"role":"user","content":"  what now?"})";
	case Role::ToolResponse:
		return R"({"role":"user","content":"　<tool_response>\n42\n</tool_response> "})";
	case Role::Assistant:
		return R"({"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]})";
	case Role::Tool:
		return R"({"role":"tool","tool_call_id":"c1","content":"42"})";
	}
	std::unreachable();
}

/**
 * Random conversations: every one the template would raise on is refused with the template rule's code, every one
 * it renders is submitted and answered.
 */
[[nodiscard]] auto random_conversations_meet_the_template(std::uint64_t seed) -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 200; ++trial) {
		auto roles = std::vector<Role>(std::uniform_int_distribution<std::size_t>{0, 5}(generator));
		for (auto& role : roles) {
			role = every_role[std::uniform_int_distribution<std::size_t>{0, every_role.size() - 1}(generator)];
		}
		auto const messages = roles | std::views::transform(message_text) | std::views::join_with(std::string_view{","}) |
		                      std::ranges::to<std::string>();
		auto const verdict = template_verdict(roles);
		auto const submitted = h->engine().submissions().size();
		if (!verdict) {
			h->engine().queue(Script{{openai::Accepted{}, finished(5, 1, 0)}});
		}
		auto const answered = h->ask(*peer, chat(with_messages(std::format("[{}]", messages)), ""));
		if (!answered) {
			return false;
		}
		if (verdict) {
			auto const code = reading->text(answered->body, {"error", "code"});
			if (answered->status != verdict->status || code != verdict->code || h->engine().submissions().size() != submitted) {
				std::println("seed {} trial {}: {} refused as {}", seed, trial, messages, code.value_or("nothing"));
				return false;
			}
		} else if (answered->status != 200 || h->engine().submissions().size() != submitted + 1) {
			std::println("seed {} trial {}: {} answered {} {}", seed, trial, messages, answered->status, answered->body);
			return false;
		}
	}
	return true;
}

/** Random sampling fields around their bounds: each is refused exactly when it is out of range, naming itself. */
[[nodiscard]] auto random_sampling_meets_its_bounds(std::uint64_t seed) -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	if (!peer || !reading) {
		return false;
	}
	auto generator = engine::testing::generator(seed);
	auto const pick = [&generator]<class T, std::size_t N>(std::array<T, N> const& values) {
		return values[std::uniform_int_distribution<std::size_t>{0, N - 1}(generator)];
	};
	constexpr auto temperatures = std::array{-1.0, -1e-9, 0.0, 0.3, 2.0, 2.0000001, 7.0};
	constexpr auto probabilities = std::array{-0.5, -1e-12, 0.0, 0.5, 1.0, 1.0000001, 3.0};
	constexpr auto bound = std::int64_t{engine::sampling::top_k_bound};
	constexpr auto top_ks = std::array<std::int64_t, 7>{-4, 0, 1, 2, bound, bound + 1, 1000};
	constexpr auto budgets = std::array<std::int64_t, 6>{-1, 0, 1, 4096, 262144, 262145};
	for (auto trial = 0; trial < 150; ++trial) {
		auto const temperature = pick(temperatures);
		auto const top_k = pick(top_ks);
		auto const top_p = pick(probabilities);
		auto const min_p = pick(probabilities);
		auto const budget = pick(budgets);
		auto expected = std::optional<std::string_view>{};
		if (budget < 1 || budget > 262144) {
			expected = "max_tokens";
		} else if (temperature < 0.0 || temperature > 2.0) {
			expected = "temperature";
		} else if (top_k < 1 || std::cmp_greater(top_k, engine::sampling::top_k_bound)) {
			expected = "top_k";
		} else if (top_p < 0.0 || top_p > 1.0) {
			expected = "top_p";
		} else if (min_p < 0.0 || min_p > 1.0) {
			expected = "min_p";
		}
		if (!expected) {
			h->engine().queue(Script{{openai::Accepted{}, finished(5, 1, 0)}});
		}
		auto const body = with(std::format(R"("temperature":{},"top_k":{},"top_p":{},"min_p":{},"max_tokens":{})", temperature, top_k,
		                                   top_p, min_p, budget));
		auto const answered = h->ask(*peer, chat(body, ""));
		if (!answered) {
			return false;
		}
		auto const param = reading->text(answered->body, {"error", "param"});
		auto const holds = expected ? answered->status == 400 && param == std::string{*expected} : answered->status == 200;
		if (!holds) {
			std::println("seed {} trial {}: {} answered {} {}", seed, trial, body, answered->status, answered->body);
			return false;
		}
	}
	return true;
}

/** The exchange of the latest submission. */
[[nodiscard]] auto latest(Harness& h) -> std::optional<http::Exchange> {
	auto const submissions = h.engine().submissions();
	if (submissions.empty()) {
		return std::nullopt;
	}
	return submissions.back().first;
}

/** Starts a streamed request on a fresh connection, accepted by the engine. */
[[nodiscard]] auto open_stream(Harness& h, client::Client& peer) -> std::optional<http::Exchange> {
	h.engine().queue(Script{{openai::Accepted{}}});
	if (!h.push(peer, chat(R"({"messages":[{"role":"user","content":"hi"}],"stream":true})", "slow")) ||
	    !h.wait([&] { return !h.engine().submissions().empty(); }, 2000ms)) {
		return std::nullopt;
	}
	return latest(h);
}

[[nodiscard]] auto content_of(Json& reading, std::vector<std::string> const& events) -> std::string {
	auto text = std::string{};
	for (auto const& event : events) {
		text += reading.text(event, {"choices", std::size_t{0}, "delta", "content"}).value_or("");
	}
	return text;
}

/**
 * A client that reads nothing while the engine writes: once the server refuses events, deltas coalesce instead of
 * queueing, and when the client reads, every delta arrives in order, in fewer chunks than the engine wrote.
 */
[[nodiscard]] auto a_slow_client_receives_coalesced_deltas() -> bool {
	auto h = serving_harness(Timeouts{.read = 5000ms, .write = 10000ms});
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto const exchange = peer ? open_stream(*h, *peer) : std::nullopt;
	if (!exchange || !reading) {
		return false;
	}
	auto written = std::string{};
	constexpr auto pieces = 3000;
	for (auto index = 0; index < pieces; ++index) {
		auto text = std::format("{:05}:{}|", index, std::string(250, static_cast<char>('a' + (index % 26))));
		written += text;
		if (!h->engine().post(*exchange, {openai::ContentText{.text = std::move(text)}})) {
			return false;
		}
		if (index % 32 == 0) {
			h->turn();
		}
	}
	if (!h->engine().post(*exchange, {finished(1, pieces, 0)})) {
		return false;
	}
	h->idle(20ms);
	auto const answered = h->ask(*peer, "");
	if (!answered) {
		return false;
	}
	auto const events = client::decode_events(answered->body);
	auto const deltas = std::ranges::count_if(events, [&](std::string const& event) {
		return reading->kind(event, {"choices", std::size_t{0}, "delta", "content"}).has_value();
	});
	return content_of(*reading, events) == written && deltas < pieces && events.back() == "[DONE]" &&
	       h->engine().cancellations().empty();
}

/**
 * A client that never reads cannot make the server hold more than the backlog bound: past it the request is
 * cancelled, and the stream ends with a stream_backlog error after a prefix of what the engine wrote.
 */
[[nodiscard]] auto a_stalled_stream_is_bounded() -> bool {
	auto h = serving_harness(Timeouts{.read = 5000ms, .write = 20000ms});
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	auto reading = Json::make();
	auto const exchange = peer ? open_stream(*h, *peer) : std::nullopt;
	if (!exchange || !reading) {
		return false;
	}
	auto written = std::string{};
	// 64 MiB without a cancellation would be an unbounded buffer.
	for (auto index = 0; index < 1024 && h->engine().cancellations().empty(); ++index) {
		auto text = std::format("{:05}{}", index, std::string(65'536 - 5, static_cast<char>('a' + (index % 26))));
		written += text;
		if (!h->engine().post(*exchange, {openai::ContentText{.text = std::move(text)}})) {
			return false;
		}
		h->turn();
		h->turn();
	}
	if (h->engine().cancellations().size() != 1 || h->engine().cancellations().front() != *exchange) {
		return false;
	}
	// What the engine posted before it saw the cancellation is dropped.
	if (!h->engine().post(*exchange, {openai::ContentText{.text = "late"}, finished(1, 1, 0)})) {
		return false;
	}
	auto const answered = h->ask(*peer, "");
	if (!answered) {
		return false;
	}
	auto const events = client::decode_events(answered->body);
	auto const received = content_of(*reading, events);
	return events.size() >= 2 && events.back() == "[DONE]" &&
	       reading->text(events[events.size() - 2], {"error", "code"}) == "stream_backlog" && written.starts_with(received) &&
	       received.size() < written.size();
}

enum class Departure : std::uint8_t {
	MidStream,
	WhileGathering,
	BeforeAcceptance,
};

/** A client that leaves cancels its request in the engine, once, and the engine's late outputs are dropped. */
[[nodiscard]] auto a_departing_client_cancels_its_request(Departure departure) -> bool {
	auto h = serving_harness(quick);
	auto peer = h ? client::connect(h->server.port()) : std::nullopt;
	if (!peer) {
		return false;
	}
	// Before acceptance the engine has posted nothing; otherwise it has accepted and written a first delta.
	if (departure != Departure::BeforeAcceptance) {
		h->engine().queue(Script{{openai::Accepted{}}, {openai::ContentText{.text = "first"}}});
	}
	auto const body = departure == Departure::MidStream ? R"({"messages":[{"role":"user","content":"hi"}],"stream":true})" : hello;
	if (!h->push(*peer, chat(body, "leaving")) || !h->wait([&] { return !h->engine().submissions().empty(); }, 2000ms)) {
		return false;
	}
	auto const found = latest(*h);
	if (!found) {
		return false;
	}
	auto const exchange = *found;
	if (departure == Departure::MidStream && !h->pump(*peer, [&] { return peer->received.contains("first"); }, 2000ms)) {
		return false;
	}
	h->idle(5ms);
	peer.reset();
	if (!h->wait([&] { return !h->engine().cancellations().empty(); }, 2000ms)) {
		return false;
	}
	if (!h->engine().post(exchange, {openai::ContentText{.text = "late"}, finished(1, 2, 0)})) {
		return false;
	}
	h->idle(20ms);
	return h->engine().cancellations().size() == 1 && h->engine().cancellations().front() == exchange;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "/health follows readiness; chat waits for it; ids are echoed", .holds = health_follows_readiness()},
	    engine::testing::Case{.name = "/v1/models lists the one model", .holds = models_lists_the_one_model()},
	    engine::testing::Case{.name = "/status writes the engine's latest snapshot", .holds = status_writes_the_latest_snapshot()},
	    engine::testing::Case{.name = "a whole completion gathers the engine's outputs", .holds = a_whole_completion_gathers_the_outputs()},
	    engine::testing::Case{.name = "sampling takes the request's fields, else its thinking mode's defaults",
	                          .holds = sampling_follows_the_thinking_mode()},
	    engine::testing::Case{.name = "a stream is framed byte for byte", .holds = a_stream_is_framed_exactly(json::Delivery::Stream)},
	    engine::testing::Case{.name = "a stream with include_usage ends with the usage chunk, byte for byte",
	                          .holds = a_stream_is_framed_exactly(json::Delivery::StreamWithUsage)},
	    engine::testing::Case{.name = "streamed tool-call fragments concatenate to the whole arguments (seed 1)",
	                          .holds = streamed_arguments_concatenate_to_the_whole(1)},
	    engine::testing::Case{.name = "streamed tool-call fragments concatenate to the whole arguments (seed 2)",
	                          .holds = streamed_arguments_concatenate_to_the_whole(2)},
	    engine::testing::Case{.name = "a call cut off by the budget is left out of a whole response", .holds = a_cut_off_call_is_left_out()},
	    engine::testing::Case{.name = "every class of invalid request is refused with its code and param", .holds = every_refusal_is_named()},
	    engine::testing::Case{.name = "a prompt too long for the context is a 400 (whole)",
	                          .holds = a_prompt_too_long_is_a_400(json::Delivery::Whole)},
	    engine::testing::Case{.name = "a prompt too long for the context is a 400 before a stream opens",
	                          .holds = a_prompt_too_long_is_a_400(json::Delivery::Stream)},
	    engine::testing::Case{.name = "random conversations are refused exactly where the template raises (seed 1)",
	                          .holds = random_conversations_meet_the_template(1)},
	    engine::testing::Case{.name = "random conversations are refused exactly where the template raises (seed 2)",
	                          .holds = random_conversations_meet_the_template(2)},
	    engine::testing::Case{.name = "random sampling fields are refused exactly out of range (seed 1)",
	                          .holds = random_sampling_meets_its_bounds(1)},
	    engine::testing::Case{.name = "a slow client receives every delta, coalesced", .holds = a_slow_client_receives_coalesced_deltas()},
	    engine::testing::Case{.name = "a client that never reads is cut off at the backlog bound", .holds = a_stalled_stream_is_bounded()},
	    engine::testing::Case{.name = "a client leaving mid-stream cancels its request",
	                          .holds = a_departing_client_cancels_its_request(Departure::MidStream)},
	    engine::testing::Case{.name = "a client leaving a whole completion cancels its request",
	                          .holds = a_departing_client_cancels_its_request(Departure::WhileGathering)},
	    engine::testing::Case{.name = "a client leaving before acceptance cancels its request",
	                          .holds = a_departing_client_cancels_its_request(Departure::BeforeAcceptance)},
	};
	return engine::testing::report(cases);
}

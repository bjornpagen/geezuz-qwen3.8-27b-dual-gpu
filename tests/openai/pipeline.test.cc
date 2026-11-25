// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.chat;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.grammar;
import engine.http;
import engine.http_test_client;
import engine.json;
import engine.kv;
import engine.model;
import engine.net;
import engine.openai;
import engine.openai_harness;
import engine.pipeline;
import engine.pipeline_doubles;
import engine.sched;
import engine.testing;
import engine.tokenizer;

/**
 * The api end to end over loopback, through the real engine loop: the service, the pipeline with the openai front
 * (the chat format's renderer and reply parser, the tokenizer), and the pipeline's own doubles of the cards and the
 * grammar. The cards double commits the script its request's seed names, so the test scripts every seed the service
 * can draw with one reply, written as the model writes it.
 */
namespace {

namespace chat = engine::chat;
namespace core = engine::core;
namespace device = engine::device;
namespace fake = engine::driver_double;
namespace grammar = engine::grammar;
namespace http = engine::http;
namespace json = engine::json;
namespace kv = engine::kv;
namespace model = engine::model;
namespace openai = engine::openai;
namespace pipeline = engine::pipeline;
namespace sched = engine::sched;
namespace tok = engine::tokenizer;
namespace doubles = engine::pipeline_doubles;
namespace client = engine::http_test_client;
using engine::openai_harness::create;
using engine::openai_harness::FixedSources;
using engine::openai_harness::get;
using engine::openai_harness::Json;
using namespace std::chrono_literals;

constexpr auto roomy =
    kv::Geometry{.block_tokens = 4, .pool_blocks = 400, .table_blocks = 96, .snapshot_blocks = 2, .snapshot_capacity = 24};
constexpr auto tuning = sched::Tuning{.chunk_tokens = 16, .share = {.numerator = 1, .denominator = 4}};

// Every request decodes under its real compiled grammar; the cards double checks each committed token against its mask.
using Cards = doubles::CardsDouble<roomy, doubles::Rules::Compiled>;
using Pipeline = pipeline::BasicPipeline<roomy, model::RowBudget::ThreeTiles, tuning, Cards, openai::Front>;

constexpr auto limits =
    openai::Limits{.requests = 16, .backlog_bytes = std::size_t{1} << 20U, .store = {.responses = 16, .bytes = std::size_t{1} << 24U}};

using Service = openai::BasicService<Pipeline, FixedSources, limits>;
using Harness = engine::openai_harness::Harness<Service>;

constexpr auto sources = FixedSources{.state = 3};

/** The reply every request is answered with, as the model writes it with thinking off. */
constexpr std::string_view reply_text = "Hello from the engine, naïve and whole.";

/** The fake driver, which the cards borrow, then the server, its service and the pipeline in it. */
struct Rig {
	std::unique_ptr<fake::Fake> driver;
	std::unique_ptr<Harness> harness;
};

[[nodiscard]] auto scripted_tokens(tok::Tokenizer const& tokenizer) -> std::vector<tok::Id> {
	auto const pieces = std::array{tok::Piece{tok::Text{.utf8 = reply_text, .spelling = tok::Spelling::Characters}}};
	auto ids = tokenizer.encode(pieces);
	core::invariant(ids.has_value());
	return *std::move(ids);
}

/** Opens the rig with every seed the service's first draws can make scripted with the reply; Ready posted. */
[[nodiscard]] auto rig(std::optional<doubles::Breakdown> breakdown) -> std::unique_ptr<Rig> {
	auto out = std::make_unique<Rig>(Rig{.driver = std::make_unique<fake::Fake>(fake::Machine{}), .harness = nullptr});
	auto tokenizer = tok::load();
	if (!tokenizer) {
		return nullptr;
	}
	auto script = scripted_tokens(*tokenizer);
	script.push_back(tok::id(tok::Added::im_end));
	if (!std::ranges::all_of(script, [](tok::Id id) {
		    return id < doubles::script_range || id == tok::id(tok::Added::im_end);
	    })) {
		std::println("the reply's tokens are outside the cards double's script range");
		return nullptr;
	}
	auto scripts = std::map<std::uint64_t, std::vector<tok::Id>>{};
	auto drawn = sources;
	for (auto draw = 0; draw < 4096; ++draw) {
		scripts.emplace(drawn.draw(), script);
	}
	auto const& driver = *out->driver;
	out->harness = engine::openai_harness::harness<Service>(
	    engine::openai_harness::quick,
	    [&](http::Poster<openai::Update> poster) {
		    auto devices = device::Devices::open(driver.driver());
		    auto front = openai::Front::make();
		    core::invariant(devices.has_value() && front.has_value());
		    auto vocabulary = grammar::engine_vocabulary(*tokenizer);
		    auto started = Pipeline::start(poster, *std::move(tokenizer), std::move(vocabulary),
			                               Cards{*std::move(devices), std::move(scripts), breakdown}, *std::move(front));
		    core::invariant(started.has_value());
		    return *std::move(started);
	    },
	    sources);
	if (!out->harness || out->harness->server.poster().post(openai::Update{openai::Ready{}}) != engine::net::Posted::Accepted) {
		return nullptr;
	}
	out->harness->idle(5ms);
	return out;
}

/** The ids a request's input renders to, as the chat format renders it with thinking off. */
[[nodiscard]] auto rendered(tok::Tokenizer const& tokenizer, std::string_view input) -> std::vector<tok::Id> {
	auto const items = std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = std::string{input}}};
	auto const conversation = chat::conversation({.instructions = std::nullopt, .items = items, .tools = {}, .effort = json::Effort::None});
	auto renderer = chat::Renderer::make();
	core::invariant(conversation.has_value() && renderer.has_value());
	auto prompt = renderer->render(tokenizer, *conversation);
	core::invariant(prompt.has_value());
	return std::move(prompt->ids);
}

/**
 * A whole response, a stream and a continuation through the engine loop: each says the scripted reply, its usage
 * counts the prompt the engine ran, and the continuation's prompt is the stored sequence and its new turn exactly.
 */
[[nodiscard]] auto requests_run_through_the_engine() -> bool {
	auto const tokenizer = tok::load();
	auto rigged = rig(std::nullopt);
	auto reading = Json::make();
	if (!tokenizer || !rigged || !reading) {
		return false;
	}
	auto& h = *rigged->harness;
	auto peer = client::connect(h.server.port());
	if (!peer) {
		return false;
	}
	auto const whole = h.ask(*peer, create(R"({"input":"hi","reasoning":{"effort":"none"}})", "e2e-whole"));
	auto const streamed = h.ask(*peer, create(R"({"input":"hi","reasoning":{"effort":"none"},"stream":true})", "e2e-stream"));
	if (!whole || !streamed || whole->status != 200 || streamed->status != 200) {
		std::println("whole or streamed failed: {}", whole ? whole->body : "none");
		return false;
	}
	auto const id = reading->text(whole->body, {"id"}).value_or("");
	auto const next =
	    h.ask(*peer, create(std::format(R"({{"input":"and then?","reasoning":{{"effort":"none"}},"previous_response_id":"{}"}})", id),
		                    "e2e-next"));
	if (!next || next->status != 200) {
		std::println("continuation failed: {}", next ? next->body : "none");
		return false;
	}
	auto const prompt = rendered(*tokenizer, "hi");
	auto const reply = scripted_tokens(*tokenizer);
	// The stored sequence is the prompt as it ran, the reply, and the token that ended its turn.
	auto const stored = prompt.size() + reply.size() + 1;
	auto renderer = chat::Renderer::make();
	auto const turns = chat::continuation(std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = "and then?"}});
	if (!renderer || !turns) {
		return false;
	}
	auto const rest = renderer->continuation(*tokenizer, *turns, chat::options_of(json::Effort::None));
	auto const events = client::decode_events(streamed->body);
	auto streamed_text = std::string{};
	for (auto const& event : events) {
		if (reading->text(event, {"type"}) == "response.output_text.delta") {
			streamed_text += reading->text(event, {"delta"}).value_or("");
		}
	}
	return rest && reading->text(whole->body, {"output", std::size_t{0}, "content", std::size_t{0}, "text"}) == reply_text &&
	       reading->text(whole->body, {"status"}) == "completed" &&
	       reading->integer(whole->body, {"usage", "input_tokens"}) == std::ssize(prompt) &&
	       reading->integer(whole->body, {"usage", "output_tokens"}) == std::ssize(reply) + 1 &&
	       reading->integer(whole->body, {"usage", "output_tokens_details", "reasoning_tokens"}) == 0 && streamed_text == reply_text &&
	       reading->text(events.back(), {"type"}) == "response.completed" &&
	       reading->text(next->body, {"output", std::size_t{0}, "content", std::size_t{0}, "text"}) == reply_text &&
	       reading->integer(next->body, {"usage", "input_tokens"}) == core::narrow<std::int64_t>(rest->ids.size() + stored) &&
	       reading->text(next->body, {"previous_response_id"}) == id;
}

/** Cards that fail under a stream end it with `response.failed`, and the service refuses what comes after. */
[[nodiscard]] auto a_card_failure_reaches_the_client() -> bool {
	auto rigged = rig(doubles::Breakdown{.verify = 1, .behind = std::nullopt});
	auto reading = Json::make();
	if (!rigged || !reading) {
		return false;
	}
	auto& h = *rigged->harness;
	auto peer = client::connect(h.server.port());
	if (!peer) {
		return false;
	}
	auto const streamed = h.ask(*peer, create(R"({"input":"hi","reasoning":{"effort":"none"},"stream":true})", ""));
	auto const health = h.ask(*peer, get("/health", ""));
	if (!streamed || !health) {
		return false;
	}
	auto const events = client::decode_events(streamed->body);
	return !events.empty() && reading->text(events.back(), {"type"}) == "response.failed" &&
	       reading->text(events.back(), {"response", "error", "code"}) == "server_error" && health->status == 503 &&
	       reading->text(health->body, {"status"}) == "failed";
}

}

auto main() -> int {
	auto const checks = std::array{
	    engine::testing::Case{.name = "whole, streamed and continued requests run through the engine loop",
		                      .holds = requests_run_through_the_engine()},
	    engine::testing::Case{.name = "a card failure under a stream reaches its client as response.failed",
		                      .holds = a_card_failure_reaches_the_client()},
	};
	return engine::testing::report(checks);
}

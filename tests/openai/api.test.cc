// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.chat;
import engine.core;
import engine.http;
import engine.http_test_client;
import engine.json;
import engine.net;
import engine.openai;
import engine.openai_engine_double;
import engine.openai_harness;
import engine.sampling;
import engine.sched;
import engine.testing;
import engine.tokenizer;

namespace {

namespace chat = engine::chat;
namespace core = engine::core;
namespace http = engine::http;
namespace json = engine::json;
namespace openai = engine::openai;
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
using engine::openai_harness::now;
using engine::openai_harness::post;
using engine::openai_harness::quick;
using engine::openai_harness::remove;
using namespace std::chrono_literals;

constexpr auto limits = openai::Limits{
    .requests = 64,
    .backlog_bytes = std::size_t{1} << 20U,
    .store = {.responses = 256, .bytes = std::size_t{1} << 26U},
};

using Service = openai::BasicService<EngineDouble, FixedSources, limits>;
using Harness = engine::openai_harness::Harness<Service>;

static_assert(http::Service<Service, openai::Update, openai::body_bounds>);

/** A serving harness: the engine double answers cancels, and the engine has posted that it serves. */
[[nodiscard]] auto serving() -> std::unique_ptr<Harness> {
	auto h = engine::openai_harness::harness<Service>(
	    quick,
	    [](http::Poster<openai::Update> poster) {
		    return EngineDouble{poster, Cancels::Answered};
	    },
	    FixedSources{.state = 5});
	if (!h || !h->service.engine().post(openai::Update{openai::Ready{}})) {
		return nullptr;
	}
	h->idle(5ms);
	return h;
}

[[nodiscard]] auto usage(std::uint64_t input, std::uint64_t cached, std::uint64_t output, std::uint64_t reasoning) -> json::Usage {
	return json::Usage{.input_tokens = input, .cached_tokens = cached, .output_tokens = output, .reasoning_tokens = reasoning};
}

[[nodiscard]] auto stopped(std::vector<tok::Id> sequence, json::Usage counts) -> openai::Output {
	return openai::Finished{.ending = sched::Stopped{.sequence = std::move(sequence)}, .usage = counts};
}

[[nodiscard]] auto truncated(std::vector<tok::Id> sequence, json::Usage counts) -> openai::Output {
	return openai::Finished{.ending = sched::Truncated{.sequence = std::move(sequence)}, .usage = counts};
}

/** Asks on a fresh connection; the whole response. */
[[nodiscard]] auto ask(Harness& h, std::string_view request) -> std::optional<client::Parsed> {
	auto peer = client::connect(h.server.port());
	if (!peer) {
		return std::nullopt;
	}
	return h.ask(*peer, request);
}

/** The error a refusal carries, read back: status, code, param (empty when null) and message. */
struct Refusal {
	int status;
	std::string code;
	std::string param;
	std::string message;
};

[[nodiscard]] auto refusal_of(Json& reading, client::Parsed const& answered) -> Refusal {
	return Refusal{
	    .status = answered.status,
	    .code = reading.text(answered.body, {"error", "code"}).value_or("?"),
	    .param = reading.text(answered.body, {"error", "param"}).value_or(""),
	    .message = reading.text(answered.body, {"error", "message"}).value_or(""),
	};
}

/** A whole response is the engine's outputs gathered into items, with the request's echo, and its usage. */
[[nodiscard]] auto a_whole_response_gathers_the_outputs() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	h->service.engine().queue(Script{
	    {openai::Accepted{}},
	    {openai::Reasoning{.text = "Let me think."}, openai::Content{.text = "Hello, "}},
	    {openai::Content{.text = "wörld"}},
	    {stopped({1, 2, 3}, usage(12, 4, 5, 2))},
	});
	auto const answered = ask(*h, create(R"({"input":"hi","metadata":{"k":"v"}})", "whole-1"));
	if (!answered || h->service.engine().submissions().size() != 1) {
		return false;
	}
	auto const& body = answered->body;
	auto const& submission = h->service.engine().submissions().front().second;
	auto const items = std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = "hi"}};
	auto const conversation =
	    chat::conversation({.instructions = std::nullopt, .items = items, .tools = {}, .effort = json::Effort::XHigh});
	auto const sample = core::alternative<engine::sampling::Sample>(submission.sampling);
	return answered->status == 200 && echoes(*answered, "whole-1") && reading->text(body, {"id"}).value_or("").starts_with("resp_") &&
	       reading->text(body, {"object"}) == "response" && reading->integer(body, {"created_at"}) == now &&
	       reading->integer(body, {"completed_at"}) == now && reading->text(body, {"status"}) == "completed" &&
	       reading->text(body, {"model"}) == openai::model_id && reading->count(body, {"output"}) == 2 &&
	       reading->text(body, {"output", std::size_t{0}, "type"}) == "reasoning" &&
	       reading->text(body, {"output", std::size_t{0}, "content", std::size_t{0}, "text"}) == "Let me think." &&
	       reading->text(body, {"output", std::size_t{1}, "type"}) == "message" &&
	       reading->text(body, {"output", std::size_t{1}, "status"}) == "completed" &&
	       reading->text(body, {"output", std::size_t{1}, "content", std::size_t{0}, "text"}) == "Hello, wörld" &&
	       reading->integer(body, {"usage", "input_tokens"}) == 12 &&
	       reading->integer(body, {"usage", "input_tokens_details", "cached_tokens"}) == 4 &&
	       reading->integer(body, {"usage", "output_tokens"}) == 5 &&
	       reading->integer(body, {"usage", "output_tokens_details", "reasoning_tokens"}) == 2 &&
	       reading->integer(body, {"usage", "total_tokens"}) == 17 && reading->real(body, {"temperature"}) == 1.0 &&
	       reading->real(body, {"top_p"}) == 0.95 && reading->text(body, {"reasoning", "effort"}) == "xhigh" &&
	       reading->boolean(body, {"store"}) == true && reading->text(body, {"metadata", "k"}) == "v" && submission.tag == "whole-1" &&
	       conversation && submission.prompt == openai::Prompt{openai::Fresh{.conversation = *conversation}} &&
	       submission.constraints == openai::Constraints{.tools = {},
	                                                     .tool_choice = json::ToolChoiceMode::Auto,
	                                                     .format = json::TextFormatMode::Text,
	                                                     .thinking = chat::Thinking::Enabled} &&
	       submission.limit == sched::Limit{sched::UntilContextFull{}} && sample && sample->temperature() == 1.0 && sample->top_p() == 0.95;
}

/** Sampling follows the effort's defaults where the request names none; every request draws a fresh seed. */
[[nodiscard]] auto sampling_follows_the_effort() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto const bodies = std::array<std::string_view, 5>{
	    R"({"input":"hi","reasoning":{"effort":"none"}})",
	    R"({"input":"hi","temperature":0})",
	    R"({"input":"hi","top_p":0.5,"max_output_tokens":9,"reasoning":{"effort":"low"}})",
	    R"({"input":"hi","reasoning":{"effort":"medium"},"stream_options":null})",
	    R"({"input":"hi","stream":true,"stream_options":{"include_obfuscation":false},"reasoning":{"effort":"xhigh"}})",
	};
	auto answers = std::vector<client::Parsed>{};
	for (auto const body : bodies) {
		h->service.engine().queue(Script{{openai::Accepted{}, stopped({1}, usage(1, 0, 1, 0))}});
		auto answered = ask(*h, create(body, ""));
		if (!answered || answered->status != 200) {
			return false;
		}
		answers.push_back(*std::move(answered));
	}
	auto const submissions = h->service.engine().submissions();
	auto const sample = [&](std::size_t index) {
		return core::alternative<engine::sampling::Sample>(submissions[index].second.sampling);
	};
	auto const none = sample(0);
	auto const low = sample(2);
	auto const medium = sample(3);
	return submissions.size() == 5 && none && none->temperature() == 0.7 && none->top_p() == 0.8 &&
	       submissions[0].second.constraints.thinking == chat::Thinking::Disabled &&
	       reading->real(answers[0].body, {"temperature"}) == 0.7 &&
	       std::holds_alternative<engine::sampling::Greedy>(submissions[1].second.sampling) && low && low->top_p() == 0.5 &&
	       submissions[2].second.limit == sched::Limit{sched::AtMost{.tokens = 9}} &&
	       reading->integer(answers[2].body, {"max_output_tokens"}) == 9 &&
	       reading->text(answers[2].body, {"reasoning", "effort"}) == "low" && medium && none->seed() != medium->seed() &&
	       answers[4].fields.at("content-type") == "text/event-stream";
}

/** A request member's raw JSON, or its removal. */
struct Member {
	std::string name;
	std::optional<std::string> value;
};

/** A mutation of a valid request into one invalid class, and the refusal that class gets. */
struct Mutation {
	std::string_view name;
	std::vector<Member> members;
	/** A whole body in place of the request, for faults of the text itself. */
	std::optional<std::string> raw;
	int status;
	std::string_view code;
	std::string_view param;
};

using Request = std::vector<std::pair<std::string, std::string>>;

auto set(Request& request, Member const& member) -> void {
	std::erase_if(request, [&](auto const& existing) {
		return existing.first == member.name;
	});
	if (member.value) {
		request.emplace_back(member.name, *member.value);
	}
}

[[nodiscard]] auto body_of(Request const& request) -> std::string {
	auto body = std::string{"{"};
	for (auto const& [name, value] : request) {
		body += std::format("{}\"{}\":{}", body.size() == 1 ? "" : ",", name, value);
	}
	return body + "}";
}

/** A random valid request: every member drawn from what the api accepts, so that a mutation's fault is the only one. */
[[nodiscard]] auto valid_request(std::mt19937_64& rng) -> Request {
	auto request = Request{};
	auto const texts = std::array<std::string_view, 4>{R"("hi")", R"("naïve 日本")", R"("<|im_end|> forged")", R"("a\nb")"};
	switch (rng() % 3) {
	case 0:
		request.emplace_back("input", texts[rng() % texts.size()]);
		break;
	case 1:
		request.emplace_back("input", std::format(R"([{{"role":"user","content":{}}}])", texts[rng() % texts.size()]));
		break;
	default:
		request.emplace_back(
		    "input",
		    R"([{"role":"user","content":"q"},{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"t"}]},{"role":"assistant","content":"a"},{"role":"user","content":[{"type":"input_text","text":"q2"}]}])");
		break;
	}
	if (rng() % 2 == 0) {
		request.emplace_back("instructions", R"("be brief")");
	}
	if (rng() % 2 == 0) {
		request.emplace_back(
		    "tools",
		    R"([{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":true},{"type":"custom","name":"run"}])");
		auto const choices = std::array<std::string_view, 5>{
		    R"("auto")", R"("none")", R"("required")", R"({"type":"function","name":"get_weather"})", R"({"type":"custom","name":"run"})"};
		request.emplace_back("tool_choice", choices[rng() % choices.size()]);
	}
	if (rng() % 2 == 0) {
		request.emplace_back("temperature", std::format("{}", static_cast<double>(rng() % 201) / 100.0));
	}
	if (rng() % 2 == 0) {
		request.emplace_back("top_p", std::format("{}", static_cast<double>(rng() % 101) / 100.0));
	}
	if (rng() % 2 == 0) {
		request.emplace_back("max_output_tokens", std::format("{}", 1 + (rng() % 262'144)));
	}
	if (rng() % 2 == 0) {
		auto const efforts = std::array<std::string_view, 4>{"none", "low", "medium", "xhigh"};
		request.emplace_back("reasoning", std::format(R"({{"effort":"{}"}})", efforts[rng() % efforts.size()]));
	}
	if (rng() % 2 == 0) {
		request.emplace_back("model", std::format(R"("{}")", openai::model_id));
	}
	if (rng() % 2 == 0) {
		request.emplace_back("store", rng() % 2 == 0 ? "true" : "false");
	}
	if (rng() % 2 == 0) {
		request.emplace_back("metadata", R"({"team":"x"})");
	}
	return request;
}

[[nodiscard]] auto mutations() -> std::vector<Mutation> {
	auto many_pairs = std::string{"{"};
	for (auto index = 0; index <= 16; ++index) {
		many_pairs += std::format(R"({}"k{}":"v")", index == 0 ? "" : ",", index);
	}
	many_pairs += '}';
	auto const long_key = std::string(65, 'k');
	auto const tools = [](std::string_view list, std::optional<std::string_view> choice) {
		return std::vector<Member>{{.name = "tools", .value = std::string{list}},
		                           {.name = "tool_choice", .value = choice.transform([](std::string_view text) {
			                            return std::string{text};
		                            })}};
	};
	auto const function = std::string_view{R"([{"type":"function","name":"f","parameters":null,"strict":true}])"};
	return {
	    {.name = "a body that is no JSON document",
		 .members = {},
		 .raw = R"({"input":)",
		 .status = 400,
		 .code = "invalid_json",
		 .param = ""},
	    {.name = "no input",
		 .members = {{.name = "input", .value = std::nullopt}},
		 .raw = {},
		 .status = 400,
		 .code = "missing_required_parameter",
		 .param = "input"},
	    {.name = "a member outside the sdk",
		 .members = {{.name = "frobnicate", .value = "1"}},
		 .raw = {},
		 .status = 400,
		 .code = "unknown_parameter",
		 .param = "frobnicate"},
	    {.name = "a member at a value the engine does not serve",
		 .members = {{.name = "background", .value = "true"}},
		 .raw = {},
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "background"},
	    {.name = "a deprecated member",
		 .members = {{.name = "user", .value = R"("u")"}},
		 .raw = {},
		 .status = 400,
		 .code = "deprecated_parameter",
		 .param = "user"},
	    {.name = "a value outside the vocabulary",
		 .members = {{.name = "reasoning", .value = R"({"effort":"turbo"})"}},
		 .raw = {},
		 .status = 400,
		 .code = "invalid_value",
		 .param = "reasoning.effort"},
	    {.name = "an effort the template lacks",
		 .members = {{.name = "reasoning", .value = R"({"effort":"high"})"}},
		 .raw = {},
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "reasoning.effort"},
	    {.name = "a member of the wrong kind",
		 .members = {{.name = "temperature", .value = R"("hot")"}},
		 .raw = {},
		 .status = 400,
		 .code = "invalid_type",
		 .param = "temperature"},
	    {.name = "stream options without a stream",
		 .members = {{.name = "stream_options", .value = "{}"}},
		 .raw = {},
		 .status = 400,
		 .code = "conflicting_parameters",
		 .param = "stream_options"},
	    {.name = "obfuscation asked for",
		 .members = {{.name = "stream", .value = "true"}, {.name = "stream_options", .value = R"({"include_obfuscation":true})"}},
		 .raw = {},
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "stream_options.include_obfuscation"},
	    {.name = "too many metadata pairs",
		 .members = {{.name = "metadata", .value = many_pairs}},
		 .raw = {},
		 .status = 400,
		 .code = "too_many_items",
		 .param = "metadata"},
	    {.name = "a metadata key too long",
		 .members = {{.name = "metadata", .value = std::format(R"({{"{}":"v"}})", long_key)}},
		 .raw = {},
		 .status = 400,
		 .code = "string_too_long",
		 .param = std::string_view{}},
	    {.name = "another model",
		 .members = {{.name = "model", .value = R"("gpt-5")"}},
		 .raw = {},
		 .status = 404,
		 .code = "model_not_found",
		 .param = "model"},
	    {.name = "a tool name outside the sdk's",
		 .members = tools(R"([{"type":"function","name":"bad name","parameters":null,"strict":true}])", std::nullopt),
		 .raw = {},
		 .status = 400,
		 .code = "invalid_tool_name",
		 .param = "tools[0].name"},
	    {.name = "a tool name defined twice",
		 .members = tools(R"([{"type":"function","name":"f","parameters":null,"strict":true},{"type":"custom","name":"f"}])", std::nullopt),
		 .raw = {},
		 .status = 400,
		 .code = "duplicate_tool_name",
		 .param = "tools[1].name"},
	    {.name = "a forced tool of the wrong kind",
		 .members = tools(function, R"({"type":"custom","name":"f"})"),
		 .raw = {},
		 .status = 400,
		 .code = "unknown_tool",
		 .param = "tool_choice.name"},
	    {.name = "a forced tool not defined",
		 .members = tools(function, R"({"type":"function","name":"g"})"),
		 .raw = {},
		 .status = 400,
		 .code = "unknown_tool",
		 .param = "tool_choice.name"},
	    {.name = "required without tools",
		 .members = {{.name = "tools", .value = std::nullopt}, {.name = "tool_choice", .value = R"("required")"}},
		 .raw = {},
		 .status = 400,
		 .code = "tool_choice_without_tools",
		 .param = "tool_choice"},
	    {.name = "a forced tool without tools",
		 .members = {{.name = "tools", .value = std::nullopt}, {.name = "tool_choice", .value = R"({"type":"function","name":"f"})"}},
		 .raw = {},
		 .status = 400,
		 .code = "tool_choice_without_tools",
		 .param = "tool_choice"},
	    {.name = "no output tokens",
		 .members = {{.name = "max_output_tokens", .value = "0"}},
		 .raw = {},
		 .status = 400,
		 .code = "max_output_tokens_below_one",
		 .param = "max_output_tokens"},
	    {.name = "more output than the context",
		 .members = {{.name = "max_output_tokens", .value = "262145"}},
		 .raw = {},
		 .status = 400,
		 .code = "max_output_tokens_above_context",
		 .param = "max_output_tokens"},
	    {.name = "a temperature out of range",
		 .members = {{.name = "temperature", .value = "2.5"}},
		 .raw = {},
		 .status = 400,
		 .code = "temperature_out_of_range",
		 .param = "temperature"},
	    {.name = "a top_p out of range",
		 .members = {{.name = "top_p", .value = "1.5"}},
		 .raw = {},
		 .status = 400,
		 .code = "top_p_out_of_range",
		 .param = "top_p"},
	    {.name = "a system message after the first",
		 .members = {{.name = "input", .value = R"([{"role":"user","content":"q"},{"role":"system","content":"s"}])"}},
		 .raw = {},
		 .status = 400,
		 .code = "system_message_not_first",
		 .param = "input[1]"},
	    {.name = "no user query",
		 .members = {{.name = "input", .value = R"([{"type":"function_call_output","call_id":"c","output":"x"}])"}},
		 .raw = {},
		 .status = 400,
		 .code = "no_user_query",
		 .param = "input"},
	    {.name = "an unknown previous response",
		 .members = {{.name = "previous_response_id", .value = R"("resp_00000000000000000000000000000000")"}},
		 .raw = {},
		 .status = 404,
		 .code = "response_not_found",
		 .param = "previous_response_id"},
	    {.name = "a previous response id no server issues",
		 .members = {{.name = "previous_response_id", .value = R"("nonsense")"}},
		 .raw = {},
		 .status = 404,
		 .code = "response_not_found",
		 .param = "previous_response_id"},
	};
}

/**
 * Testing by search: random valid requests, each mutated into every invalid class, are refused with that class's
 * status, code and param, and never reach the engine; the valid requests themselves are served.
 */
[[nodiscard]] auto mutated_requests_are_refused_by_name(std::uint64_t seed) -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto rng = engine::testing::generator(seed);
	auto const all = mutations();
	for (auto round = 0; round < 6; ++round) {
		auto const base = valid_request(rng);
		h->service.engine().queue(Script{{openai::Accepted{}, stopped({1}, usage(1, 0, 1, 0))}});
		auto const served = ask(*h, create(body_of(base), ""));
		if (!served || served->status != 200) {
			std::println("valid request refused: {} -> {}", body_of(base), served ? served->body : "no answer");
			return false;
		}
		for (auto const& mutation : all) {
			auto request = base;
			for (auto const& member : mutation.members) {
				set(request, member);
			}
			auto const before = h->service.engine().submissions().size();
			auto const answered = ask(*h, create(mutation.raw.value_or(body_of(request)), "mutated"));
			if (!answered) {
				return false;
			}
			auto const refused = refusal_of(*reading, *answered);
			auto const param_holds = mutation.param.empty() ? true : refused.param == mutation.param;
			if (refused.status != mutation.status || refused.code != mutation.code || !param_holds || !echoes(*answered, "mutated") ||
			    h->service.engine().submissions().size() != before) {
				std::println("{}: got {} {} '{}' for {}", mutation.name, refused.status, refused.code, refused.param, body_of(request));
				return false;
			}
		}
	}
	return true;
}

/** The sdk's efforts the template lacks are refused listing the four it has (the owner's decision). */
[[nodiscard]] auto unserved_efforts_name_the_served_ones() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	for (auto const effort : std::array<std::string_view, 3>{"minimal", "high", "max"}) {
		auto const answered = ask(*h, create(std::format(R"({{"input":"hi","reasoning":{{"effort":"{}"}}}})", effort), ""));
		if (!answered || !refusal_of(*reading, *answered).message.contains("'none', 'low', 'medium', 'xhigh'")) {
			return false;
		}
	}
	return true;
}

/** A tool whose definition outgrows one JSON fragment of the chat format is refused at the boundary, not rendered. */
[[nodiscard]] auto oversized_tools_are_refused() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto const description = std::string(json::fragment_bounds.document_bytes - 8, 'd');
	auto const body = std::format(R"({{"input":"hi","tools":[{{"type":"custom","name":"run","description":"{}"}}]}})", description);
	auto const answered = ask(*h, create(body, ""));
	if (!answered) {
		return false;
	}
	auto const refused = refusal_of(*reading, *answered);
	return refused.status == 400 && refused.code == "tool_too_large" && refused.param == "tools[0]" &&
	       h->service.engine().submissions().empty();
}

/** One event as the oracle checks it: its type and the item it concerns. */
struct Seen {
	std::string type;
	std::optional<std::int64_t> output_index;

	[[nodiscard]] auto operator==(Seen const&) const -> bool = default;
};

/** The members the sdk's event of each type has, besides `type` and `sequence_number`. */
[[nodiscard]] auto members_of(std::string_view type) -> std::vector<std::string_view> {
	if (type.starts_with("response.output_item.")) {
		return {"output_index", "item"};
	}
	if (type.starts_with("response.content_part.")) {
		return {"item_id", "output_index", "content_index", "part"};
	}
	if (type == "response.output_text.delta") {
		return {"item_id", "output_index", "content_index", "delta", "logprobs"};
	}
	if (type == "response.output_text.done") {
		return {"item_id", "output_index", "content_index", "text", "logprobs"};
	}
	if (type == "response.reasoning_text.delta") {
		return {"item_id", "output_index", "content_index", "delta"};
	}
	if (type == "response.reasoning_text.done") {
		return {"item_id", "output_index", "content_index", "text"};
	}
	if (type == "response.function_call_arguments.delta" || type == "response.custom_tool_call_input.delta") {
		return {"item_id", "output_index", "delta"};
	}
	if (type == "response.function_call_arguments.done") {
		return {"item_id", "output_index", "arguments"};
	}
	if (type == "response.custom_tool_call_input.done") {
		return {"item_id", "output_index", "input"};
	}
	if (type == "error") {
		return {"code", "message", "param"};
	}
	return {"response"};
}

/**
 * The events of a stream as the oracle reads them: every event numbered by its position and carrying its type's
 * members, each delta naming the item its item's added event introduced; runs of one item's deltas merged.
 */
[[nodiscard]] auto read_events(Json& reading, std::vector<std::string> const& events) -> std::optional<std::vector<Seen>> {
	auto seen = std::vector<Seen>{};
	auto ids = std::map<std::int64_t, std::string>{};
	for (auto const [position, event] : std::views::enumerate(events)) {
		auto const type = reading.text(event, {"type"});
		if (!type || reading.integer(event, {"sequence_number"}) != position) {
			return std::nullopt;
		}
		for (auto const member : members_of(*type)) {
			if (!reading.kind(event, {member})) {
				std::println("event {} lacks {}", *type, member);
				return std::nullopt;
			}
		}
		auto const index = reading.integer(event, {"output_index"});
		if (type == "response.output_item.added") {
			ids[index.value()] = reading.text(event, {"item", "id"}).value_or("");
		}
		if (auto const item = reading.text(event, {"item_id"}); item && (!index || ids[*index] != *item)) {
			return std::nullopt;
		}
		auto const merged = type->ends_with(".delta") && !seen.empty() && seen.back() == Seen{.type = *type, .output_index = index};
		if (!merged) {
			seen.push_back(Seen{.type = *type, .output_index = index});
		}
	}
	return seen;
}

/** The events a text item's run makes, and a call's, at `index`. */
auto text_run(std::vector<Seen>& out, std::int64_t index, std::string_view stream) -> void {
	out.push_back({.type = "response.output_item.added", .output_index = index});
	out.push_back({.type = "response.content_part.added", .output_index = index});
	out.push_back({.type = std::format("response.{}.delta", stream), .output_index = index});
	out.push_back({.type = std::format("response.{}.done", stream), .output_index = index});
	out.push_back({.type = "response.content_part.done", .output_index = index});
	out.push_back({.type = "response.output_item.done", .output_index = index});
}

auto call_run(std::vector<Seen>& out, std::int64_t index, std::string_view stream) -> void {
	out.push_back({.type = "response.output_item.added", .output_index = index});
	out.push_back({.type = std::format("response.{}.delta", stream), .output_index = index});
	out.push_back({.type = std::format("response.{}.done", stream), .output_index = index});
	out.push_back({.type = "response.output_item.done", .output_index = index});
}

/** The whole body of a request answered by a stream, which ends with the server's last chunk. */
[[nodiscard]] auto streamed(Harness& h, std::string_view body, std::string_view id) -> std::optional<client::Parsed> {
	return ask(h, create(body, id));
}

/**
 * A streamed response is the sdk's events in order, numbered from 0: the response opens, each item is added, its
 * text streams and is done, it is done, and the response completes carrying every item and the usage. Each item's
 * deltas join to its done text, and the stored response is the one the last event carried.
 */
[[nodiscard]] auto a_stream_follows_the_sdk_events() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	h->service.engine().queue(Script{
	    {openai::Accepted{}},
	    {openai::Reasoning{.text = "Think "}},
	    {openai::Reasoning{.text = "hard."}, openai::Content{.text = "Hi "}},
	    {openai::Content{.text = "there"}, openai::CallBegin{.index = 0, .name = "get_weather", .callee = openai::Callee::Function}},
	    {openai::CallEnd{.index = 0, .text = R"({"city": "Oslo"})", .closing = openai::Closing::Closed}},
	    {openai::CallBegin{.index = 1, .name = "run", .callee = openai::Callee::Custom},
		 openai::CallEnd{.index = 1, .text = "ls -la", .closing = openai::Closing::Closed}},
	    {openai::Content{.text = "bye"}},
	    {stopped({4, 5}, usage(30, 0, 20, 6))},
	});
	auto const body =
	    R"({"input":"weather?","stream":true,"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"},"strict":true},{"type":"custom","name":"run"}]})";
	auto const answered = streamed(*h, body, "stream-1");
	if (!answered) {
		return false;
	}
	auto const events = client::decode_events(answered->body);
	auto const seen = read_events(*reading, events);
	auto expected = std::vector<Seen>{{.type = "response.created", .output_index = std::nullopt},
	                                  {.type = "response.in_progress", .output_index = std::nullopt}};
	text_run(expected, 0, "reasoning_text");
	text_run(expected, 1, "output_text");
	call_run(expected, 2, "function_call_arguments");
	call_run(expected, 3, "custom_tool_call_input");
	text_run(expected, 4, "output_text");
	expected.push_back({.type = "response.completed", .output_index = std::nullopt});
	if (!seen || *seen != expected) {
		std::println("events: {}", events.size());
		for (auto const& event : events) {
			std::println("  {}", event.substr(0, 120));
		}
		return false;
	}
	auto const joined = [&](std::int64_t index) {
		auto text = std::string{};
		for (auto const& event : events) {
			if (reading->integer(event, {"output_index"}) == index && reading->text(event, {"type"}).value_or("").ends_with(".delta")) {
				text += reading->text(event, {"delta"}).value_or("");
			}
		}
		return text;
	};
	auto const& last = events.back();
	auto const id = reading->text(last, {"response", "id"}).value_or("");
	auto const stored = ask(*h, get(std::format("/v1/responses/{}", id), ""));
	auto const carried = last.substr(last.find(R"("response":)") + 11);
	return answered->status == 200 && echoes(*answered, "stream-1") && answered->fields.at("content-type") == "text/event-stream" &&
	       reading->text(events[0], {"response", "status"}) == "in_progress" && reading->count(events[0], {"response", "output"}) == 0 &&
	       joined(0) == "Think hard." && joined(1) == "Hi there" && joined(2) == R"({"city": "Oslo"})" && joined(3) == "ls -la" &&
	       joined(4) == "bye" && reading->text(last, {"response", "status"}) == "completed" &&
	       reading->count(last, {"response", "output"}) == 5 &&
	       reading->text(last, {"response", "output", std::size_t{2}, "type"}) == "function_call" &&
	       reading->text(last, {"response", "output", std::size_t{2}, "name"}) == "get_weather" &&
	       reading->text(last, {"response", "output", std::size_t{2}, "call_id"}).value_or("").starts_with("call_") &&
	       reading->text(last, {"response", "output", std::size_t{3}, "type"}) == "custom_tool_call" &&
	       reading->text(last, {"response", "output", std::size_t{3}, "input"}) == "ls -la" &&
	       reading->integer(last, {"response", "usage", "output_tokens_details", "reasoning_tokens"}) == 6 && stored &&
	       stored->status == 200 && carried.starts_with(stored->body);
}

/** A reply the budget ended is incomplete, and so is the item it cut off, whole or streamed. */
[[nodiscard]] auto a_truncated_reply_is_incomplete() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	h->service.engine().queue(Script{{openai::Accepted{}, openai::Content{.text = "abc"}, truncated({1, 2}, usage(3, 0, 2, 0))}});
	auto const whole = ask(*h, create(R"({"input":"hi","max_output_tokens":2})", ""));
	h->service.engine().queue(Script{{openai::Accepted{}, openai::Content{.text = "done. "},
	                                  openai::CallBegin{.index = 0, .name = "f", .callee = openai::Callee::Function},
	                                  openai::CallEnd{.index = 0, .text = R"({"a": "x"})", .closing = openai::Closing::CutOff},
	                                  truncated({1}, usage(3, 0, 9, 0))}});
	auto const cut =
	    streamed(*h, R"({"input":"hi","stream":true,"tools":[{"type":"function","name":"f","parameters":null,"strict":true}]})", "");
	if (!whole || !cut) {
		return false;
	}
	auto const events = client::decode_events(cut->body);
	auto const& last = events.back();
	return reading->text(whole->body, {"status"}) == "incomplete" &&
	       reading->text(whole->body, {"incomplete_details", "reason"}) == "max_output_tokens" &&
	       reading->text(whole->body, {"output", std::size_t{0}, "status"}) == "incomplete" &&
	       reading->text(last, {"type"}) == "response.incomplete" &&
	       reading->text(last, {"response", "output", std::size_t{0}, "status"}) == "completed" &&
	       reading->text(last, {"response", "output", std::size_t{1}, "status"}) == "incomplete" &&
	       reading->text(last, {"response", "output", std::size_t{1}, "arguments"}) == R"({"a": "x"})";
}

/** The id prefix the server gives an input item of each type. */
[[nodiscard]] auto prefix_of(std::string_view type) -> std::string_view {
	if (type == "message") {
		return "msg_";
	}
	if (type == "function_call") {
		return "fc_";
	}
	if (type == "function_call_output") {
		return "fco_";
	}
	return "rs_";
}

/**
 * A stored response is served back whole by GET, its input items page by page in either order from any cursor as a
 * plain model of the list says, and a deletion removes it; an unstored one was never kept.
 */
[[nodiscard]] auto stored_responses_are_served_back(std::uint64_t seed) -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto rng = engine::testing::generator(seed);
	auto const kinds = std::array<std::string_view, 4>{R"({"role":"user","content":"u"})", R"({"role":"assistant","content":"a"})",
	                                                   R"({"type":"function_call","call_id":"c","name":"f","arguments":"{\"x\": 1}"})",
	                                                   R"({"type":"function_call_output","call_id":"c","output":"o"})"};
	auto const count = core::narrow<std::size_t>(1 + (rng() % 40));
	auto items = std::string{R"([{"role":"user","content":"first"})"};
	auto types = std::vector<std::string_view>{"message"};
	for (auto index = std::size_t{1}; index < count; ++index) {
		auto const kind = rng() % kinds.size();
		items += std::format(",{}", kinds[kind]);
		types.emplace_back(kind < 2 ? "message" : kind == 2 ? "function_call" : "function_call_output");
	}
	items += ']';
	h->service.engine().queue(Script{{openai::Accepted{}, openai::Content{.text = "ok"}, stopped({7}, usage(1, 0, 1, 0))}});
	auto const made = ask(*h, create(std::format(R"({{"input":{}}})", items), ""));
	if (!made || made->status != 200) {
		return false;
	}
	auto const id = reading->text(made->body, {"id"}).value_or("");
	auto const path = std::format("/v1/responses/{}", id);
	auto const retrieved = ask(*h, get(path, "get-1"));
	auto const first_page = ask(*h, get(std::format("{}/input_items?order=asc&limit=100", path), ""));
	if (!retrieved || retrieved->body != made->body || !echoes(*retrieved, "get-1") || !first_page) {
		return false;
	}
	// The model of the list: every item's id in input order, each with its type's prefix.
	auto ids = std::vector<std::string>{};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto item_id = reading->text(first_page->body, {"data", index, "id"}).value_or("");
		if (!item_id.starts_with(prefix_of(types[index])) || reading->text(first_page->body, {"data", index, "type"}) != types[index]) {
			return false;
		}
		ids.push_back(std::move(item_id));
	}
	for (auto round = 0; round < 30; ++round) {
		auto const descending = rng() % 2 == 0;
		auto ordered = ids;
		if (descending) {
			std::ranges::reverse(ordered);
		}
		auto query = std::vector<client::Parameter>{};
		auto limit = std::size_t{20};
		if (rng() % 3 != 0) {
			limit = core::narrow<std::size_t>(1 + (rng() % 100));
			query.push_back({.name = "limit", .value = std::to_string(limit)});
		}
		if (rng() % 2 == 0) {
			query.push_back({.name = "order", .value = descending ? "desc" : "asc"});
		} else if (!descending) {
			query.push_back({.name = "order", .value = "asc"});
		}
		auto start = std::size_t{0};
		if (rng() % 2 == 0) {
			start = core::narrow<std::size_t>(rng() % count);
			query.push_back({.name = "after", .value = ordered[start]});
			start += 1;
		}
		std::ranges::shuffle(query, rng);
		auto const page = ask(*h, get(std::format("{}/input_items?{}", path, client::query_string(query, rng)), ""));
		auto const expected = std::span{ordered}.subspan(start, std::min(limit, count - start));
		if (!page || page->status != 200 || reading->count(page->body, {"data"}) != expected.size() ||
		    reading->boolean(page->body, {"has_more"}) != (start + expected.size() < count)) {
			return false;
		}
		for (auto const [index, item_id] : std::views::enumerate(expected)) {
			if (reading->text(page->body, {"data", core::narrow<std::size_t>(index), "id"}) != item_id) {
				return false;
			}
		}
		if (!expected.empty() &&
		    (reading->text(page->body, {"first_id"}) != expected.front() || reading->text(page->body, {"last_id"}) != expected.back())) {
			return false;
		}
	}
	auto const deleted = ask(*h, remove(path, ""));
	auto const gone = ask(*h, get(path, ""));
	auto const again = ask(*h, remove(path, ""));
	auto const no_items = ask(*h, get(std::format("{}/input_items", path), ""));
	h->service.engine().queue(Script{{openai::Accepted{}, stopped({7}, usage(1, 0, 1, 0))}});
	auto const unstored = ask(*h, create(R"({"input":"hi","store":false})", ""));
	auto const never =
	    unstored ? ask(*h, get(std::format("/v1/responses/{}", reading->text(unstored->body, {"id"}).value_or("")), "")) : std::nullopt;
	return deleted && deleted->status == 200 && reading->text(deleted->body, {"id"}) == id &&
	       reading->text(deleted->body, {"object"}) == "response.deleted" && reading->boolean(deleted->body, {"deleted"}) == true && gone &&
	       refusal_of(*reading, *gone).code == "response_not_found" && gone->status == 404 && again && again->status == 404 && no_items &&
	       no_items->status == 404 && unstored && reading->boolean(unstored->body, {"store"}) == false && never && never->status == 404;
}

/** The stored routes' queries and the cancel route refuse what the server does not serve, by name. */
[[nodiscard]] auto stored_route_queries_are_refused_by_name() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	h->service.engine().queue(Script{{openai::Accepted{}, stopped({7}, usage(1, 0, 1, 0))}});
	auto const made = ask(*h, create(R"({"input":"hi"})", ""));
	if (!made) {
		return false;
	}
	auto const path = std::format("/v1/responses/{}", reading->text(made->body, {"id"}).value_or(""));

	struct Expectation {
		std::string target;
		int status;
		std::string_view code;
		std::string_view param;
	};

	auto const expectations = std::vector<Expectation>{
	    {.target = get(std::format("{}?stream=true", path), ""), .status = 400, .code = "unsupported_parameter", .param = "stream"},
	    {.target = get(std::format("{}?stream=yes", path), ""), .status = 400, .code = "invalid_value", .param = "stream"},
	    {.target = get(std::format("{}?include=reasoning.encrypted_content", path), ""),
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "include"},
	    {.target = get(std::format("{}?include_obfuscation=true", path), ""),
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "include_obfuscation"},
	    {.target = get(std::format("{}?starting_after=3", path), ""),
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "starting_after"},
	    {.target = get(std::format("{}?frobnicate=1", path), ""), .status = 400, .code = "unknown_parameter", .param = "frobnicate"},
	    {.target = get(std::format("{}/input_items?limit=0", path), ""), .status = 400, .code = "invalid_value", .param = "limit"},
	    {.target = get(std::format("{}/input_items?limit=101", path), ""), .status = 400, .code = "invalid_value", .param = "limit"},
	    {.target = get(std::format("{}/input_items?limit=x", path), ""), .status = 400, .code = "invalid_value", .param = "limit"},
	    {.target = get(std::format("{}/input_items?order=up", path), ""), .status = 400, .code = "invalid_value", .param = "order"},
	    {.target = get(std::format("{}/input_items?after=msg_nope", path), ""), .status = 400, .code = "invalid_value", .param = "after"},
	    {.target = get(std::format("{}/input_items?include=x", path), ""),
		 .status = 400,
		 .code = "unsupported_parameter",
		 .param = "include"},
	    {.target = get(std::format("{}/input_items?limit=2&limit=3", path), ""), .status = 400, .code = "invalid_value", .param = "limit"},
	    {.target = post(std::format("{}/cancel", path), "", ""), .status = 400, .code = "unsupported_parameter", .param = "background"},
	    {.target = get("/v1/responses/resp_nope", ""), .status = 404, .code = "response_not_found", .param = ""},
	    {.target = post("/v1/responses/compact", "{}", ""), .status = 405, .code = "method_not_allowed", .param = ""},
	    {.target = post("/v1/responses/input_tokens", "{}", ""), .status = 405, .code = "method_not_allowed", .param = ""},
	};
	for (auto const& expectation : expectations) {
		auto const answered = ask(*h, expectation.target);
		if (!answered) {
			return false;
		}
		auto const refused = refusal_of(*reading, *answered);
		if (refused.status != expectation.status || refused.code != expectation.code || refused.param != expectation.param) {
			std::println("{}: got {} {} '{}'", expectation.target.substr(0, 80), refused.status, refused.code, refused.param);
			return false;
		}
	}
	auto const plain = ask(*h, get(std::format("{}?stream=false&include_obfuscation=false", path), ""));
	return plain && plain->status == 200;
}

/** Creates a stored response the engine answers with `sequence`; its id. */
[[nodiscard]] auto stored(Harness& h, Json& reading, std::string_view body, openai::Output finish) -> std::optional<std::string> {
	h.service.engine().queue(Script{{openai::Accepted{}, openai::Content{.text = "answer"}, std::move(finish)}});
	auto const made = ask(h, create(body, ""));
	if (!made || made->status != 200) {
		return std::nullopt;
	}
	return reading.text(made->body, {"id"});
}

/**
 * A continuation reaches the engine as the stored response's exact sequence and its new turns, never a re-render;
 * one that changes the stored instructions, tools or effort, continues an incomplete or unstored response, or puts a
 * system message in its new input, is refused by name.
 */
[[nodiscard]] auto continuations_carry_the_stored_sequence() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto const tools = std::string_view{R"([{"type":"function","name":"f","parameters":{"type":"object"},"strict":true}])"};
	auto const sequence = std::vector<tok::Id>{5, 6, 7, tok::id(tok::Added::im_end)};
	auto const parent =
	    stored(*h, *reading, std::format(R"({{"input":"hi","instructions":"be brief","tools":{},"reasoning":{{"effort":"low"}}}})", tools),
		       stopped(sequence, usage(3, 0, 2, 0)));
	auto const cut = stored(*h, *reading, R"({"input":"hi"})", truncated({1, 2}, usage(1, 0, 1, 0)));
	auto const unkept = stored(*h, *reading, R"({"input":"hi","store":false})", stopped({1}, usage(1, 0, 1, 0)));
	if (!parent || !cut || !unkept) {
		return false;
	}
	auto const items =
	    std::string_view{R"([{"type":"function_call_output","call_id":"c","output":"sunny"},{"role":"user","content":"thanks"}])"};
	auto const continuing = [&](std::string_view previous, std::string_view instructions, std::string_view tool_list,
	                            std::string_view effort, std::string_view input) {
		return std::format(R"({{"previous_response_id":"{}","input":{},"instructions":"{}","tools":{},"reasoning":{{"effort":"{}"}}}})",
		                   previous, input, instructions, tool_list, effort);
	};
	h->service.engine().queue(Script{{openai::Accepted{}, stopped({1}, usage(1, 0, 1, 0))}});
	auto const continued = ask(*h, create(continuing(*parent, "be brief", tools, "low", items), ""));
	auto const before = h->service.engine().submissions().size();

	struct Refused {
		std::string body;
		int status;
		std::string_view code;
		std::string_view param;
	};

	auto const refusals = std::vector<Refused>{
	    {.body = continuing(*parent, "be long", tools, "low", items),
		 .status = 400,
		 .code = "previous_response_mismatch",
		 .param = "instructions"},
	    {.body = continuing(*parent, "be brief", "[]", "low", items),
		 .status = 400,
		 .code = "previous_response_mismatch",
		 .param = "tools"},
	    {.body = continuing(*parent, "be brief", tools, "xhigh", items),
		 .status = 400,
		 .code = "previous_response_mismatch",
		 .param = "reasoning.effort"},
	    {.body = continuing(*parent, "be brief", tools, "low", R"([{"role":"developer","content":"x"}])"),
		 .status = 400,
		 .code = "system_message_not_first",
		 .param = "input[0]"},
	    {.body = std::format(R"({{"previous_response_id":"{}","input":"more"}})", *cut),
		 .status = 400,
		 .code = "previous_response_incomplete",
		 .param = "previous_response_id"},
	    {.body = std::format(R"({{"previous_response_id":"{}","input":"more"}})", *unkept),
		 .status = 404,
		 .code = "response_not_found",
		 .param = "previous_response_id"},
	};
	for (auto const& refused : refusals) {
		auto const answered = ask(*h, create(refused.body, ""));
		auto const read = answered ? std::optional{refusal_of(*reading, *answered)} : std::nullopt;
		if (!read || read->status != refused.status || read->code != refused.code || read->param != refused.param) {
			return false;
		}
	}
	if (!continued || continued->status != 200 || h->service.engine().submissions().size() != before) {
		return false;
	}
	auto const& submission = h->service.engine().submissions()[before - 1].second;
	auto const new_items = std::vector<json::InputItem>{json::FunctionOutputItem{.call_id = "c", .output = "sunny"},
	                                                    json::MessageItem{.role = json::Role::User, .text = "thanks"}};
	auto const turns = chat::continuation(new_items);
	return turns &&
	       submission.prompt ==
	           openai::Prompt{openai::Continued{.sequence = sequence, .turns = *turns, .options = chat::options_of(json::Effort::Low)}} &&
	       submission.constraints.thinking == chat::Thinking::Enabled && submission.constraints.tools.size() == 1 &&
	       reading->text(continued->body, {"previous_response_id"}) == *parent;
}

/**
 * Branching: several forks of one stored response, sent at once on their own connections, each continue from its
 * exact sequence with their own new turns; each fork is stored with its own sequence and continues in turn.
 */
[[nodiscard]] auto forks_of_one_response_continue_independently() -> bool {
	auto h = serving();
	auto reading = Json::make();
	if (!h || !reading) {
		return false;
	}
	auto const root = std::vector<tok::Id>{9, 8, 7, tok::id(tok::Added::im_end)};
	auto const parent = stored(*h, *reading, R"({"input":"root"})", stopped(root, usage(3, 0, 1, 0)));
	if (!parent) {
		return false;
	}
	constexpr auto forks = std::size_t{5};
	auto peers = std::vector<client::Client>{};
	auto const before = h->service.engine().submissions().size();
	for (auto fork = std::size_t{0}; fork < forks; ++fork) {
		auto peer = client::connect(h->server.port());
		if (!peer) {
			return false;
		}
		auto const body = std::format(R"({{"previous_response_id":"{}","input":"fork {}"}})", *parent, fork);
		if (!h->push(*peer, create(body, std::format("fork-{}", fork)))) {
			return false;
		}
		peers.push_back(*std::move(peer));
	}
	if (!h->wait(
	        [&] {
		        return h->service.engine().submissions().size() == before + forks;
	        },
	        2000ms)) {
		return false;
	}
	// The forks are answered in reverse order of their arrival, each with a sequence of its own.
	for (auto fork = forks; fork > 0; --fork) {
		auto const& [exchange, submission] = h->service.engine().submissions()[before + fork - 1];
		auto const turns = chat::continuation(
		    std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = std::format("fork {}", fork - 1)}});
		if (!turns || submission.prompt != openai::Prompt{openai::Continued{
		                                       .sequence = root, .turns = *turns, .options = chat::options_of(json::Effort::XHigh)}}) {
			return false;
		}
		auto sequence = root;
		sequence.push_back(static_cast<tok::Id>(100 + fork));
		sequence.push_back(tok::id(tok::Added::im_end));
		if (!h->service.engine().post(exchange, {openai::Accepted{}, openai::Content{.text = std::format("fork {} says", fork - 1)},
		                                         stopped(sequence, usage(6, 4, 2, 0))})) {
			return false;
		}
	}
	auto fork_ids = std::vector<std::string>{};
	for (auto& peer : peers) {
		auto const answered = h->ask(peer, "");
		if (!answered || answered->status != 200 || reading->text(answered->body, {"previous_response_id"}) != *parent) {
			return false;
		}
		fork_ids.push_back(reading->text(answered->body, {"id"}).value_or(""));
	}
	// Each fork continues from its own stored sequence.
	for (auto const [fork, id] : std::views::enumerate(fork_ids)) {
		h->service.engine().queue(Script{{openai::Accepted{}, stopped({1}, usage(1, 0, 1, 0))}});
		auto const next = ask(*h, create(std::format(R"({{"previous_response_id":"{}","input":"again"}})", id), ""));
		auto const& submission = h->service.engine().submissions().back().second;
		auto const continued = core::alternative<openai::Continued>(submission.prompt);
		auto expected = root;
		expected.push_back(static_cast<tok::Id>(100 + fork + 1));
		expected.push_back(tok::id(tok::Added::im_end));
		if (!next || next->status != 200 || !continued || continued->sequence != expected) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const checks = std::array{
	    engine::testing::Case{.name = "a whole response gathers the engine's outputs into items, with the echo and usage",
		                      .holds = a_whole_response_gathers_the_outputs()},
	    engine::testing::Case{.name = "sampling follows the effort's defaults; every request draws a fresh seed",
		                      .holds = sampling_follows_the_effort()},
	    engine::testing::Case{.name = "valid requests mutated into each invalid class are refused by name (seed 1)",
		                      .holds = mutated_requests_are_refused_by_name(1)},
	    engine::testing::Case{.name = "valid requests mutated into each invalid class are refused by name (seed 2)",
		                      .holds = mutated_requests_are_refused_by_name(2)},
	    engine::testing::Case{.name = "the efforts the template lacks are refused listing the four it has",
		                      .holds = unserved_efforts_name_the_served_ones()},
	    engine::testing::Case{.name = "a tool too large for the chat format is refused at the boundary",
		                      .holds = oversized_tools_are_refused()},
	    engine::testing::Case{.name = "a stream is the sdk's events, in order, numbered from 0",
		                      .holds = a_stream_follows_the_sdk_events()},
	    engine::testing::Case{.name = "a reply the budget ended is incomplete, with the item it cut off",
		                      .holds = a_truncated_reply_is_incomplete()},
	    engine::testing::Case{.name = "stored responses are served back, paged and deleted (seed 1)",
		                      .holds = stored_responses_are_served_back(1)},
	    engine::testing::Case{.name = "stored responses are served back, paged and deleted (seed 2)",
		                      .holds = stored_responses_are_served_back(2)},
	    engine::testing::Case{.name = "the stored routes refuse unserved queries by name",
		                      .holds = stored_route_queries_are_refused_by_name()},
	    engine::testing::Case{.name = "a continuation carries the stored sequence and refuses what would re-render it",
		                      .holds = continuations_carry_the_stored_sequence()},
	    engine::testing::Case{.name = "concurrent forks of one stored response continue independently",
		                      .holds = forks_of_one_response_continue_independently()},
	};
	return engine::testing::report(checks);
}

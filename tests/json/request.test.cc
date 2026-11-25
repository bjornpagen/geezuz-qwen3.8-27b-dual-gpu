// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.json_model;
import engine.testing;

namespace {

namespace json = engine::json;
namespace model = engine::json_model;

using json::Violation;

/** A sink that keeps what is written: the test writes each request once, choosing its spellings as it goes. */
struct Text {
	std::string bytes;

	auto put(std::string_view text) -> void {
		bytes += text;
	}
};

using Members = json::Members<Text>;
using Elements = json::Elements<Text>;

[[nodiscard]] auto make_decoder() -> std::unique_ptr<json::RequestDecoder> {
	auto decoder = json::RequestDecoder::make();
	engine::core::invariant(decoder.has_value());
	return std::make_unique<json::RequestDecoder>(std::move(*decoder));
}

using Scratch = json::Reader<json::request_bounds>;

[[nodiscard]] auto make_scratch() -> std::unique_ptr<Scratch> {
	auto reader = Scratch::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<Scratch>(std::move(*reader));
}

[[nodiscard]] auto json_text(Scratch& scratch, std::string_view text) -> json::JsonText {
	auto held = json::JsonText::from(scratch, text);
	engine::core::invariant(held.has_value());
	return *std::move(held);
}

[[nodiscard]] auto coin(std::mt19937_64& rng) -> bool {
	return (rng() % 2) == 0;
}

[[nodiscard]] auto compact_object(std::mt19937_64& rng, std::size_t members) -> std::string {
	auto object = model::Members{};
	for (auto count = members; count > 0; --count) {
		object.emplace_back(model::random_text(rng, 4) + std::to_string(count), model::generate(rng, model::random_shape(rng)));
	}
	return model::serialize(model::Model{.value = std::move(object)}, model::Style{.spelling = model::Spelling::Raw, .spacing = model::Spacing::Compact}, rng);
}

[[nodiscard]] auto random_tool(std::mt19937_64& rng, Scratch& scratch) -> json::Tool {
	auto const name = "fn_" + std::to_string(rng() % 1000);
	auto const parameters = coin(rng) ? std::optional<json::JsonText>{json_text(scratch, compact_object(rng, rng() % 4))} : std::nullopt;
	auto definition = Text{};
	json::write_object(definition, [&](Members& tool) {
		tool.string("type", "function");
		tool.object("function", [&](Members& function) {
			function.string("name", name);
			if (coin(rng)) {
				function.string("description", model::random_text(rng, 10));
			}
			if (parameters) {
				function.json("parameters", *parameters);
			} else if (coin(rng)) {
				function.null("parameters");
			}
			if (coin(rng)) {
				function.boolean("strict", false);
			}
		});
	});
	return json::Tool{.name = name, .parameters = parameters, .definition = json_text(scratch, definition.bytes)};
}

[[nodiscard]] auto random_call(std::mt19937_64& rng, Scratch& scratch) -> json::ToolCall {
	auto call = json::ToolCall{.id = "call_" + std::to_string(rng()), .name = "fn_" + std::to_string(rng() % 1000), .arguments = {}};
	for (auto count = rng() % 4; count > 0; --count) {
		auto name = "arg" + std::to_string(count);
		if (coin(rng)) {
			call.arguments.push_back(json::Argument{.name = std::move(name), .value = model::random_text(rng, 12)});
		} else {
			auto const value = model::generate(rng, model::random_shape(rng));
			auto text = model::serialize(value, model::Style{.spelling = model::Spelling::Raw, .spacing = model::Spacing::Compact}, rng);
			// A string argument is the other alternative; generate a non-string here.
			if (std::holds_alternative<std::string>(value.value)) {
				text.insert(0, "[").append("]");
			}
			call.arguments.push_back(json::Argument{.name = std::move(name), .value = json_text(scratch, text)});
		}
	}
	return call;
}

[[nodiscard]] auto random_message(std::mt19937_64& rng, Scratch& scratch) -> json::Message {
	switch (rng() % 4) {
	case 0:
		return json::SystemMessage{.content = model::random_text(rng, 20)};
	case 1:
		return json::UserMessage{.content = model::random_text(rng, 20)};
	case 2: {
		auto message = json::AssistantMessage{.content = coin(rng) ? model::random_text(rng, 20) : "", .reasoning_content = coin(rng) ? model::random_text(rng, 20) : "", .tool_calls = {}};
		for (auto count = rng() % 3; count > 0; --count) {
			message.tool_calls.push_back(random_call(rng, scratch));
		}
		return message;
	}
	default:
		return json::ToolMessage{.tool_call_id = "call_" + std::to_string(rng()), .content = model::random_text(rng, 20)};
	}
}

template<class T>
[[nodiscard]] auto maybe(std::mt19937_64& rng, T value) -> std::optional<T> {
	return coin(rng) ? std::optional<T>{std::move(value)} : std::nullopt;
}

[[nodiscard]] auto random_request(std::mt19937_64& rng, Scratch& scratch) -> json::ChatRequest {
	auto request = json::ChatRequest{
	    .model = maybe(rng, model::random_text(rng, 8)),
	    .messages = {},
	    .tools = {},
	    .tool_choice = std::nullopt,
	    .delivery = std::array{json::Delivery::Whole, json::Delivery::Stream, json::Delivery::StreamWithUsage}[rng() % 3],
	    .max_tokens = maybe(rng, static_cast<std::int64_t>(rng() % 100000)),
	    .temperature = maybe(rng, model::random_finite(rng)),
	    .top_p = maybe(rng, static_cast<double>(rng() % 1000) / 1000.0),
	    .top_k = maybe(rng, static_cast<std::int64_t>(rng() % 100) - 1),
	    .min_p = maybe(rng, static_cast<double>(rng() % 1000) / 1000.0),
	    .seed = maybe(rng, std::bit_cast<std::int64_t>(rng())),
	    .stop = {},
	    .template_options = {
	        .thinking = maybe(rng, coin(rng) ? json::Thinking::Enabled : json::Thinking::Disabled),
	        .reasoning_effort = maybe(rng, std::array{json::ReasoningEffort::Low, json::ReasoningEffort::Medium, json::ReasoningEffort::XHigh}[rng() % 3]),
	        .past_thinking = maybe(rng, coin(rng) ? json::PastThinking::Preserved : json::PastThinking::Dropped),
	    },
	};
	for (auto count = rng() % 6; count > 0; --count) {
		request.messages.push_back(random_message(rng, scratch));
	}
	for (auto count = rng() % 3; count > 0; --count) {
		request.tools.push_back(random_tool(rng, scratch));
	}
	if (coin(rng)) {
		request.tool_choice = coin(rng) ? json::ToolChoice{std::array{json::ToolChoiceMode::None, json::ToolChoiceMode::Auto, json::ToolChoiceMode::Required}[rng() % 3]}
		                                : json::ToolChoice{json::ForcedFunction{.name = "fn_" + std::to_string(rng() % 1000)}};
	}
	for (auto count = rng() % (json::max_stop_sequences + 1); count > 0; --count) {
		request.stop.push_back(model::random_text(rng, 4));
	}
	return request;
}

/** Writes message content in one of its spellings: a string, text parts, or (when empty) null or nothing. */
auto write_content(Members& message, std::string_view content, std::mt19937_64& rng) -> void {
	auto const spelling = rng() % 4;
	if (content.empty() && spelling == 0) {
		return;
	}
	if (content.empty() && spelling == 1) {
		message.null("content");
		return;
	}
	if (spelling == 2) {
		message.array("content", [&](Elements& parts) {
			auto at = std::size_t{0};
			while (at < content.size() || (at == 0 && coin(rng))) {
				auto end = std::min<std::size_t>(content.size(), at + 1 + rng() % 12);
				while (end < content.size() && (std::bit_cast<std::uint8_t>(content[end]) & 0xC0U) == 0x80U) {
					++end;
				}
				parts.object([&](Members& part) {
					if (coin(rng)) {
						part.string("type", "text");
						part.string("text", content.substr(at, end - at));
					} else {
						part.string("text", content.substr(at, end - at));
						part.string("type", "text");
					}
				});
				at = end;
				if (at == 0) {
					break;
				}
			}
		});
		return;
	}
	message.string("content", content);
}

auto write_arguments(Members& function, std::vector<json::Argument> const& arguments, std::mt19937_64& rng) -> void {
	auto const object = [&arguments](Members& written) {
		for (auto const& argument : arguments) {
			if (std::holds_alternative<std::string>(argument.value)) {
				written.string(argument.name, std::get<std::string>(argument.value));
			} else {
				written.json(argument.name, std::get<json::JsonText>(argument.value));
			}
		}
	};
	auto const spelling = rng() % 4;
	if (arguments.empty() && spelling == 0) {
		return;
	}
	if (arguments.empty() && spelling == 1) {
		function.string("arguments", "");
		return;
	}
	if (spelling == 2) {
		function.object("arguments", object);
		return;
	}
	auto held = Text{};
	json::write_object(held, object);
	function.string("arguments", held.bytes);
}

auto write_message(Elements& messages, json::Message const& message, std::mt19937_64& rng) -> void {
	messages.object([&](Members& written) {
		if (std::holds_alternative<json::SystemMessage>(message)) {
			written.string("role", coin(rng) ? "system" : "developer");
			write_content(written, std::get<json::SystemMessage>(message).content, rng);
		} else if (std::holds_alternative<json::UserMessage>(message)) {
			write_content(written, std::get<json::UserMessage>(message).content, rng);
			written.string("role", "user");
			if (coin(rng)) {
				written.string("name", "someone");
			}
		} else if (std::holds_alternative<json::ToolMessage>(message)) {
			auto const& tool = std::get<json::ToolMessage>(message);
			written.string("tool_call_id", tool.tool_call_id);
			written.string("role", "tool");
			write_content(written, tool.content, rng);
		} else {
			auto const& assistant = std::get<json::AssistantMessage>(message);
			written.string("role", "assistant");
			write_content(written, assistant.content, rng);
			if (!assistant.reasoning_content.empty() || coin(rng)) {
				written.string("reasoning_content", assistant.reasoning_content);
			} else if (coin(rng)) {
				written.null("reasoning_content");
			}
			if (coin(rng)) {
				written.null("refusal");
			}
			if (!assistant.tool_calls.empty() || coin(rng)) {
				written.array("tool_calls", [&](Elements& calls) {
					auto index = std::int64_t{0};
					for (auto const& call : assistant.tool_calls) {
						calls.object([&](Members& written_call) {
							if (coin(rng)) {
								written_call.integer("index", index);
							}
							written_call.string("id", call.id);
							if (coin(rng)) {
								written_call.string("type", "function");
							}
							written_call.object("function", [&](Members& function) {
								function.string("name", call.name);
								write_arguments(function, call.arguments, rng);
							});
						});
						++index;
					}
				});
			} else if (coin(rng)) {
				written.null("tool_calls");
			}
		}
	});
}

enum class Field : std::uint8_t {
	Messages,
	Model,
	Tools,
	ToolChoice,
	Delivery,
	MaxTokens,
	Sampling,
	Stop,
	Template,
	Neutral,
};

/** Writes a request in a random order of members and a random choice of the spellings each member allows. */
[[nodiscard]] auto encode(json::ChatRequest const& request, std::mt19937_64& rng) -> std::string {
	auto fields = std::vector<Field>{Field::Messages, Field::Model, Field::Tools, Field::ToolChoice, Field::Delivery, Field::MaxTokens, Field::Sampling, Field::Stop, Field::Template, Field::Neutral};
	std::ranges::shuffle(fields, rng);
	auto out = Text{};
	json::write_object(out, [&](Members& written) {
		for (auto const field : fields) {
			switch (field) {
			case Field::Messages:
				written.array("messages", [&](Elements& messages) {
					for (auto const& message : request.messages) {
						write_message(messages, message, rng);
					}
				});
				break;
			case Field::Model:
				if (request.model) {
					written.string("model", *request.model);
				}
				break;
			case Field::Tools:
				if (!request.tools.empty() || coin(rng)) {
					written.array("tools", [&](Elements& tools) {
						for (auto const& tool : request.tools) {
							tools.json(tool.definition);
						}
					});
				}
				break;
			case Field::ToolChoice:
				if (request.tool_choice && std::holds_alternative<json::ToolChoiceMode>(*request.tool_choice)) {
					auto constexpr modes = std::array<std::string_view, 3>{"none", "auto", "required"};
					written.string("tool_choice", modes[std::to_underlying(std::get<json::ToolChoiceMode>(*request.tool_choice))]);
				} else if (request.tool_choice) {
					written.object("tool_choice", [&](Members& choice) {
						choice.string("type", "function");
						choice.object("function", [&](Members& function) { function.string("name", std::get<json::ForcedFunction>(*request.tool_choice).name); });
					});
				}
				break;
			case Field::Delivery:
				if (request.delivery == json::Delivery::Whole) {
					if (coin(rng)) {
						written.boolean("stream", false);
					}
				} else {
					written.boolean("stream", true);
					if (request.delivery == json::Delivery::StreamWithUsage) {
						written.object("stream_options", [](Members& options) { options.boolean("include_usage", true); });
					} else if (coin(rng)) {
						written.object("stream_options", [&](Members& options) {
							if (coin(rng)) {
								options.boolean("include_usage", false);
							}
						});
					}
				}
				break;
			case Field::MaxTokens:
				if (request.max_tokens) {
					written.integer(coin(rng) ? "max_tokens" : "max_completion_tokens", *request.max_tokens);
				}
				break;
			case Field::Sampling:
				for (auto const& [name, value] : {std::pair{"temperature", request.temperature}, std::pair{"top_p", request.top_p}, std::pair{"min_p", request.min_p}}) {
					if (value) {
						written.real(name, *value);
					} else if (coin(rng)) {
						written.null(name);
					}
				}
				for (auto const& [name, value] : {std::pair{"top_k", request.top_k}, std::pair{"seed", request.seed}}) {
					if (value) {
						written.integer(name, *value);
					}
				}
				break;
			case Field::Stop:
				if (request.stop.size() == 1 && coin(rng)) {
					written.string("stop", request.stop.front());
				} else if (!request.stop.empty() || coin(rng)) {
					written.array("stop", [&](Elements& stops) {
						for (auto const& stop : request.stop) {
							stops.string(stop);
						}
					});
				}
				break;
			case Field::Template: {
				auto const& options = request.template_options;
				if (options.thinking || options.reasoning_effort || options.past_thinking || coin(rng)) {
					written.object("chat_template_kwargs", [&](Members& kwargs) {
						if (options.thinking) {
							kwargs.boolean("enable_thinking", *options.thinking == json::Thinking::Enabled);
						}
						if (options.reasoning_effort) {
							auto constexpr efforts = std::array<std::string_view, 3>{"low", "medium", "xhigh"};
							kwargs.string("reasoning_effort", efforts[std::to_underlying(*options.reasoning_effort)]);
						}
						if (options.past_thinking) {
							kwargs.boolean("preserve_thinking", *options.past_thinking == json::PastThinking::Preserved);
						}
					});
				}
				break;
			}
			case Field::Neutral:
				if (coin(rng)) {
					written.integer("n", 1);
					written.real("presence_penalty", 0.0);
					written.integer("frequency_penalty", 0);
					written.boolean("logprobs", false);
					written.object("logit_bias", [](Members&) {});
					written.string("user", "client-42");
					written.object("response_format", [](Members& format) { format.string("type", "text"); });
					written.array("modalities", [](Elements& modalities) { modalities.string("text"); });
					written.boolean("parallel_tool_calls", true);
					written.null("metadata");
				}
				break;
			}
		}
	});
	return out.bytes;
}

/** Every generated request, written in any order and any spelling its members allow, decodes to itself. */
[[nodiscard]] auto requests_round_trip(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto decoder = make_decoder();
	auto scratch = make_scratch();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const request = random_request(rng, *scratch);
		auto const body = encode(request, rng);
		auto const decoded = decoder->decode_copy(body);
		if (!decoded || *decoded != request) {
			std::println("request did not decode to itself: {}\n{:?}", decoded ? "different" : json::describe(decoded.error()), body);
			return false;
		}
	}
	return true;
}

struct Refusal {
	std::string_view body;
	json::DecodeError error;
};

[[nodiscard]] auto at(json::DecodeFault fault, std::initializer_list<json::PathStep> path) -> json::DecodeError {
	return json::DecodeError{.fault = std::move(fault), .path = path};
}

[[nodiscard]] auto shape(json::Expected expected, json::Kind found) -> json::DecodeFault {
	return json::WrongShape{.expected = expected, .found = found};
}

/** Requests refused with the documented fault at the documented member, one per rule the decoder states. */
[[nodiscard]] auto refusals_name_the_member() -> bool {
	using json::Kind;
	using json::Expected;
	using P = json::PathStep;
	auto const parse = [](json::ParseFault fault) { return json::DecodeFault{json::ParseError{.fault = fault}}; };
	auto const refusals = std::vector<Refusal>{
	    {.body = R"([])", .error = at(shape(Expected::Object, Kind::Array), {})},
	    {.body = R"({"messages":[]} {})", .error = at(parse(json::ParseFault::TrailingContent), {})},
	    {.body = R"({"messages":[]}x)", .error = at(parse(json::ParseFault::TrailingContent), {})},
	    {.body = R"({"messages":[{"role":"user","content":"]}"}]} x)", .error = at(parse(json::ParseFault::TrailingContent), {})},
	    {.body = R"({"messages":[)", .error = at(parse(json::ParseFault::Truncated), {})},
	    {.body = R"({})", .error = at(Violation::MissingField, {P{"messages"}})},
	    {.body = R"({"messages":[],"best_of":2})", .error = at(Violation::UnknownField, {P{"best_of"}})},
	    {.body = R"({"messages":[],"messages":[]})", .error = at(parse(json::ParseFault::DuplicateKey), {})},
	    {.body = R"({"messages":{}})", .error = at(shape(Expected::Array, Kind::Object), {P{"messages"}})},
	    {.body = R"({"messages":[{"content":"x"}]})", .error = at(Violation::MissingField, {P{"messages"}, P{std::size_t{0}}, P{"role"}})},
	    {.body = R"({"messages":[{"role":"robot","content":"x"}]})", .error = at(Violation::UnknownValue, {P{"messages"}, P{std::size_t{0}}, P{"role"}})},
	    {.body = R"({"messages":[{"role":"function","content":"x","name":"f"}]})", .error = at(Violation::Unsupported, {P{"messages"}, P{std::size_t{0}}, P{"role"}})},
	    {.body = R"({"messages":[{"role":"user","content":"x","mood":"good"}]})", .error = at(Violation::UnknownField, {P{"messages"}, P{std::size_t{0}}, P{"mood"}})},
	    {.body = R"({"messages":[{"role":"user","content":"x","tool_call_id":"c"}]})", .error = at(Violation::UnknownField, {P{"messages"}, P{std::size_t{0}}, P{"tool_call_id"}})},
	    {.body = R"({"messages":[{"role":"tool","content":"x"}]})", .error = at(Violation::MissingField, {P{"messages"}, P{std::size_t{0}}, P{"tool_call_id"}})},
	    {.body = R"({"messages":[{"role":"user","content":7}]})", .error = at(shape(Expected::Content, Kind::Integer), {P{"messages"}, P{std::size_t{0}}, P{"content"}})},
	    {.body = R"({"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u"}}]}]})",
	     .error = at(Violation::Unsupported, {P{"messages"}, P{std::size_t{0}}, P{"content"}, P{std::size_t{1}}, P{"type"}})},
	    {.body = R"({"messages":[{"role":"user","content":[{"type":"sticker"}]}]})", .error = at(Violation::UnknownValue, {P{"messages"}, P{std::size_t{0}}, P{"content"}, P{std::size_t{0}}, P{"type"}})},
	    {.body = R"({"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{}}]}]})",
	     .error = at(Violation::UnknownField, {P{"messages"}, P{std::size_t{0}}, P{"content"}, P{std::size_t{0}}, P{"cache_control"}})},
	    {.body = R"({"messages":[{"role":"user","content":[{"text":"a"}]}]})", .error = at(Violation::MissingField, {P{"messages"}, P{std::size_t{0}}, P{"content"}, P{std::size_t{0}}, P{"type"}})},
	    {.body = R"({"messages":[{"role":"assistant","refusal":"no"}]})", .error = at(Violation::Unsupported, {P{"messages"}, P{std::size_t{0}}, P{"refusal"}})},
	    {.body = R"({"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"custom","custom":{}}]}]})",
	     .error = at(Violation::Unsupported, {P{"messages"}, P{std::size_t{0}}, P{"tool_calls"}, P{std::size_t{0}}, P{"type"}})},
	    {.body = R"({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f"}}]}]})",
	     .error = at(Violation::MissingField, {P{"messages"}, P{std::size_t{0}}, P{"tool_calls"}, P{std::size_t{0}}, P{"id"}})},
	    {.body = R"({"messages":[{"role":"assistant","tool_calls":[{"id":"c","function":{"name":"f","arguments":"{\"a\":"}}]}]})",
	     .error = at(parse(json::ParseFault::Truncated), {P{"messages"}, P{std::size_t{0}}, P{"tool_calls"}, P{std::size_t{0}}, P{"function"}, P{"arguments"}})},
	    {.body = R"({"messages":[{"role":"assistant","tool_calls":[{"id":"c","function":{"name":"f","arguments":"[1]"}}]}]})",
	     .error = at(shape(Expected::Arguments, Kind::Array), {P{"messages"}, P{std::size_t{0}}, P{"tool_calls"}, P{std::size_t{0}}, P{"function"}, P{"arguments"}})},
	    {.body = R"({"messages":[{"role":"assistant","tool_calls":[{"id":"c","function":{"name":"f","arguments":"{\"a\":1,\"a\":2}"}}]}]})",
	     .error = at(parse(json::ParseFault::DuplicateKey), {P{"messages"}, P{std::size_t{0}}, P{"tool_calls"}, P{std::size_t{0}}, P{"function"}, P{"arguments"}})},
	    {.body = R"({"messages":[{"role":"assistant","tool_calls":[{"id":"c","function":{"name":"f","extra":1}}]}]})",
	     .error = at(Violation::UnknownField, {P{"messages"}, P{std::size_t{0}}, P{"tool_calls"}, P{std::size_t{0}}, P{"function"}, P{"extra"}})},
	    {.body = R"({"messages":[],"tools":[{"type":"custom","custom":{"name":"x"}}]})", .error = at(Violation::Unsupported, {P{"tools"}, P{std::size_t{0}}, P{"type"}})},
	    {.body = R"({"messages":[],"tools":[{"type":"function","function":{"name":"f"},"cache_control":{}}]})", .error = at(Violation::UnknownField, {P{"tools"}, P{std::size_t{0}}, P{"cache_control"}})},
	    {.body = R"({"messages":[],"tools":[{"type":"function","function":{"name":"f","strict":true}}]})", .error = at(Violation::Unsupported, {P{"tools"}, P{std::size_t{0}}, P{"function"}, P{"strict"}})},
	    {.body = R"({"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":[]}}]})",
	     .error = at(shape(Expected::Object, Kind::Array), {P{"tools"}, P{std::size_t{0}}, P{"function"}, P{"parameters"}})},
	    {.body = R"({"messages":[],"tools":[{"type":"function","function":{"name":"f","parameters":{"a":tru}}}]})", .error = at(parse(json::ParseFault::InvalidLiteral), {P{"tools"}, P{std::size_t{0}}})},
	    {.body = R"({"messages":[],"tools":[{"type":"function","function":{}}]})", .error = at(Violation::MissingField, {P{"tools"}, P{std::size_t{0}}, P{"function"}, P{"name"}})},
	    {.body = R"({"messages":[],"tool_choice":"maybe"})", .error = at(Violation::UnknownValue, {P{"tool_choice"}})},
	    {.body = R"({"messages":[],"tool_choice":{"type":"allowed_tools","allowed_tools":{}}})", .error = at(Violation::Unsupported, {P{"tool_choice"}, P{"type"}})},
	    {.body = R"({"messages":[],"tool_choice":3})", .error = at(shape(Expected::ToolChoice, Kind::Integer), {P{"tool_choice"}})},
	    {.body = R"({"messages":[],"max_tokens":5,"max_completion_tokens":5})", .error = at(json::Conflict{.other = "max_tokens"}, {P{"max_completion_tokens"}})},
	    {.body = R"({"messages":[],"max_tokens":5.5})", .error = at(shape(Expected::Integer, Kind::Real), {P{"max_tokens"}})},
	    {.body = R"({"messages":[],"stream_options":{"include_usage":true}})", .error = at(json::Conflict{.other = "stream"}, {P{"stream_options"}})},
	    {.body = R"({"messages":[],"stream":true,"stream_options":{"include_obfuscation":true}})", .error = at(Violation::Unsupported, {P{"stream_options"}, P{"include_obfuscation"}})},
	    {.body = R"({"messages":[],"stream":"yes"})", .error = at(shape(Expected::Boolean, Kind::String), {P{"stream"}})},
	    {.body = R"({"messages":[],"temperature":"hot"})", .error = at(shape(Expected::Number, Kind::String), {P{"temperature"}})},
	    {.body = R"({"messages":[],"stop":["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q"]})", .error = at(json::TooMany{.limit = json::max_stop_sequences}, {P{"stop"}})},
	    {.body = R"({"messages":[],"stop":{}})", .error = at(shape(Expected::Stop, Kind::Object), {P{"stop"}})},
	    {.body = R"({"messages":[],"stop":["a",1]})", .error = at(shape(Expected::String, Kind::Integer), {P{"stop"}, P{std::size_t{1}}})},
	    {.body = R"({"messages":[],"chat_template_kwargs":{"reasoning_effort":"max"}})", .error = at(Violation::UnknownValue, {P{"chat_template_kwargs"}, P{"reasoning_effort"}})},
	    {.body = R"({"messages":[],"chat_template_kwargs":{"thinking_budget":100}})", .error = at(Violation::UnknownField, {P{"chat_template_kwargs"}, P{"thinking_budget"}})},
	    {.body = R"({"messages":[],"n":2})", .error = at(Violation::Unsupported, {P{"n"}})},
	    {.body = R"({"messages":[],"presence_penalty":0.5})", .error = at(Violation::Unsupported, {P{"presence_penalty"}})},
	    {.body = R"({"messages":[],"logprobs":true})", .error = at(Violation::Unsupported, {P{"logprobs"}})},
	    {.body = R"({"messages":[],"logit_bias":{"50256":-100}})", .error = at(Violation::Unsupported, {P{"logit_bias"}})},
	    {.body = R"({"messages":[],"response_format":{"type":"json_object"}})", .error = at(Violation::Unsupported, {P{"response_format"}})},
	    {.body = R"({"messages":[],"parallel_tool_calls":false})", .error = at(Violation::Unsupported, {P{"parallel_tool_calls"}})},
	    {.body = R"({"messages":[],"reasoning_effort":"high"})", .error = at(Violation::Unsupported, {P{"reasoning_effort"}})},
	    {.body = R"({"messages":[],"modalities":["text","audio"]})", .error = at(Violation::Unsupported, {P{"modalities"}})},
	    {.body = R"({"messages":[],"user":5})", .error = at(shape(Expected::String, Kind::Integer), {P{"user"}})},
	};
	auto decoder = make_decoder();
	return std::ranges::all_of(refusals, [&decoder](Refusal const& refusal) {
		auto const decoded = decoder->decode_copy(refusal.body);
		if (decoded || decoded.error() != refusal.error) {
			std::println("{} -> {}", refusal.body, decoded ? std::string{"accepted"} : json::describe(decoded.error()));
			return false;
		}
		return true;
	});
}

/** The shapes harnesses resend that the endpoint must take: null content, null tool calls, echoed nulls. */
[[nodiscard]] auto harness_resends_accepted() -> bool {
	auto decoder = make_decoder();
	auto const decoded = decoder->decode_copy(R"({
		"model": "qwen3.8",
		"messages": [
			{"role": "user", "content": "list files"},
			{"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{\"path\": \".\", \"depth\": 2}"}}], "refusal": null, "annotations": []},
			{"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
			{"role": "assistant", "content": null, "reasoning_content": "done", "tool_calls": null, "function_call": null, "audio": null}
		],
		"stream": true, "stream_options": {"include_usage": true},
		"temperature": 0, "max_tokens": 4096, "n": 1, "presence_penalty": 0, "frequency_penalty": 0.0,
		"chat_template_kwargs": {"enable_thinking": false}
	})");
	if (!decoded) {
		std::println("{}", json::describe(decoded.error()));
		return false;
	}
	auto const& messages = decoded->messages;
	auto const& call = std::get<json::AssistantMessage>(messages[1]);
	return messages.size() == 4 && call.content.empty() && call.tool_calls.size() == 1 && call.tool_calls[0].arguments.size() == 2 &&
	       std::get<std::string>(call.tool_calls[0].arguments[0].value) == "." && std::get<json::JsonText>(call.tool_calls[0].arguments[1].value).text() == "2" &&
	       std::get<json::AssistantMessage>(messages[3]).reasoning_content == "done" && decoded->delivery == json::Delivery::StreamWithUsage &&
	       decoded->temperature == 0.0 && decoded->max_tokens == 4096 && decoded->template_options.thinking == json::Thinking::Disabled;
}

/** Mutated requests never crash the decoder (the sanitizers are the oracle): each decodes or is refused. */
[[nodiscard]] auto mutated_requests_handled(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto decoder = make_decoder();
	auto scratch = make_scratch();
	auto refused = std::size_t{0};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto body = encode(random_request(rng, *scratch), rng);
		for (auto edits = 1 + rng() % 3; edits > 0 && !body.empty(); --edits) {
			auto const at = rng() % body.size();
			auto constexpr bytes = std::string_view{"{}[],:\"\\0tfn \x80"};
			if (coin(rng)) {
				body[at] = bytes[rng() % bytes.size()];
			} else {
				body.erase(at, 1);
			}
		}
		refused += decoder->decode_copy(body).has_value() ? 0 : 1;
	}
	return refused > 0;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "generated requests in any order and spelling decode to themselves (seed 1)", .holds = requests_round_trip(1, 3000)},
	    engine::testing::Case{.name = "generated requests in any order and spelling decode to themselves (seed 2)", .holds = requests_round_trip(2, 3000)},
	    engine::testing::Case{.name = "each refusal names its fault and the member at fault", .holds = refusals_name_the_member()},
	    engine::testing::Case{.name = "harness resends of tool-call and reasoning-only turns decode", .holds = harness_resends_accepted()},
	    engine::testing::Case{.name = "mutated requests decode or are refused, never crash (seed 3)", .holds = mutated_requests_handled(3, 5000)},
	};
	return engine::testing::report(cases);
}

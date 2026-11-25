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

template<class T>
[[nodiscard]] auto maybe(std::mt19937_64& rng, T value) -> std::optional<T> {
	return coin(rng) ? std::optional<T>{std::move(value)} : std::nullopt;
}

[[nodiscard]] auto id(std::mt19937_64& rng, std::string_view prefix) -> std::string {
	return std::format("{}{}", prefix, rng() % 100000);
}

[[nodiscard]] auto random_item(std::mt19937_64& rng, Scratch& scratch) -> json::InputItem {
	switch (rng() % 6) {
	case 0: {
		auto constexpr roles = std::array{json::Role::User, json::Role::Assistant, json::Role::System, json::Role::Developer};
		return json::MessageItem{.role = roles[rng() % roles.size()], .text = model::random_text(rng, 20)};
	}
	case 1:
		return json::FunctionCallItem{.call_id = id(rng, "call_"), .name = id(rng, "fn_"), .arguments = json_text(scratch, compact_object(rng, rng() % 4))};
	case 2:
		return json::FunctionOutputItem{.call_id = id(rng, "call_"), .output = model::random_text(rng, 20)};
	case 3:
		return json::CustomCallItem{.call_id = id(rng, "call_"), .name = id(rng, "custom_"), .input = model::random_text(rng, 20)};
	case 4:
		return json::CustomOutputItem{.call_id = id(rng, "call_"), .output = model::random_text(rng, 20)};
	default:
		return json::ReasoningItem{.text = model::random_text(rng, 20)};
	}
}

[[nodiscard]] auto random_tool(std::mt19937_64& rng, Scratch& scratch) -> json::Tool {
	if (coin(rng)) {
		return json::Tool{
		    .name = id(rng, "fn_"),
		    .description = maybe(rng, model::random_text(rng, 10)),
		    .input = json::FunctionTool{.parameters = coin(rng) ? std::optional{json_text(scratch, compact_object(rng, rng() % 4))} : std::nullopt},
		};
	}
	auto grammar = std::optional<json::Grammar>{};
	if (coin(rng)) {
		grammar = json::Grammar{.syntax = coin(rng) ? json::GrammarSyntax::Lark : json::GrammarSyntax::Regex, .definition = model::random_text(rng, 30)};
	}
	return json::Tool{.name = id(rng, "custom_"), .description = maybe(rng, model::random_text(rng, 10)), .input = json::CustomTool{.grammar = grammar}};
}

[[nodiscard]] auto random_choice(std::mt19937_64& rng) -> json::ToolChoice {
	switch (rng() % 5) {
	case 0:
		return json::ToolChoiceMode::None;
	case 1:
		return json::ToolChoiceMode::Auto;
	case 2:
		return json::ToolChoiceMode::Required;
	case 3:
		return json::ForcedFunction{.name = id(rng, "fn_")};
	default:
		return json::ForcedCustom{.name = id(rng, "custom_")};
	}
}

[[nodiscard]] auto random_format(std::mt19937_64& rng, Scratch& scratch) -> json::TextFormat {
	switch (rng() % 3) {
	case 0:
		return json::TextFormatMode::Text;
	case 1:
		return json::TextFormatMode::JsonObject;
	default:
		return json::JsonSchemaFormat{
		    .name = id(rng, "schema_"),
		    .description = maybe(rng, model::random_text(rng, 10)),
		    .schema = json_text(scratch, compact_object(rng, rng() % 4)),
		};
	}
}

[[nodiscard]] auto random_request(std::mt19937_64& rng, Scratch& scratch) -> json::CreateRequest {
	auto constexpr efforts = std::array{json::Effort::None, json::Effort::Low, json::Effort::Medium, json::Effort::XHigh};
	auto request = json::CreateRequest{
	    .model = maybe(rng, model::random_text(rng, 8)),
	    .input = {},
	    .instructions = maybe(rng, model::random_text(rng, 20)),
	    .tools = {},
	    .tool_choice = random_choice(rng),
	    .format = random_format(rng, scratch),
	    .effort = efforts[rng() % efforts.size()],
	    .max_output_tokens = maybe(rng, static_cast<std::int64_t>(rng() % 300000) - 10),
	    .temperature = maybe(rng, model::random_finite(rng)),
	    .top_p = maybe(rng, static_cast<double>(rng() % 1000) / 1000.0),
	    .delivery = coin(rng) ? json::Delivery::Whole : json::Delivery::Stream,
	    .storage = coin(rng) ? json::Storage::Stored : json::Storage::Unstored,
	    .previous_response_id = maybe(rng, id(rng, "resp_")),
	    .metadata = {},
	};
	for (auto count = rng() % 7; count > 0; --count) {
		request.input.push_back(random_item(rng, scratch));
	}
	for (auto count = rng() % 3; count > 0; --count) {
		request.tools.push_back(random_tool(rng, scratch));
	}
	for (auto count = rng() % (json::max_metadata_pairs + 1); count > 0; --count) {
		request.metadata.push_back(json::MetadataPair{.key = std::format("k{}", count), .value = model::random_text(rng, 30)});
	}
	return request;
}

/** Writes text content as a string or as parts split at character boundaries, the parts' type chosen among `types`. */
auto write_content(Members& item, std::string_view member, std::string_view content, std::span<std::string_view const> types, std::mt19937_64& rng) -> void {
	if (coin(rng)) {
		item.string(member, content);
		return;
	}
	item.array(member, [&](Elements& parts) {
		auto at = std::size_t{0};
		while (at < content.size()) {
			auto end = std::min<std::size_t>(content.size(), at + 1 + rng() % 12);
			while (end < content.size() && (std::bit_cast<std::uint8_t>(content[end]) & 0xC0U) == 0x80U) {
				++end;
			}
			auto const type = types[rng() % types.size()];
			parts.object([&](Members& part) {
				if (coin(rng)) {
					part.string("type", type);
					part.string("text", content.substr(at, end - at));
				} else {
					part.string("text", content.substr(at, end - at));
					part.string("type", type);
				}
				if (type == "input_text" && coin(rng)) {
					part.null("prompt_cache_breakpoint");
				}
				if (type == "output_text" && coin(rng)) {
					part.array("annotations", [](Elements&) {});
					part.array("logprobs", [](Elements&) {});
				}
			});
			at = end;
		}
	});
}

constexpr auto input_parts = std::to_array<std::string_view>({"input_text"});
constexpr auto assistant_parts = std::to_array<std::string_view>({"input_text", "output_text"});
constexpr auto reasoning_parts = std::to_array<std::string_view>({"reasoning_text"});

[[nodiscard]] auto role_text(json::Role role) -> std::string_view {
	switch (role) {
	case json::Role::User:
		return "user";
	case json::Role::Assistant:
		return "assistant";
	case json::Role::System:
		return "system";
	case json::Role::Developer:
		return "developer";
	}
	std::unreachable();
}

/** The members a resent item may carry at their neutral values. */
auto write_item_extras(Members& item, std::mt19937_64& rng, bool status, bool caller) -> void {
	if (coin(rng)) {
		item.string("id", id(rng, "item_"));
	}
	if (status && coin(rng)) {
		item.string("status", "completed");
	}
	if (caller && coin(rng)) {
		item.object("caller", [](Members& written) { written.string("type", "direct"); });
	}
}

auto write_item(Elements& items, json::InputItem const& item, std::mt19937_64& rng) -> void {
	items.object([&](Members& written) {
		if (auto const message = engine::core::alternative<json::MessageItem>(item)) {
			if (coin(rng)) {
				written.string("type", "message");
			}
			written.string("role", role_text(message->role));
			write_content(written, "content", message->text, message->role == json::Role::Assistant ? std::span<std::string_view const>{assistant_parts} : std::span<std::string_view const>{input_parts}, rng);
			write_item_extras(written, rng, true, false);
			if (coin(rng)) {
				written.null("phase");
			}
		} else if (auto const call = engine::core::alternative<json::FunctionCallItem>(item)) {
			written.string("type", "function_call");
			written.string("call_id", call->call_id);
			written.string("arguments", call->arguments.text());
			written.string("name", call->name);
			write_item_extras(written, rng, true, true);
			if (coin(rng)) {
				written.null("namespace");
			}
		} else if (auto const output = engine::core::alternative<json::FunctionOutputItem>(item)) {
			written.string("call_id", output->call_id);
			written.string("type", "function_call_output");
			write_content(written, "output", output->output, input_parts, rng);
			write_item_extras(written, rng, true, true);
		} else if (auto const custom = engine::core::alternative<json::CustomCallItem>(item)) {
			written.string("input", custom->input);
			written.string("type", "custom_tool_call");
			written.string("name", custom->name);
			written.string("call_id", custom->call_id);
			write_item_extras(written, rng, false, true);
		} else if (auto const custom_output = engine::core::alternative<json::CustomOutputItem>(item)) {
			written.string("type", "custom_tool_call_output");
			written.string("call_id", custom_output->call_id);
			write_content(written, "output", custom_output->output, input_parts, rng);
			write_item_extras(written, rng, false, true);
		} else {
			auto const& reasoning = std::get<json::ReasoningItem>(item);
			written.string("type", "reasoning");
			if (coin(rng)) {
				written.array("summary", [](Elements&) {});
			}
			if (!reasoning.text.empty() || coin(rng)) {
				written.array("content", [&](Elements& parts) {
					auto const text = std::string_view{reasoning.text};
					auto at = std::size_t{0};
					while (at < text.size()) {
						auto end = std::min<std::size_t>(text.size(), at + 1 + rng() % 12);
						while (end < text.size() && (std::bit_cast<std::uint8_t>(text[end]) & 0xC0U) == 0x80U) {
							++end;
						}
						parts.object([&](Members& part) {
							part.string("type", reasoning_parts[0]);
							part.string("text", text.substr(at, end - at));
						});
						at = end;
					}
				});
			}
			if (coin(rng)) {
				written.null("encrypted_content");
			}
			write_item_extras(written, rng, true, false);
		}
	});
}

auto write_tool(Elements& tools, json::Tool const& tool, std::mt19937_64& rng) -> void {
	tools.object([&](Members& written) {
		auto const describe = [&] {
			if (tool.description) {
				written.string("description", *tool.description);
			} else if (coin(rng)) {
				written.null("description");
			}
		};
		if (auto const function = engine::core::alternative<json::FunctionTool>(tool.input)) {
			written.string("name", tool.name);
			written.string("type", "function");
			if (function->parameters) {
				written.json("parameters", *function->parameters);
			} else {
				written.null("parameters");
			}
			describe();
			auto const strict = rng() % 3;
			if (strict == 0) {
				written.null("strict");
			} else {
				written.boolean("strict", strict == 1);
			}
			if (coin(rng)) {
				written.boolean("defer_loading", false);
				written.array("allowed_callers", [](Elements& callers) { callers.string("direct"); });
				written.null("output_schema");
			}
			return;
		}
		auto const& custom = std::get<json::CustomTool>(tool.input);
		written.string("type", "custom");
		describe();
		written.string("name", tool.name);
		if (custom.grammar) {
			written.object("format", [&](Members& format) {
				auto const syntax = custom.grammar->syntax == json::GrammarSyntax::Lark ? "lark" : "regex";
				if (coin(rng)) {
					format.string("definition", custom.grammar->definition);
					format.string("type", "grammar");
					format.string("syntax", syntax);
				} else {
					format.string("type", "grammar");
					format.string("syntax", syntax);
					format.string("definition", custom.grammar->definition);
				}
			});
		} else if (auto const spelling = rng() % 3; spelling == 0) {
			written.object("format", [](Members& format) { format.string("type", "text"); });
		} else if (spelling == 1) {
			written.null("format");
		}
	});
}

auto write_choice(Members& written, json::ToolChoice const& choice, std::mt19937_64& rng) -> void {
	if (auto const mode = engine::core::alternative<json::ToolChoiceMode>(choice)) {
		if (*mode == json::ToolChoiceMode::Auto && coin(rng)) {
			if (coin(rng)) {
				written.null("tool_choice");
			}
			return;
		}
		auto constexpr modes = std::array<std::string_view, 3>{"none", "auto", "required"};
		written.string("tool_choice", modes[std::to_underlying(*mode)]);
		return;
	}
	written.object("tool_choice", [&](Members& forced) {
		if (auto const function = engine::core::alternative<json::ForcedFunction>(choice)) {
			forced.string("name", function->name);
			forced.string("type", "function");
		} else {
			forced.string("type", "custom");
			forced.string("name", std::get<json::ForcedCustom>(choice).name);
		}
	});
}

auto write_text(Members& written, json::TextFormat const& format, std::mt19937_64& rng) -> void {
	if (auto const mode = engine::core::alternative<json::TextFormatMode>(format); mode && *mode == json::TextFormatMode::Text) {
		switch (rng() % 5) {
		case 0:
			return;
		case 1:
			written.null("text");
			return;
		case 2:
			written.object("text", [](Members&) {});
			return;
		case 3:
			written.object("text", [](Members& text) { text.null("format"); });
			return;
		default:
			written.object("text", [&](Members& text) {
				text.object("format", [](Members& written_format) { written_format.string("type", "text"); });
				if (coin(rng)) {
					text.string("verbosity", "medium");
				}
			});
			return;
		}
	}
	written.object("text", [&](Members& text) {
		if (coin(rng)) {
			text.null("verbosity");
		}
		text.object("format", [&](Members& written_format) {
			if (engine::core::alternative<json::TextFormatMode>(format)) {
				written_format.string("type", "json_object");
				return;
			}
			auto const& schema = std::get<json::JsonSchemaFormat>(format);
			written_format.json("schema", schema.schema);
			written_format.string("type", "json_schema");
			written_format.string("name", schema.name);
			if (schema.description) {
				written_format.string("description", *schema.description);
			} else if (coin(rng)) {
				written_format.null("description");
			}
			if (coin(rng)) {
				written_format.boolean("strict", coin(rng));
			}
		});
	});
}

auto write_reasoning(Members& written, json::Effort effort, std::mt19937_64& rng) -> void {
	auto constexpr efforts = std::array<std::string_view, 4>{"none", "low", "medium", "xhigh"};
	if (effort == json::Effort::XHigh) {
		switch (rng() % 4) {
		case 0:
			return;
		case 1:
			written.null("reasoning");
			return;
		case 2:
			written.object("reasoning", [](Members& reasoning) { reasoning.null("effort"); });
			return;
		default:
			break;
		}
	}
	written.object("reasoning", [&](Members& reasoning) {
		if (coin(rng)) {
			reasoning.null("summary");
			reasoning.string("context", coin(rng) ? "auto" : "all_turns");
			reasoning.string("mode", "standard");
		}
		reasoning.string("effort", efforts[std::to_underlying(effort)]);
	});
}

enum class Field : std::uint8_t {
	Input,
	Model,
	Instructions,
	Tools,
	ToolChoice,
	Text,
	Reasoning,
	Budget,
	Sampling,
	Delivery,
	Storage,
	Previous,
	Metadata,
	Neutral,
};

/** Writes a request in a random order of members and a random choice of the spellings each member allows. */
[[nodiscard]] auto encode(json::CreateRequest const& request, std::mt19937_64& rng) -> std::string {
	auto fields = std::vector<Field>{Field::Input,  Field::Model,  Field::Instructions, Field::Tools,    Field::ToolChoice, Field::Text,     Field::Reasoning,
	                                 Field::Budget, Field::Sampling, Field::Delivery,   Field::Storage, Field::Previous,   Field::Metadata, Field::Neutral};
	std::ranges::shuffle(fields, rng);
	auto out = Text{};
	json::write_object(out, [&](Members& written) {
		for (auto const field : fields) {
			switch (field) {
			case Field::Input: {
				auto const one_user = request.input.size() == 1 && engine::core::alternative<json::MessageItem>(request.input.front()) &&
				                      std::get<json::MessageItem>(request.input.front()).role == json::Role::User;
				if (one_user && coin(rng)) {
					written.string("input", std::get<json::MessageItem>(request.input.front()).text);
				} else {
					written.array("input", [&](Elements& items) {
						for (auto const& item : request.input) {
							write_item(items, item, rng);
						}
					});
				}
				break;
			}
			case Field::Model:
				if (request.model) {
					written.string("model", *request.model);
				}
				break;
			case Field::Instructions:
				if (request.instructions) {
					written.string("instructions", *request.instructions);
				} else if (coin(rng)) {
					written.null("instructions");
				}
				break;
			case Field::Tools:
				if (!request.tools.empty() || coin(rng)) {
					written.array("tools", [&](Elements& tools) {
						for (auto const& tool : request.tools) {
							write_tool(tools, tool, rng);
						}
					});
				}
				break;
			case Field::ToolChoice:
				write_choice(written, request.tool_choice, rng);
				break;
			case Field::Text:
				write_text(written, request.format, rng);
				break;
			case Field::Reasoning:
				write_reasoning(written, request.effort, rng);
				break;
			case Field::Budget:
				if (request.max_output_tokens) {
					written.integer("max_output_tokens", *request.max_output_tokens);
				} else if (coin(rng)) {
					written.null("max_output_tokens");
				}
				break;
			case Field::Sampling:
				for (auto const& [name, value] : {std::pair{"temperature", request.temperature}, std::pair{"top_p", request.top_p}}) {
					if (value) {
						written.real(name, *value);
					} else if (coin(rng)) {
						written.null(name);
					}
				}
				break;
			case Field::Delivery:
				if (request.delivery == json::Delivery::Stream) {
					written.boolean("stream", true);
					if (coin(rng)) {
						written.object("stream_options", [&](Members& options) {
							if (coin(rng)) {
								options.boolean("include_obfuscation", false);
							}
						});
					}
				} else if (auto const spelling = rng() % 3; spelling == 0) {
					written.boolean("stream", false);
				} else if (spelling == 1) {
					written.null("stream");
				}
				break;
			case Field::Storage:
				if (request.storage == json::Storage::Unstored) {
					written.boolean("store", false);
				} else if (auto const spelling = rng() % 3; spelling == 0) {
					written.boolean("store", true);
				} else if (spelling == 1) {
					written.null("store");
				}
				break;
			case Field::Previous:
				if (request.previous_response_id) {
					written.string("previous_response_id", *request.previous_response_id);
				} else if (coin(rng)) {
					written.null("previous_response_id");
				}
				break;
			case Field::Metadata:
				if (!request.metadata.empty() || coin(rng)) {
					written.object("metadata", [&](Members& pairs) {
						for (auto const& pair : request.metadata) {
							pairs.string(pair.key, pair.value);
						}
					});
				} else if (coin(rng)) {
					written.null("metadata");
				}
				break;
			case Field::Neutral:
				if (coin(rng)) {
					written.boolean("background", false);
					written.boolean("parallel_tool_calls", true);
					written.string("truncation", "disabled");
					written.string("service_tier", coin(rng) ? "auto" : "default");
					written.array("include", [](Elements&) {});
					written.integer("top_logprobs", 0);
					written.null("max_tool_calls");
					written.null("safety_identifier");
					written.null("user");
					written.null("prompt_cache_key");
					written.null("conversation");
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
	std::string body;
	json::DecodeError error;
};

[[nodiscard]] auto at(json::DecodeFault fault, std::initializer_list<json::PathStep> path) -> json::DecodeError {
	return json::DecodeError{.fault = std::move(fault), .path = path};
}

[[nodiscard]] auto shape(json::Expected expected, json::Kind found) -> json::DecodeFault {
	return json::WrongShape{.expected = expected, .found = found};
}

/** A valid request with `members` spliced in after its input. */
[[nodiscard]] auto with(std::string_view members) -> std::string {
	return std::format(R"({{"input":"hi",{}}})", members);
}

/** A request whose input is the one item `item`. */
[[nodiscard]] auto item(std::string_view item) -> std::string {
	return std::format(R"({{"input":[{}]}})", item);
}

/** A request whose tools are the one tool `tool`. */
[[nodiscard]] auto tool(std::string_view tool) -> std::string {
	return std::format(R"({{"input":"hi","tools":[{}]}})", tool);
}

/** Requests refused with the documented fault at the documented member, one per rule the decoder states. */
[[nodiscard]] auto refusals_name_the_member() -> bool {
	using json::Kind;
	using json::Expected;
	using P = json::PathStep;
	auto const parse = [](json::ParseFault fault) { return json::DecodeFault{json::ParseError{.fault = fault}}; };
	auto const zero = P{std::size_t{0}};
	auto many_pairs = std::string{};
	for (auto index = std::size_t{0}; index <= json::max_metadata_pairs; ++index) {
		many_pairs += std::format(R"({}"k{}":"v")", index == 0 ? "" : ",", index);
	}
	auto const long_key = std::string(json::max_metadata_key_bytes + 1, 'k');
	auto const long_value = std::string(json::max_metadata_value_bytes + 1, 'v');
	auto const refusals = std::vector<Refusal>{
	    {.body = R"([])", .error = at(shape(Expected::Object, Kind::Array), {})},
	    {.body = R"({"input":"x"} x)", .error = at(parse(json::ParseFault::TrailingContent), {})},
	    {.body = R"({"input":[)", .error = at(parse(json::ParseFault::Truncated), {})},
	    {.body = R"({})", .error = at(Violation::MissingField, {P{"input"}})},
	    {.body = R"({"input":"x","input":"y"})", .error = at(parse(json::ParseFault::DuplicateKey), {})},
	    {.body = R"({"input":3})", .error = at(shape(Expected::Input, Kind::Integer), {P{"input"}})},
	    {.body = with(R"("frobnicate":1)"), .error = at(Violation::UnknownField, {P{"frobnicate"}})},
	    {.body = with(R"("messages":[])"), .error = at(Violation::UnknownField, {P{"messages"}})},
	    {.body = with(R"("seed":1)"), .error = at(Violation::UnknownField, {P{"seed"}})},
	    {.body = with(R"("model":null)"), .error = at(shape(Expected::String, Kind::Null), {P{"model"}})},
	    {.body = with(R"("user":"u")"), .error = at(Violation::Deprecated, {P{"user"}})},
	    {.body = with(R"("prompt_cache_retention":"24h")"), .error = at(Violation::Deprecated, {P{"prompt_cache_retention"}})},
	    {.body = with(R"("reasoning":{"generate_summary":"auto"})"), .error = at(Violation::Deprecated, {P{"reasoning"}, P{"generate_summary"}})},
	    {.body = with(R"("reasoning":{"effort":"high"})"), .error = at(Violation::Unsupported, {P{"reasoning"}, P{"effort"}})},
	    {.body = with(R"("reasoning":{"effort":"minimal"})"), .error = at(Violation::Unsupported, {P{"reasoning"}, P{"effort"}})},
	    {.body = with(R"("reasoning":{"effort":"max"})"), .error = at(Violation::Unsupported, {P{"reasoning"}, P{"effort"}})},
	    {.body = with(R"("reasoning":{"effort":"turbo"})"), .error = at(Violation::UnknownValue, {P{"reasoning"}, P{"effort"}})},
	    {.body = with(R"("reasoning":{"summary":"auto"})"), .error = at(Violation::Unsupported, {P{"reasoning"}, P{"summary"}})},
	    {.body = with(R"("reasoning":{"context":"current_turn"})"), .error = at(Violation::Unsupported, {P{"reasoning"}, P{"context"}})},
	    {.body = with(R"("reasoning":{"budget":5})"), .error = at(Violation::UnknownField, {P{"reasoning"}, P{"budget"}})},
	    {.body = with(R"("background":true)"), .error = at(Violation::Unsupported, {P{"background"}})},
	    {.body = with(R"("conversation":"conv_1")"), .error = at(Violation::Unsupported, {P{"conversation"}})},
	    {.body = with(R"("prompt":{"id":"p"})"), .error = at(Violation::Unsupported, {P{"prompt"}})},
	    {.body = with(R"("include":["reasoning.encrypted_content"])"), .error = at(Violation::Unsupported, {P{"include"}})},
	    {.body = with(R"("truncation":"auto")"), .error = at(Violation::Unsupported, {P{"truncation"}})},
	    {.body = with(R"("service_tier":"flex")"), .error = at(Violation::Unsupported, {P{"service_tier"}})},
	    {.body = with(R"("parallel_tool_calls":false)"), .error = at(Violation::Unsupported, {P{"parallel_tool_calls"}})},
	    {.body = with(R"("max_tool_calls":3)"), .error = at(Violation::Unsupported, {P{"max_tool_calls"}})},
	    {.body = with(R"("prompt_cache_key":"k")"), .error = at(Violation::Unsupported, {P{"prompt_cache_key"}})},
	    {.body = with(R"("top_logprobs":2)"), .error = at(Violation::Unsupported, {P{"top_logprobs"}})},
	    {.body = with(R"("max_output_tokens":5.5)"), .error = at(shape(Expected::Integer, Kind::Real), {P{"max_output_tokens"}})},
	    {.body = with(R"("temperature":"hot")"), .error = at(shape(Expected::Number, Kind::String), {P{"temperature"}})},
	    {.body = with(R"("stream":"yes")"), .error = at(shape(Expected::Boolean, Kind::String), {P{"stream"}})},
	    {.body = with(R"("stream_options":{})"), .error = at(json::Conflict{.other = "stream"}, {P{"stream_options"}})},
	    {.body = with(R"("stream":true,"stream_options":{"include_obfuscation":true})"), .error = at(Violation::Unsupported, {P{"stream_options"}, P{"include_obfuscation"}})},
	    {.body = with(R"("stream":true,"stream_options":{"include_usage":true})"), .error = at(Violation::UnknownField, {P{"stream_options"}, P{"include_usage"}})},
	    {.body = with(std::format(R"("metadata":{{{}}})", many_pairs)), .error = at(json::TooMany{.limit = json::max_metadata_pairs}, {P{"metadata"}})},
	    {.body = with(std::format(R"("metadata":{{"{}":"v"}})", long_key)), .error = at(json::TooLong{.limit = json::max_metadata_key_bytes}, {P{"metadata"}, P{long_key}})},
	    {.body = with(std::format(R"("metadata":{{"k":"{}"}})", long_value)), .error = at(json::TooLong{.limit = json::max_metadata_value_bytes}, {P{"metadata"}, P{"k"}})},
	    {.body = with(R"("metadata":{"k":1})"), .error = at(shape(Expected::String, Kind::Integer), {P{"metadata"}, P{"k"}})},
	    {.body = item(R"({"type":"web_search_call","id":"w"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"type"}})},
	    {.body = item(R"({"type":"item_reference","id":"r"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"type"}})},
	    {.body = item(R"({"type":"wat"})"), .error = at(Violation::UnknownValue, {P{"input"}, zero, P{"type"}})},
	    {.body = item(R"({"content":"x"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"type"}})},
	    {.body = item(R"({"type":"message","content":"x"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"role"}})},
	    {.body = item(R"({"role":"robot","content":"x"})"), .error = at(Violation::UnknownValue, {P{"input"}, zero, P{"role"}})},
	    {.body = item(R"({"role":"user"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"content"}})},
	    {.body = item(R"({"role":"user","content":7})"), .error = at(shape(Expected::Content, Kind::Integer), {P{"input"}, zero, P{"content"}})},
	    {.body = item(R"({"role":"user","content":"x","tool_call_id":"c"})"), .error = at(Violation::UnknownField, {P{"input"}, zero, P{"tool_call_id"}})},
	    {.body = item(R"({"role":"user","content":"x","status":"in_progress"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"status"}})},
	    {.body = item(R"({"role":"user","content":"x","phase":"final_answer"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"phase"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"input_text","text":"a"},{"type":"input_image","image_url":"u"}]})"),
	     .error = at(Violation::Unsupported, {P{"input"}, zero, P{"content"}, P{std::size_t{1}}, P{"type"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"input_file","file_id":"f"}]})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"content"}, zero, P{"type"}})},
	    {.body = item(R"({"role":"assistant","content":[{"type":"refusal","refusal":"no"}]})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"content"}, zero, P{"type"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"output_text","text":"a"}]})"), .error = at(Violation::UnknownValue, {P{"input"}, zero, P{"content"}, zero, P{"type"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"sticker"}]})"), .error = at(Violation::UnknownValue, {P{"input"}, zero, P{"content"}, zero, P{"type"}})},
	    {.body = item(R"({"role":"user","content":[{"text":"a"}]})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"content"}, zero, P{"type"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"input_text"}]})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"content"}, zero, P{"text"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"input_text","text":"a","cache_control":{}}]})"),
	     .error = at(Violation::UnknownField, {P{"input"}, zero, P{"content"}, zero, P{"cache_control"}})},
	    {.body = item(R"({"role":"user","content":[{"type":"input_text","text":"a","prompt_cache_breakpoint":{"mode":"explicit"}}]})"),
	     .error = at(Violation::Unsupported, {P{"input"}, zero, P{"content"}, zero, P{"prompt_cache_breakpoint"}})},
	    {.body = item(R"({"role":"assistant","content":[{"type":"output_text","text":"a","annotations":[{"type":"url_citation"}]}]})"),
	     .error = at(Violation::Unsupported, {P{"input"}, zero, P{"content"}, zero, P{"annotations"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":"[1]"})"), .error = at(shape(Expected::Object, Kind::Array), {P{"input"}, zero, P{"arguments"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":"{\"a\":"})"), .error = at(parse(json::ParseFault::Truncated), {P{"input"}, zero, P{"arguments"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":"{\"a\":1,\"a\":2}"})"),
	     .error = at(parse(json::ParseFault::DuplicateKey), {P{"input"}, zero, P{"arguments"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":{}})"), .error = at(shape(Expected::String, Kind::Object), {P{"input"}, zero, P{"arguments"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"arguments"}})},
	    {.body = item(R"({"type":"function_call","name":"f","arguments":"{}"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"call_id"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":"{}","caller":{"type":"program","caller_id":"p"}})"),
	     .error = at(Violation::Unsupported, {P{"input"}, zero, P{"caller"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":"{}","namespace":"n"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"namespace"}})},
	    {.body = item(R"({"type":"function_call","call_id":"c","name":"f","arguments":"{}","output":"o"})"), .error = at(Violation::UnknownField, {P{"input"}, zero, P{"output"}})},
	    {.body = item(R"({"type":"function_call_output","output":"x"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"call_id"}})},
	    {.body = item(R"({"type":"function_call_output","call_id":"c"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"output"}})},
	    {.body = item(R"({"type":"function_call_output","call_id":"c","output":"o","name":"f"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"name"}})},
	    {.body = item(R"({"type":"function_call_output","call_id":"c","output":[{"type":"input_image","image_url":"u"}]})"),
	     .error = at(Violation::Unsupported, {P{"input"}, zero, P{"output"}, zero, P{"type"}})},
	    {.body = item(R"({"type":"custom_tool_call","call_id":"c","name":"t"})"), .error = at(Violation::MissingField, {P{"input"}, zero, P{"input"}})},
	    {.body = item(R"({"type":"custom_tool_call","call_id":"c","name":"t","input":"i","status":"completed"})"), .error = at(Violation::UnknownField, {P{"input"}, zero, P{"status"}})},
	    {.body = item(R"({"type":"custom_tool_call_output","call_id":"c","output":"o","name":"t"})"), .error = at(Violation::UnknownField, {P{"input"}, zero, P{"name"}})},
	    {.body = item(R"({"type":"reasoning","summary":[{"type":"summary_text","text":"s"}]})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"summary"}})},
	    {.body = item(R"({"type":"reasoning","encrypted_content":"e"})"), .error = at(Violation::Unsupported, {P{"input"}, zero, P{"encrypted_content"}})},
	    {.body = item(R"({"type":"reasoning","content":"x"})"), .error = at(shape(Expected::Array, Kind::String), {P{"input"}, zero, P{"content"}})},
	    {.body = item(R"({"type":"reasoning","content":[{"type":"input_text","text":"x"}]})"), .error = at(Violation::UnknownValue, {P{"input"}, zero, P{"content"}, zero, P{"type"}})},
	    {.body = item(R"({"type":"reasoning","role":"assistant"})"), .error = at(Violation::UnknownField, {P{"input"}, zero, P{"role"}})},
	    {.body = tool(R"({"type":"web_search"})"), .error = at(Violation::Unsupported, {P{"tools"}, zero, P{"type"}})},
	    {.body = tool(R"({"type":"mcp","server_label":"s"})"), .error = at(Violation::Unsupported, {P{"tools"}, zero, P{"type"}})},
	    {.body = tool(R"({"type":"magic"})"), .error = at(Violation::UnknownValue, {P{"tools"}, zero, P{"type"}})},
	    {.body = tool(R"({"name":"f"})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"type"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"strict"}})},
	    {.body = tool(R"({"type":"function","name":"f","strict":true})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"parameters"}})},
	    {.body = tool(R"({"type":"function","parameters":null,"strict":true})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"name"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":[],"strict":true})"), .error = at(shape(Expected::Object, Kind::Array), {P{"tools"}, zero, P{"parameters"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null,"strict":"yes"})"), .error = at(shape(Expected::Boolean, Kind::String), {P{"tools"}, zero, P{"strict"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null,"strict":true,"defer_loading":true})"), .error = at(Violation::Unsupported, {P{"tools"}, zero, P{"defer_loading"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null,"strict":true,"allowed_callers":["programmatic"]})"),
	     .error = at(Violation::Unsupported, {P{"tools"}, zero, P{"allowed_callers"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null,"strict":true,"output_schema":{}})"), .error = at(Violation::Unsupported, {P{"tools"}, zero, P{"output_schema"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null,"strict":true,"function":{}})"), .error = at(Violation::UnknownField, {P{"tools"}, zero, P{"function"}})},
	    {.body = tool(R"({"type":"function","name":"f","parameters":null,"strict":true,"format":null})"), .error = at(Violation::UnknownField, {P{"tools"}, zero, P{"format"}})},
	    {.body = tool(R"({"type":"custom","name":"c","parameters":{}})"), .error = at(Violation::UnknownField, {P{"tools"}, zero, P{"parameters"}})},
	    {.body = tool(R"({"type":"custom"})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"name"}})},
	    {.body = tool(R"({"type":"custom","name":"c","format":{"type":"grammar","syntax":"ebnf","definition":"x"}})"),
	     .error = at(Violation::UnknownValue, {P{"tools"}, zero, P{"format"}, P{"syntax"}})},
	    {.body = tool(R"({"type":"custom","name":"c","format":{"type":"grammar","syntax":"lark"}})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"format"}, P{"definition"}})},
	    {.body = tool(R"({"type":"custom","name":"c","format":{"type":"grammar","definition":"x"}})"), .error = at(Violation::MissingField, {P{"tools"}, zero, P{"format"}, P{"syntax"}})},
	    {.body = tool(R"({"type":"custom","name":"c","format":{"type":"text","definition":"x"}})"), .error = at(Violation::UnknownField, {P{"tools"}, zero, P{"format"}, P{"definition"}})},
	    {.body = tool(R"({"type":"custom","name":"c","format":{"type":"grammar","grammar":{"syntax":"lark","definition":"x"}}})"),
	     .error = at(Violation::UnknownField, {P{"tools"}, zero, P{"format"}, P{"grammar"}})},
	    {.body = tool(R"({"type":"custom","name":"c","format":{"type":"cfg"}})"), .error = at(Violation::UnknownValue, {P{"tools"}, zero, P{"format"}, P{"type"}})},
	    {.body = with(R"("tool_choice":"maybe")"), .error = at(Violation::UnknownValue, {P{"tool_choice"}})},
	    {.body = with(R"("tool_choice":3)"), .error = at(shape(Expected::ToolChoice, Kind::Integer), {P{"tool_choice"}})},
	    {.body = with(R"("tool_choice":{"type":"allowed_tools","mode":"auto","tools":[]})"), .error = at(Violation::Unsupported, {P{"tool_choice"}, P{"type"}})},
	    {.body = with(R"("tool_choice":{"type":"web_search_preview"})"), .error = at(Violation::Unsupported, {P{"tool_choice"}, P{"type"}})},
	    {.body = with(R"("tool_choice":{"type":"function"})"), .error = at(Violation::MissingField, {P{"tool_choice"}, P{"name"}})},
	    {.body = with(R"("tool_choice":{"type":"function","function":{"name":"f"}})"), .error = at(Violation::UnknownField, {P{"tool_choice"}, P{"function"}})},
	    {.body = with(R"("text":{"format":{"type":"xml"}})"), .error = at(Violation::UnknownValue, {P{"text"}, P{"format"}, P{"type"}})},
	    {.body = with(R"("text":{"format":{"type":"json_schema","name":"n"}})"), .error = at(Violation::MissingField, {P{"text"}, P{"format"}, P{"schema"}})},
	    {.body = with(R"("text":{"format":{"type":"json_schema","schema":{}}})"), .error = at(Violation::MissingField, {P{"text"}, P{"format"}, P{"name"}})},
	    {.body = with(R"("text":{"format":{"type":"json_schema","name":"n","schema":[]}})"),
	     .error = at(shape(Expected::Object, Kind::Array), {P{"text"}, P{"format"}, P{"schema"}})},
	    {.body = with(R"("text":{"format":{"type":"json_object","name":"n"}})"), .error = at(Violation::UnknownField, {P{"text"}, P{"format"}, P{"name"}})},
	    {.body = with(R"("text":{"verbosity":"high"})"), .error = at(Violation::Unsupported, {P{"text"}, P{"verbosity"}})},
	    {.body = with(R"("text":{"style":"terse"})"), .error = at(Violation::UnknownField, {P{"text"}, P{"style"}})},
	    {.body = with(R"("response_format":{"type":"text"})"), .error = at(Violation::UnknownField, {P{"response_format"}})},
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

/** The defaults an absent member takes: auto tool choice, free text, `xhigh` thinking, stored, whole. */
[[nodiscard]] auto absent_members_take_the_defaults() -> bool {
	auto decoder = make_decoder();
	auto const decoded = decoder->decode_copy(R"({"input":"hello"})");
	return decoded && decoded->input == std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = "hello"}} &&
	       decoded->tool_choice == json::ToolChoice{json::ToolChoiceMode::Auto} && decoded->format == json::TextFormat{json::TextFormatMode::Text} &&
	       decoded->effort == json::Effort::XHigh && decoded->storage == json::Storage::Stored && decoded->delivery == json::Delivery::Whole &&
	       !decoded->model && !decoded->instructions && decoded->tools.empty() && decoded->metadata.empty() && !decoded->max_output_tokens &&
	       !decoded->temperature && !decoded->top_p && !decoded->previous_response_id;
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
			auto const position = rng() % body.size();
			auto constexpr bytes = std::string_view{"{}[],:\"\\0tfn \x80"};
			if (coin(rng)) {
				body[position] = bytes[rng() % bytes.size()];
			} else {
				body.erase(position, 1);
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
	    engine::testing::Case{.name = "absent members take the sdk's defaults", .holds = absent_members_take_the_defaults()},
	    engine::testing::Case{.name = "mutated requests decode or are refused, never crash (seed 3)", .holds = mutated_requests_handled(3, 5000)},
	};
	return engine::testing::report(cases);
}

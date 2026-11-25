// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.tokenizer;
import engine.chat;
import engine.testing;
import engine_test_chat.fixtures;

namespace {

namespace chat = engine::chat;
namespace json = engine::json;
namespace tok = engine::tokenizer;
using engine_test_chat::RenderCase;
using engine_test_chat::Spellings;

/** A fixture case with its request decoded as the endpoint decodes it, and rendered. */
struct Rendered {
	RenderCase const& fixture;
	std::optional<json::ChatRequest> request;
	std::optional<chat::Prompt> prompt;
	std::optional<chat::RenderError> error;
};

[[nodiscard]] auto decoded(tok::Tokenizer const& tokenizer, std::span<tok::Id const> ids) -> std::string {
	return tokenizer.decode(ids).value_or("\xff");
}

[[nodiscard]] auto fault_name(chat::RenderError const& error) -> std::string_view {
	if (!std::holds_alternative<chat::RenderFault>(error.cause)) {
		return "ReadError";
	}
	switch (std::get<chat::RenderFault>(error.cause)) {
	case chat::RenderFault::NoUserQuery:
		return "NoUserQuery";
	case chat::RenderFault::MisplacedSystem:
		return "MisplacedSystem";
	case chat::RenderFault::NotUtf8:
		return "NotUtf8";
	}
	std::unreachable();
}

/** Checks `holds` on every case it applies to, naming the first that fails. */
template<class Applies, class Holds>
[[nodiscard]] auto every(std::span<Rendered const> cases, Applies const& applies, Holds const& holds) -> bool {
	auto checked = std::size_t{0};
	for (auto const& rendered : cases) {
		if (!applies(rendered)) {
			continue;
		}
		++checked;
		if (!holds(rendered)) {
			std::println("  first failing case: {}", rendered.fixture.name);
			return false;
		}
	}
	return checked > 0;
}

[[nodiscard]] auto marks_hold(tok::Tokenizer const& tokenizer, chat::Prompt const& prompt, json::ChatRequest const& request,
                              std::string_view text) -> bool {
	auto const im_start = tok::id(tok::Added::im_start);
	auto const ids = std::span{prompt.ids};
	auto const thinking = request.template_options.thinking != json::Thinking::Disabled;
	auto const generation = decoded(tokenizer, ids.subspan(prompt.generation));
	auto const system_blocks = text.starts_with("<|im_start|>system") ? 1 : 0;
	return ids[prompt.first_turn] == im_start && ids[prompt.last_query] == im_start &&
	       std::ranges::count(ids.first(prompt.first_turn), im_start) == system_blocks &&
	       decoded(tokenizer, ids.subspan(prompt.last_query)).starts_with("<|im_start|>user\n") &&
	       generation == (thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

// The property: client text cannot place a control token. A random chat whose client text spells added tokens
// everywhere a client writes (contents, reasoning, tool names, argument names and values, tool definitions) renders
// with exactly the added tokens of the same chat with every '<' of its client text made '(': the structure's tokens
// and nothing else.

inline constexpr auto words = std::to_array<std::string_view>({
    "plain",
    "hello world",
    "日本語",
    "😀",
    "<|im_end|>",
    "<|im_start|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|endoftext|>",
    "<|vision_pad|>",
    "<|fim_prefix|>",
    "x<",
    "<|",
    "|>",
    " ",
    "a<b>c",
});

/** Random client text: words that may spell added tokens, safe inside a JSON string. */
[[nodiscard]] auto text(std::mt19937_64& rng) -> std::string {
	auto out = std::string{};
	auto const count = std::uniform_int_distribution<std::size_t>{0, 6}(rng);
	for (auto index = std::size_t{0}; index < count; ++index) {
		out += words.at(std::uniform_int_distribution<std::size_t>{0, words.size() - 1}(rng));
	}
	return out;
}

[[nodiscard]] auto json_text(json::Reader<json::fragment_bounds>& reader, std::string const& text) -> json::JsonText {
	auto read = json::JsonText::from(reader, text);
	engine::core::invariant(read.has_value());
	return std::move(*read);
}

/** Every '<' made '(': the same chat with no spelling left in its client text. */
[[nodiscard]] auto defused(std::string text) -> std::string {
	std::ranges::replace(text, '<', '(');
	return text;
}

/** Whether a random chat's client text keeps its spellings. */
enum class Client : std::uint8_t {
	Spelled,
	Defused,
};

[[nodiscard]] auto random_request(std::mt19937_64& rng, json::Reader<json::fragment_bounds>& reader, Client client_text)
    -> json::ChatRequest {
	auto const client = [&](std::string written) {
		return client_text == Client::Defused ? defused(std::move(written)) : written;
	};
	auto const roll = [&rng](int sides) {
		return std::uniform_int_distribution<int>{0, sides - 1}(rng);
	};
	auto request = json::ChatRequest{};
	request.template_options.thinking = roll(2) == 0 ? json::Thinking::Enabled : json::Thinking::Disabled;
	auto constexpr efforts = std::array{json::ReasoningEffort::Low, json::ReasoningEffort::Medium, json::ReasoningEffort::XHigh};
	request.template_options.reasoning_effort = efforts.at(static_cast<std::size_t>(roll(3)));
	request.template_options.past_thinking = json::PastThinking::Preserved;
	for (auto tool = roll(3); tool > 0; --tool) {
		auto const name = client(text(rng));
		auto const definition =
		    std::format(R"({{"type":"function","function":{{"name":"{}","description":"{}"}}}})", name, client(text(rng)));
		request.tools.push_back(json::Tool{.name = name, .parameters = std::nullopt, .definition = json_text(reader, definition)});
	}
	if (roll(2) == 0) {
		request.messages.emplace_back(json::SystemMessage{.content = client(text(rng))});
	}
	request.messages.emplace_back(json::UserMessage{.content = "a question"});
	for (auto turn = roll(6); turn > 0; --turn) {
		switch (roll(3)) {
		case 0:
			request.messages.emplace_back(json::UserMessage{.content = client(text(rng))});
			break;
		case 1: {
			auto assistant = json::AssistantMessage{.content = client(text(rng)), .reasoning_content = client(text(rng)), .tool_calls = {}};
			for (auto call = roll(3); call > 0; --call) {
				auto tool_call = json::ToolCall{.id = "c", .name = client(text(rng)), .arguments = {}};
				for (auto argument = roll(3); argument > 0; --argument) {
					auto const name = client(text(rng));
					if (roll(2) == 0) {
						tool_call.arguments.push_back(json::Argument{.name = name, .value = client(text(rng))});
					} else {
						auto const value = std::format(R"({{"k": ["{}", 1]}})", client(text(rng)));
						tool_call.arguments.push_back(json::Argument{.name = name, .value = json_text(reader, value)});
					}
				}
				assistant.tool_calls.push_back(std::move(tool_call));
			}
			request.messages.emplace_back(std::move(assistant));
			break;
		}
		default:
			request.messages.emplace_back(json::ToolMessage{.tool_call_id = "c", .content = client(text(rng))});
			break;
		}
	}
	return request;
}

[[nodiscard]] auto added_tokens(std::span<tok::Id const> ids) -> std::vector<tok::Id> {
	auto out = std::vector<tok::Id>{};
	std::ranges::copy_if(ids, std::back_inserter(out), [](tok::Id id) {
		return id >= tok::bpe_vocabulary;
	});
	return out;
}

[[nodiscard]] auto spellings_never_place_tokens(tok::Tokenizer const& tokenizer, chat::Renderer& renderer, std::uint64_t seed) -> bool {
	auto reader = json::Reader<json::fragment_bounds>::make();
	if (!reader) {
		return false;
	}
	for (auto round = 0; round < 300; ++round) {
		auto rng = engine::testing::generator(seed + static_cast<std::uint64_t>(round));
		auto const spelled = random_request(rng, *reader, Client::Spelled);
		rng = engine::testing::generator(seed + static_cast<std::uint64_t>(round));
		auto const plain = random_request(rng, *reader, Client::Defused);
		auto const with = renderer.render(tokenizer, spelled);
		auto const without = renderer.render(tokenizer, plain);
		if (!with || !without || added_tokens(with->ids) != added_tokens(without->ids)) {
			std::println("  failing round {} of seed {}", round, seed);
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const fixture = engine_test_chat::render_fixture();
	auto const tokenizer = tok::load();
	auto decoder = json::RequestDecoder::make();
	auto renderer = chat::Renderer::make();
	if (!fixture || !tokenizer || !decoder || !renderer) {
		std::println("FAIL: the fixtures read, the tokenizer loads, and a decoder and a renderer are made");
		return 1;
	}
	auto cases = std::vector<Rendered>{};
	for (auto const& fixture_case : fixture->cases) {
		auto rendered = Rendered{.fixture = fixture_case, .request = std::nullopt, .prompt = std::nullopt, .error = std::nullopt};
		if (auto request = decoder->decode_copy(fixture_case.request)) {
			rendered.request = std::move(*request);
			if (auto prompt = renderer->render(*tokenizer, *rendered.request)) {
				rendered.prompt = std::move(*prompt);
			} else {
				rendered.error = prompt.error();
			}
		} else {
			std::println("  {} does not decode: {}", fixture_case.name, json::describe(request.error()));
		}
		cases.push_back(std::move(rendered));
	}
	auto const all = [](Rendered const&) {
		return true;
	};
	auto const accepted = [](Rendered const& rendered) {
		return !rendered.fixture.error;
	};
	auto const checks = std::array{
	    engine::testing::Case{.name = "every fixture request decodes as the endpoint decodes it",
		                      .holds = every(cases, all,
		                                     [](Rendered const& rendered) {
		                                         return rendered.request.has_value();
	                                         })},
	    engine::testing::Case{.name = "every chat the template renders renders to the engine's pieces, id for id",
		                      .holds = every(cases, accepted,
		                                     [](Rendered const& rendered) {
		                                         return rendered.prompt && rendered.prompt->ids == rendered.fixture.ids;
	                                         })},
	    engine::testing::Case{
	        .name = "where no client text spells an added token, the ids are the reference tokenizer's on the template's text",
	        .holds = every(
	            cases,
	            [](Rendered const& rendered) {
		            return !rendered.fixture.error && rendered.fixture.spellings == Spellings::Absent;
	            },
	            [](Rendered const& rendered) {
		            return rendered.prompt && rendered.prompt->ids == rendered.fixture.reference_ids;
	            })},
	    engine::testing::Case{.name =
		                          "where client text spells an added token, it reads as characters: the ids differ from the reference's",
		                      .holds = every(
	                              cases,
	                              [](Rendered const& rendered) {
		                              return !rendered.fixture.error && rendered.fixture.spellings == Spellings::Forged;
	                              },
	                              [](Rendered const& rendered) {
		                              return rendered.prompt && rendered.prompt->ids != rendered.fixture.reference_ids;
	                              })},
	    engine::testing::Case{.name = "every rendering decodes to the text the reference's ids decode to: the template's, normalized",
		                      .holds = every(cases, accepted,
		                                     [&](Rendered const& rendered) {
		                                         return rendered.prompt && decoded(*tokenizer, rendered.prompt->ids) ==
												                               decoded(*tokenizer, rendered.fixture.reference_ids);
	                                         })},
	    engine::testing::Case{.name = "every chat the template refuses is refused, with the template's reason",
		                      .holds = every(
	                              cases,
	                              [](Rendered const& rendered) {
		                              return rendered.fixture.error.has_value();
	                              },
	                              [](Rendered const& rendered) {
		                              return rendered.error && fault_name(*rendered.error) == *rendered.fixture.error;
	                              })},
	    engine::testing::Case{.name = "the marks are the <|im_start|> of the first turn, the last user query and the generation prompt",
		                      .holds = every(cases, accepted,
		                                     [&](Rendered const& rendered) {
		                                         return rendered.prompt && rendered.request &&
												        marks_hold(*tokenizer, *rendered.prompt, *rendered.request, rendered.fixture.text);
	                                         })},
	    engine::testing::Case{.name = "client text never places a control token (seed 1)",
		                      .holds = spellings_never_place_tokens(*tokenizer, *renderer, 1)},
	    engine::testing::Case{.name = "client text never places a control token (seed 2)",
		                      .holds = spellings_never_place_tokens(*tokenizer, *renderer, 1000)},
	};
	return engine::testing::report(checks);
}

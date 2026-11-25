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
using engine_test_chat::TemplateInput;

/** A fixture case with its request read as the template takes it, made a conversation, and rendered. */
struct Rendered {
	RenderCase const& fixture;
	std::optional<TemplateInput> input;
	std::optional<chat::Prompt> prompt;
	/** The template's refusal, or the renderer's, by name. */
	std::optional<std::string_view> error;
};

[[nodiscard]] auto decoded(tok::Tokenizer const& tokenizer, std::span<tok::Id const> ids) -> std::string {
	return tokenizer.decode(ids).value_or("\xff");
}

[[nodiscard]] auto fault_name(chat::ShapeError const& error) -> std::string_view {
	switch (error.fault) {
	case chat::ShapeFault::NoUserQuery:
		return "NoUserQuery";
	case chat::ShapeFault::MisplacedSystem:
		return "MisplacedSystem";
	}
	std::unreachable();
}

[[nodiscard]] auto fault_name(chat::RenderError const& error) -> std::string_view {
	if (!std::holds_alternative<chat::RenderFault>(error.cause)) {
		return "ReadError";
	}
	switch (std::get<chat::RenderFault>(error.cause)) {
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

[[nodiscard]] auto marks_hold(tok::Tokenizer const& tokenizer, chat::Prompt const& prompt, chat::Options options, std::string_view text) -> bool {
	auto const im_start = tok::id(tok::Added::im_start);
	auto const ids = std::span{prompt.ids};
	auto const thinking = options.thinking == chat::Thinking::Enabled;
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

/** A random chat the template renders; `past` says whether it may drop past reasoning. */
[[nodiscard]] auto random_conversation(std::mt19937_64& rng, json::Reader<json::fragment_bounds>& reader, Client client_text, chat::PastThinking past)
    -> chat::Conversation {
	auto const client = [&](std::string written) {
		return client_text == Client::Defused ? defused(std::move(written)) : written;
	};
	auto const roll = [&rng](int sides) {
		return std::uniform_int_distribution<int>{0, sides - 1}(rng);
	};
	auto constexpr efforts = std::array{chat::Effort::Low, chat::Effort::Medium, chat::Effort::XHigh};
	auto const options = chat::Options{
	    .thinking = roll(2) == 0 ? chat::Thinking::Enabled : chat::Thinking::Disabled,
	    .effort = efforts.at(static_cast<std::size_t>(roll(3))),
	    .past_thinking = past,
	};
	auto tools = std::vector<std::string>{};
	for (auto tool = roll(3); tool > 0; --tool) {
		tools.push_back(std::format(R"({{"type":"function","function":{{"name":"{}","description":"{}"}}}})", client(text(rng)), client(text(rng))));
	}
	auto turns = std::vector<chat::Turn>{};
	if (roll(2) == 0) {
		turns.emplace_back(chat::SystemTurn{.text = client(text(rng))});
	}
	turns.emplace_back(chat::UserTurn{.text = "a question"});
	for (auto turn = roll(6); turn > 0; --turn) {
		switch (roll(3)) {
		case 0:
			turns.emplace_back(chat::UserTurn{.text = client(text(rng))});
			break;
		case 1: {
			auto assistant = chat::AssistantTurn{.reasoning = client(text(rng)), .content = client(text(rng)), .calls = {}};
			for (auto call = roll(3); call > 0; --call) {
				if (roll(4) == 0) {
					assistant.calls.push_back(chat::Call{.name = client(text(rng)), .arguments = chat::CustomInput{.text = client(text(rng))}});
					continue;
				}
				auto object = std::string{"{"};
				for (auto argument = roll(3); argument > 0; --argument) {
					object += object.size() > 1 ? "," : "";
					auto const name = std::format(R"("{}{}")", client(text(rng)), argument);
					object += roll(2) == 0 ? std::format(R"({}:"{}")", name, client(text(rng))) : std::format(R"({}:{{"k": ["{}", 1]}})", name, client(text(rng)));
				}
				object += '}';
				assistant.calls.push_back(chat::Call{.name = client(text(rng)), .arguments = json_text(reader, object)});
			}
			turns.emplace_back(std::move(assistant));
			break;
		}
		default:
			turns.emplace_back(chat::ToolTurn{.content = client(text(rng))});
			break;
		}
	}
	auto made = chat::Conversation::make(std::move(turns), std::move(tools), options);
	engine::core::invariant(made.has_value());
	return *std::move(made);
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
		auto const past = round % 2 == 0 ? chat::PastThinking::Preserved : chat::PastThinking::Dropped;
		auto const spelled = random_conversation(rng, *reader, Client::Spelled, past);
		rng = engine::testing::generator(seed + static_cast<std::uint64_t>(round));
		auto const plain = random_conversation(rng, *reader, Client::Defused, past);
		auto const with = renderer.render(tokenizer, spelled);
		auto const without = renderer.render(tokenizer, plain);
		if (!with || !without || added_tokens(with->ids) != added_tokens(without->ids)) {
			std::println("  failing round {} of seed {}", round, seed);
			return false;
		}
	}
	return true;
}

/**
 * The continuation property: for random chats and every assistant turn in them, the whole chat's ids are the ids of
 * the chat cut after that turn, up to and including its `<|im_end|>`, followed by the continuation of the rest.
 */
[[nodiscard]] auto continuations_extend_exactly(tok::Tokenizer const& tokenizer, chat::Renderer& renderer, std::uint64_t seed) -> bool {
	auto reader = json::Reader<json::fragment_bounds>::make();
	if (!reader) {
		return false;
	}
	auto checked = 0;
	for (auto round = 0; round < 300; ++round) {
		auto rng = engine::testing::generator(seed + static_cast<std::uint64_t>(round));
		auto const past = round % 2 == 0 ? chat::PastThinking::Preserved : chat::PastThinking::Dropped;
		auto const whole = random_conversation(rng, *reader, Client::Spelled, past);
		auto const rendered = renderer.render(tokenizer, whole);
		if (!rendered) {
			return false;
		}
		auto const turns = whole.turns();
		for (auto cut = std::size_t{1}; cut < turns.size(); ++cut) {
			// A stored response ends with an assistant turn; dropping past reasoning would make the earlier part depend
			// on where the last query falls, so those chats continue only when they preserve it.
			auto const before = std::vector<chat::Turn>{turns.begin(), turns.begin() + static_cast<std::ptrdiff_t>(cut)};
			if (!std::holds_alternative<chat::AssistantTurn>(turns[cut - 1]) || past == chat::PastThinking::Dropped) {
				continue;
			}
			auto const head = chat::Conversation::make(before, std::vector<std::string>{whole.tools().begin(), whole.tools().end()}, whole.options());
			if (!head) {
				continue;
			}
			auto const prefix = renderer.render(tokenizer, *head);
			auto const rest = renderer.continuation(tokenizer, turns.subspan(cut), whole.options());
			if (!prefix || !rest) {
				return false;
			}
			// The stored tokens end with the assistant turn's `<|im_end|>`, which the generation prompt follows.
			auto stored = prefix->generation;
			while (stored > 0 && prefix->ids[stored - 1] != tok::id(tok::Added::im_end)) {
				--stored;
			}
			auto const end = std::span{prefix->ids}.first(stored);
			auto joined = std::vector<tok::Id>{end.begin(), end.end()};
			joined.insert(joined.end(), rest->ids.begin(), rest->ids.end());
			auto const generation_at = rest->ids.at(rest->generation) == tok::id(tok::Added::im_start) &&
			                           joined.size() - rest->ids.size() + rest->generation == rendered->generation;
			if (end.empty() || joined != rendered->ids || !generation_at) {
				std::println("  failing round {} of seed {}, cut {}", round, seed, cut);
				return false;
			}
			++checked;
		}
	}
	return checked > 0;
}

}

auto main() -> int {
	auto const fixture = engine_test_chat::render_fixture();
	auto const tokenizer = tok::load();
	auto renderer = chat::Renderer::make();
	if (!fixture || !tokenizer || !renderer) {
		std::println("FAIL: the fixtures read, the tokenizer loads, and a renderer is made");
		return 1;
	}
	auto cases = std::vector<Rendered>{};
	for (auto const& fixture_case : fixture->cases) {
		auto rendered = Rendered{.fixture = fixture_case, .input = engine_test_chat::template_input(fixture_case.request), .prompt = std::nullopt, .error = std::nullopt};
		if (rendered.input) {
			auto const conversation = chat::Conversation::make(rendered.input->turns, rendered.input->tools, rendered.input->options);
			if (!conversation) {
				rendered.error = fault_name(conversation.error());
			} else if (auto prompt = renderer->render(*tokenizer, *conversation)) {
				rendered.prompt = std::move(*prompt);
			} else {
				rendered.error = fault_name(prompt.error());
			}
		} else {
			std::println("  {} does not read as the template's input", fixture_case.name);
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
	    engine::testing::Case{.name = "every fixture request reads as the template's input",
		                      .holds = every(cases, all,
		                                     [](Rendered const& rendered) {
		                                         return rendered.input.has_value();
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
	    engine::testing::Case{.name = "every chat the template refuses is refused when made a conversation, with the template's reason",
		                      .holds = every(
	                              cases,
	                              [](Rendered const& rendered) {
		                              return rendered.fixture.error.has_value();
	                              },
	                              [](Rendered const& rendered) {
		                              return rendered.error && *rendered.error == *rendered.fixture.error;
	                              })},
	    engine::testing::Case{.name = "the marks are the <|im_start|> of the first turn, the last user query and the generation prompt",
		                      .holds = every(cases, accepted,
		                                     [&](Rendered const& rendered) {
		                                         return rendered.prompt && rendered.input &&
												        marks_hold(*tokenizer, *rendered.prompt, rendered.input->options, rendered.fixture.text);
	                                         })},
	    engine::testing::Case{.name = "client text never places a control token (seed 1)",
		                      .holds = spellings_never_place_tokens(*tokenizer, *renderer, 1)},
	    engine::testing::Case{.name = "client text never places a control token (seed 2)",
		                      .holds = spellings_never_place_tokens(*tokenizer, *renderer, 1000)},
	    engine::testing::Case{.name = "a continuation appends exactly what the whole chat renders after the stored turn (seed 1)",
		                      .holds = continuations_extend_exactly(*tokenizer, *renderer, 1)},
	    engine::testing::Case{.name = "a continuation appends exactly what the whole chat renders after the stored turn (seed 2)",
		                      .holds = continuations_extend_exactly(*tokenizer, *renderer, 5000)},
	};
	return engine::testing::report(checks);
}

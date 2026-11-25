// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.chat;
import engine.core;
import engine.grammar;
import engine.json;
import engine.openai;
import engine.pipeline;
import engine.sampling;
import engine.sched;
import engine.testing;
import engine.tokenizer;

namespace {

namespace chat = engine::chat;
namespace core = engine::core;
namespace json = engine::json;
namespace openai = engine::openai;
namespace pipeline = engine::pipeline;
namespace sched = engine::sched;
namespace tok = engine::tokenizer;

/** A card error of the tests' own: the front is generic over the cards' error, as the pipeline is. */
struct [[nodiscard]] CardFailure {
	engine::core::Retry retried;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return retried;
	}

	[[nodiscard]] constexpr auto operator==(CardFailure const&) const -> bool = default;
};

static_assert(pipeline::Front<openai::Front, CardFailure>);

[[nodiscard]] auto object(std::string_view text) -> json::JsonText {
	auto reader = json::Reader<json::fragment_bounds>::make();
	core::invariant(reader.has_value());
	auto read = json::JsonText::from(*reader, text);
	core::invariant(read.has_value());
	return *std::move(read);
}

inline constexpr auto words = std::to_array<std::string_view>({"hello", "world", "naïve", "日本語", "😀", "<think>", "</think>",
                                                               "<tool_call>", "{\"a\": 1}", "\\", "\"", "\n", "  ", "x<y", "<|im_end|>"});

[[nodiscard]] auto phrase(std::mt19937_64& rng) -> std::string {
	auto text = std::string{};
	for (auto count = 1 + (rng() % 6); count > 0; --count) {
		text += words[rng() % words.size()];
		text += ' ';
	}
	return text;
}

[[nodiscard]] auto tools_of(std::mt19937_64& rng) -> std::vector<json::Tool> {
	auto tools = std::vector<json::Tool>{};
	if (rng() % 2 == 0) {
		tools.push_back(json::Tool{
		    .name = "get_weather",
		    .description = "the weather",
		    .input = json::FunctionTool{
		        .parameters = object(R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}})")}});
	}
	if (rng() % 2 == 0) {
		tools.push_back(json::Tool{.name = "run", .description = std::nullopt, .input = json::CustomTool{.grammar = std::nullopt}});
	}
	return tools;
}

/** A random chat as Responses input items: a user query first, then any of the item kinds. */
[[nodiscard]] auto items_of(std::mt19937_64& rng, std::size_t count) -> std::vector<json::InputItem> {
	auto items = std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = phrase(rng)}};
	for (auto index = std::size_t{1}; index < count; ++index) {
		switch (rng() % 6) {
		case 0:
			items.emplace_back(json::MessageItem{.role = json::Role::User, .text = phrase(rng)});
			break;
		case 1:
			items.emplace_back(json::MessageItem{.role = json::Role::Assistant, .text = phrase(rng)});
			break;
		case 2:
			items.emplace_back(json::ReasoningItem{.text = phrase(rng)});
			break;
		case 3:
			items.emplace_back(
			    json::FunctionCallItem{.call_id = "c1", .name = "get_weather", .arguments = object(R"({"city": "Oslo", "days": 2})")});
			break;
		case 4:
			items.emplace_back(json::CustomCallItem{.call_id = "c2", .name = "run", .input = phrase(rng)});
			break;
		default:
			items.emplace_back(json::FunctionOutputItem{.call_id = "c1", .output = phrase(rng)});
			break;
		}
	}
	return items;
}

constexpr auto efforts = std::array{json::Effort::None, json::Effort::Low, json::Effort::Medium, json::Effort::XHigh};

[[nodiscard]] auto submission(openai::Prompt prompt, std::vector<json::Tool> tools, json::Effort effort, std::uint64_t seed)
    -> openai::Submission {
	auto const params =
	    engine::sampling::parse({.temperature = std::nullopt, .top_p = std::nullopt}, engine::sampling::thinking_defaults, seed);
	core::invariant(params.has_value());
	auto compiled = openai::compile_output(openai::output_request(tools, json::ToolChoiceMode::Auto, json::TextFormatMode::Text, effort));
	core::invariant(compiled.has_value());
	return openai::Submission{
	    .tag = std::format("req-{}", seed),
	    .prompt = std::move(prompt),
	    .grammar = *std::move(compiled),
	    .tools = std::move(tools),
	    .sampling = *params,
	    .limit = sched::AtMost{.tokens = 1 + (seed % 100)},
	};
}

/**
 * A fresh submission runs as the chat format renders its conversation: the prompt is the renderer's, its marks are
 * the renderer's block offsets, and its sampling, limit and tag reach the engine as they were given.
 */
[[nodiscard]] auto fresh_prompts_are_the_render(std::uint64_t seed, tok::Tokenizer const& tokenizer) -> bool {
	auto rng = engine::testing::generator(seed);
	auto front = openai::Front::make();
	auto renderer = chat::Renderer::make();
	if (!front || !renderer) {
		return false;
	}
	for (auto round = 0; round < 40; ++round) {
		auto const tools = tools_of(rng);
		auto const items = items_of(rng, 1 + (rng() % 8));
		auto const effort = efforts[rng() % efforts.size()];
		auto const instructions = rng() % 2 == 0 ? std::optional<std::string_view>{"be brief"} : std::nullopt;
		auto conversation = chat::conversation({.instructions = instructions, .items = items, .tools = tools, .effort = effort});
		if (!conversation) {
			return false;
		}
		auto const expected = renderer->render(tokenizer, *conversation);
		if (!expected) {
			return false;
		}
		auto const asked = submission(openai::Fresh{.conversation = *conversation}, tools, effort, rng());
		auto const prepared = front->prepare(tokenizer, asked);
		auto const& request = prepared.request;
		auto const system_end = expected->first_turn > 0 ? std::optional<sched::Count>{expected->first_turn} : std::nullopt;
		if (request.prompt != expected->ids || request.marks.system_end != system_end ||
		    request.marks.last_user_start != expected->last_query || request.marks.generation_start != expected->generation ||
		    request.sampling != asked.sampling || request.limit != asked.limit || prepared.tag != asked.tag) {
			return false;
		}
	}
	return true;
}

/**
 * A continuation runs as the stored sequence exactly, then the new turns as the chat format renders them after a
 * finished assistant turn; its generation prompt is marked past the stored part.
 */
[[nodiscard]] auto continuations_are_the_sequence_then_the_turns(std::uint64_t seed, tok::Tokenizer const& tokenizer) -> bool {
	auto rng = engine::testing::generator(seed);
	auto front = openai::Front::make();
	auto renderer = chat::Renderer::make();
	if (!front || !renderer) {
		return false;
	}
	for (auto round = 0; round < 40; ++round) {
		auto sequence = std::vector<tok::Id>{};
		for (auto length = rng() % 300; length > 0; --length) {
			sequence.push_back(static_cast<tok::Id>(rng() % tok::bpe_vocabulary));
		}
		sequence.push_back(tok::id(tok::Added::im_end));
		auto const items = items_of(rng, rng() % 6);
		auto const effort = efforts[rng() % efforts.size()];
		auto turns = chat::continuation(items);
		if (!turns) {
			return false;
		}
		auto const options = chat::options_of(effort);
		auto const rest = renderer->continuation(tokenizer, *turns, options);
		if (!rest) {
			return false;
		}
		auto expected = sequence;
		expected.insert(expected.end(), rest->ids.begin(), rest->ids.end());
		auto const asked = submission(openai::Continued{.sequence = sequence, .turns = *std::move(turns), .options = options},
		                              tools_of(rng), effort, rng());
		auto const prepared = front->prepare(tokenizer, asked);
		if (prepared.request.prompt != expected || prepared.request.marks.generation_start != sequence.size() + rest->generation) {
			return false;
		}
	}
	return true;
}

/** The token ids of model output as the model writes it: the chat format's tags as their added tokens. */
[[nodiscard]] auto output_tokens(tok::Tokenizer const& tokenizer, std::string_view text) -> std::vector<tok::Id> {
	auto const pieces = std::array{tok::Piece{tok::Text{.utf8 = text, .spelling = tok::Spelling::Token}}};
	auto ids = tokenizer.encode(pieces);
	core::invariant(ids.has_value());
	return *std::move(ids);
}

/**
 * The engine starts each request's matcher from the grammar its request compiled to at the boundary: a request whose
 * format is one JSON object, with thinking off, may begin with the object but not with a word, as a matcher of the same
 * grammar says.
 */
[[nodiscard]] auto matchers_start_from_the_compiled_grammar(tok::Tokenizer const& tokenizer) -> bool {
	auto front = openai::Front::make();
	if (!front) {
		return false;
	}
	auto const vocabulary = engine::grammar::engine_vocabulary(tokenizer);
	auto const tools =
	    std::vector<json::Tool>{json::Tool{.name = "run", .description = std::nullopt, .input = json::CustomTool{.grammar = std::nullopt}}};
	auto const items = std::vector<json::InputItem>{json::MessageItem{.role = json::Role::User, .text = "go"}};
	auto conversation = chat::conversation({.instructions = std::nullopt, .items = items, .tools = tools, .effort = json::Effort::None});
	auto compiled = openai::compile_output(
	    openai::output_request(tools, json::ToolChoiceMode::None, json::TextFormatMode::JsonObject, json::Effort::None));
	if (!conversation || !compiled) {
		return false;
	}
	auto direct = engine::grammar::Matcher::make(*compiled, engine::grammar::engine_layout);
	auto asked = submission(openai::Fresh{.conversation = *std::move(conversation)}, tools, json::Effort::None, 7);
	asked.grammar = *std::move(compiled);
	auto prepared = front->prepare(tokenizer, std::move(asked));
	auto const started = prepared.grammar.mask(vocabulary);
	auto const expected = direct.mask(vocabulary);
	auto const word = output_tokens(tokenizer, "Hello").front();
	return started && expected && std::ranges::equal(started->tokens, expected->tokens) &&
	       started->allows(output_tokens(tokenizer, "{").front()) && !started->allows(word);
}

/** What a reply's outputs say, gathered. */
struct Heard {
	std::string reasoning;
	std::string content;
	std::vector<openai::CallBegin> begun;
	std::vector<openai::CallEnd> ended;
	std::optional<openai::Finished> finished;
	bool split = false;
};

[[nodiscard]] auto valid_utf8(std::string_view text) -> bool {
	auto at = std::size_t{0};
	while (at < text.size()) {
		auto const lead = std::bit_cast<std::uint8_t>(text[at]);
		auto const length = lead < 0x80U ? 1U : lead >= 0xf0U ? 4U : lead >= 0xe0U ? 3U : lead >= 0xc0U ? 2U : 0U;
		if (length == 0 || at + length > text.size()) {
			return false;
		}
		for (auto next = 1U; next < length; ++next) {
			if ((std::bit_cast<std::uint8_t>(text[at + next]) & 0xc0U) != 0x80U) {
				return false;
			}
		}
		at += length;
	}
	return true;
}

auto hear(Heard& heard, std::vector<openai::Output> const& outputs) -> void {
	for (auto const& output : outputs) {
		if (auto const reasoning = core::alternative<openai::Reasoning>(output)) {
			heard.split = heard.split || !valid_utf8(reasoning->text);
			heard.reasoning += reasoning->text;
		} else if (auto const content = core::alternative<openai::Content>(output)) {
			heard.split = heard.split || !valid_utf8(content->text);
			heard.content += content->text;
		} else if (auto const begin = core::alternative<openai::CallBegin>(output)) {
			heard.begun.push_back(*begin);
		} else if (auto const end = core::alternative<openai::CallEnd>(output)) {
			heard.ended.push_back(*end);
		} else if (auto const finished = core::alternative<openai::Finished>(output)) {
			heard.finished = *finished;
		}
	}
}

/** Feeds a reply's tokens one by one, then ends it. */
[[nodiscard]] auto run(tok::Tokenizer const& tokenizer, std::span<tok::Id const> tokens, std::span<json::Tool const> tools,
                       chat::Thinking thinking, pipeline::Ending ending) -> Heard {
	auto parser = chat::Parser::make(tools, thinking);
	core::invariant(parser.has_value());
	auto reply = openai::Reply{*std::move(parser), tools, thinking};
	auto heard = Heard{};
	for (auto const token : tokens) {
		hear(heard, reply.feed(tokenizer, token));
	}
	auto const usage = sched::Usage{.prompt_tokens = 50, .cached_tokens = 20, .generated_tokens = tokens.size() + 1};
	hear(heard, reply.end(tokenizer, pipeline::Finish{.ending = std::move(ending), .usage = usage}));
	return heard;
}

/**
 * A reply fed token by token says what the chat format's parser says of its whole text: the reasoning, the answer,
 * each call (a custom tool's as its input), every piece whole utf-8; the usage counts the thinking block's tokens.
 */
[[nodiscard]] auto replies_say_what_the_parser_says(tok::Tokenizer const& tokenizer) -> bool {
	auto const tools = std::vector<json::Tool>{
	    json::Tool{.name = "get_weather",
		           .description = std::nullopt,
		           .input = json::FunctionTool{.parameters = object(R"({"type":"object","properties":{"days":{"type":"integer"}}})")}},
	    json::Tool{.name = "run", .description = std::nullopt, .input = json::CustomTool{.grammar = std::nullopt}},
	};
	auto const thought = std::string_view{"Let me think about 日本語 \"quotes\"."};
	auto const text =
	    std::format("{}\n</think>\n\nThe answer is naïve 😀.\n\n<tool_call>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n"
		            "</function>\n</tool_call>\n<tool_call>\n<function=run>\n<parameter=input>\nls -la \"x\\y\"\n</parameter>\n"
		            "</function>\n</tool_call>",
		            thought);
	auto const tokens = output_tokens(tokenizer, text);
	auto const reasoning_tokens = output_tokens(tokenizer, std::format("{}\n</think>", thought)).size();
	auto const whole = chat::parse(text, tools, chat::Thinking::Enabled);
	auto const heard = run(tokenizer, tokens, tools, chat::Thinking::Enabled, sched::Stopped{.sequence = {1, 2, 3}});
	return whole && !heard.split && heard.reasoning == whole->reasoning && heard.content == whole->content && heard.begun.size() == 2 &&
	       heard.ended.size() == 2 &&
	       heard.begun[0] == openai::CallBegin{.index = 0, .name = "get_weather", .callee = openai::Callee::Function} &&
	       heard.begun[1] == openai::CallBegin{.index = 1, .name = "run", .callee = openai::Callee::Custom} &&
	       heard.ended[0] == openai::CallEnd{.index = 0, .text = whole->calls[0].arguments, .closing = openai::Closing::Closed} &&
	       heard.ended[1] == openai::CallEnd{.index = 1, .text = R"(ls -la "x\y")", .closing = openai::Closing::Closed} && heard.finished &&
	       heard.finished->ending == pipeline::Ending{sched::Stopped{.sequence = {1, 2, 3}}} &&
	       heard.finished->usage ==
	           json::Usage{
	               .input_tokens = 50, .cached_tokens = 20, .output_tokens = tokens.size() + 1, .reasoning_tokens = reasoning_tokens};
}

/** The token that is exactly one byte, found by search: the tokenizer is byte level, so every byte has one. */
[[nodiscard]] auto byte_token(tok::Tokenizer const& tokenizer, char byte) -> tok::Id {
	auto const wanted = std::string(1, byte);
	for (auto id = tok::Id{0}; id < tok::bpe_vocabulary; ++id) {
		if (tokenizer.token(id) == std::string_view{wanted}) {
			return id;
		}
	}
	core::invariant(false);
	return 0;
}

/**
 * A character whose bytes come in separate tokens is said once, whole, when its last byte arrives; bytes that never
 * complete a character become U+FFFD at the reply's end. A thinking-off reply has no reasoning tokens.
 */
[[nodiscard]] auto split_characters_arrive_whole(tok::Tokenizer const& tokenizer) -> bool {
	auto tokens = output_tokens(tokenizer, "word ");
	for (auto const byte : std::string_view{"日本"}) {
		tokens.push_back(byte_token(tokenizer, byte));
	}
	tokens.push_back(byte_token(tokenizer, '\xe6'));
	auto const heard = run(tokenizer, tokens, {}, chat::Thinking::Disabled, sched::Truncated{.sequence = {}});
	return !heard.split && heard.content == "word 日本\xef\xbf\xbd" && heard.finished && heard.finished->usage.reasoning_tokens == 0;
}

/** A call the budget cut off is closed with what was parsed of it, marked cut off; one the reply closed is not. */
[[nodiscard]] auto cut_calls_are_marked(tok::Tokenizer const& tokenizer) -> bool {
	auto const tools =
	    std::vector<json::Tool>{json::Tool{.name = "run", .description = std::nullopt, .input = json::CustomTool{.grammar = std::nullopt}}};
	auto const open = output_tokens(tokenizer, "<tool_call>\n<function=run>\n<parameter=input>\nrm\n</parameter>\n");
	auto const cut = run(tokenizer, open, tools, chat::Thinking::Disabled, sched::Truncated{.sequence = {}});
	auto const ended = run(tokenizer, open, tools, chat::Thinking::Disabled, sched::Stopped{.sequence = {}});
	return cut.ended.size() == 1 && cut.ended[0] == openai::CallEnd{.index = 0, .text = "rm", .closing = openai::Closing::CutOff} &&
	       ended.ended.size() == 1 && ended.ended[0].closing == openai::Closing::Closed;
}

/** The front answers the engine's verdicts and posts its batches, snapshots and failure as the service reads them. */
[[nodiscard]] auto the_front_maps_the_engine() -> bool {
	auto front = openai::Front::make();
	if (!front) {
		return false;
	}
	auto const batch = std::vector<openai::Addressed>{openai::Addressed{.exchange = {}, .output = openai::Accepted{}}};
	auto const snapshot = openai::Snapshot{.slots = {}, .queued = 3, .pool = {}, .counters = {}};
	auto const transient = pipeline::EngineError<CardFailure>{.cause = CardFailure{.retried = core::Retry::Transient}};
	auto const failed = front->failed(transient);
	auto const fault = engine::grammar::MatchError{.fault = engine::grammar::MatchFault::RowTooLarge};
	return front->accepted() == openai::Output{openai::Accepted{}} &&
	       front->refused(sched::PromptTooLong{.prompt_tokens = 9}) == openai::Output{openai::PromptTooLong{.prompt_tokens = 9}} &&
	       front->cancelled() == openai::Output{openai::Cancelled{}} &&
	       front->overran(fault) == openai::Output{openai::Overran{.fault = fault}} &&
	       front->batch(batch) == openai::Update{openai::Outputs{.items = batch}} && front->status(snapshot) == openai::Update{snapshot} &&
	       std::holds_alternative<openai::EngineFailed>(failed) && std::get<openai::EngineFailed>(failed).reason.contains("transient");
}

}

auto main() -> int {
	auto const tokenizer = tok::load();
	if (!tokenizer) {
		std::println("FAIL: the tokenizer image does not load");
		return 1;
	}
	auto const checks = std::array{
	    engine::testing::Case{.name = "a fresh prompt is the chat format's render, with its marks (seed 1)",
		                      .holds = fresh_prompts_are_the_render(1, *tokenizer)},
	    engine::testing::Case{.name = "a fresh prompt is the chat format's render, with its marks (seed 2)",
		                      .holds = fresh_prompts_are_the_render(2, *tokenizer)},
	    engine::testing::Case{.name = "a continuation is the stored sequence, then the new turns rendered after it (seed 1)",
		                      .holds = continuations_are_the_sequence_then_the_turns(1, *tokenizer)},
	    engine::testing::Case{.name = "a continuation is the stored sequence, then the new turns rendered after it (seed 2)",
		                      .holds = continuations_are_the_sequence_then_the_turns(2, *tokenizer)},
	    engine::testing::Case{.name = "the engine's matcher starts from the grammar compiled at the boundary",
		                      .holds = matchers_start_from_the_compiled_grammar(*tokenizer)},
	    engine::testing::Case{.name = "a reply fed token by token says what the parser says of its whole text",
		                      .holds = replies_say_what_the_parser_says(*tokenizer)},
	    engine::testing::Case{.name = "a character split across tokens arrives whole", .holds = split_characters_arrive_whole(*tokenizer)},
	    engine::testing::Case{.name = "a call the budget cut off is marked cut off", .holds = cut_calls_are_marked(*tokenizer)},
	    engine::testing::Case{.name = "the front maps the engine's verdicts, batches, snapshots and failure",
		                      .holds = the_front_maps_the_engine()},
	};
	return engine::testing::report(checks);
}

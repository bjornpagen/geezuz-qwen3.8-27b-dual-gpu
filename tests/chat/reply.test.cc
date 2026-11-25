// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.simdjson;
import engine.tokenizer;
import engine.chat;
import engine.testing;
import engine_test_chat.fixtures;

namespace {

namespace chat = engine::chat;
namespace json = engine::json;
namespace tok = engine::tokenizer;
using engine_test_chat::ParseCase;

/**
 * Feeds `pieces` to a parser and gathers its events, checking each as it comes: text is never empty, always utf-8,
 * and never inside a tool call; calls open in order and close in the order they opened. Nothing when a check fails.
 */
[[nodiscard]] auto gather(chat::Parser parser, std::span<std::string const> pieces) -> std::optional<chat::Reply> {
	auto reply = chat::Reply{};
	auto open = std::optional<std::size_t>{};
	auto holds = true;
	auto const take = [&](std::vector<chat::Event> const& events) {
		for (auto const& event : events) {
			if (std::holds_alternative<chat::ReasoningDelta>(event) || std::holds_alternative<chat::ContentDelta>(event)) {
				auto const& text = std::holds_alternative<chat::ReasoningDelta>(event) ? std::get<chat::ReasoningDelta>(event).text
				                                                                       : std::get<chat::ContentDelta>(event).text;
				holds = holds && !text.empty() && engine::simdjson::valid_utf8(text) && !open;
			} else if (std::holds_alternative<chat::ToolCallStart>(event)) {
				holds = holds && !open && std::get<chat::ToolCallStart>(event).index == reply.calls.size();
				open = std::get<chat::ToolCallStart>(event).index;
			} else {
				holds = holds && open == std::get<chat::ToolCallEnd>(event).index;
				open.reset();
			}
			if (holds) {
				reply.add(event);
			}
		}
	};
	for (auto const& piece : pieces) {
		take(parser.push(piece));
	}
	take(parser.finish());
	if (!holds || open) {
		return std::nullopt;
	}
	return reply;
}

/** The output cut at random byte boundaries, each boundary cut with probability `rate`: characters split too. */
[[nodiscard]] auto split(std::string_view output, std::mt19937_64& rng, double rate) -> std::vector<std::string> {
	auto pieces = std::vector<std::string>{""};
	for (auto const byte : output) {
		pieces.back().push_back(byte);
		if (std::bernoulli_distribution{rate}(rng)) {
			pieces.emplace_back();
		}
	}
	return pieces;
}

/** The output as a model streams it: encoded, then decoded token by token. */
[[nodiscard]] auto token_pieces(tok::Tokenizer const& tokenizer, std::string_view output) -> std::vector<std::string> {
	auto const piece = std::array{tok::Piece{tok::Text{.utf8 = output, .spelling = tok::Spelling::Token}}};
	auto const ids = tokenizer.encode(piece);
	auto pieces = std::vector<std::string>{};
	if (!ids) {
		return pieces;
	}
	auto stream = tok::Detokenizer{};
	for (auto const id : *ids) {
		pieces.push_back(stream.push(tokenizer, id).value_or(""));
	}
	pieces.push_back(stream.finish());
	return pieces;
}

/** A parse case with the tools its reply parser casts by. */
struct Parsed {
	ParseCase const& fixture;
	std::optional<std::vector<json::Tool>> tools;
};

/** What a reply is parsed under: its tools, and whether it starts inside the thinking block. */
struct Under {
	std::span<json::Tool const> tools;
	chat::Thinking thinking;
};

/** Checks `holds` on every case with its tools, naming the first that fails. */
template<class Holds>
[[nodiscard]] auto every(std::span<Parsed const> cases, Holds const& holds) -> bool {
	for (auto const& parsed : cases) {
		if (!parsed.tools || !holds(parsed.fixture, Under{.tools = *parsed.tools, .thinking = parsed.fixture.thinking})) {
			std::println("  first failing case: {}", parsed.fixture.name);
			return false;
		}
	}
	return !cases.empty();
}

[[nodiscard]] auto one_shot(Under under, std::string_view output) -> std::optional<chat::Reply> {
	auto reply = chat::parse(output, under.tools, under.thinking);
	return reply ? std::optional{std::move(*reply)} : std::nullopt;
}

[[nodiscard]] auto pieces_gather(Under under, std::span<std::string const> pieces, std::string_view whole) -> bool {
	auto parser = chat::Parser::make(under.tools, under.thinking);
	if (!parser) {
		return false;
	}
	auto const streamed = gather(std::move(*parser), pieces);
	return streamed && streamed == one_shot(under, whole);
}

[[nodiscard]] auto random_splits_gather(std::span<Parsed const> cases, std::uint64_t seed, double rate) -> bool {
	auto rng = engine::testing::generator(seed);
	return every(cases, [&](ParseCase const& reply, Under under) {
		auto const pieces = split(reply.output, rng, rate);
		return pieces_gather(under, pieces, reply.output);
	});
}

// Where the parser differs from the reference on purpose.

[[nodiscard]] auto reasoning_regions_join() -> bool {
	auto const reply = one_shot(Under{.tools = {}, .thinking = chat::Thinking::Enabled}, "first\n</think>\n\nanswer <think> second </think> more");
	return reply == chat::Reply{.reasoning = "firstsecond", .content = "answermore", .calls = {}};
}

[[nodiscard]] auto structured_casts_stay_json() -> bool {
	auto fragments = json::Reader<json::fragment_bounds>::make();
	auto const tool = fragments ? engine_test_chat::parser_tool(R"({"type":"function","function":{"name":"f",)"
	                                                            R"("parameters":{"type":"object","properties":{"o":{"type":"object"},"a":{"type":"array"}}}}})",
	                                                            *fragments)
	                            : std::nullopt;
	if (!tool) {
		return false;
	}
	auto const tools = std::array{*tool};
	auto const reply = one_shot(Under{.tools = tools, .thinking = chat::Thinking::Enabled},
	                              "</think><tool_call>\n<function=f>\n<parameter=o>\n{\"k\": 1, \"k\": "
	                              "2}\n</parameter>\n<parameter=a>\n[NaN, 1e999]\n</parameter>\n</function>\n</tool_call>");
	return reply && reply->calls.size() == 1 && reply->calls.front().arguments == R"({"o": "{\"k\": 1, \"k\": 2}", "a": "[NaN, 1e999]"})";
}

/** A custom tool's `input` stays the text the model wrote, even where a function's string-free schema would cast. */
[[nodiscard]] auto custom_input_is_never_cast() -> bool {
	auto const tools = std::array{json::Tool{.name = "run", .description = std::nullopt, .input = json::CustomTool{.grammar = std::nullopt}}};
	auto const reply = one_shot(Under{.tools = tools, .thinking = chat::Thinking::Disabled},
	                            "<tool_call>\n<function=run>\n<parameter=input>\n42\n</parameter>\n</function>\n</tool_call>");
	return reply && reply->calls.size() == 1 && reply->calls.front().arguments == R"({"input": "42"})";
}

// The round trip: a reply written in the template's own format parses back to itself.

inline constexpr auto solid = std::to_array<std::string_view>(
    {"Let", "x<y", "a>b", "<tool_", "</think", "<para", "</parameter", "<function", "日本語", "😀", "é", "{\"a\": 1}", "\\", "\""});
inline constexpr auto spaces = std::to_array<std::string_view>({" ", "  ", "\n", "\n\n", "\t", "　", " ", "\r\n", ""});
inline constexpr auto names =
    std::to_array<std::string_view>({"get_weather", "f", "search the web", "名前", "a<b", "x=y", "</function", " pad "});

template<std::size_t N>
[[nodiscard]] auto pick(std::mt19937_64& rng, std::array<std::string_view, N> const& pool) -> std::string_view {
	return pool.at(std::uniform_int_distribution<std::size_t>{0, N - 1}(rng));
}

/** Text whose first and last characters are not whitespace, or nothing. */
[[nodiscard]] auto core_text(std::mt19937_64& rng) -> std::string {
	auto out = std::string{};
	auto const count = std::uniform_int_distribution<std::size_t>{0, 5}(rng);
	for (auto index = std::size_t{0}; index < count; ++index) {
		out += index == 0 ? std::string_view{} : pick(rng, spaces);
		out += pick(rng, solid);
	}
	return out;
}

/** The reference's JSON for a string: Python's `json.dumps(text, ensure_ascii=False)`. */
[[nodiscard]] auto python_string(std::string_view text) -> std::string {
	auto out = std::string{"\""};
	for (auto const byte : text) {
		auto const code = std::bit_cast<std::uint8_t>(byte);
		if (byte == '"' || byte == '\\') {
			out += '\\';
			out += byte;
		} else if (byte == '\n') {
			out += "\\n";
		} else if (byte == '\r') {
			out += "\\r";
		} else if (byte == '\t') {
			out += "\\t";
		} else if (code < 0x20U) {
			out += std::format("\\u{:04x}", code);
		} else {
			out += byte;
		}
	}
	return out + "\"";
}

struct Written {
	std::string output;
	chat::Reply reply;
};

/** A random reply, and the model's text for it as the chat template writes an assistant turn after `<think>\n`. */
[[nodiscard]] auto random_reply(std::mt19937_64& rng, chat::Thinking thinking) -> Written {
	auto written = Written{};
	if (thinking == chat::Thinking::Enabled) {
		written.reply.reasoning = core_text(rng);
		written.output = std::string{pick(rng, spaces)} + written.reply.reasoning + std::string{pick(rng, spaces)} + "\n</think>\n\n";
	}
	written.reply.content = core_text(rng);
	written.output += std::string{pick(rng, spaces)} + written.reply.content;
	auto const calls = std::uniform_int_distribution<std::size_t>{0, 3}(rng);
	for (auto index = std::size_t{0}; index < calls; ++index) {
		written.output += index > 0 ? "\n" : written.reply.content.empty() ? "" : "\n\n";
		auto call = chat::ParsedCall{.name = std::string{pick(rng, names)}, .arguments = "{"};
		written.output.append("<tool_call>\n<function=").append(call.name).append(">\n");
		auto const parameters = std::uniform_int_distribution<std::size_t>{0, 3}(rng);
		for (auto parameter = std::size_t{0}; parameter < parameters; ++parameter) {
			auto const name = std::format("p{}", parameter);
			auto const value = std::string{pick(rng, spaces)} + core_text(rng) + std::string{pick(rng, spaces)};
			written.output.append("<parameter=").append(name).append(">\n").append(value).append("\n</parameter>\n");
			call.arguments.append(parameter > 0 ? ", " : "").append(python_string(name)).append(": ").append(python_string(value));
		}
		call.arguments.push_back('}');
		written.output += "</function>\n</tool_call>";
		written.reply.calls.push_back(std::move(call));
	}
	return written;
}

[[nodiscard]] auto replies_round_trip(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	for (auto const thinking : {chat::Thinking::Enabled, chat::Thinking::Disabled}) {
		for (auto round = 0; round < 500; ++round) {
			auto const written = random_reply(rng, thinking);
			auto parser = chat::Parser::make({}, thinking);
			auto const pieces = split(written.output, rng, 0.2);
			if (!parser || chat::parse(written.output, {}, thinking) != written.reply || gather(std::move(*parser), pieces) != written.reply) {
				std::println("  failing output: {:?}", written.output);
				return false;
			}
		}
	}
	return true;
}

}

auto main() -> int {
	auto const fixture = engine_test_chat::parse_fixture();
	auto const tokenizer = tok::load();
	if (!fixture || !tokenizer) {
		std::println("FAIL: the fixtures read and the tokenizer loads");
		return 1;
	}
	auto cases = std::vector<Parsed>{};
	for (auto const& fixture_case : fixture->cases) {
		auto tools = engine_test_chat::parser_tools(fixture_case);
		if (!tools) {
			std::println("  {}: its tools do not read", fixture_case.name);
		}
		cases.push_back(Parsed{.fixture = fixture_case, .tools = std::move(tools)});
	}
	auto const checks = std::array{
	    engine::testing::Case{.name = "every fixture reply parses as the checkpoint's response template parses it",
		                      .holds = every(cases,
		                                     [](ParseCase const& reply, Under under) {
		                                         return one_shot(under, reply.output) == reply.expected;
	                                         })},
	    engine::testing::Case{.name = "cut at random bytes, every fixture reply streams to its whole parse (seed 1, dense cuts)",
		                      .holds = random_splits_gather(cases, 1, 0.5)},
	    engine::testing::Case{.name = "cut at random bytes, every fixture reply streams to its whole parse (seed 2, sparse cuts)",
		                      .holds = random_splits_gather(cases, 2, 0.05)},
	    engine::testing::Case{.name = "cut at random bytes, every fixture reply streams to its whole parse (seed 3, every byte)",
		                      .holds = random_splits_gather(cases, 3, 1.0)},
	    engine::testing::Case{.name = "decoded token by token, every fixture reply streams to the parse of its decoding",
		                      .holds = every(cases,
		                                     [&](ParseCase const& reply, Under under) {
		                                         auto const pieces = token_pieces(*tokenizer, reply.output);
		                                         auto whole = std::string{};
		                                         std::ranges::for_each(pieces, [&whole](std::string const& piece) {
			                                         whole += piece;
		                                         });
		                                         return !pieces.empty() && pieces_gather(under, pieces, whole);
	                                         })},
	    engine::testing::Case{.name = "reasoning regions join, where the reference keeps only the last", .holds = reasoning_regions_join()},
	    engine::testing::Case{.name = "object and array casts read strict JSON: repeated keys and NaN stay text",
		                      .holds = structured_casts_stay_json()},
	    engine::testing::Case{.name = "a custom tool's input is never cast", .holds = custom_input_is_never_cast()},
	    engine::testing::Case{.name = "replies written as the template writes them parse back to themselves (seed 1)",
		                      .holds = replies_round_trip(1)},
	    engine::testing::Case{.name = "replies written as the template writes them parse back to themselves (seed 2)",
		                      .holds = replies_round_trip(2)},
	};
	return engine::testing::report(checks);
}

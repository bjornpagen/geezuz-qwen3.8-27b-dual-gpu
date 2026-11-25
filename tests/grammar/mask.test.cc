// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.grammar;
import engine.regex;
import engine.testing;

namespace {

namespace grammar = engine::grammar;
namespace regex = engine::regex;

/**
 * Pieces random tokens are made of: single bytes the grammars care about, whole and split multi-byte characters,
 * markers and their parts, and words; tokens join one to three of them, so a token may span lexemes, characters
 * and markers the way real tokens do.
 */
inline constexpr auto pieces = std::to_array<std::string_view>({
    "a", "b", "c", "x", "0", "1", "7", "-", ".", "e", " ", "\n", "\t", "\"", "\\", "{", "}", "[", "]", ":", ",", "<", ">", "/", "=",
    "é", "\xc3", "\xa9", "😀", "\xf0\x9f", "\x98\x80", "true", "false", "null", "<tool_call>", "</tool_call>", "<tool_", "call>", "</think>",
    "<parameter=", "</parameter>", "</function>", "<function=", "\n<", "abc", "name", "id", "12", "3.5", "\"a\"", ": ", ", ", "<|im_end|>",
});

struct Sample {
	grammar::Vocabulary vocabulary;
	/** Every token's bytes, by id; the end token last. */
	std::vector<std::string> bytes;
	grammar::Token end;
	grammar::Layout layout;
	/** Each token's row bit: a permutation, so the two layouts differ. */
	std::vector<std::uint32_t> rows;
};

/** Every single byte, then random joins of pieces, then one end token; rows are a random permutation of ids. */
[[nodiscard]] auto sample_vocabulary(std::uint64_t seed, std::size_t joined) -> Sample {
	auto random = engine::testing::generator(seed);
	auto bytes = std::vector<std::string>{};
	for (auto byte = 0U; byte < 256U; ++byte) {
		bytes.emplace_back(1, static_cast<char>(static_cast<std::uint8_t>(byte)));
	}
	auto const pick = [&random](std::size_t count) { return std::uniform_int_distribution<std::size_t>{0, count - 1}(random); };
	while (bytes.size() < 256 + joined) {
		auto token = std::string{};
		for (auto count = pick(3) + 1; count > 0; --count) {
			token.append(pieces.at(pick(pieces.size())));
		}
		if (!std::ranges::contains(bytes, token)) {
			bytes.push_back(token);
		}
	}
	auto const end = static_cast<grammar::Token>(bytes.size());
	auto const layout = grammar::Layout{.tokens = bytes.size() + 1, .rows = bytes.size() + 1};
	auto rows = std::vector<std::uint32_t>(layout.rows);
	std::ranges::iota(rows, 0U);
	std::ranges::shuffle(rows, random);
	auto tokens = std::vector<grammar::TokenBytes>{};
	for (auto const [id, text] : std::views::enumerate(bytes)) {
		auto const token = static_cast<grammar::Token>(id);
		tokens.push_back(grammar::TokenBytes{.token = token, .bytes = text, .row = rows[token]});
	}
	auto const ends = std::array{grammar::EndToken{.token = end, .row = rows[end]}};
	return Sample{.vocabulary = grammar::Vocabulary::make(tokens, ends, layout), .bytes = std::move(bytes), .end = end, .layout = layout, .rows = std::move(rows)};
}

/**
 * The mask's oracle is the matcher's own byte-by-byte advance: a token is in the mask exactly when committing it
 * succeeds, and the end token exactly when the output may end. Checked at every position of random walks, in both
 * layouts; the walk also never reaches a position with nothing allowed before it ends.
 */
[[nodiscard]] auto masks_match_advance(grammar::Grammar const& compiled, Sample const& sample, std::uint64_t seed, std::size_t walks, std::size_t steps,
                                       std::span<std::string_view const> prefixes) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto walk = std::size_t{0}; walk < walks; ++walk) {
		auto matcher = grammar::Matcher::make(compiled, sample.layout);
		// Each walk starts from one of the prefixes, written byte by byte, so walks reach deep into the grammar.
		auto const prefix = prefixes.empty() ? std::string_view{} : prefixes[walk % prefixes.size()];
		for (auto const letter : prefix) {
			if (!matcher.advance(sample.vocabulary, std::bit_cast<std::uint8_t>(letter))) {
				std::println("  prefix refused: {:?}", prefix);
				return false;
			}
		}
		for (auto step = std::size_t{0}; step < steps && !matcher.ended(); ++step) {
			auto const mask = matcher.mask(sample.vocabulary);
			if (!mask) {
				std::println("  mask failed: {}", std::to_underlying(mask.error().fault));
				return false;
			}
			auto const tokens = std::vector<std::uint64_t>{mask->tokens.begin(), mask->tokens.end()};
			auto const rows = std::vector<std::uint64_t>{mask->rows.begin(), mask->rows.end()};
			auto const allows = [&tokens](grammar::Token token) { return ((tokens[token / 64] >> (token % 64U)) & 1U) != 0; };
			auto const allows_row = [&rows](std::uint32_t row) { return ((rows[row / 64] >> (row % 64U)) & 1U) != 0; };
			auto allowed = std::vector<grammar::Token>{};
			for (auto token = grammar::Token{0}; token <= sample.end; ++token) {
				auto const committed = matcher.advance(sample.vocabulary, token);
				if (committed) {
					matcher.rollback(1);
					allowed.push_back(token);
				}
				if (allows(token) != committed.has_value() || allows_row(sample.rows[token]) != committed.has_value()) {
					auto const text = token == sample.end ? std::string{"<end>"} : sample.bytes[token];
					std::println("  walk {} step {}: token {:?} mask {} row {} advance {}", walk, step, text, allows(token), allows_row(sample.rows[token]), committed.has_value());
					return false;
				}
			}
			if (allowed.empty()) {
				std::println("  walk {} step {}: nothing allowed", walk, step);
				return false;
			}
			auto const chosen = allowed[std::uniform_int_distribution<std::size_t>{0, allowed.size() - 1}(random)];
			if (!matcher.advance(sample.vocabulary, chosen)) {
				return false;
			}
		}
	}
	return true;
}

/** Rolling back k tokens restores the exact mask and acceptance of k tokens before. */
[[nodiscard]] auto rollback_restores(grammar::Grammar const& compiled, Sample const& sample, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto matcher = grammar::Matcher::make(compiled, sample.layout);
	auto const snapshot = [&] {
		auto const mask = matcher.mask(sample.vocabulary);
		auto words = std::vector<std::uint64_t>{mask->tokens.begin(), mask->tokens.end()};
		words.insert(words.end(), mask->rows.begin(), mask->rows.end());
		words.push_back(matcher.accepting() ? 1 : 0);
		return words;
	};
	auto history = std::vector<std::vector<std::uint64_t>>{snapshot()};
	for (auto step = 0; step < 200; ++step) {
		auto const back = std::uniform_int_distribution<std::size_t>{0, std::min<std::size_t>(5, matcher.committed())}(random);
		if (back > 0 && std::uniform_int_distribution<int>{0, 2}(random) == 0) {
			matcher.rollback(back);
			history.resize(history.size() - back);
			if (snapshot() != history.back()) {
				std::println("  rollback of {} at step {} differs", back, step);
				return false;
			}
			continue;
		}
		auto const mask = matcher.mask(sample.vocabulary);
		auto allowed = std::vector<grammar::Token>{};
		for (auto token = grammar::Token{0}; token < sample.end; ++token) {
			if (mask->allows(token)) {
				allowed.push_back(token);
			}
		}
		if (allowed.empty()) {
			break;
		}
		std::ignore = matcher.advance(sample.vocabulary, allowed[std::uniform_int_distribution<std::size_t>{0, allowed.size() - 1}(random)]);
		history.push_back(snapshot());
	}
	return true;
}

/**
 * For a regular expression the prefix language has an independent oracle: the expression's own automaton, built
 * apart from any grammar. A token is allowed exactly when the text with its bytes still leads to a live state.
 */
[[nodiscard]] auto regex_masks_match_prefixes(std::string_view pattern, Sample const& sample, std::uint64_t seed) -> bool {
	auto compiled = grammar::compile_regex(pattern);
	if (!compiled) {
		return false;
	}
	auto store = regex::Store{std::size_t{1} << 20U};
	auto const automaton = *regex::Automaton::build(store, *regex::parse(store, pattern, regex::Matching::Whole), 100000);
	auto random = engine::testing::generator(seed);
	auto matcher = grammar::Matcher::make(std::move(*compiled), sample.layout);
	auto text = std::string{};
	for (auto step = 0; step < 40 && !matcher.ended(); ++step) {
		auto const mask = matcher.mask(sample.vocabulary);
		auto state = automaton.start();
		for (auto const letter : text) {
			state = automaton.step(state, std::bit_cast<std::uint8_t>(letter));
		}
		auto allowed = std::vector<grammar::Token>{};
		for (auto token = grammar::Token{0}; token < sample.end; ++token) {
			auto next = state;
			for (auto const letter : sample.bytes[token]) {
				next = automaton.step(next, std::bit_cast<std::uint8_t>(letter));
			}
			if (mask->allows(token) != (next != regex::dead)) {
				std::println("  {:?} after {:?}: token {:?}", pattern, text, sample.bytes[token]);
				return false;
			}
			if (next != regex::dead) {
				allowed.push_back(token);
			}
		}
		if (mask->allows(sample.end) != automaton.accepting(state)) {
			return false;
		}
		if (allowed.empty()) {
			break;
		}
		auto const chosen = allowed[std::uniform_int_distribution<std::size_t>{0, allowed.size() - 1}(random)];
		text.append(sample.bytes[chosen]);
		std::ignore = matcher.advance(sample.vocabulary, chosen);
	}
	return true;
}

[[nodiscard]] auto lark(std::string_view text) -> grammar::Grammar {
	auto compiled = grammar::compile_lark(text);
	engine::core::invariant(compiled.has_value());
	return std::move(*compiled);
}

[[nodiscard]] auto output(grammar::OutputRequest const& request) -> grammar::Grammar {
	auto compiled = grammar::compile(request);
	if (!compiled) {
		std::println("  refused {} at {}", std::to_underlying(compiled.error().refusal), compiled.error().at);
	}
	engine::core::invariant(compiled.has_value());
	return std::move(*compiled);
}

[[nodiscard]] auto weather() -> grammar::OutputRequest {
	return grammar::OutputRequest{
	    .reasoning = grammar::Reasoning::Thinking,
	    .tools = {grammar::FunctionTool{.name = "get_weather",
	                                    .parameters = R"({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 14}, "units": {"enum": ["c", "f"]}}, "required": ["city"], "additionalProperties": false})"},
	              grammar::CustomTool{.name = "calc", .input = grammar::LarkInput{.definition = "start: expr\nexpr: NUM | expr \"+\" NUM\nNUM: /[0-9]+/\n"}}},
	    .tool_choice = grammar::Calls::Auto,
	    .parallel = grammar::Parallel::Many,
	    .format = grammar::PlainText{},
	};
}

}

auto main() -> int {
	auto const small = sample_vocabulary(11, 300);
	auto const other = sample_vocabulary(12, 600);
	auto const json = R"({"type": "object", "properties": {"a": {"type": "string", "maxLength": 4}, "b": {"type": "array", "items": {"type": "number"}}, "c": {"anyOf": [{"type": "null"}, {"type": "boolean"}]}}, "required": ["a"], "additionalProperties": false})";
	auto schema = grammar::compile_schema(json);
	auto const none = std::vector<std::string_view>{};
	auto const json_prefixes = std::vector<std::string_view>{"", R"({"a": "x)", R"({"a": "xy", "b": [1, 2.)", R"({"a": "", "c": )"};
	auto const call_prefixes = std::vector<std::string_view>{
	    "",
	    "hmm</think>",
	    "hmm</think>\n\nok<tool_call>\n<function=get_weather>\n<parameter=city>\nPa",
	    "a</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n1",
	    "a</think><tool_call>\n<function=calc>\n<parameter=input>\n12+3",
	    "a</think><tool_call>\n<function=calc>\n<parameter=input>\n12+3\n</parameter>\n</function>\n</tool_call>",
	    "a</think><tool_call>\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n<parameter=units>\n",
	};
	auto const custom_prefixes = std::vector<std::string_view>{"", "<tool_call>\n<function=calc>\n<parameter=input>\nabc", "<tool_call>\n<function=calc>\n<parameter=input>\nabc\n</para"};
	auto const content_prefixes = std::vector<std::string_view>{"", R"({"a": "x")", "{\"a\": \"x\"}\n<tool_call>\n<function=get_weather>\n<parameter=city>\nx"};
	auto const custom = grammar::OutputRequest{.reasoning = grammar::Reasoning::Direct,
	                                           .tools = {grammar::CustomTool{.name = "calc", .input = grammar::LarkInput{.definition = "start: TEXT\nTEXT: /(.|\\n)+/\n"}}},
	                                           .tool_choice = grammar::ForcedTool{.name = "calc"},
	                                           .parallel = grammar::Parallel::One,
	                                           .format = grammar::PlainText{}};
	auto const content = grammar::OutputRequest{.reasoning = grammar::Reasoning::Direct,
	                                            .tools = weather().tools,
	                                            .tool_choice = grammar::Calls::Required,
	                                            .parallel = grammar::Parallel::Many,
	                                            .format = grammar::JsonSchema{.schema = json}};
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "masks match advance: a lark grammar with ignore (seed 1)",
	     .holds = masks_match_advance(lark("start: item (\",\" item)*\nitem: WORD | \"<\" WORD \">\"\nWORD: /[a-zé😀]+/\n%ignore /[ \\n]+/\n"), small, 1, 8, 40, none)},
	    {.name = "masks match advance: lazy lexemes (seed 2)",
	     .holds = masks_match_advance(lark("start: head \"=\" /[0-9]+/ tail\nhead[lazy]: /(.|\\n)*<function/\ntail: /[^<]*/\n"), small, 2, 8, 40, none)},
	    {.name = "masks match advance: a JSON schema (seed 3)", .holds = schema && masks_match_advance(*schema, other, 3, 8, 60, json_prefixes)},
	    {.name = "masks match advance: the composed output with tools (seed 4)", .holds = masks_match_advance(output(weather()), other, 4, 14, 50, call_prefixes)},
	    {.name = "masks match advance: a custom lark tool's region and closing (seed 5)", .holds = masks_match_advance(output(custom), small, 5, 9, 50, custom_prefixes)},
	    {.name = "masks match advance: JSON content and required calls (seed 6)", .holds = masks_match_advance(output(content), other, 6, 9, 50, content_prefixes)},
	    {.name = "rollback restores masks exactly (seed 7)", .holds = rollback_restores(output(weather()), other, 7)},
	    {.name = "rollback restores masks exactly in JSON (seed 8)", .holds = schema && rollback_restores(*schema, small, 8)},
	    {.name = "regex masks are the prefix language (dates)", .holds = regex_masks_match_prefixes("[0-9]{4}-[0-9]{2}-[0-9]{2}", small, 9)},
	    {.name = "regex masks are the prefix language (free text)", .holds = regex_masks_match_prefixes("(?s:.*)", other, 10)},
	    {.name = "regex masks are the prefix language (alternatives and unicode)", .holds = regex_masks_match_prefixes("(?:abc|a(?:é|😀)+|[0-9]+\\.[0-9]+)\\n?", other, 11)},
	});
	return engine::testing::report(cases);
}

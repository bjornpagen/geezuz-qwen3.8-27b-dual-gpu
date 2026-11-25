// SPDX-License-Identifier: AGPL-3.0-only
//
// Not a test: a benchmark run by hand in the release preset, so it never runs by default. It measures the compile
// time of request grammars at the compile-time bounds and the mask latency over random walks of realistic requests
// with the real tokenizer, and prints percentiles.
import std;
import engine.core;
import engine.grammar;
import engine.grammar_support;
import engine.tokenizer;

namespace {

namespace grammar = engine::grammar;
namespace support = engine::grammar_support;

using Clock = std::chrono::steady_clock;

[[nodiscard]] auto microseconds(Clock::duration duration) -> double {
	return std::chrono::duration<double, std::micro>{duration}.count();
}

auto percentiles(std::string_view name, std::vector<Clock::duration> samples) -> void {
	if (samples.empty()) {
		return;
	}
	std::ranges::sort(samples);
	auto const at = [&samples](double q) { return microseconds(samples[static_cast<std::size_t>(q * static_cast<double>(samples.size() - 1))]); };
	std::println("{:<44} n={:<6} p50={:>8.1f}us p90={:>8.1f}us p99={:>8.1f}us max={:>8.1f}us", name, samples.size(), at(0.5), at(0.9), at(0.99), at(1.0));
}

/** The best of `runs` compiles of a request, and whether it compiled. */
template<class Compile>
auto time_compile(std::string_view name, std::size_t runs, Compile compile) -> void {
	auto best = Clock::duration::max();
	auto outcome = std::string{"compiled"};
	for (auto run = std::size_t{0}; run < runs; ++run) {
		auto const began = Clock::now();
		auto const compiled = compile();
		best = std::min(best, Clock::now() - began);
		if (!compiled) {
			outcome = std::format("refused {}", std::to_underlying(compiled.error().refusal));
		} else {
			outcome = std::format("compiled: {} lexemes, {} rules, {} states", compiled->lexeme_count(), compiled->rule_count(), compiled->state_count());
		}
	}
	std::println("{:<44} best of {}: {:>9.1f}us  {}", name, runs, microseconds(best), outcome);
}

[[nodiscard]] auto many_properties(std::size_t count) -> std::string {
	auto out = std::string{R"({"type": "object", "properties": {)"};
	for (auto index = std::size_t{0}; index < count; ++index) {
		out.append(std::format(R"({}"p{}": {{"type": "{}"}})", index == 0 ? "" : ", ", index, index % 3 == 0 ? "string" : index % 3 == 1 ? "integer" : "boolean"));
	}
	out.append(R"(}, "additionalProperties": false})");
	return out;
}

[[nodiscard]] auto many_enum_values(std::size_t count) -> std::string {
	auto out = std::string{R"({"type": "object", "properties": {"e": {"enum": [)"};
	for (auto index = std::size_t{0}; index < count; ++index) {
		out.append(std::format(R"({}"value_{}")", index == 0 ? "" : ", ", index));
	}
	out.append(R"(]}}, "required": ["e"], "additionalProperties": false})");
	return out;
}

[[nodiscard]] auto deep(std::size_t depth) -> std::string {
	auto out = std::string{};
	for (auto level = std::size_t{0}; level < depth; ++level) {
		out.append(R"({"type": "object", "properties": {"x": )");
	}
	out.append(R"({"type": "string"})");
	for (auto level = std::size_t{0}; level < depth; ++level) {
		out.append(R"(}, "required": ["x"], "additionalProperties": false})");
	}
	return out;
}

[[nodiscard]] auto many_rules(std::size_t count) -> std::string {
	auto out = std::string{"start: r0\n"};
	for (auto index = std::size_t{0}; index + 1 < count; ++index) {
		out.append(std::format("r{}: \"w{}\" r{} | /[a-z]{{1,8}}/\n", index, index, index + 1));
	}
	out.append(std::format("r{}: \"end\"\n", count - 1));
	return out;
}

[[nodiscard]] auto tool(std::string name, std::string schema) -> grammar::Tool {
	return grammar::FunctionTool{.name = std::move(name), .parameters = std::move(schema)};
}

[[nodiscard]] auto request(std::vector<grammar::Tool> tools, grammar::ToolChoice choice, grammar::ContentFormat format, grammar::Reasoning reasoning) -> grammar::OutputRequest {
	return {.reasoning = reasoning, .tools = std::move(tools), .tool_choice = std::move(choice), .parallel = grammar::Parallel::Many, .format = std::move(format)};
}

inline constexpr std::string_view weather =
    R"({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 14}, "units": {"enum": ["celsius", "fahrenheit"]}}, "required": ["city", "days", "units"], "additionalProperties": false})";
inline constexpr std::string_view answer =
    R"({"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["answer", "confidence", "tags"], "additionalProperties": false})";

}

auto main() -> int {
	auto const tokenizer = engine::tokenizer::load().value();
	auto const began = Clock::now();
	auto const vocabulary = grammar::engine_vocabulary(tokenizer);
	std::println("vocabulary built in {:.1f} ms", microseconds(Clock::now() - began) / 1000.0);
	auto const walker = support::make_walker(tokenizer);

	std::println("\ncompile times (the http boundary compiles every request):");
	auto const plain = request({}, grammar::Calls::None, grammar::PlainText{}, grammar::Reasoning::Thinking);
	time_compile("plain chat, thinking", 20, [&] { return grammar::compile(plain); });
	auto const tools = request({tool("get_weather", std::string{weather}), grammar::CustomTool{.name = "calc", .input = grammar::LarkInput{.definition = "start: NUMBER (\"+\" NUMBER)*\n%import common.NUMBER\n"}}},
	                           grammar::Calls::Auto, grammar::PlainText{}, grammar::Reasoning::Thinking);
	time_compile("two tools, auto", 20, [&] { return grammar::compile(tools); });
	time_compile("json schema content", 20, [&] { return grammar::compile(request({}, grammar::Calls::None, grammar::JsonSchema{.schema = std::string{answer}}, grammar::Reasoning::Direct)); });
	auto const wide = many_properties(grammar::request_bounds.properties);
	time_compile("5000 properties (the bound)", 3, [&] { return grammar::compile(request({}, grammar::Calls::None, grammar::JsonSchema{.schema = wide}, grammar::Reasoning::Direct)); });
	time_compile("5000 properties as tool parameters", 3, [&] { return grammar::compile(request({tool("wide", wide)}, grammar::Calls::Auto, grammar::PlainText{}, grammar::Reasoning::Direct)); });
	auto const enums = many_enum_values(grammar::request_bounds.enum_values);
	time_compile("2000 enum values (the bound)", 3, [&] { return grammar::compile(request({}, grammar::Calls::None, grammar::JsonSchema{.schema = enums}, grammar::Reasoning::Direct)); });
	auto const nested = deep(grammar::request_bounds.schema_depth - 1);
	time_compile("schema nested 31 deep", 3, [&] { return grammar::compile(request({}, grammar::Calls::None, grammar::JsonSchema{.schema = nested}, grammar::Reasoning::Direct)); });
	auto const rules = many_rules(grammar::request_bounds.lark_rules - 1);
	time_compile("lark with 1000 definitions (the bound)", 3, [&] { return grammar::compile(request({grammar::CustomTool{.name = "big", .input = grammar::LarkInput{.definition = rules}}}, grammar::Calls::Auto, grammar::PlainText{}, grammar::Reasoning::Direct)); });
	time_compile("string of maxLength 1000", 3, [&] { return grammar::compile_schema(R"({"type": "string", "maxLength": 1000})"); });
	time_compile("string of maxLength 5000 (past the bound)", 3, [&] { return grammar::compile_schema(R"({"type": "string", "maxLength": 5000})"); });
	time_compile("regex of 16 KiB", 3, [&] {
		auto text = std::string{};
		while (text.size() + 16 < grammar::request_bounds.regex_bytes) {
			text.append("(?:ab|cd)[0-9]x");
		}
		return grammar::compile_regex(text);
	});

	std::println("\nmask latency over random walks (real tokenizer, {} tokens):", engine::tokenizer::vocabulary);
	auto const scenarios = std::to_array<std::pair<std::string_view, grammar::OutputRequest>>({
	    {"reasoning and free content", plain},
	    {"thinking, tools auto", tools},
	    {"tool call required", request({tool("get_weather", std::string{weather})}, grammar::Calls::Required, grammar::PlainText{}, grammar::Reasoning::Direct)},
	    {"json schema content", request({}, grammar::Calls::None, grammar::JsonSchema{.schema = std::string{answer}}, grammar::Reasoning::Direct)},
	    {"json object content", request({}, grammar::Calls::None, grammar::JsonObject{}, grammar::Reasoning::Direct)},
	});
	auto everything = std::vector<Clock::duration>{};
	for (auto const& [name, scenario] : scenarios) {
		auto compiled = grammar::compile(scenario);
		engine::core::invariant(compiled.has_value());
		auto samples = std::vector<Clock::duration>{};
		for (auto seed = std::uint64_t{1}; seed <= 10; ++seed) {
			auto const taken = support::walk(*compiled, vocabulary, walker, seed, 200, 600).value();
			std::ranges::copy(taken.masks, std::back_inserter(samples));
		}
		std::ranges::copy(samples, std::back_inserter(everything));
		percentiles(name, std::move(samples));
	}
	percentiles("all positions", std::move(everything));
	return 0;
}

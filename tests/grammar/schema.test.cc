// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.grammar;
import engine.grammar_support;
import engine.testing;

namespace {

namespace grammar = engine::grammar;
namespace support = engine::grammar_support;

using Texts = std::vector<std::string_view>;

/** The schema's grammar accepts exactly the texts listed as valid among those listed (`FINAL_REJECT:` for prefixes). */
[[nodiscard]] auto behaves(grammar::Vocabulary const& bytes, std::string_view schema, Texts const& accepted, Texts const& rejected) -> bool {
	auto compiled = grammar::compile_schema(schema);
	if (!compiled) {
		std::println("  refused: {} at {:?}", std::to_underlying(compiled.error().refusal), compiled.error().at);
		return false;
	}
	return support::agrees(*compiled, bytes, accepted, rejected);
}

[[nodiscard]] auto refuses(std::string_view schema, grammar::Refusal refusal) -> bool {
	auto const compiled = grammar::compile_schema(schema);
	if (compiled || compiled.error().refusal != refusal) {
		std::println("  expected refusal {}, got {}", std::to_underlying(refusal), compiled ? -1 : std::to_underlying(compiled.error().refusal));
		return false;
	}
	return true;
}

/** Weights that steer a byte walk through JSON: toward closing quotes and brackets, digits and short words. */
[[nodiscard]] auto weight(std::uint8_t byte, double urgency) -> double {
	if (byte == '"' || byte == '}' || byte == ']') {
		return 4.0 + (40.0 * urgency);
	}
	if (byte == ',' || byte == ':' || byte == '{' || byte == '[') {
		return 4.0;
	}
	if ((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'z') || byte == '-' || byte == '.') {
		return 2.0;
	}
	if (byte == ' ' || byte == '\n') {
		return 0.5;
	}
	return byte < 0x80 ? 0.3 : 0.05;
}

/**
 * Random byte walks under the masks end, and every text they end with is JSON the independent validator finds
 * valid for the schema. The walk picks among allowed bytes by weight, so it reaches objects, arrays and numbers.
 */
[[nodiscard]] auto walks_validate(grammar::Vocabulary const& bytes, std::string_view schema, std::uint64_t seed, std::size_t walks) -> bool {
	auto compiled = grammar::compile_schema(schema);
	auto const document = support::parse_json(schema);
	if (!compiled || !document) {
		std::println("  did not compile");
		return false;
	}
	auto random = engine::testing::generator(seed);
	for (auto walk = std::size_t{0}; walk < walks; ++walk) {
		auto matcher = grammar::Matcher::make(*compiled, support::byte_layout);
		auto text = std::string{};
		auto ended = false;
		for (auto step = 0; step < 3000 && !ended; ++step) {
			auto const mask = matcher.mask(bytes);
			auto weights = std::vector<double>(257, 0.0);
			auto const urgency = static_cast<double>(step) / 300.0;
			for (auto byte = 0U; byte < 256U; ++byte) {
				weights[byte] = mask->allows(byte) ? weight(static_cast<std::uint8_t>(byte), urgency) : 0.0;
			}
			weights[support::byte_end] = mask->allows(support::byte_end) ? 1.0 + (100.0 * urgency) : 0.0;
			if (std::ranges::all_of(weights, [](double w) { return w == 0.0; })) {
				std::println("  stuck after {:?}", text);
				return false;
			}
			auto const chosen = static_cast<grammar::Token>(std::discrete_distribution<std::size_t>{weights.begin(), weights.end()}(random));
			if (!matcher.advance(bytes, chosen)) {
				return false;
			}
			ended = chosen == support::byte_end;
			if (!ended) {
				text.push_back(static_cast<char>(static_cast<std::uint8_t>(chosen)));
			}
		}
		auto const instance = support::parse_json(text);
		if (!ended || !instance || !support::validates(*document, *document, *instance)) {
			std::println("  {} {:?}", ended ? "invalid" : "did not end", text);
			return false;
		}
	}
	return true;
}

/**
 * A length bound is counted beside the automaton, not written into it: a bound of any size up to the schema limit
 * compiles to an automaton no larger than the unbounded string's, in content and in a tool argument alike.
 */
[[nodiscard]] auto bounds_do_not_grow_automata() -> bool {
	auto const unbounded = grammar::compile_schema(R"({"type": "string"})");
	auto const huge = grammar::compile_schema(R"({"type": "string", "minLength": 3, "maxLength": 100000})");
	auto const argument = grammar::compile(grammar::OutputRequest{
	    .reasoning = grammar::Reasoning::Direct,
	    .tools = {grammar::FunctionTool{.name = "f", .parameters = R"({"type": "object", "properties": {"s": {"type": "string", "maxLength": 100000}}, "additionalProperties": false})"}},
	    .tool_choice = grammar::Calls::Auto,
	    .parallel = grammar::Parallel::Many,
	    .format = grammar::PlainText{}});
	if (!unbounded || !huge || !argument) {
		std::println("  a bounded string was refused");
		return false;
	}
	// Counting adds the unit automaton's few states; the bound itself adds none.
	return huge->state_count() <= unbounded->state_count() + 32 && argument->state_count() < 2000;
}

/** A string exactly `count` characters long, as JSON text. */
[[nodiscard]] auto string_of(std::size_t count) -> std::string {
	return "\"" + std::string(count, 'a') + "\"";
}

/** The bounds hold exactly far past the counts the remaining-count table holds bit by bit. */
[[nodiscard]] auto long_bounds_are_exact(grammar::Vocabulary const& bytes) -> bool {
	auto compiled = grammar::compile_schema(R"({"type": "string", "minLength": 700, "maxLength": 1200})");
	if (!compiled) {
		return false;
	}
	auto const verdict = [&](std::string const& text) { return support::judge(*compiled, bytes, text); };
	return verdict(string_of(700)) == support::Verdict::Accepted && verdict(string_of(1200)) == support::Verdict::Accepted &&
	       verdict(string_of(950)) == support::Verdict::Accepted && verdict(string_of(699)) == support::Verdict::Refused &&
	       verdict(string_of(1201)) == support::Verdict::Refused && verdict("\"" + std::string(699, 'a')) == support::Verdict::Prefix;
}

inline constexpr std::string_view person = R"({
  "type": "object",
  "properties": {
    "name": {"type": "string", "minLength": 1, "maxLength": 12},
    "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "email": {"type": "string", "format": "email"},
    "born": {"type": "string", "format": "date"},
    "score": {"type": "number", "exclusiveMinimum": -1.5, "maximum": 99.25},
    "tags": {"type": "array", "items": {"type": "string", "pattern": "^[a-z]{1,5}$"}, "minItems": 1, "maxItems": 3},
    "kind": {"enum": ["a", "b", 3, null]},
    "nick": {"type": ["string", "null"]},
    "id": {"type": "string", "format": "uuid"}
  },
  "required": ["name", "age", "tags"],
  "additionalProperties": false
})";

inline constexpr std::string_view tree = R"({
  "$defs": {"node": {"type": "object", "properties": {"value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "maxItems": 2}}, "required": ["value"], "additionalProperties": false}},
  "$ref": "#/$defs/node"
})";

inline constexpr std::string_view choices = R"({
  "anyOf": [
    {"type": "object", "properties": {"a": {"const": 1}}, "required": ["a"], "additionalProperties": false},
    {"type": "array", "items": {"type": "boolean"}, "maxItems": 4},
    {"type": "string", "format": "time"},
    {"type": "number", "minimum": 10, "exclusiveMaximum": 10.5}
  ]
})";

[[nodiscard]] auto nested(std::size_t depth) -> std::string {
	auto out = std::string{};
	for (auto level = std::size_t{0}; level < depth; ++level) {
		out.append(R"({"type": "array", "items": )");
	}
	out.append("true");
	out.append(std::string(depth, '}'));
	return out;
}

inline constexpr std::string_view open = R"({"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": {"type": "string", "maxLength": 3}})";

}

auto main() -> int {
	auto const bytes = support::byte_vocabulary();
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "integers within bounds, written as JSON writes them",
	     .holds = behaves(bytes, R"({"type": "integer", "minimum": -12, "exclusiveMaximum": 105})", {"-12", "0", "7", "99", "104"}, {"-13", "105", "1000", "07", "-0", "1.0", "FINAL_REJECT:-"})},
	    {.name = "integers past every written bound are no integers the engine writes", .holds = refuses(R"({"type": "integer", "minimum": 1e30})", grammar::Refusal::Empty)},
	    {.name = "reals within inclusive bounds",
	     .holds = behaves(bytes, R"({"type": "number", "minimum": -2.5, "maximum": 3.25})", {"-2.5", "-2.49", "0", "3.25", "3.2500", "1.125", "-0.5"},
	                      {"-2.51", "3.26", "3.250001", "4", "-3", "FINAL_REJECT:-0", "1e2"})},
	    {.name = "reals within exclusive bounds",
	     .holds = behaves(bytes, R"({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1})", {"0.5", "0.000001", "0.999"}, {"FINAL_REJECT:0", "FINAL_REJECT:0.0", "1", "1.0", "FINAL_REJECT:0.000", "-0.5", "2"})},
	    {.name = "unbounded numbers take exponents and no negative zero",
	     .holds = behaves(bytes, R"({"type": "number"})", {"1e5", "-2.5E-3", "0", "123.456"}, {"FINAL_REJECT:-0", "FINAL_REJECT:-0.0", "1e100", "01", "FINAL_REJECT:1."})},
	    {.name = "strings: lengths count characters", .holds = behaves(bytes, R"({"type": "string", "minLength": 2, "maxLength": 3})", {R"("ab")", R"("é😀")", R"("abc")"}, {R"("abcd")", R"("a")"})},
	    {.name = "strings: a length bound counts each character once, escapes and multi-byte ones included",
	     .holds = behaves(bytes, R"({"type": "string", "minLength": 2, "maxLength": 3})", {R"("é😀")", R"("\n\t")", R"("a\u001eb")", R"("ab")", R"("\"\\\/")"},
	                      {R"("a")", R"("\u001e")", R"("abcd")", R"("é😀éa")", R"("\n\n\n\n")", "FINAL_REJECT:\"a"})},
	    {.name = "strings: large bounds are counted, not built into automata", .holds = bounds_do_not_grow_automata()},
	    {.name = "strings: bounds far past the counting table are exact", .holds = long_bounds_are_exact(bytes)},
	    {.name = "strings: escapes only for control characters",
	     .holds = behaves(bytes, R"({"type": "string"})", {R"("a\"b\\c\n")", R"("\u001e")", R"("é")"}, {R"("\u00e9")", "\"\x01\"", R"("\x")"})},
	    {.name = "strings: patterns search unless anchored",
	     .holds = behaves(bytes, R"({"type": "object", "properties": {"a": {"type": "string", "pattern": "b[0-9]"}, "c": {"type": "string", "pattern": "^x+$"}}, "required": ["a", "c"], "additionalProperties": false})",
	                      {R"({"a": "ab1c", "c": "xx"})", R"({"a":"b9","c":"x"})"}, {R"({"a": "abc", "c": "x"})", R"({"a": "b1", "c": "xy"})", R"({"c": "x", "a": "b1"})"})},
	    {.name = "formats: dates with real calendars",
	     .holds = behaves(bytes, R"({"type": "string", "format": "date"})", {R"("2024-02-29")", R"("2000-02-29")", R"("2023-12-31")"}, {R"("2023-02-29")", R"("1900-02-29")", R"("2023-04-31")", R"("2023-13-01")"})},
	    {.name = "formats: date-times and times with offsets",
	     .holds = behaves(bytes, R"({"type": "string", "format": "date-time"})", {R"("2024-01-02T03:04:05Z")", R"("2024-01-02T23:59:59.123+05:30")"}, {R"("2024-01-02T24:00:00Z")", R"("2024-01-02T03:04:05")"})},
	    {.name = "objects: properties in order, optional ones skippable",
	     .holds = behaves(bytes, R"({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}}, "required": ["b"], "additionalProperties": false})",
	                      {R"({"a": 1, "b": true})", R"({"b": false})", "{\n  \"b\": true\n}"}, {"{}", R"({"b": true, "a": 1})", R"({"a": 1})", R"({"b": true, "c": 1})"})},
	    {.name = "objects: additional properties by their schema",
	     .holds = behaves(bytes, open, {R"({"x": 1, "y": "abc"})", R"({"y": "a", "z": ""})", "{}"}, {R"({"y": "abcd"})", R"({"y": 1})", R"({"x": "a"})"})},
	    {.name = "enums and consts of any JSON value",
	     .holds = behaves(bytes, R"({"enum": [1, "two", [3], {"four": 4}, null, true]})", {"1", R"("two")", "[3]", R"({"four": 4})", "null", "true"}, {"2", R"("three")", "false"})},
	    {.name = "type lists and nullable",
	     .holds = behaves(bytes, R"({"type": ["integer", "null"], "minimum": 5})", {"5", "null", "99"}, {"FINAL_REJECT:4", "-4", R"("5")"}) &&
	              behaves(bytes, R"({"type": "string", "nullable": true})", {R"("a")", "null"}, {"1"})},
	    {.name = "oneOf over disjoint types",
	     .holds = behaves(bytes, R"({"oneOf": [{"type": "string"}, {"type": "integer"}]})", {R"("a")", "3"}, {"true", "1.5"})},
	    {.name = "recursion through $defs", .holds = behaves(bytes, tree, {R"({"value": 1})", R"({"value": 1, "children": [{"value": 2, "children": []}, {"value": 3}]})"}, {R"({"value": 1, "children": [{}]})"})},
	    {.name = "arrays: item counts",
	     .holds = behaves(bytes, R"({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3})", {"[1, 2]", "[1,2,3]", "[\n  1,\n  2\n]"}, {"[1]", "[1, 2, 3, 4]", "[]"})},
	    {.name = "whitespace: one space, or a newline and bounded indentation",
	     .holds = behaves(bytes, R"({"type": "array", "items": {"type": "integer"}})", {"[1, 2]", "[1,2]", "[\n\t1\n]", "[ ]"}, {"[1,  2]", "[\n\n1]"})},
	    {.name = "random walks validate: a person", .holds = walks_validate(bytes, person, 1, 20)},
	    {.name = "random walks validate: a recursive tree", .holds = walks_validate(bytes, tree, 2, 20)},
	    {.name = "random walks validate: unions", .holds = walks_validate(bytes, choices, 3, 30)},
	    {.name = "random walks validate: open objects", .holds = walks_validate(bytes, open, 4, 20)},
	    {.name = "random walks validate: any JSON", .holds = walks_validate(bytes, "true", 5, 20)},
	    {.name = "multipleOf is refused", .holds = refuses(R"({"type": "integer", "multipleOf": 3})", grammar::Refusal::SchemaMultipleOf)},
	    {.name = "unknown keywords are refused", .holds = refuses(R"({"type": "string", "contentEncoding": "base64"})", grammar::Refusal::SchemaKeyword)},
	    {.name = "allOf is refused", .holds = refuses(R"({"allOf": [{"type": "string"}]})", grammar::Refusal::SchemaKeyword)},
	    {.name = "unknown types are refused", .holds = refuses(R"({"type": "text"})", grammar::Refusal::SchemaType)},
	    {.name = "remote references are refused", .holds = refuses(R"({"$ref": "https://example.com/s.json"})", grammar::Refusal::SchemaReference)},
	    {.name = "dangling references are refused", .holds = refuses(R"({"$ref": "#/$defs/missing"})", grammar::Refusal::SchemaReference)},
	    {.name = "overlapping oneOf is refused", .holds = refuses(R"({"oneOf": [{"type": "integer"}, {"type": "number"}]})", grammar::Refusal::SchemaOneOf)},
	    {.name = "unsupported formats are refused", .holds = refuses(R"({"type": "string", "format": "uri"})", grammar::Refusal::SchemaFormat)},
	    {.name = "unsupported pattern syntax is refused", .holds = refuses(R"({"type": "string", "pattern": "(?<=a)b"})", grammar::Refusal::Regex)},
	    {.name = "an exclusive bound past 15 digits is refused", .holds = refuses(R"({"type": "number", "exclusiveMinimum": 0.1234567890123456})", grammar::Refusal::SchemaBound)},
	    {.name = "a schema no value satisfies is refused", .holds = refuses(R"({"type": "integer", "minimum": 5, "maximum": 4})", grammar::Refusal::Empty)},
	    {.name = "a required property the schema does not describe is refused",
	     .holds = refuses(R"({"type": "object", "properties": {}, "required": ["a"]})", grammar::Refusal::SchemaKeyword)},
	    {.name = "schemas nested past the bound are refused", .holds = refuses(nested(40), grammar::Refusal::SchemaTooLarge)},
	});
	return engine::testing::report(cases);
}

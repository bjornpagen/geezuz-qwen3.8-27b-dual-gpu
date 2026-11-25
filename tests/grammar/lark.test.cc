// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.grammar;
import engine.grammar_support;
import engine.testing;

namespace {

namespace grammar = engine::grammar;
namespace support = engine::grammar_support;

using Texts = std::vector<std::string_view>;

/** The grammar accepts and refuses exactly as the cases say (llguidance's form, `FINAL_REJECT:` for prefixes). */
[[nodiscard]] auto behaves(grammar::Vocabulary const& bytes, std::string_view lark, Texts const& accepted, Texts const& rejected) -> bool {
	auto compiled = grammar::compile_lark(lark);
	if (!compiled) {
		std::println("  refused: {} at {:?}", std::to_underlying(compiled.error().refusal), compiled.error().at);
		return false;
	}
	return support::agrees(*compiled, bytes, accepted, rejected);
}

[[nodiscard]] auto refuses(std::string_view lark, grammar::Refusal refusal) -> bool {
	auto const compiled = grammar::compile_lark(lark);
	if (compiled) {
		std::println("  compiled, expected refusal {}", std::to_underlying(refusal));
		return false;
	}
	if (compiled.error().refusal != refusal) {
		std::println("  refused with {}, expected {}", std::to_underlying(compiled.error().refusal), std::to_underlying(refusal));
		return false;
	}
	return true;
}

}

auto main() -> int {
	auto const bytes = support::byte_vocabulary();
	// Cases taken from llguidance's own lark tests and syntax examples, where the engine
	// implements the feature they test.
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "dot matches one character, multi-byte ones included",
	     .holds = behaves(bytes, R"(start: /.../ "abc" /.../)", {"abcabcabc", "aaaabcccc", "🔵🟠✅abc❌🟠🔵"},
	                      {"aaabcccc", "aaaaabcccc", "FINAL_REJECT:aaaabccc", "aaaabccccc", "🔵🟠✅❌abc❌✅🟠🔵", "🔵🟠abc🟠🔵"})},
	    {.name = "bounded repetition of a rule",
	     .holds = behaves(bytes, "start:  ab{3,5}\nab:  \"a\" | \"b\"\n", {"aba", "abaa", "aaaaa", "aabaa"}, {"FINAL_REJECT:aa", "FINAL_REJECT:ab", "aaaaaa"})},
	    {.name = "repetition without an upper bound",
	     .holds = behaves(bytes, "start:  ab{3,}\nab:  \"a\" | \"b\"\n", {"aba", "abaa", "aaaaa", "aabaa", "aaaaaa"}, {"FINAL_REJECT:aa", "FINAL_REJECT:ab"})},
	    {.name = "repetition without a lower bound",
	     .holds = behaves(bytes, "start:  ab{,5}\nab:  \"a\" | \"b\"\n", {"", "aa", "b", "aba", "abaa", "aaaaa", "aabaa"}, {"aaaaaa"})},
	    {.name = "lark's ~ repetition of a terminal",
	     .holds = behaves(bytes, "start:  HI~0..2\nHI:  \"hi\"\n", {"", "hi", "hihi"}, {"hihihi", "hi hi"})},
	    {.name = "a lexeme two lexemes match is scanned as both",
	     .holds = behaves(bytes, "start: \"'foo'\" /a+/ | STRING /b+/\nSTRING: /'[^']*'/\n", {"'foo'a", "'foo'aaa", "'bar'b", "'bar'bbb", "'foo'bb"},
	                      {"'bar'a", "'bar'c"})},
	    {.name = "a lazy rule ends at its first match",
	     .holds = behaves(bytes, "start: with_lazy\nwith_lazy[lazy]: /(.|\\n)*<end>/\n", {"foo<end>", "<end>"}, {"foo<end>bar<end>", "FINAL_REJECT:foo<en"})},
	    {.name = "a suffix rule ends at its suffix's first match",
	     .holds = behaves(bytes, "start: with_suffix \"!\"\nwith_suffix[suffix=\"<end>\"]: /.*/\n", {"foo<end>!", "<end>!"}, {"foo<end>bar<end>!"})},
	    {.name = "lazy heads let tool calls interleave with text",
	     .holds = behaves(bytes,
	                      "start: ( f_foo | f_bar )* f_end\n"
	                      "f_end: TEXT\n"
	                      "TEXT: /(.|\\n)*/\n"
	                      "f_foo_hd[lazy]: TEXT \"<function\"\n"
	                      "f_foo: f_foo_hd \"=foo>\" %json { \"type\": \"object\" } \"</function>\"\n"
	                      "f_bar_hd[lazy]: TEXT \"<function\"\n"
	                      "f_bar: f_bar_hd \"=bar>\" /[0-9]+/ \"</function>\"\n",
	                      {"hello", "", "x<function=bar>123</function>tail", "a<function=foo>{}</function>", "<function=foo>{\"a\": 1}</function>b<function=bar>7</function>"},
	                      {"x<function=baz>", "x<function=bar>12a", "FINAL_REJECT:x<function=foo>{\"a\": 1}"})},
	    {.name = "ignored text may stand between any two lexemes and around the whole",
	     .holds = behaves(bytes, "start: \"a\" \"b\"+\n%ignore \" \"\n", {"ab", "a b", " a  b ", "a b b"}, {"ab c", "FINAL_REJECT: a "})},
	    {.name = "common terminals import, with an alias",
	     .holds = behaves(bytes, "start: NUM (\",\" NUM)*\n%import common.NUMBER -> NUM\n%import common.WS\n%ignore WS\n", {"1", "1, 2.5,3e4", " 7 "}, {"1,,2", "FINAL_REJECT:1,"})},
	    {.name = "intersection and complement in a terminal",
	     .holds = behaves(bytes, "start: ASCII_LINES\nASCII_LINES: /[a-zA-Z \\n]*/ & ~/(?s:.*)\\n\\n(?s:.*)/\n", {"ab\ncd", "", "a b"}, {"ab\n\ncd"})},
	    {.name = "case-insensitive literals and regex flags",
	     .holds = behaves(bytes, "start: \"select\"i \" \" /[a-z]+/i\n", {"SELECT abc", "Select ABC", "select x"}, {"selec x", "select 1"})},
	    {.name = "a schema embedded with %json, with integer bounds",
	     .holds = behaves(bytes, "start: %json {\"type\": \"integer\", \"minimum\": 3, \"maximum\": 12}\n", {"3", "12", "7"}, {"2", "13", "03", "-1"})},
	    {.name = "recursive rules: arithmetic",
	     .holds = behaves(bytes,
	                      "start: expr\nexpr: expr \"+\" term | term\nterm: term \"*\" factor | factor\nfactor: \"(\" expr \")\" | NUMBER\nNUMBER: /[0-9]+/\n",
	                      {"1", "1+2*3", "(1+2)*3", "((4))"}, {"FINAL_REJECT:1+", "FINAL_REJECT:(1+2", "()", "1++"})},
	    {.name = "a character range literal", .holds = behaves(bytes, "start: (\"a\"..\"f\")+\n", {"abc", "fed"}, {"g", "abg"})},
	    {.name = "multi-line alternatives continue with |",
	     .holds = behaves(bytes, "start: \"a\"\n     | \"b\"\n     | (\"c\"\n        \"d\")\n", {"a", "b", "cd"}, {"FINAL_REJECT:c", "ab"})},
	    {.name = "no start rule is refused", .holds = refuses("root: \"abc\"\n", grammar::Refusal::LarkNoStart)},
	    {.name = "terminal priorities are refused", .holds = refuses("start: A\nA.2: \"a\"\n", grammar::Refusal::LarkPriority)},
	    {.name = "special tokens are refused", .holds = refuses("start: <|im_end|>\n", grammar::Refusal::LarkToken)},
	    {.name = "grammar references are refused", .holds = refuses("start: @sub\n", grammar::Refusal::LarkToken)},
	    {.name = "lookaround is refused", .holds = refuses("start: /a(?=b)/\n", grammar::Refusal::Regex)},
	    {.name = "lazy quantifiers are refused", .holds = refuses("start: /a*?/\n", grammar::Refusal::Regex)},
	    {.name = "%declare is refused", .holds = refuses("%declare FOO\nstart: \"a\"\n", grammar::Refusal::LarkDirective)},
	    {.name = "%import of anything but common is refused", .holds = refuses("%import other.FOO\nstart: \"a\"\n", grammar::Refusal::LarkDirective)},
	    {.name = "stop is refused", .holds = refuses("start: foo\nfoo[stop=\"x\"]: /.*/\n", grammar::Refusal::LarkOption)},
	    {.name = "max_tokens is refused", .holds = refuses("start: foo\nfoo[max_tokens=5]: /.*/\n", grammar::Refusal::LarkOption)},
	    {.name = "an undefined rule is refused", .holds = refuses("start: foo\n", grammar::Refusal::LarkUndefined)},
	    {.name = "an undefined terminal is refused", .holds = refuses("start: FOO\n", grammar::Refusal::LarkUndefined)},
	    {.name = "a recursive terminal is refused", .holds = refuses("start: A\nA: \"a\" A?\n", grammar::Refusal::LarkRecursiveTerminal)},
	    {.name = "& in a rule is refused", .holds = refuses("start: \"a\" & \"b\"\n", grammar::Refusal::LarkOperator)},
	    {.name = "~ in a rule is refused", .holds = refuses("start: ~\"a\"\n", grammar::Refusal::LarkOperator)},
	    {.name = "templates are refused", .holds = refuses("start: pair{\"a\"}\npair{x}: x x\n", grammar::Refusal::LarkTemplate)},
	    {.name = "a reversed character range is refused", .holds = refuses("start: \"d\"..\"a\"\n", grammar::Refusal::LarkSyntax)},
	    {.name = "a case-insensitive literal past ascii is refused", .holds = refuses("start: \"é\"i\n", grammar::Refusal::Regex)},
	    {.name = "an unterminated string is refused", .holds = refuses("start: \"abc\n", grammar::Refusal::LarkSyntax)},
	    {.name = "a grammar with no finite derivation is refused", .holds = refuses("start: a\na: \"x\" a\n", grammar::Refusal::Empty)},
	});
	return engine::testing::report(cases);
}

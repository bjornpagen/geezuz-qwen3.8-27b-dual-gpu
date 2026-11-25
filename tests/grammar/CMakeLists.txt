# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_grammar_support)
target_sources(engine_grammar_support PUBLIC FILE_SET CXX_MODULES FILES support.cc)
target_link_libraries(engine_grammar_support PUBLIC engine_language_profile engine_core engine_grammar engine_json engine_testing engine_tokenizer)

engine_test(grammar.regex SOURCES regex.test.cc LINK engine_regex)
engine_test(grammar.lark SOURCES lark.test.cc LINK engine_grammar engine_grammar_support)
engine_test(grammar.schema SOURCES schema.test.cc LINK engine_core engine_grammar engine_grammar_support)
engine_test(grammar.mask SOURCES mask.test.cc LINK engine_core engine_grammar engine_regex)
engine_test(grammar.output SOURCES output.test.cc LINK engine_core engine_chat engine_grammar engine_grammar_support engine_json engine_tokenizer)

# A benchmark, not a test: built with the tests and run by hand in `release`, so it never runs by default.
add_executable(engine_bench_grammar bench.cc)
target_link_libraries(engine_bench_grammar PRIVATE engine_language_profile engine_core engine_grammar engine_grammar_support engine_tokenizer)

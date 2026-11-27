# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_grammar)
target_sources(engine_grammar PUBLIC FILE_SET CXX_MODULES FILES
	grammar.cc error.cc builder.cc counting.cc compiled.cc lexer.cc parser.cc tokens.cc matcher.cc document.cc numbers.cc strings.cc schema.cc arguments.cc lark_syntax.cc lark.cc output.cc engine_tokens.cc)
target_link_libraries(engine_grammar PUBLIC engine_language_profile engine_chat engine_core engine_regex engine_json engine_model engine_tokenizer engine_vocabulary)

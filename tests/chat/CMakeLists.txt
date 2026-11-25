# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_test_chat_fixtures)
target_sources(engine_test_chat_fixtures PUBLIC FILE_SET CXX_MODULES FILES fixtures.cc)
target_link_libraries(engine_test_chat_fixtures PUBLIC engine_language_profile engine_core engine_json engine_tokenizer engine_chat engine_chat_fixtures)

engine_test(chat.render SOURCES render.test.cc LINK engine_chat engine_json engine_tokenizer engine_test_chat_fixtures)
engine_test(chat.reply SOURCES reply.test.cc LINK engine_chat engine_json engine_simdjson engine_tokenizer engine_test_chat_fixtures)
engine_test(chat.numbers SOURCES numbers.test.cc LINK engine_chat engine_json engine_tokenizer engine_test_chat_fixtures)
engine_test(chat.conversation SOURCES conversation.test.cc LINK engine_chat engine_json)
engine_checkpoint_test(chat.checkpoint SOURCES checkpoint.test.cc LINK engine_file engine_sha256 engine_test_chat_fixtures)

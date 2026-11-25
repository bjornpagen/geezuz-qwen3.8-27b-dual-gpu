# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_chat)
target_sources(engine_chat PUBLIC FILE_SET CXX_MODULES FILES chat.cc text.cc dumps.cc cast.cc conversation.cc render.cc reply.cc)
target_link_libraries(engine_chat PUBLIC engine_language_profile engine_core engine_json engine_tokenizer)

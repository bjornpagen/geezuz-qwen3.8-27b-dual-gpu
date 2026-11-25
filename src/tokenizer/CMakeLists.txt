# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_tokenizer)
target_sources(engine_tokenizer PUBLIC FILE_SET CXX_MODULES FILES
	tokenizer.cc added.cc utf8.cc unicode.cc split.cc bpe.cc image.cc codec.cc)
target_link_libraries(engine_tokenizer PUBLIC engine_language_profile engine_core engine_model engine_tokenizer_image)

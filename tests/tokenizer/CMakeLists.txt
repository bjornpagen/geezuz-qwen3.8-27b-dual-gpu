# SPDX-License-Identifier: AGPL-3.0-only
engine_test(tokenizer SOURCES tokenizer.test.cc LINK engine_tokenizer engine_tokenizer_fixtures)
target_sources(engine_test_tokenizer PRIVATE FILE_SET CXX_MODULES FILES fixtures.cc)
engine_checkpoint_test(tokenizer.checkpoint SOURCES checkpoint.test.cc LINK engine_tokenizer engine_file)

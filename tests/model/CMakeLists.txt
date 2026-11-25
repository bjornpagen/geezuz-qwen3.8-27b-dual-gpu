# SPDX-License-Identifier: AGPL-3.0-only
engine_test(model SOURCES model.test.cc LINK engine_model)

engine_test(model.draft_vocabulary SOURCES draft_vocabulary.test.cc LINK engine_model engine_file)
# The draft head's row count is checked against the vocabulary file it gathers.
set_tests_properties(model.draft_vocabulary PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/data")

engine_checkpoint_test(model.checkpoint SOURCES checkpoint.test.cc LINK engine_model engine_file engine_json engine_safetensors)

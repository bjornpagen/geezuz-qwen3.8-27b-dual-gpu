# SPDX-License-Identifier: AGPL-3.0-only
# The oracle side of the json tests: a plain model of JSON values, its own serializer, generator and measures.
add_library(engine_json_model)
target_sources(engine_json_model PUBLIC FILE_SET CXX_MODULES FILES model.cc)
target_link_libraries(engine_json_model PUBLIC engine_language_profile engine_core engine_json)

engine_test(json.read SOURCES read.test.cc LINK engine_json engine_json_model)
engine_test(json.bounds SOURCES bounds.test.cc LINK engine_json engine_json_model)
engine_test(json.write SOURCES write.test.cc LINK engine_json engine_json_model)
engine_test(json.numbers SOURCES numbers.test.cc LINK engine_json engine_json_model)
engine_test(json.utf8 SOURCES utf8.test.cc LINK engine_json engine_json_model)
engine_test(json.response SOURCES response.test.cc LINK engine_json engine_json_model)
engine_test(json.request SOURCES request.test.cc LINK engine_json engine_json_model)

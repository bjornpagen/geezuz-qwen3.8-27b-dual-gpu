// SPDX-License-Identifier: AGPL-3.0-only
//
// The model's sampled vocabulary is the tokenizer's: `tokenizer.checkpoint` pins the tokenizer's ids to the
// checkpoint's tokenizer.json, so together the two tests pin the padding rows to the ids no tokenizer can produce.
import std;
import engine.model;
import engine.testing;
import engine.tokenizer;

auto main() -> int {
	namespace model = engine::model;
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the sampled vocabulary is exactly the tokenizer's ids, and lm_head rows past it are padding",
	        .holds = model::sampled_vocabulary == engine::tokenizer::vocabulary && model::sampled_vocabulary < model::vocab,
	    },
	};
	return engine::testing::report(cases);
}

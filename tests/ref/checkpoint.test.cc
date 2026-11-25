// SPDX-License-Identifier: AGPL-3.0-only
//
// The reference on the real checkpoint: a short counting prompt, its greedy continuation, and the mtp head's drafts.
//
// "Sensible" is pinned against Hugging Face transformers' Qwen3.5 text model, run once on the cpu from this checkpoint
// with every e4m3 weight dequantized exactly per matmul (the fixture below): the prompt's ids are the reference
// tokenizer's, the greedy continuation is the same six tokens, and each chosen token's logit is within 0.2 of
// transformers'. Transformers keeps the residual stream and every activation in bf16 and accumulates in another order,
// so logits differ by a few hundredths (at most 0.058 on the run that checked this, against transformers' bf16 logits,
// quantized to 0.0625 near 16); at every step its best token leads the second by 3.75 logits or more, so no such
// difference can change a choice. The mtp head, given the trunk's "," after " 6", must draft the two
// tokens the trunk then decodes.
import std;
import engine.model;
import engine.ref;
import engine.testing;
import engine.tokenizer;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;

using model::Count;
using ref::Token;

// Captured from transformers 5.24.1 (Qwen3_5ForCausalLM, eager attention, torch cpu): "1, 2, 3, 4, 5," and its
// greedy continuation " 6, 7,", with the logit of each chosen token.
constexpr auto prompt_text = std::string_view{"1, 2, 3, 4, 5,"};
constexpr auto prompt_ids = std::array<Token, 14>{16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11};
constexpr auto continuation = std::array<Token, 6>{220, 21, 11, 220, 22, 11};
constexpr auto chosen_logits = std::array{15.0625F, 19.0F, 13.875F, 15.6875F, 19.5F, 14.6875F};
constexpr auto logit_tolerance = 0.2F;

struct Greedy {
	std::vector<Token> tokens;
	std::vector<float> logits;
	std::vector<Token> drafts;
	double seconds;
};

[[nodiscard]] auto greedy(ref::Model& model, std::span<Token const> prompt) -> Greedy {
	auto const started = std::chrono::steady_clock::now();
	auto sequence = ref::Sequence::empty();
	auto result = Greedy{.tokens = {}, .logits = {}, .drafts = {}, .seconds = 0.0};
	auto feed = std::vector<Token>{prompt.begin(), prompt.end()};
	for (auto step = Count{0}; step < continuation.size(); ++step) {
		auto const logits = ref::extend<ref::Tensor>(model, sequence, feed);
		auto const best = ref::summarize(logits.row(feed.size() - 1)).best.front();
		result.tokens.push_back(best.id);
		result.logits.push_back(best.logit);
		std::println("step {}: token {} logit {:.4f} ({:.1f} s so far)", step, best.id, best.logit,
		             std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
		if (step == 2) {
			result.drafts = ref::draft<ref::Tensor>(model, sequence, best.id, 2);
			std::println("drafts after step 2: {} {}", result.drafts[0], result.drafts[1]);
		}
		feed = {best.id};
	}
	result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	return result;
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 2) {
		std::println("FAIL: the checkpoint directory is the one argument");
		return 1;
	}
	auto const tokenizer = engine::tokenizer::load();
	auto const pieces = std::array{engine::tokenizer::Piece{engine::tokenizer::Text{.utf8 = prompt_text, .spelling = engine::tokenizer::Spelling::Characters}}};
	auto const encoded = tokenizer ? tokenizer->encode(pieces) : std::unexpected(engine::tokenizer::EncodeError{.piece = 0, .offset = 0});
	auto const opened_at = std::chrono::steady_clock::now();
	auto model = ref::Model::open(std::filesystem::path{arguments[1]});
	if (!model) {
		std::println("FAIL: the checkpoint opens (failure {}: {})", std::to_underlying(model.error().failure), model.error().subject);
		return 1;
	}
	std::println("opened and scanned in {:.1f} s", std::chrono::duration<double>(std::chrono::steady_clock::now() - opened_at).count());
	auto const run = greedy(*model, prompt_ids);
	std::println("{} rows ({} prompt, {} decoded one at a time, 2 drafts) in {:.1f} s", prompt_ids.size() + continuation.size() - 1,
	             prompt_ids.size(), continuation.size() - 1, run.seconds);
	auto close = true;
	for (auto const [got, want] : std::views::zip(run.logits, chosen_logits)) {
		std::println("logit {:.4f} against transformers' {:.4f}", got, want);
		close = close && std::abs(got - want) <= logit_tolerance;
	}
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the prompt encodes to transformers' ids",
	        .holds = encoded.has_value() && std::ranges::equal(*encoded, prompt_ids),
	    },
	    engine::testing::Case{
	        .name = "the greedy continuation is transformers' six tokens, token for token",
	        .holds = std::ranges::equal(run.tokens, continuation),
	    },
	    engine::testing::Case{.name = "each chosen token's logit is within 0.2 of transformers'", .holds = close},
	    engine::testing::Case{
	        .name = "given the trunk's comma after six, the mtp head drafts the two tokens the trunk then decodes",
	        .holds = run.drafts.size() == 2 && run.drafts[0] == continuation[3] && run.drafts[1] == continuation[4],
	    },
	};
	return engine::testing::report(cases);
}

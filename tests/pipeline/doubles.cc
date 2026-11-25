// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline_doubles;

import std;
import engine.card_scheduler;
import engine.core;
import engine.device;
import engine.exec;
import engine.http;
import engine.kv;
import engine.kv_fake;
import engine.model;
import engine.pipeline;
import engine.sampling;
import engine.sched;
import engine.tokenizer;

/**
 * Test doubles of the parts the pipeline composes, named as doubles: the cards (decode and prefill), the grammar and
 * the api in front. Together they let the engine loop run whole on a machine without cards, and each checks the
 * pipeline from its side: the cards double aborts on a committed token its verify row's mask forbids, on a mask that
 * allows the token the grammar forbids at that position, and on a pool command the content-checking kv device
 * rejects.
 */
namespace engine::pipeline_doubles {

namespace ex = engine::exec;
namespace pipeline = engine::pipeline;
using pipeline::Token;

/** The tokens the grammar double forbids, one per position in turn: none of them is ever scripted. */
export inline constexpr Token forbidden_base = 200'000;
export inline constexpr Token forbidden_period = 7;

/** The token the grammar double forbids after `taken` tokens of a completion. */
export [[nodiscard]] constexpr auto forbidden(std::size_t taken) -> Token {
	return forbidden_base + core::narrow<Token>(taken % forbidden_period);
}

/** Scripted tokens are drawn below this, far from the forbidden ones and from every added token. */
export inline constexpr Token script_range = 100'000;

inline constexpr Token end_of_turn = tokenizer::id(tokenizer::Added::im_end);

export struct TokenMaskDouble {
	std::array<std::uint64_t, pipeline::token_words> bits;

	[[nodiscard]] auto words() const -> std::span<std::uint64_t const, pipeline::token_words> {
		return bits;
	}
};

/**
 * A test double of a grammar's parser state: after k tokens it allows every token but `forbidden(k)`, so each
 * position's mask differs from its neighbours' and a mask computed for the wrong prefix is caught.
 */
export class GrammarDouble {
public:
	[[nodiscard]] auto mask() const -> TokenMaskDouble {
		auto mask = TokenMaskDouble{};
		mask.bits.fill(~std::uint64_t{0});
		auto const tail = model::sampled_vocabulary % 64;
		mask.bits.back() = tail == 0 ? ~std::uint64_t{0} : (std::uint64_t{1} << tail) - 1;
		auto const banned = forbidden(taken_);
		mask.bits[banned / 64] &= ~(std::uint64_t{1} << (banned % 64));
		return mask;
	}

	[[nodiscard]] auto advance(Token token) -> pipeline::Step {
		if (token == forbidden(taken_)) {
			return pipeline::Step::Refused;
		}
		taken_ += 1;
		return pipeline::Step::Taken;
	}

	auto rollback(std::size_t count) -> void {
		core::invariant(count <= taken_);
		taken_ -= count;
	}

private:
	std::size_t taken_ = 0;
};

static_assert(pipeline::GrammarState<GrammarDouble>);

/** What a test asks of the engine: a prompt, the script its completion follows (by seed), and its budget. */
export struct Ask {
	std::uint64_t tag;
	std::vector<Token> prompt;
	std::uint64_t seed;
	sched::Limit limit;
};

export struct Accepted {
	[[nodiscard]] constexpr auto operator==(Accepted const&) const -> bool = default;
};

export struct Refused {
	std::uint64_t prompt_tokens;

	[[nodiscard]] constexpr auto operator==(Refused const&) const -> bool = default;
};

export struct Committed {
	Token token;

	[[nodiscard]] constexpr auto operator==(Committed const&) const -> bool = default;
};

export struct Ended {
	pipeline::Finish finish;

	[[nodiscard]] constexpr auto operator==(Ended const&) const -> bool = default;
};

export using Output = std::variant<Accepted, Refused, Committed, Ended>;

export using Batch = std::vector<pipeline::Addressed<Output>>;

export using Update = std::variant<Batch, pipeline::Snapshot<std::uint64_t>>;

/** A reply double: each committed token is one output, and the finish another. */
export struct ReplyDouble {
	[[nodiscard]] auto feed(tokenizer::Tokenizer const& /*tokenizer*/, Token token) -> std::vector<Output> {
		return {Committed{.token = token}};
	}

	[[nodiscard]] auto end(tokenizer::Tokenizer const& /*tokenizer*/, pipeline::Finish finish) -> std::vector<Output> {
		return {Ended{.finish = finish}};
	}
};

/** A test double of the api in front of the engine: submissions are token prompts, outputs are tokens. */
export struct FrontDouble {
	using Submission = Ask;
	using Output = pipeline_doubles::Output;
	using Update = pipeline_doubles::Update;
	using Tag = std::uint64_t;
	using Grammar = GrammarDouble;
	using Reply = ReplyDouble;

	[[nodiscard]] auto prepare(tokenizer::Tokenizer const& /*tokenizer*/, Ask ask) -> pipeline::Prepared<Tag, Grammar, Reply> {
		// Sampled, with the ask's seed as the fresh one, so the seed reaches the cards and names the script.
		auto const params = sampling::parse(sampling::Request{.temperature = 1.0, .top_p = std::nullopt}, sampling::thinking_defaults, ask.seed);
		core::invariant(params.has_value());
		return {
		    .tag = ask.tag,
		    .request = {.prompt = std::move(ask.prompt), .marks = {}, .sampling = *params, .limit = ask.limit},
		    .grammar = GrammarDouble{},
		    .reply = ReplyDouble{},
		};
	}

	[[nodiscard]] auto accepted() const -> Output {
		return Accepted{};
	}

	[[nodiscard]] auto refused(sched::PromptTooLong refusal) const -> Output {
		return Refused{.prompt_tokens = refusal.prompt_tokens};
	}

	[[nodiscard]] auto batch(Batch outputs) const -> Update {
		return outputs;
	}

	[[nodiscard]] auto status(pipeline::Snapshot<Tag> snapshot) const -> Update {
		return snapshot;
	}
};

static_assert(pipeline::Front<FrontDouble>);

/** The completion a seed scripts: `length` tokens below script_range, then the end of the turn. */
export [[nodiscard]] auto script(std::uint64_t seed, std::size_t length) -> std::vector<Token> {
	auto random = std::mt19937_64{seed};
	auto tokens = std::vector<Token>{};
	for (auto index = std::size_t{0}; index < length; ++index) {
		tokens.push_back(core::narrow<Token>(random() % script_range));
	}
	tokens.push_back(end_of_turn);
	return tokens;
}

[[nodiscard]] constexpr auto mix(std::uint64_t value) -> std::uint64_t {
	value += 0x9e37'79b9'7f4a'7c15ULL;
	value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
	value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
	return value ^ (value >> 31U);
}

/**
 * A test double of the cards: it decodes nothing, and commits for each request the script its seed names. Each row
 * accepts a pseudo-random number of drafts; the draft after them is wrong, and half the time it is the very token the
 * grammar forbids there, so the pipeline's handling of refused drafts runs. Every step completes on both fake cards'
 * events through the card scheduler, so the engine's waits poll a driver.
 *
 * The cards learn a request's prompt length from its first sampling chunk, and index its script by position from
 * there, so a paused and resumed request continues its script.
 */
export template<kv::Geometry G>
class CardsDouble {
public:
	CardsDouble(device::Devices devices, std::map<std::uint64_t, std::vector<Token>> scripts)
	    : devices_{std::make_unique<device::Devices>(std::move(devices))},
	      kv_{std::make_unique<kv_fake::FakeDevice<G>>()},
	      scripts_{std::move(scripts)} {
		for (auto const card : device::card_pair) {
			auto stream = devices_->context(card).create_stream();
			core::invariant(stream.has_value());
			streams_.push_back(std::move(*stream));
		}
		for (auto const& [seed, tokens] : scripts_) {
			core::invariant(std::ranges::all_of(tokens, [](Token token) {
				return token < script_range || token == end_of_turn;
			}));
		}
	}

	auto apply(kv::Command<G> const& command) -> void {
		kv_->apply(command);
		core::invariant(kv_->violations().empty());
	}

	[[nodiscard]] auto draft(sched::Batch batch) {
		auto drafts = pipeline::Drafts{};
		auto const depth = model::drafts(batch.depth);
		for (auto const& row : batch.rows) {
			auto const at = index_of(row.rule.seed, row.position + 1);
			auto const accepts = mix(row.rule.seed ^ mix(row.position)) % (depth + 1);
			auto drafted = pipeline::DraftRow{};
			for (auto draft = std::size_t{0}; draft < depth; ++draft) {
				drafted[draft] = truth(row.rule.seed, at + draft);
			}
			if (accepts < depth) {
				auto const banned = forbidden(at + accepts);
				auto const right = truth(row.rule.seed, at + accepts);
				drafted[accepts] = mix(row.position) % 2 == 0 ? banned : (right == 1 ? 2 : 1);
			}
			drafts.rows.push_back(drafted);
			accepted_.push_back(accepts);
		}
		batch_ = std::move(batch);
		return both() | ex::then([proposed = std::move(drafts)] {
			       return proposed;
		       });
	}

	[[nodiscard]] auto verify(pipeline::Masks masks) {
		core::invariant(batch_.has_value());
		auto const depth = model::drafts(batch_->depth);
		core::invariant(masks.rows.size() == batch_->rows.size() * (1 + depth));
		auto done = sched::BatchDone{.rows = {}, .took = std::chrono::milliseconds{20}};
		for (auto const [index, row] : std::views::enumerate(batch_->rows)) {
			auto const at = index_of(row.rule.seed, row.position + 1);
			auto const accepts = accepted_[core::narrow<std::size_t>(index)];
			auto accepted = sched::Accepted{.tokens = {}, .count = accepts + 1};
			for (auto step = std::size_t{0}; step <= accepts; ++step) {
				auto const& mask = masks.rows[(core::narrow<std::size_t>(index) * (1 + depth)) + step];
				auto const token = truth(row.rule.seed, at + step);
				// Each committed token is sampled under its own verify row's mask: the grammar after the tokens before it.
				core::invariant(mask.allows(token) && !mask.allows(forbidden(at + step)));
				accepted.tokens[step] = token;
			}
			done.rows.push_back(accepted);
		}
		batch_.reset();
		accepted_.clear();
		return both() | ex::then([verified = pipeline::Verified{.done = std::move(done), .masks = std::move(masks)}] mutable {
			       return std::move(verified);
		       });
	}

	[[nodiscard]] auto prefill(pipeline::Prefill work) {
		auto done = std::visit(
		    [&]<class W>(W const& chosen) {
			    if constexpr (requires { chosen.mask; }) {
				    auto const& chunk = chosen.chunk;
				    auto const end = chunk.from + chunk.tokens.size();
				    auto const seed = chunk.rule.seed;
				    prompts_.try_emplace(seed, end);
				    auto const at = index_of(seed, end);
				    auto const token = truth(seed, at);
				    core::invariant(chosen.mask.allows(token) && !chosen.mask.allows(forbidden(at)));
				    return sched::ChunkDone{.sampled = token, .took = std::chrono::microseconds{core::narrow<std::int64_t>(chunk.tokens.size())}};
			    } else {
				    return sched::ChunkDone{.sampled = std::nullopt, .took = std::chrono::microseconds{core::narrow<std::int64_t>(chosen.chunk.tokens.size())}};
			    }
		    },
		    work);
		return both() | ex::then([outcome = done] {
			       return outcome;
		       });
	}

private:
	/** Both cards have finished what their streams hold: the step's completion as the engine waits for it. */
	[[nodiscard]] auto both() const {
		return ex::when_all(card_scheduler::Card{devices_->context(model::Card::Zero), streams_[0]}.schedule(),
		                    card_scheduler::Card{devices_->context(model::Card::One), streams_[1]}.schedule());
	}

	/** The completion's index of the token at sequence position `position`. */
	[[nodiscard]] auto index_of(std::uint64_t seed, sched::Count position) const -> std::size_t {
		auto const prompt = prompts_.find(seed);
		core::invariant(prompt != prompts_.end() && position >= prompt->second);
		return position - prompt->second;
	}

	[[nodiscard]] auto truth(std::uint64_t seed, std::size_t index) const -> Token {
		auto const& tokens = scripts_.at(seed);
		return index < tokens.size() ? tokens[index] : end_of_turn;
	}

	std::unique_ptr<device::Devices> devices_;
	std::vector<device::Stream> streams_;
	std::unique_ptr<kv_fake::FakeDevice<G>> kv_;
	std::map<std::uint64_t, std::vector<Token>> scripts_;
	std::map<std::uint64_t, sched::Count> prompts_;
	std::optional<sched::Batch> batch_;
	std::vector<std::size_t> accepted_;
};

}

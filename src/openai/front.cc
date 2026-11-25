// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:front;

import std;
import engine.chat;
import engine.core;
import engine.grammar;
import engine.json;
import engine.pipeline;
import engine.sched;
import engine.tokenizer;
import :boundary;

/**
 * The api's front of the engine: what a submission becomes on the engine's thread, and what its committed tokens
 * become. It satisfies `pipeline::Front`, so `pipeline::BasicPipeline` runs it.
 */
namespace engine::openai {

/** Where the reply stands in the chat format, as its tokens show: inside a thinking block, or past it. */
enum class Region : std::uint8_t {
	Thinking,
	Answering,
};

[[nodiscard]] constexpr auto opening_region(chat::Thinking thinking) -> Region {
	switch (thinking) {
	case chat::Thinking::Enabled:
		return Region::Thinking;
	case chat::Thinking::Disabled:
		return Region::Answering;
	}
	std::unreachable();
}

/**
 * Turns one request's committed tokens into its outputs: tokens to utf-8 text (a character split across tokens waits
 * for its last byte), text through the chat format's reply parser, and the parser's events into outputs. It counts
 * the reasoning tokens: those inside a thinking block, its closing tag included.
 */
export class Reply {
public:
	Reply(chat::Parser parser, std::span<json::Tool const> tools, chat::Thinking thinking)
	    : parser_{std::move(parser)},
	      custom_{tools | std::views::filter([](json::Tool const& tool) {
		              return std::holds_alternative<json::CustomTool>(tool.input);
	              }) |
		          std::views::transform(&json::Tool::name) | std::ranges::to<std::vector<std::string>>()},
	      region_{opening_region(thinking)} {}

	[[nodiscard]] auto feed(tokenizer::Tokenizer const& tokenizer, tokenizer::Id token) -> std::vector<Output> {
		count(token);
		auto text = text_.push(tokenizer, token);
		// The sampler draws only rows of the sampled vocabulary, every one of them a token the tokenizer names.
		core::invariant(text.has_value());
		auto outputs = std::vector<Output>{};
		take(outputs, parser_.push(*text), Closing::Closed);
		return outputs;
	}

	[[nodiscard]] auto end(tokenizer::Tokenizer const& /*tokenizer*/, pipeline::Finish finish) -> std::vector<Output> {
		auto outputs = std::vector<Output>{};
		if (auto const rest = text_.finish(); !rest.empty()) {
			take(outputs, parser_.push(rest), Closing::Closed);
		}
		// The parser closes a call still open at the reply's end; when the budget ended the reply, that call was cut off.
		auto const last = std::holds_alternative<sched::Truncated>(finish.ending) ? Closing::CutOff : Closing::Closed;
		take(outputs, parser_.finish(), last);
		auto const& usage = finish.usage;
		outputs.emplace_back(Finished{
		    .ending = std::move(finish.ending),
		    .usage =
		        json::Usage{
		            .input_tokens = usage.prompt_tokens,
		            .cached_tokens = usage.cached_tokens,
		            .output_tokens = usage.generated_tokens,
		            .reasoning_tokens = reasoning_tokens_,
		        },
		});
		return outputs;
	}

private:
	auto count(tokenizer::Id token) -> void {
		if (token == tokenizer::id(tokenizer::Added::think)) {
			region_ = Region::Thinking;
		}
		if (region_ == Region::Thinking) {
			reasoning_tokens_ += 1;
		}
		if (token == tokenizer::id(tokenizer::Added::think_end)) {
			region_ = Region::Answering;
		}
	}

	[[nodiscard]] auto callee(std::string_view name) const -> Callee {
		return std::ranges::contains(custom_, name) ? Callee::Custom : Callee::Function;
	}

	auto take(std::vector<Output>& outputs, std::vector<chat::Event> events, Closing closing) -> void {
		for (auto& event : events) {
			if (auto const reasoning = core::alternative<chat::ReasoningDelta>(event)) {
				outputs.emplace_back(Reasoning{.text = std::move(reasoning->text)});
			} else if (auto const content = core::alternative<chat::ContentDelta>(event)) {
				outputs.emplace_back(Content{.text = std::move(content->text)});
			} else if (auto const start = core::alternative<chat::ToolCallStart>(event)) {
				auto const called = callee(start->name);
				open_ = called;
				outputs.emplace_back(
				    CallBegin{.index = core::narrow<std::uint32_t>(start->index), .name = std::move(start->name), .callee = called});
			} else {
				auto& ending = std::get<chat::ToolCallEnd>(event);
				// The parser closes only the call it opened.
				core::invariant(open_.has_value());
				// A custom tool's one parameter is its input; a call that wrote none has an empty one.
				auto text = open_.value() == Callee::Custom ? chat::custom_input(ending.arguments).value_or(std::string{})
				                                            : std::move(ending.arguments);
				outputs.emplace_back(
				    CallEnd{.index = core::narrow<std::uint32_t>(ending.index), .text = std::move(text), .closing = closing});
				open_.reset();
			}
		}
	}

	chat::Parser parser_;
	tokenizer::Detokenizer text_;
	/** The names of the request's custom tools. */
	std::vector<std::string> custom_;
	Region region_;
	std::uint64_t reasoning_tokens_ = 0;
	/** The callee of the call the parser has open. */
	std::optional<Callee> open_;
};

/**
 * What starts a request's grammar: its constraints, compiled, to a parser state. The engine's grammar plugs in here;
 * the request's constraints were checked against it at the boundary, so starting one does not fail.
 */
export template<class P>
concept Grammars = std::movable<P> && requires(P& grammars, Constraints const& constraints) {
	{ grammars.start(constraints) } -> pipeline::GrammarState;
};

/** A submission's prompt as the engine runs it, and where its blocks begin. */
struct RenderedPrompt {
	std::vector<tokenizer::Id> prompt;
	sched::Marks marks;
};

/** The marks of a fresh prompt: a block the prompt opens first has no system block before it. */
[[nodiscard]] auto marks_of(chat::Prompt const& prompt) -> sched::Marks {
	return sched::Marks{
	    .system_end = prompt.first_turn > 0 ? std::optional<sched::Count>{prompt.first_turn} : std::nullopt,
	    .last_user_start = prompt.last_query,
	    .generation_start = prompt.generation,
	};
}

/**
 * The api's adapter in front of the engine, for `pipeline::BasicPipeline`: it renders each submission with the chat
 * format (a continuation as its stored sequence and the new turns after it), starts its grammar with `P`, and answers
 * it with a `Reply`. It runs on the engine's thread only.
 */
export template<Grammars P>
class Front {
public:
	using Submission = openai::Submission;
	using Output = openai::Output;
	using Update = openai::Update;
	using Tag = std::string;
	using Grammar = decltype(std::declval<P&>().start(std::declval<Constraints const&>()));
	using Reply = openai::Reply;

	[[nodiscard]] static auto make(P grammars) -> core::Result<Front, json::ReadError> {
		auto renderer = chat::Renderer::make();
		if (!renderer) {
			return std::unexpected(renderer.error());
		}
		return Front{std::move(grammars), *std::move(renderer)};
	}

	[[nodiscard]] auto prepare(tokenizer::Tokenizer const& tokenizer, Submission submission) -> pipeline::Prepared<Tag, Grammar, Reply> {
		auto rendered = std::visit(
		    [&](auto const& prompt) {
			    return render(tokenizer, prompt);
		    },
		    submission.prompt);
		auto request = sched::Submission{
		    .prompt = std::move(rendered.prompt),
		    .marks = rendered.marks,
		    .sampling = submission.sampling,
		    .limit = submission.limit,
		};
		auto const& constraints = submission.constraints;
		auto parser = chat::Parser::make(constraints.tools, constraints.thinking);
		// Every tool's definition fits one fragment's bounds, which the boundary checked, so its schema reads back.
		core::invariant(parser.has_value());
		return pipeline::Prepared<Tag, Grammar, Reply>{
		    .tag = std::move(submission.tag),
		    .request = std::move(request),
		    .grammar = grammars_.start(constraints),
		    .reply = Reply{*std::move(parser), constraints.tools, constraints.thinking},
		};
	}

	[[nodiscard]] auto accepted() const -> Output {
		return Accepted{};
	}

	[[nodiscard]] auto refused(sched::PromptTooLong refusal) const -> Output {
		return PromptTooLong{.prompt_tokens = refusal.prompt_tokens};
	}

	[[nodiscard]] auto cancelled() const -> Output {
		return Cancelled{};
	}

	[[nodiscard]] auto overran(grammar::MatchError fault) const -> Output {
		return Overran{.fault = fault};
	}

	[[nodiscard]] auto batch(std::vector<Addressed> outputs) const -> Update {
		return Outputs{.items = std::move(outputs)};
	}

	[[nodiscard]] auto status(Snapshot snapshot) const -> Update {
		return snapshot;
	}

	/**
	 * The engine's last update, after it failed. The engine's error carries its cards' own, which the api does not
	 * know, so the reason says only whether restarting the engine could help.
	 */
	template<class E>
	    requires requires(E const& error) {
		    { error.retry() } -> std::same_as<core::Retry>;
	    }
	[[nodiscard]] auto failed(E const& error) const -> Update {
		switch (error.retry()) {
		case core::Retry::Transient:
			return EngineFailed{.reason = "the engine stopped serving after a transient failure of its cards or its thread"};
		case core::Retry::Permanent:
			return EngineFailed{.reason = "the engine stopped serving after a permanent failure of its cards or its thread"};
		}
		std::unreachable();
	}

private:
	Front(P grammars, chat::Renderer renderer) : grammars_{std::move(grammars)}, renderer_{std::move(renderer)} {}

	[[nodiscard]] auto render(tokenizer::Tokenizer const& tokenizer, Fresh const& fresh) -> RenderedPrompt {
		auto prompt = renderer_.render(tokenizer, fresh.conversation);
		// Client text is utf-8 (the decoder checked it) and every definition fits a fragment (the boundary checked it).
		core::invariant(prompt.has_value());
		auto const marks = marks_of(*prompt);
		return RenderedPrompt{.prompt = std::move(prompt->ids), .marks = marks};
	}

	[[nodiscard]] auto render(tokenizer::Tokenizer const& tokenizer, Continued const& continued) -> RenderedPrompt {
		auto rest = renderer_.continuation(tokenizer, continued.turns, continued.options);
		// As for a fresh prompt: the new turns' text is utf-8 and their calls' arguments were read under fragment bounds.
		core::invariant(rest.has_value());
		auto prompt = continued.sequence;
		prompt.insert(prompt.end(), rest->ids.begin(), rest->ids.end());
		// The stored part's own blocks were marked when it ran, and the cache holds its end.
		return RenderedPrompt{
		    .prompt = std::move(prompt),
		    .marks = {.system_end = std::nullopt,
			          .last_user_start = std::nullopt,
			          .generation_start = continued.sequence.size() + rest->generation},
		};
	}

	P grammars_;
	chat::Renderer renderer_;
};

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai_engine_double;

import std;
import engine.core;
import engine.grammar;
import engine.http;
import engine.net;
import engine.openai;
import engine.pipeline;
import engine.pipeline_doubles;

namespace engine::openai_engine_double {

namespace openai = engine::openai;
namespace pipeline = engine::pipeline;
namespace pipeline_doubles = engine::pipeline_doubles;

/** The outputs one request is answered with, in batches as an engine posts them: one batch per round. */
export using Script = std::vector<std::vector<openai::Output>>;

/** Whether the double answers a cancel with `Cancelled`, as the engine does, or leaves the request unanswered. */
export enum class Cancels : std::uint8_t {
	Answered,
	/** The cancel is not answered yet: the test posts the answer itself, when it wants the engine to let go. */
	Held,
};

/**
 * A test double of the engine: it renders and decodes nothing. Each submission takes the next script the test queued
 * and posts its batches at once, through the server's mailbox as the real pipeline would; a test that scripts only
 * part of a request posts the rest itself with `post`. It records every submission and cancellation.
 */
export class EngineDouble {
public:
	EngineDouble(http::Poster<openai::Update> poster, Cancels cancels) : poster_{poster}, cancels_{cancels} {}

	auto submit(http::Exchange exchange, openai::Submission submission) -> void {
		submissions_.emplace_back(exchange, std::move(submission));
		if (scripts_.empty()) {
			return;
		}
		auto script = std::move(scripts_.front());
		scripts_.pop_front();
		for (auto& batch : script) {
			// The double runs on the owner thread, which drains the mailbox each turn; a full one is a broken test.
			core::invariant(post(exchange, std::move(batch)));
		}
	}

	auto cancel(http::Exchange exchange) -> void {
		cancellations_.push_back(exchange);
		if (cancels_ == Cancels::Answered) {
			core::invariant(post(exchange, {openai::Cancelled{}}));
		}
	}

	/** The next submission is answered with `script`. */
	auto queue(Script script) -> void {
		scripts_.push_back(std::move(script));
	}

	/** Posts one batch of outputs for one request. */
	[[nodiscard]] auto post(http::Exchange exchange, std::vector<openai::Output> outputs) const -> bool {
		auto batch = openai::Outputs{};
		for (auto& output : outputs) {
			batch.items.push_back(openai::Addressed{.exchange = exchange, .output = std::move(output)});
		}
		return poster_.post(openai::Update{std::move(batch)}) == net::Posted::Accepted;
	}

	[[nodiscard]] auto post(openai::Update update) const -> bool {
		return poster_.post(std::move(update)) == net::Posted::Accepted;
	}

	[[nodiscard]] auto submissions() const -> std::span<std::pair<http::Exchange, openai::Submission> const> {
		return submissions_;
	}

	[[nodiscard]] auto cancellations() const -> std::span<http::Exchange const> {
		return cancellations_;
	}

private:
	http::Poster<openai::Update> poster_;
	Cancels cancels_;
	std::deque<Script> scripts_;
	std::vector<std::pair<http::Exchange, openai::Submission>> submissions_;
	std::vector<http::Exchange> cancellations_;
};

static_assert(openai::Engine<EngineDouble>);

/**
 * A test double of a grammar's matcher that allows every token everywhere, and keeps the constraints it was started
 * with, so a test sees what reached the grammar.
 */
export class PermissiveGrammar {
public:
	PermissiveGrammar(openai::Constraints constraints, pipeline_doubles::Everything everything)
	    : constraints_{std::move(constraints)}, everything_{std::move(everything)} {}

	[[nodiscard]] auto advance(grammar::Vocabulary const& /*vocabulary*/, pipeline::Token /*token*/)
	    -> core::Result<void, grammar::MatchError> {
		taken_ += 1;
		return {};
	}

	auto rollback(std::size_t count) -> void {
		core::invariant(count <= taken_);
		taken_ -= count;
	}

	[[nodiscard]] auto mask(grammar::Vocabulary const& /*vocabulary*/) -> core::Result<grammar::MaskView, grammar::MatchError> {
		return grammar::MaskView{.tokens = everything_.tokens, .rows = everything_.rows};
	}

	[[nodiscard]] auto constraints() const -> openai::Constraints const& {
		return constraints_;
	}

private:
	openai::Constraints constraints_;
	pipeline_doubles::Everything everything_;
	std::size_t taken_ = 0;
};

static_assert(pipeline::GrammarState<PermissiveGrammar>);

/** A test double of the engine's grammars: every request's grammar allows every token the tokenizer has. */
export struct PermissiveGrammars {
	pipeline_doubles::Everything everything;

	[[nodiscard]] auto start(openai::Constraints const& constraints) const -> PermissiveGrammar {
		return PermissiveGrammar{constraints, everything};
	}
};

static_assert(openai::Grammars<PermissiveGrammars>);

}

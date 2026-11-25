// SPDX-License-Identifier: AGPL-3.0-only
export module engine.pipeline:facade;

import std;
import engine.card_scheduler;
import engine.core;
import engine.http;
import engine.kv;
import engine.model;
import engine.net;
import engine.sched;
import engine.tokenizer;
import :engine;
import :seams;

namespace engine::pipeline {

/**
 * The engine as the api's service drives it, on the reactor's owner thread: `submit` and `cancel` post to the
 * engine's own thread and return at once, and the engine posts its outputs and status back through the server's
 * poster. The engine thread starts serving (and posts its first status) as soon as the pipeline starts; the cards
 * are opened and loaded before, by whoever assembles them.
 *
 * The server whose poster it is must outlive the pipeline: the pipeline stops and joins its thread when it is
 * destroyed or stopped.
 */
export template<kv::Geometry G, model::RowBudget B, sched::Tuning T, Cards<G> C, Front F>
class BasicPipeline {
	using Worker = card_scheduler::Worker<Intake<typename F::Submission>, intake_capacity, core::Result<void, EngineError>>;

public:
	[[nodiscard]] static auto start(http::Poster<typename F::Update> poster, tokenizer::Tokenizer tokenizer, C cards, F front)
	    -> core::Result<BasicPipeline, net::NetError> {
		auto worker = Worker::start([loop = Loop<G, B, T, C, F>{poster, std::move(tokenizer), std::move(cards), std::move(front)}](
		                                Inbox<typename F::Submission>& inbox) mutable {
			return loop.run(inbox);
		});
		if (!worker) {
			return std::unexpected(worker.error());
		}
		return BasicPipeline{std::move(*worker)};
	}

	auto submit(http::Exchange exchange, F::Submission submission) -> void {
		post(Submit<typename F::Submission>{.exchange = exchange, .submission = std::move(submission)});
	}

	auto cancel(http::Exchange exchange) -> void {
		post(Cancel{.exchange = exchange});
	}

	/**
	 * Stops the engine's thread and returns why it ended: a failure of the cards or of its wait, or nothing when it
	 * served until now. A failed engine posts nothing more; its owner decides what the process does next.
	 */
	[[nodiscard]] auto stop() -> core::Result<void, EngineError> {
		return worker_->stop();
	}

private:
	explicit BasicPipeline(std::unique_ptr<Worker> worker) : worker_{std::move(worker)} {}

	auto post(Intake<typename F::Submission> message) -> void {
		// Bounded by the connections, which the intake's capacity covers twice over at least.
		core::invariant(worker_->post(std::move(message)) == net::Posted::Accepted);
	}

	std::unique_ptr<Worker> worker_;
};

/** The engine's pipeline: the model's pool, the shipped row budget and the prefill share. */
export template<Cards<kv::model_geometry> C, Front F>
using Pipeline = BasicPipeline<kv::model_geometry, model::row_budget, sched::tuning, C, F>;

}

// SPDX-License-Identifier: AGPL-3.0-only
//
// The engine as the programs start it: the cards brought up, the tokenizer and the grammar's vocabulary built, and the
// pipeline started over them with the responses api's front. The server starts it on a thread of its own, so its socket
// answers /health while the cards load, and adopts the pipeline once it serves; the cli starts it in place.
export module engine.cards:serving;

import std;
import engine.card_scheduler;
import engine.core;
import engine.decode;
import engine.grammar;
import engine.http;
import engine.json;
import engine.net;
import engine.openai;
import engine.pipeline;
import engine.tokenizer;
import :adapter;
import :start;

namespace engine::cards {

/** The engine's pipeline: the cards, behind the responses api's front. */
export using Pipeline = pipeline::Pipeline<Cards, openai::Front>;

/** Why the engine could not start: the cards, or a part built into the program that would not load. */
export struct [[nodiscard]] LoadError {
	std::variant<StartError, tokenizer::LoadError, json::ReadError, net::NetError> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(LoadError const&) const -> bool = default;
};

struct DescribeLoad {
	[[nodiscard]] auto operator()(StartError const& reason) const -> std::string {
		return describe(reason);
	}

	[[nodiscard]] auto operator()(tokenizer::LoadError const& /*reason*/) const -> std::string {
		return "the tokenizer built into the program does not load";
	}

	[[nodiscard]] auto operator()(json::ReadError const& /*reason*/) const -> std::string {
		return "the chat format built into the program does not load";
	}

	[[nodiscard]] auto operator()(net::NetError const& reason) const -> std::string {
		return std::format("the engine's thread could not open its wait (system error {})", reason.code);
	}
};

/** The failure as one line: the reason the server's /health gives and the programs print. */
export [[nodiscard]] auto describe(LoadError const& error) -> std::string {
	return std::visit(DescribeLoad{}, error.reason);
}

/**
 * Brings the engine up from the images in `directory` and starts its pipeline, which posts its updates through
 * `poster` from then on. Blocks for the whole bring-up: seconds to minutes on the cards.
 */
export [[nodiscard]] auto load(std::filesystem::path const& directory, http::Poster<openai::Update> poster)
    -> core::Result<Pipeline, LoadError> {
	auto tokenizer = tokenizer::load();
	if (!tokenizer) {
		return std::unexpected(LoadError{.reason = tokenizer.error()});
	}
	auto front = openai::Front::make();
	if (!front) {
		return std::unexpected(LoadError{.reason = front.error()});
	}
	auto cards = start(directory);
	if (!cards) {
		return std::unexpected(LoadError{.reason = cards.error()});
	}
	auto vocabulary = grammar::engine_vocabulary(*tokenizer);
	return Pipeline::start(poster, std::move(*tokenizer), std::move(vocabulary), std::move(*cards), std::move(*front))
	    .transform_error([](net::NetError failed) {
		    return LoadError{.reason = failed};
	    });
}

/** The loader's thread takes no messages: its one job is the bring-up. */
struct NoMessage {};

using Loader = card_scheduler::Worker<NoMessage, 1, core::Result<Pipeline, LoadError>>;

/** The engine is loading on the loader's thread; once it posted `Ready`, the loader holds the started pipeline. */
struct Loading {
	std::unique_ptr<Loader> loader;
};

struct Serving {
	Pipeline pipeline;
};

/** Why the engine ended other than by being stopped while it served: it never started, or it failed serving. */
export struct [[nodiscard]] ServingError {
	std::variant<LoadError, pipeline::EngineError<Cards::Error>> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(ServingError const&) const -> bool = default;
};

struct DescribeServing {
	[[nodiscard]] auto operator()(LoadError const& reason) const -> std::string {
		return describe(reason);
	}

	[[nodiscard]] auto operator()(pipeline::EngineError<Cards::Error> const& reason) const -> std::string {
		return std::visit(*this, reason.cause);
	}

	[[nodiscard]] auto operator()(Cards::Error const& reason) const -> std::string {
		return std::format("the engine stopped serving: {}", decode::describe(reason));
	}

	[[nodiscard]] auto operator()(net::NetError const& reason) const -> std::string {
		return std::format("the engine stopped serving: its thread could not wait (system error {})", reason.code);
	}
};

export [[nodiscard]] auto describe(ServingError const& error) -> std::string {
	return std::visit(DescribeServing{}, error.reason);
}

/**
 * The engine as the responses service drives it, on the server's owner thread. It starts loading at once on a thread
 * of its own, which posts `Ready` or `EngineFailed` through the server's poster when the bring-up ends. The service
 * submits nothing before `Ready`, so the first submit or cancel adopts the pipeline the loader left; that join is
 * immediate, since the loader posts `Ready` as its last act.
 */
export class Engine {
public:
	/**
	 * Starts loading. `report` hears, on the loader's thread, the reason a bring-up failed, as the service's /health
	 * will give it: the program says so the moment it happens.
	 */
	template<std::invocable<std::string_view> Report>
	[[nodiscard]] static auto start(std::filesystem::path images, http::Poster<openai::Update> updates, Report reporter)
	    -> core::Result<Engine, net::NetError> {
		auto loader = Loader::start([directory = std::move(images), poster = updates, report = std::move(reporter)](
		                                card_scheduler::Inbox<NoMessage, 1>& /*inbox*/) mutable -> core::Result<Pipeline, LoadError> {
			auto loaded = load(directory, poster);
			auto told = net::Posted::Accepted;
			if (loaded) {
				told = poster.post(openai::Ready{});
			} else {
				auto reason = describe(loaded.error());
				std::invoke(report, std::string_view{reason});
				told = poster.post(openai::EngineFailed{.reason = std::move(reason)});
			}
			// The bring-up's word is the first the loader posts, and the mailbox holds thousands.
			core::invariant(told == net::Posted::Accepted);
			return loaded;
		});
		if (!loader) {
			return std::unexpected(loader.error());
		}
		return Engine{Loading{.loader = std::move(*loader)}};
	}

	auto submit(http::Exchange exchange, openai::Submission submission) -> void {
		serving().submit(exchange, std::move(submission));
	}

	auto cancel(http::Exchange exchange) -> void {
		serving().cancel(exchange);
	}

	/**
	 * Ends the engine: the pipeline stopped, or the loader joined first (a bring-up still running is waited for: the
	 * cards are never left half loaded). Returns the failure that ended serving, if one did.
	 */
	[[nodiscard]] auto stop() -> core::Result<void, ServingError> {
		if (auto const loading = core::alternative<Loading>(state_)) {
			auto loaded = loading->loader->stop();
			if (!loaded) {
				return std::unexpected(ServingError{.reason = loaded.error()});
			}
			state_ = Serving{.pipeline = std::move(*loaded)};
		}
		return std::get<Serving>(state_).pipeline.stop().transform_error([](pipeline::EngineError<Cards::Error> failed) {
			return ServingError{.reason = std::move(failed)};
		});
	}

private:
	explicit Engine(Loading loading) : state_{std::move(loading)} {}

	[[nodiscard]] auto serving() -> Pipeline& {
		if (auto const loading = core::alternative<Loading>(state_)) {
			auto loaded = loading->loader->stop();
			// The service submits only after `Ready`, which the loader posts only for a pipeline it returns.
			core::invariant(loaded.has_value());
			state_ = Serving{.pipeline = std::move(*loaded)};
		}
		return std::get<Serving>(state_).pipeline;
	}

	std::variant<Loading, Serving> state_;
};

}

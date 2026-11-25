// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:service;

import std;
import engine.core;
import engine.grammar;
import engine.http;
import engine.json;
import engine.model;
import engine.pipeline;
import engine.store;
import :boundary;
import :error;
import :validate;
import :assembly;
import :stream;
import :routes;

namespace engine::openai {

/** The create route's body bounds: the decoder's document bound, padded so it reads the body in place. */
export inline constexpr http::BodyBounds body_bounds{.bytes = json::request_bounds.document_bytes, .padding = json::padding};

/** The http server the service runs behind; the engine posts its updates through this server's poster. */
export using Server = http::Server<Update, body_bounds>;

export using Port = http::Port<body_bounds>;

/** The service's bounds, each a compile-time fact of the server. */
export struct Limits {
	/** Engine requests outstanding at once (submitted, without their terminal output): past it, creates are refused. */
	std::size_t requests;
	/** Bytes a streaming client may leave unread before its stream ends with `stream_backlog`. */
	std::size_t backlog_bytes;
	store::Bounds store;
};

/**
 * The server's bounds. Half the engine's intake: every request outstanding may still owe its submit and its cancel to
 * the intake at once. A client four megabytes behind the model is not reading.
 */
export inline constexpr Limits server_limits{
    .requests = pipeline::intake_capacity / 2,
    .backlog_bytes = std::size_t{4} << 20U,
    .store = store::server_bounds,
};

/** Submitted; the engine has neither accepted nor refused it. */
struct Waiting {
	json::Delivery delivery;
};

/** Accepted, answered whole once it finishes. */
struct Gathering {};

using Phase = std::variant<Waiting, Gathering, Streaming>;

/** Whether the engine still owes a request its terminal output. */
enum class Producer : std::uint8_t {
	Running,
	/** Finished, or cancelled by the service: anything more the engine posts for it is not this exchange's to hear. */
	Done,
};

/** A request the service owes an answer, keyed by its exchange. */
struct Live {
	http::Exchange exchange;
	Assembly assembly;
	std::vector<json::StoredInput> input;
	Phase phase;
	Producer producer;
};

[[nodiscard]] auto terminal(Output const& output) -> bool {
	return std::holds_alternative<PromptTooLong>(output) || std::holds_alternative<Finished>(output) ||
	       std::holds_alternative<Overran>(output) || std::holds_alternative<Cancelled>(output);
}

[[nodiscard]] auto overrun(Overran const& overran) -> ApiError {
	// The engine answers `overran` only for a grammar that outgrew its bounds, never for a token it refused.
	core::invariant(pipeline::outgrown(overran.fault));
	auto const bound = overran.fault.fault == grammar::MatchFault::LexerTooLarge ? "its lexer's" : "a parser row's";
	return ApiError{
	    .code = Code::GrammarTooLarge,
	    .message = std::format("the request's grammar outgrew {} bound while the reply was decoded; the reply ended there", bound),
	    .param = std::nullopt,
	};
}

[[nodiscard]] auto not_found(std::string_view id) -> ApiError {
	return ApiError{
	    .code = Code::ResponseNotFound, .message = std::format("no stored response has the id '{}'", id), .param = std::nullopt};
}

[[nodiscard]] auto ok(std::string body) -> http::Response {
	return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = std::move(body)};
}

[[nodiscard]] auto too_long(PromptTooLong refused, std::optional<std::int64_t> budget) -> ApiError {
	auto message = budget ? std::format("the prompt's {} tokens and max_output_tokens {} exceed the context of {} tokens",
	                                    refused.prompt_tokens, *budget, model::max_positions)
	                      : std::format("the prompt's {} tokens leave no room in the context of {} tokens", refused.prompt_tokens,
	                                    model::max_positions);
	return ApiError{.code = Code::ContextLengthExceeded, .message = std::move(message), .param = "input"};
}

[[nodiscard]] auto outgrown() -> ApiError {
	return ApiError{.code = Code::ResponseTooLarge, .message = "the response grew past what one response may carry", .param = std::nullopt};
}

[[nodiscard]] auto backlogged() -> ApiError {
	return ApiError{
	    .code = Code::StreamBacklog, .message = "the client read the stream too slowly; the response was cancelled", .param = std::nullopt};
}

/**
 * The openai Responses api over an engine, as an `http::Service` on the server's owner thread: it validates and
 * refuses at the boundary, submits typed requests, turns the engine's outputs into whole responses and streams, and
 * keeps the stored responses. Its readiness is the one fact of whether the engine serves; every request it owes an
 * answer is in `live_`, at most one per connection, and every engine request without its terminal output is counted.
 */
export template<Engine E, Sources S, Limits L>
class BasicService {
public:
	[[nodiscard]] static auto open(E engine, S sources) -> core::Result<BasicService, json::ReadError> {
		auto decoder = json::RequestDecoder::make();
		if (!decoder) {
			return std::unexpected(decoder.error());
		}
		return BasicService{std::move(engine), std::move(sources), *std::move(decoder)};
	}

	[[nodiscard]] auto serve(http::Exchange exchange, http::Request<body_bounds> request) -> http::Answer {
		if (std::holds_alternative<http::Health>(request.route)) {
			return health_response(readiness_);
		}
		if (auto const refused = unready(readiness_)) {
			return error_response(*refused);
		}
		return std::visit(
		    [&](auto& route) {
			    return answer(exchange, request.id, route);
		    },
		    request.route);
	}

	auto deliver(Update update, Port& port) -> void {
		if (auto const outputs = core::alternative<Outputs>(update)) {
			for (auto& item : outputs->items) {
				advance(item, port);
			}
		} else if (auto const snapshot = core::alternative<Snapshot>(update)) {
			snapshot_ = std::move(*snapshot);
		} else if (std::holds_alternative<Ready>(update)) {
			// The engine starts once; a second start, or one after it failed, is a bug in whoever posted it.
			core::invariant(std::holds_alternative<Loading>(readiness_));
			readiness_ = Ready{};
		} else {
			fail(std::get<EngineFailed>(update), port);
		}
	}

	auto drained(http::Exchange exchange, Port& port) -> void {
		auto const index = find(exchange);
		if (!index) {
			return;
		}
		auto const stream = core::alternative<Streaming>(live_[*index].phase);
		// Only a stream refuses events, so only a stream is drained.
		core::invariant(stream.has_value());
		stream->flowing = Flowing::Moving;
		flush(*index, port);
	}

	auto abandoned(http::Exchange exchange) -> void {
		auto const index = find(exchange);
		if (!index) {
			return;
		}
		if (live_[*index].producer == Producer::Running) {
			cancel(live_[*index]);
		}
		forget(*index);
	}

	/** The engine, for the caller that owns the service. */
	[[nodiscard]] auto engine() -> E& {
		return engine_;
	}

private:
	BasicService(E engine, S sources, json::RequestDecoder decoder)
	    : engine_{std::move(engine)}, sources_{std::move(sources)}, decoder_{std::move(decoder)}, started_{sources_.unix_seconds()} {
		live_.reserve(http::max_connections);
		orphans_.reserve(L.requests);
	}

	[[nodiscard]] auto answer(http::Exchange exchange, http::RequestId const& id, http::CreateResponse<body_bounds>& route)
	    -> http::Answer {
		if (outstanding_ >= L.requests) {
			return error_response(ApiError{
			    .code = Code::EngineBusy,
			    .message = std::format("the engine holds {} requests, its bound; retry once some finish", L.requests),
			    .param = std::nullopt,
			});
		}
		auto const& body = route.body;
		auto decoded = decoder_.decode(json::PaddedView::over(body.padded(), body.length()));
		if (!decoded) {
			return error_response(refusal(decoded.error()));
		}
		auto admitted = admit(*std::move(decoded), id.text(), sources_, [this](store::ResponseId stored) {
			return store_.find(stored);
		});
		if (!admitted) {
			return error_response(admitted.error());
		}
		// A connection carries one exchange at a time, so the requests owed an answer never outnumber connections.
		core::invariant(live_.size() < http::max_connections);
		live_.push_back(Live{
		    .exchange = exchange,
		    .assembly = Assembly{std::move(admitted->response)},
		    .input = std::move(admitted->input),
		    .phase = Waiting{.delivery = admitted->delivery},
		    .producer = Producer::Running,
		});
		outstanding_ += 1;
		engine_.submit(exchange, std::move(admitted->submission));
		return http::Deferred{};
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::RetrieveResponse const& route)
	    -> http::Answer {
		if (auto const checked = check_retrieve(route.query); !checked) {
			return error_response(checked.error());
		}
		auto const stored = lookup(route.id);
		if (!stored) {
			return error_response(not_found(route.id.text));
		}
		return ok(json::encode(stored->response));
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::DeleteResponse const& route)
	    -> http::Answer {
		auto const id = store::ResponseId::parse(route.id.text);
		if (!id || store_.remove(*id) == store::Removal::Absent) {
			return error_response(not_found(route.id.text));
		}
		return ok(json::encode_deleted(route.id.text));
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::ListInputItems const& route)
	    -> http::Answer {
		auto const paging = paging_of(route.query);
		if (!paging) {
			return error_response(paging.error());
		}
		auto const stored = lookup(route.id);
		if (!stored) {
			return error_response(not_found(route.id.text));
		}
		auto page = page_of(stored->input, *paging);
		if (!page) {
			return error_response(page.error());
		}
		return ok(*std::move(page));
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::CancelResponse const& /*route*/) const
	    -> http::Answer {
		return error_response(ApiError{
		    .code = Code::UnsupportedParameter,
		    .message = "only background responses can be cancelled, and background responses are not supported",
		    .param = "background",
		});
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::ListModels const& /*route*/) const
	    -> http::Answer {
		return ok(models_body(started_));
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::Health const& /*route*/) const
	    -> http::Answer {
		return health_response(readiness_);
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::ServerStatus const& /*route*/) const
	    -> http::Answer {
		return ok(status_body(snapshot_));
	}

	[[nodiscard]] auto lookup(http::ResourceId const& named) -> std::optional<store::Stored const&> {
		auto const id = store::ResponseId::parse(named.text);
		if (!id) {
			return std::nullopt;
		}
		return store_.find(*id);
	}

	[[nodiscard]] auto find(http::Exchange exchange) const -> std::optional<std::size_t> {
		auto const found = std::ranges::find(live_, exchange, &Live::exchange);
		if (found == live_.end()) {
			return std::nullopt;
		}
		return core::narrow<std::size_t>(found - live_.begin());
	}

	auto forget(std::size_t index) -> void {
		if (index + 1 != live_.size()) {
			live_[index] = std::move(live_.back());
		}
		live_.pop_back();
	}

	/** Lets go of a request whose answer is settled without the engine: what it still posts for it is dropped. */
	auto cancel(Live& live) -> void {
		core::invariant(live.producer == Producer::Running && orphans_.size() < L.requests);
		engine_.cancel(live.exchange);
		orphans_.push_back(live.exchange);
		live.producer = Producer::Done;
	}

	auto advance(Addressed& item, Port& port) -> void {
		// A cancelled request's outputs end with its terminal one, which frees its place in the bound.
		if (auto const orphan = std::ranges::find(orphans_, item.exchange); orphan != orphans_.end()) {
			if (terminal(item.output)) {
				orphans_.erase(orphan);
				outstanding_ -= 1;
			}
			return;
		}
		auto const index = find(item.exchange);
		// The engine posts only for requests it holds, and nothing after a request's terminal output.
		core::invariant(index.has_value() && live_[*index].producer == Producer::Running);
		auto& live = live_[*index];
		if (terminal(item.output)) {
			outstanding_ -= 1;
		}
		if (auto const waiting = core::alternative<Waiting>(live.phase)) {
			start(*index, waiting->delivery, item.output, port);
			return;
		}
		produce(*index, item.output, port);
	}

	/** The engine's first output for a request: it is accepted or refused. */
	auto start(std::size_t index, json::Delivery delivery, Output const& verdict, Port& port) -> void {
		auto& live = live_[index];
		if (auto const refused = core::alternative<PromptTooLong>(verdict)) {
			live.producer = Producer::Done;
			std::ignore = port.respond(live.exchange, error_response(too_long(*refused, live.assembly.response().max_output_tokens)));
			forget(index);
			return;
		}
		// A request's first output is its verdict.
		core::invariant(std::holds_alternative<Accepted>(verdict));
		switch (delivery) {
		case json::Delivery::Whole:
			live.phase = Gathering{};
			return;
		case json::Delivery::Stream:
			break;
		}
		if (port.open_stream(live.exchange) == http::Delivery::Gone) {
			cancel(live);
			forget(index);
			return;
		}
		auto stream = Streaming{.backlog = {}, .sequence = 0, .flowing = Flowing::Moving};
		if (auto const opened = opening(live.assembly)) {
			for (auto const& event : *opened) {
				stream.backlog.push(event);
			}
		} else {
			cancel(live);
			stream.backlog.push(Fault{.error = outgrown()});
		}
		live.phase = std::move(stream);
		flush(index, port);
	}

	[[nodiscard]] auto steps(Live& live, Output const& output) -> std::vector<Step> {
		if (auto const reasoning = core::alternative<Reasoning>(output)) {
			return live.assembly.text(Text::Reasoning, reasoning->text, sources_);
		}
		if (auto const content = core::alternative<Content>(output)) {
			return live.assembly.text(Text::Answer, content->text, sources_);
		}
		if (auto const begin = core::alternative<CallBegin>(output)) {
			return live.assembly.begin(*begin, sources_);
		}
		if (auto const end = core::alternative<CallEnd>(output)) {
			return live.assembly.end(*end);
		}
		// After its verdict a request's outputs are its reply's, then its finish.
		auto const finished = core::alternative<Finished>(output);
		core::invariant(finished.has_value());
		live.producer = Producer::Done;
		auto finishing = live.assembly.finish(*finished, sources_.unix_seconds());
		keep(live, finished->ending);
		return finishing;
	}

	/** Stores a finished response that asked to be stored, with its input items and the exact sequence it leaves. */
	auto keep(Live& live, pipeline::Ending const& ending) -> void {
		auto const& response = live.assembly.response();
		if (response.storage == json::Storage::Unstored) {
			return;
		}
		auto sequence = std::visit(
		    [](auto const& end) {
			    return end.sequence;
		    },
		    ending);
		auto const id = store::ResponseId::parse(response.id);
		// The service issued the id.
		core::invariant(id.has_value());
		store_.insert(*id, store::Stored{.response = response, .input = std::move(live.input), .sequence = std::move(sequence)});
	}

	/** What an accepted request's reply makes of the response. */
	auto produce(std::size_t index, Output const& output, Port& port) -> void {
		auto& live = live_[index];
		if (auto const overran = core::alternative<Overran>(output)) {
			live.producer = Producer::Done;
			if (auto const stream = core::alternative<Streaming>(live.phase)) {
				stream->backlog.push(Fault{.error = overrun(*overran)});
				flush(index, port);
				return;
			}
			std::ignore = port.respond(live.exchange, error_response(overrun(*overran)));
			forget(index);
			return;
		}
		auto made = steps(live, output);
		auto const outgrew = live.producer == Producer::Running && live.assembly.text_bytes() > http::max_response_body_bytes;
		if (std::holds_alternative<Gathering>(live.phase)) {
			if (outgrew) {
				cancel(live);
				std::ignore = port.respond(live.exchange, error_response(outgrown()));
				forget(index);
			} else if (live.producer == Producer::Done) {
				// The http writer answers a body past its bound with its own 500 in place of this one.
				std::ignore = port.respond(live.exchange, ok(json::encode(live.assembly.response())));
				forget(index);
			}
			return;
		}
		auto& stream = std::get<Streaming>(live.phase);
		stream.backlog.push(std::move(made));
		if (live.producer == Producer::Done) {
			stream.backlog.push(Terminal{});
		} else if (outgrew) {
			cancel(live);
			stream.backlog.push(Fault{.error = outgrown()});
		} else if (stream.backlog.bytes() > L.backlog_bytes) {
			cancel(live);
			stream.backlog.clear();
			stream.backlog.push(Fault{.error = backlogged()});
		}
		flush(index, port);
	}

	/** Sends what a stream holds, unless its client is behind; forgets it once it ends or is gone. */
	auto flush(std::size_t index, Port& port) -> void {
		auto& live = live_[index];
		auto& stream = std::get<Streaming>(live.phase);
		for (;;) {
			switch (pump(stream, live.assembly, port, live.exchange)) {
			case Pumped::Idle:
			case Pumped::Blocked:
				return;
			case Pumped::Outgrown:
				if (live.producer == Producer::Running) {
					cancel(live);
				}
				continue;
			case Pumped::Ended:
				forget(index);
				return;
			case Pumped::Gone:
				if (live.producer == Producer::Running) {
					cancel(live);
				}
				forget(index);
				return;
			}
		}
	}

	/**
	 * The engine failed: it answers nothing more. Every request it still owed an answer ends now (a stream with
	 * `response.failed`, a whole response with a 500), every request it held is let go, and new ones are refused.
	 */
	auto fail(EngineFailed const& failed, Port& port) -> void {
		readiness_ = failed;
		orphans_.clear();
		outstanding_ = 0;
		for (auto index = live_.size(); index > 0; --index) {
			auto& live = live_[index - 1];
			if (live.producer == Producer::Done) {
				continue;
			}
			live.producer = Producer::Done;
			if (auto const stream = core::alternative<Streaming>(live.phase)) {
				stream->backlog.push(Failing{.reason = failed.reason});
				flush(index - 1, port);
				continue;
			}
			std::ignore = port.respond(live.exchange, error_response(ApiError{
			                                              .code = Code::ServerError,
			                                              .message = std::format("the engine failed while answering: {}", failed.reason),
			                                              .param = std::nullopt,
			                                          }));
			forget(index - 1);
		}
	}

	E engine_;
	S sources_;
	json::RequestDecoder decoder_;
	store::BasicStore<L.store> store_;
	std::int64_t started_;
	Readiness readiness_ = Loading{};
	std::optional<Snapshot> snapshot_;
	std::vector<Live> live_;
	/** Requests the service cancelled whose terminal output the engine has not posted yet. */
	std::vector<http::Exchange> orphans_;
	/** Engine requests submitted without their terminal output: the live ones still running, and the orphans. */
	std::size_t outstanding_ = 0;
};

/** The service with the server's bounds. */
export template<Engine E, Sources S>
using Service = BasicService<E, S, server_limits>;

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:service;

import std;
import engine.core;
import engine.http;
import engine.json;
import engine.model;
import :engine;
import :error;
import :gather;
import :render;
import :stream;
import :validate;

namespace engine::openai {

/** The chat route's body bounds: the decoder's document bound, padded so it reads the body in place. */
export inline constexpr http::BodyBounds body_bounds{.bytes = json::request_bounds.document_bytes, .padding = json::padding};

/** The http server the service runs behind; the engine posts its updates through this server's poster. */
export using Server = http::Server<Update, body_bounds>;

export using Port = http::Port<body_bounds>;

/**
 * Where the service's time and randomness come from, passed in so that nothing it does draws on hidden state:
 * `unix_seconds` stamps completions, `draw` seeds requests that name no seed and serials completion ids.
 */
export template<class S>
concept Sources = requires(S& sources) {
	{ sources.unix_seconds() } -> std::same_as<std::int64_t>;
	{ sources.draw() } -> std::same_as<std::uint64_t>;
};

/** Submitted; the engine has neither accepted nor refused it yet. */
struct Waiting {
	json::Delivery delivery;
};

using Phase = std::variant<Waiting, Streaming, Gathering>;

/** A request the service owes an answer: everything about it, keyed by its exchange. */
struct Live {
	http::Exchange exchange;
	Frame frame;
	Budget budget;
	Phase phase;
};

enum class Next : std::uint8_t {
	Keep,
	/** The exchange is answered or gone; the service forgets it. */
	Forget,
};

[[nodiscard]] auto too_long(PromptTooLong refused, Budget budget) -> ApiError {
	auto message = std::visit(
	    [&refused]<class B>(B const& chosen) {
		    if constexpr (requires { chosen.tokens; }) {
			    return std::format("the prompt's {} tokens and max_tokens {} exceed the context of {} tokens", refused.prompt_tokens,
			                       chosen.tokens, model::max_positions);
		    } else {
			    return std::format("the prompt's {} tokens leave no room in the context of {} tokens", refused.prompt_tokens,
			                       model::max_positions);
		    }
	    },
	    budget);
	return ApiError{.code = Code::ContextLengthExceeded, .message = std::move(message), .param = "messages"};
}

[[nodiscard]] auto outgrown() -> ApiError {
	return ApiError{.code = Code::ResponseTooLarge, .message = "the completion grew past what one response may carry", .param = std::nullopt};
}

[[nodiscard]] auto backlogged() -> ApiError {
	return ApiError{
	    .code = Code::StreamBacklog,
	    .message = "the client read the stream too slowly; the request was cancelled",
	    .param = std::nullopt,
	};
}

/**
 * The openai api over an engine: `/v1/chat/completions` (whole and streamed), `/v1/models`, `/health` and
 * `/status`, as an http::Service on the server's owner thread. It validates and refuses at the boundary, submits
 * typed requests, and turns the engine's typed outputs into responses and events; it never renders or tokenizes.
 * Every request it owes an answer is in `live_`, at most one per connection.
 */
export template<Engine E, Sources S>
class Service {
public:
	[[nodiscard]] static auto open(E engine, S sources) -> core::Result<Service, json::ReadError> {
		auto decoder = json::RequestDecoder::make();
		if (!decoder) {
			return std::unexpected(decoder.error());
		}
		auto arguments = json::Reader<json::fragment_bounds>::make();
		if (!arguments) {
			return std::unexpected(arguments.error());
		}
		auto const started = sources.unix_seconds();
		return Service{std::move(engine), std::move(sources), *std::move(decoder), *std::move(arguments), started};
	}

	[[nodiscard]] auto serve(http::Exchange exchange, http::Request<body_bounds> request) -> http::Answer {
		return std::visit([&](auto& route) { return answer(exchange, request.id, route); }, request.route);
	}

	auto deliver(Update update, Port& port) -> void {
		if (auto const status = core::alternative<Status>(update)) {
			status_ = std::move(*status);
			return;
		}
		for (auto& item : std::get<Outputs>(update).items) {
			advance(item, port);
		}
	}

	auto drained(http::Exchange exchange, Port& port) -> void {
		auto const index = find(exchange);
		if (!index) {
			return;
		}
		auto& live = live_[*index];
		auto stream = core::alternative<Streaming>(live.phase);
		// Only a stream refuses events, so only a stream is drained.
		core::invariant(stream.has_value());
		stream->flowing = Flowing::Moving;
		settle(*index, flush(live, *stream, port));
	}

	auto abandoned(http::Exchange exchange) -> void {
		auto const index = find(exchange);
		if (!index) {
			return;
		}
		auto const& live = live_[*index];
		auto const stream = core::alternative<Streaming>(live.phase);
		if (!stream || stream->producer == Producer::Running) {
			engine_.cancel(exchange);
		}
		forget(*index);
	}

	/** The engine, for the caller that owns the service. */
	[[nodiscard]] auto engine() -> E& {
		return engine_;
	}

private:
	Service(E engine, S sources, json::RequestDecoder decoder, json::Reader<json::fragment_bounds> arguments, std::int64_t started)
	    : engine_{std::move(engine)},
	      sources_{std::move(sources)},
	      decoder_{std::move(decoder)},
	      arguments_{std::move(arguments)},
	      started_{started},
	      status_{loading_status()} {
		live_.reserve(http::max_connections);
	}

	[[nodiscard]] auto answer(http::Exchange exchange, http::RequestId const& id, http::ChatCompletions<body_bounds>& route) -> http::Answer {
		if (status_.readiness == Readiness::Loading) {
			return error_response(
			    ApiError{.code = Code::EngineLoading, .message = "the model is still loading; retry shortly", .param = std::nullopt});
		}
		auto const& body = route.body;
		auto decoded = decoder_.decode(json::PaddedView::over(body.padded(), body.length()));
		if (!decoded) {
			return error_response(refusal(decoded.error()));
		}
		auto admitted = admit(*std::move(decoded), id, sources_.draw());
		if (!admitted) {
			return error_response(admitted.error());
		}
		// A connection carries one exchange at a time, so the requests owed an answer never outnumber connections.
		core::invariant(live_.size() < http::max_connections);
		auto const frame = Frame{.serial = sources_.draw(), .created = sources_.unix_seconds()};
		live_.push_back(Live{
		    .exchange = exchange,
		    .frame = frame,
		    .budget = admitted->submission.budget,
		    .phase = Waiting{.delivery = admitted->delivery},
		});
		engine_.submit(exchange, std::move(admitted->submission));
		return http::Deferred{};
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::ListModels const& /*route*/) const
	    -> http::Answer {
		return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = models_body(started_)};
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::Health const& /*route*/) const -> http::Answer {
		return health_response(status_.readiness);
	}

	[[nodiscard]] auto answer(http::Exchange /*exchange*/, http::RequestId const& /*id*/, http::ServerStatus const& /*route*/) const
	    -> http::Answer {
		return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = status_body(status_)};
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

	auto settle(std::size_t index, Next next) -> void {
		if (next == Next::Forget) {
			forget(index);
		}
	}

	auto advance(Addressed& item, Port& port) -> void {
		auto const index = find(item.exchange);
		// An exchange the service has forgotten was answered or cancelled. Nothing follows Finished, so these are what
		// the engine posted before it saw a cancellation: its client left or fell behind.
		if (!index) {
			return;
		}
		auto& live = live_[*index];
		if (auto const waiting = core::alternative<Waiting>(live.phase)) {
			// A request's first output is its verdict; `value()` aborts on an engine that sends anything else first.
			settle(*index, start(live, waiting->delivery, core::alternative<Verdict>(item.output).value(), port));
			return;
		}
		// After the verdict an engine sends only what it produced.
		auto& produced = core::alternative<Produced>(item.output).value();
		if (auto stream = core::alternative<Streaming>(live.phase)) {
			settle(*index, step(live, *stream, produced, port));
		} else {
			settle(*index, gather(live, std::get<Gathering>(live.phase), produced, port));
		}
	}

	/** The engine's first output for a request: it is accepted or refused. */
	[[nodiscard]] auto start(Live& live, json::Delivery delivery, Verdict const& verdict, Port& port) -> Next {
		if (auto const refused = core::alternative<PromptTooLong>(verdict)) {
			std::ignore = port.respond(live.exchange, error_response(too_long(*refused, live.budget)));
			return Next::Forget;
		}
		switch (delivery) {
		case json::Delivery::Whole:
			live.phase = Gathering{.content = {}, .reasoning = {}, .calls = {}, .open = std::nullopt, .bytes = 0};
			return Next::Keep;
		case json::Delivery::Stream:
		case json::Delivery::StreamWithUsage:
			break;
		}
		if (port.open_stream(live.exchange) == http::Delivery::Gone) {
			engine_.cancel(live.exchange);
			return Next::Forget;
		}
		auto backlog = Backlog{};
		backlog.push(Opening{}, 0);
		live.phase = Streaming{
		    .delivery = delivery,
		    .backlog = std::move(backlog),
		    .calls = {.begun = 0, .ended = 0},
		    .producer = Producer::Running,
		    .flowing = Flowing::Moving,
		};
		return flush(live, std::get<Streaming>(live.phase), port);
	}

	[[nodiscard]] auto step(Live& live, Streaming& stream, Produced& output, Port& port) -> Next {
		switch (stream.producer) {
		case Producer::Cancelled:
			return Next::Keep;
		case Producer::Finished:
			// Finished is a request's last output.
			core::invariant(false);
			break;
		case Producer::Running:
			break;
		}
		std::visit(StreamStep{.stream = stream}, output);
		if (stream.producer == Producer::Running && stream.backlog.bytes() > max_backlog_bytes) {
			engine_.cancel(live.exchange);
			stream.producer = Producer::Cancelled;
			stream.backlog.clear();
			stream.backlog.push(Failure{.error = backlogged()}, 0);
			stream.backlog.push(Done{}, 0);
		}
		return flush(live, stream, port);
	}

	/** Sends what the stream holds, unless its client is behind. */
	[[nodiscard]] auto flush(Live& live, Streaming& stream, Port& port) -> Next {
		if (stream.flowing == Flowing::Blocked) {
			return Next::Keep;
		}
		switch (pump(stream.backlog, live.frame, events_, port, live.exchange)) {
		case Pumped::Idle:
			return Next::Keep;
		case Pumped::Blocked:
			stream.flowing = Flowing::Blocked;
			return Next::Keep;
		case Pumped::Ended:
			return Next::Forget;
		case Pumped::Gone:
			if (stream.producer == Producer::Running) {
				engine_.cancel(live.exchange);
			}
			return Next::Forget;
		}
		std::unreachable();
	}

	[[nodiscard]] auto gather(Live& live, Gathering& gathering, Produced& output, Port& port) -> Next {
		auto gathered = std::visit(Gather{.gathering = gathering, .reader = arguments_, .frame = live.frame}, output);
		if (std::holds_alternative<Gathered>(gathered)) {
			return Next::Keep;
		}
		if (auto const complete = core::alternative<Complete>(gathered)) {
			std::ignore = port.respond(live.exchange, http::Response{
			                                              .status = http::StatusCode::Ok,
			                                              .content_type = http::ContentType::Json,
			                                              .body = json::encode(complete->completion),
			                                          });
			return Next::Forget;
		}
		engine_.cancel(live.exchange);
		std::ignore = port.respond(live.exchange, error_response(outgrown()));
		return Next::Forget;
	}

	E engine_;
	S sources_;
	json::RequestDecoder decoder_;
	json::Reader<json::fragment_bounds> arguments_;
	EventWriter events_;
	std::int64_t started_;
	Status status_;
	std::vector<Live> live_;
};

}

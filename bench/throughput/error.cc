// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:error;

import std;
import engine.core;
import engine.net;
import :api;
import :wire;
import :workload;

namespace engine::throughput {

/** The read of /status between waves. */
export struct StatusRead {
	[[nodiscard]] constexpr auto operator==(StatusRead const&) const -> bool = default;
};

/** The stream that sends one prompt of the workload, named by the prompt's place in it. */
export struct PromptStream {
	std::size_t prompt;

	[[nodiscard]] constexpr auto operator==(PromptStream const&) const -> bool = default;
};

/** Which of the client's exchanges a failure happened on. */
export using Origin = std::variant<StatusRead, PromptStream>;

/** The server could not be reached, or the client's own readiness queue failed. */
export struct Unreachable {
	net::NetError cause;

	[[nodiscard]] constexpr auto operator==(Unreachable const&) const -> bool = default;
};

/** No memory for the client's JSON reader. */
export struct NoMemory {
	[[nodiscard]] constexpr auto operator==(NoMemory const&) const -> bool = default;
};

/** The server closed or reset the connection before its response ended. */
export struct Disconnected {
	Origin origin;

	[[nodiscard]] constexpr auto operator==(Disconnected const&) const -> bool = default;
};

/** No byte moved on the connection for the protocol's silence. */
export struct Silent {
	Origin origin;

	[[nodiscard]] constexpr auto operator==(Silent const&) const -> bool = default;
};

/** The response is not one this server writes. */
export struct Malformed {
	Origin origin;
	Fault fault;

	[[nodiscard]] constexpr auto operator==(Malformed const&) const -> bool = default;
};

/** The server answered with a refusal. */
export struct Refused {
	Origin origin;
	Refusal refusal;

	[[nodiscard]] auto operator==(Refused const&) const -> bool = default;
};

/** The engine failed under a stream (`response.failed`): it serves nothing more until it is restarted. */
export struct StreamFailed {
	std::size_t prompt;
	std::string message;

	[[nodiscard]] auto operator==(StreamFailed const&) const -> bool = default;
};

/** A stream ended with the api's `error` event, without its response. */
export struct StreamFaulted {
	std::size_t prompt;
	std::string code;
	std::string message;

	[[nodiscard]] auto operator==(StreamFaulted const&) const -> bool = default;
};

/** What an engine that is not at rest showed: requests of its own, or no snapshot where a wave must have left one. */
export using Restless = std::variant<Unreported, Occupied>;

/** The engine did not come to rest within the protocol's patience; `seen` is what its last snapshot showed. */
export struct Unsettled {
	Restless seen;

	[[nodiscard]] constexpr auto operator==(Unsettled const&) const -> bool = default;
};

/** Why a run that finished yields no measurement. */
export enum class Unmeasured : std::uint8_t {
	/** A stream's first output event was its last: its tokens arrived at one instant, with no span to time them over. */
	NoSpan,
	/** A stream finished with no output token. */
	NoTokens,
	/** The engine's cycle count did not advance across a wave: its counters are not this run's. */
	NoCycles,
};

export struct [[nodiscard]] Unmeasurable {
	Unmeasured why;
	/** The run, counted from zero. */
	std::size_t run;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Unmeasurable const&) const -> bool = default;
};

struct Retryable {
	[[nodiscard]] constexpr auto operator()(Unreachable const& unreachable) const -> core::Retry {
		return unreachable.cause.retry();
	}

	[[nodiscard]] constexpr auto operator()(NoMemory const& /*short_of*/) const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator()(Disconnected const& /*gone*/) const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator()(Silent const& /*silent*/) const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator()(Malformed const& /*malformed*/) const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator()(Refused const& refused) const -> core::Retry {
		switch (refused.refusal.standing) {
		case Standing::Loading:
		case Standing::Busy:
		case Standing::Short:
			return core::Retry::Transient;
		case Standing::Final:
			return core::Retry::Permanent;
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator()(StreamFailed const& /*failed*/) const -> core::Retry {
		return core::Retry::Permanent;
	}

	// Every `error` event is of the request as sent or of how this client reads: the same again ends the same.
	[[nodiscard]] constexpr auto operator()(StreamFaulted const& /*faulted*/) const -> core::Retry {
		return core::Retry::Permanent;
	}

	// Another client's requests end.
	[[nodiscard]] constexpr auto operator()(Unsettled const& /*unsettled*/) const -> core::Retry {
		return core::Retry::Transient;
	}

	[[nodiscard]] constexpr auto operator()(Unmeasurable const& unmeasurable) const -> core::Retry {
		return unmeasurable.retry();
	}
};

/** Every way a benchmark fails to yield its measurement. */
export struct [[nodiscard]] BenchError {
	std::variant<Unreachable, NoMemory, Disconnected, Silent, Malformed, Refused, StreamFailed, StreamFaulted, Unsettled, Unmeasurable> reason;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return std::visit(Retryable{}, reason);
	}

	[[nodiscard]] auto operator==(BenchError const&) const -> bool = default;
};

export template<class T>
using Result = core::Result<T, BenchError>;

template<class Reason>
[[nodiscard]] auto failure(Reason reason) -> std::unexpected<BenchError> {
	return std::unexpected(BenchError{.reason = std::move(reason)});
}

[[nodiscard]] constexpr auto describe(Fault fault) -> std::string_view {
	switch (fault) {
	case Fault::HeadTooLarge:
		return "its head is too long";
	case Fault::BadHead:
		return "its head is not an http/1.1 response's";
	case Fault::BadFraming:
		return "its body's end cannot be found";
	case Fault::TrailingBytes:
		return "bytes follow it";
	case Fault::BodyTooLarge:
		return "its body is too long";
	case Fault::EventTooLarge:
		return "one of its events is too long";
	case Fault::UnexpectedMedia:
		return "it is of another media type than its route answers with";
	case Fault::BadJson:
		return "its JSON is not what the api writes";
	case Fault::UnknownEvent:
		return "one of its events is of no type the api has";
	case Fault::EventAfterEnd:
		return "an event follows its terminal event";
	case Fault::NoTerminalEvent:
		return "it ended without a terminal event";
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto describe(Unmeasured why) -> std::string_view {
	switch (why) {
	case Unmeasured::NoSpan:
		return "a stream's whole reply arrived at one instant, so there is no span to time it over (the prompt's reply is too short)";
	case Unmeasured::NoTokens:
		return "a stream finished without an output token";
	case Unmeasured::NoCycles:
		return "the engine's cycle count did not advance across a wave, so its counters are not this run's";
	}
	std::unreachable();
}

struct Describe {
	Workload const& workload;

	[[nodiscard]] auto exchange(Origin const& origin) const -> std::string {
		if (auto const stream = core::alternative<PromptStream>(origin)) {
			return std::format("the stream of {}", workload.prompts()[stream->prompt].name());
		}
		return "the read of /status";
	}

	[[nodiscard]] auto operator()(Unreachable const& unreachable) const -> std::string {
		return net::describe(unreachable.cause);
	}

	[[nodiscard]] auto operator()(NoMemory const& /*short_of*/) const -> std::string {
		return "no memory for the client's JSON reader";
	}

	[[nodiscard]] auto operator()(Disconnected const& gone) const -> std::string {
		return std::format("the server closed {} before its response ended", exchange(gone.origin));
	}

	[[nodiscard]] auto operator()(Silent const& silent) const -> std::string {
		return std::format("no byte moved on {} for the protocol's silence", exchange(silent.origin));
	}

	[[nodiscard]] auto operator()(Malformed const& malformed) const -> std::string {
		return std::format("the response on {} is not this server's: {}", exchange(malformed.origin), describe(malformed.fault));
	}

	[[nodiscard]] auto operator()(Refused const& refused) const -> std::string {
		return std::format("the server refused {} with {} {}: {}", exchange(refused.origin), refused.refusal.status, refused.refusal.code,
		                   refused.refusal.message);
	}

	[[nodiscard]] auto operator()(StreamFailed const& failed) const -> std::string {
		return std::format("the engine failed under the stream of {}: {}", workload.prompts()[failed.prompt].name(), failed.message);
	}

	[[nodiscard]] auto operator()(StreamFaulted const& faulted) const -> std::string {
		return std::format("the stream of {} ended with the error {}: {}", workload.prompts()[faulted.prompt].name(), faulted.code,
		                   faulted.message);
	}

	[[nodiscard]] auto operator()(Unsettled const& unsettled) const -> std::string {
		if (auto const occupied = core::alternative<Occupied>(unsettled.seen)) {
			return std::format("the engine did not come to rest: {} requests running and {} waiting, so another client is using it",
			                   occupied->running, occupied->queued);
		}
		return "the engine posted no snapshot after it ran a wave";
	}

	[[nodiscard]] auto operator()(Unmeasurable const& unmeasurable) const -> std::string {
		return std::format("run {} yields no measurement: {}", unmeasurable.run + 1, describe(unmeasurable.why));
	}
};

/** One sentence for an operator: what failed and why; `workload` names the prompt a stream sent. */
export [[nodiscard]] auto describe(BenchError const& error, Workload const& workload) -> std::string {
	return std::visit(Describe{.workload = workload}, error.reason);
}

}

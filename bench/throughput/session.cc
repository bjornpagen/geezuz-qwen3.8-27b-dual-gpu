// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:session;

import std;
import engine.core;
import engine.handle;
import engine.net;
import :api;
import :error;
import :events;
import :limits;
import :measure;
import :request;
import :wire;
import :workload;

/**
 * The client as one owner-driven state machine over its own readiness queue: it reads /status until the engine is at
 * rest, runs a wave of streams, reads /status again for what the wave cost the engine, and so on through every wave
 * of every run. One thread owns it; each turn is one wait and everything the wait made possible.
 */
namespace engine::throughput {

/** Where the client reads time. Arrival stamps and expiries come from the one source, so a test can script both. */
export template<class C>
concept TimeSource = requires(C& clock) {
	{ clock.now() } -> std::same_as<Instant>;
};

/** The machine's monotonic clock: the one a measurement is taken with. */
export struct Monotonic {
	[[nodiscard]] auto now() const -> Instant {
		return net::Clock::now();
	}
};

/** The runs are not over; `events` counts the stream events read so far. */
export struct Underway {
	std::uint64_t events;

	[[nodiscard]] constexpr auto operator==(Underway const&) const -> bool = default;
};

/** Every run is over: what each recorded. */
export struct Complete {
	std::vector<RunFacts> runs;

	[[nodiscard]] auto operator==(Complete const&) const -> bool = default;
};

export using Progress = std::variant<Underway, Complete>;

/** A generational reference to one of the session's connections. */
struct WireHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator<=>(WireHandle const&) const = default;
};

/** One request and its response on a connection of its own. */
struct Wire {
	net::Socket socket;
	std::string request;
	std::size_t sent;
	Inbound inbound;
	/** When a byte must next have moved, either way. */
	Instant quiet_by;
};

constexpr std::uint8_t wire_generation_bits = 40;

using Wires = handle::Table<Wire, WireHandle, max_streams, wire_generation_bits>;

// A wire's token is its handle's bits, which stay below the poller's own tokens.
static_assert(std::bit_width(max_streams - 1) + wire_generation_bits < 63);

enum class Moved : std::uint8_t {
	No,
	Yes,
};

/** What one read of a wire brought: the body bytes it completed, whether the response is over, whether bytes moved. */
struct Arrival {
	std::string body;
	Extent extent;
	Moved moved;
};

/** Sends what the kernel takes now of the request's unsent rest. */
[[nodiscard]] auto push(Wire& wire, Origin const& origin) -> Result<Moved> {
	auto moved = Moved::No;
	while (wire.sent < wire.request.size()) {
		auto const sent = wire.socket.send(std::string_view{wire.request}.substr(wire.sent));
		if (auto const transferred = core::alternative<net::Transferred>(sent)) {
			wire.sent += transferred->count;
			moved = Moved::Yes;
			continue;
		}
		if (std::holds_alternative<net::WouldBlock>(sent)) {
			break;
		}
		return failure(Disconnected{.origin = origin});
	}
	return moved;
}

/** Reads once: at most `into` of the response's bytes, so what a turn holds of a stream is bounded by its buffer. */
[[nodiscard]] auto pull(Wire& wire, std::span<char> into, Origin const& origin) -> Result<Arrival> {
	auto const received = wire.socket.receive(into);
	if (std::holds_alternative<net::WouldBlock>(received)) {
		return Arrival{.body = {}, .extent = Extent::Open, .moved = Moved::No};
	}
	auto const transferred = core::alternative<net::Transferred>(received);
	// The server ends a response by its framing and only then closes: an end or a reset before that cut it short.
	if (!transferred) {
		return failure(Disconnected{.origin = origin});
	}
	auto fed = wire.inbound.feed(std::string_view{into.data(), transferred->count});
	if (!fed) {
		return failure(Malformed{.origin = origin, .fault = fed.error().fault});
	}
	return Arrival{.body = std::move(fed->body), .extent = fed->extent, .moved = Moved::Yes};
}

/** A read of /status under way: its body so far. */
struct Asking {
	WireHandle wire;
	std::string body;
};

/** The engine was not at rest; it is asked again at `again`. */
struct Pausing {
	Instant again;
};

using Attempt = std::variant<Asking, Pausing>;

/** Before the first wave: waiting for the engine to be at rest. One that has posted no snapshot has run nothing. */
struct Opening {
	Attempt attempt;
	Instant give_up;
};

/** No output event has arrived. */
struct Awaited {};

/** Output is arriving; `first` is when its first event did. */
struct Flowing {
	Instant first;
};

/** The terminal event arrived; the response's own end follows it. */
struct Ended {
	Instant first;
	Instant last;
	Usage usage;
	Ending ending;
};

/** One stream of the wave under way. */
struct Lane {
	std::size_t prompt;
	WireHandle wire;
	EventDecoder events;
	/** The body of an answer that is no stream: a refusal's, kept to be read whole. */
	std::string whole;
	std::variant<Awaited, Flowing, Ended> progress;
	/** Whether the response has been read to its end and its connection let go. */
	Extent extent;
};

/** A wave's streams are under way; `before` is the engine's counters at rest when it began. */
struct Waving {
	std::vector<Lane> lanes;
	Instant sent;
	Counters before;
};

/** After a wave: waiting for the engine to be at rest again, so its counters hold the wave whole. */
struct Closing {
	Attempt attempt;
	Instant give_up;
	std::vector<StreamFacts> streams;
	Counters before;
};

/** Every run is over, or the session failed: nothing more is turned. */
struct Finished {};

using Stage = std::variant<Opening, Waving, Closing, Finished>;

/** The counters of an engine at rest, or what it showed instead. */
using Rest = std::variant<Counters, Restless>;

/** Before the first wave an engine without a snapshot has run no cycle: its counts are zero. */
struct RestBeforeAnyWave {
	[[nodiscard]] constexpr auto operator()(Unreported const& /*unreported*/) const -> Rest {
		return Counters{.cycles = 0, .drafted_tokens = 0, .accepted_tokens = 0};
	}

	[[nodiscard]] constexpr auto operator()(Occupied const& occupied) const -> Rest {
		return Restless{occupied};
	}

	[[nodiscard]] constexpr auto operator()(Idle const& idle) const -> Rest {
		return idle.counters;
	}
};

/** After a wave the engine has run cycles: a missing snapshot is one that has not arrived yet. */
struct RestAfterAWave {
	[[nodiscard]] constexpr auto operator()(Unreported const& unreported) const -> Rest {
		return Restless{unreported};
	}

	[[nodiscard]] constexpr auto operator()(Occupied const& occupied) const -> Rest {
		return Restless{occupied};
	}

	[[nodiscard]] constexpr auto operator()(Idle const& idle) const -> Rest {
		return idle.counters;
	}
};

[[nodiscard]] constexpr auto rest_of(Opening const& /*stage*/, EngineState const& state) -> Rest {
	return std::visit(RestBeforeAnyWave{}, state);
}

[[nodiscard]] constexpr auto rest_of(Closing const& /*stage*/, EngineState const& state) -> Rest {
	return std::visit(RestAfterAWave{}, state);
}

/**
 * One benchmark session against one server: `protocol.runs` runs of `workload`, each wave between two reads of the
 * engine's counters at rest. `turn` it until it is complete; a failure ends it.
 */
export template<TimeSource C>
class Session {
public:
	[[nodiscard]] static auto open(net::Endpoint server, Workload workload, Protocol protocol, C clock) -> Result<Session> {
		core::invariant(protocol.runs > 0 && protocol.silence > std::chrono::milliseconds::zero());
		auto poller = net::Poller::open();
		if (!poller) {
			return failure(Unreachable{.cause = poller.error()});
		}
		auto reader = WireReader::make();
		if (!reader) {
			return failure(NoMemory{});
		}
		auto const now = clock.now();
		return Session{server, std::move(workload), protocol, std::move(clock), std::move(*poller), std::move(*reader),
		               Opening{.attempt = Pausing{.again = now}, .give_up = now + protocol.settle_patience}};
	}

	/** The session's clock, for a test that scripts time. */
	[[nodiscard]] auto clock() -> C& {
		return clock_;
	}

	/** The nearest instant, on the session's clock, at which a turn has something to do without a readiness fact. */
	[[nodiscard]] auto deadline() const -> std::optional<Instant> {
		auto nearest = std::optional<Instant>{};
		auto const consider = [&nearest](Instant candidate) {
			if (!nearest || candidate < *nearest) {
				nearest = candidate;
			}
		};
		for (auto const handle : wires_->handles()) {
			consider(wires_->get(handle).quiet_by);
		}
		if (auto const pause = paused()) {
			consider(pause->again);
		}
		return nearest;
	}

	/**
	 * One wait, until a readiness fact or `until` on the monotonic clock, then everything it made possible, stamped
	 * with the session's clock. After it reports the session complete, or failed, the session is over.
	 */
	[[nodiscard]] auto turn(std::optional<net::Deadline> until) -> Result<Progress> {
		core::invariant(!std::holds_alternative<Finished>(stage_));
		auto advanced = advance(until);
		if (!advanced) {
			stage_ = Finished{};
			return std::unexpected(std::move(advanced.error()));
		}
		if (std::holds_alternative<Finished>(stage_)) {
			return Complete{.runs = std::move(runs_)};
		}
		return Underway{.events = events_};
	}

private:
	Session(net::Endpoint server, Workload workload, Protocol protocol, C clock, net::Poller poller, WireReader reader, Stage stage)
	    : server_{server},
	      workload_{std::move(workload)},
	      protocol_{protocol},
	      clock_{std::move(clock)},
	      poller_{std::move(poller)},
	      reader_{std::move(reader)},
	      wires_{std::make_unique<Wires>()},
	      buffer_{std::make_unique<std::array<char, receive_bytes>>()},
	      stage_{std::move(stage)} {}

	/** The pause the session is in, if it waits to ask the engine again. */
	[[nodiscard]] auto paused() const -> std::optional<Pausing const&> {
		if (auto const opening = core::alternative<Opening>(stage_)) {
			return core::alternative<Pausing>(opening->attempt);
		}
		if (auto const closing = core::alternative<Closing>(stage_)) {
			return core::alternative<Pausing>(closing->attempt);
		}
		return std::nullopt;
	}

	[[nodiscard]] auto advance(std::optional<net::Deadline> until) -> Result<void> {
		auto const facts = poller_.wait_until(until);
		if (!facts) {
			return failure(Unreachable{.cause = facts.error()});
		}
		auto const now = clock_.now();
		for (auto const& fact : facts->view()) {
			auto const ready = core::alternative<net::Ready>(fact);
			if (!ready) {
				continue;
			}
			if (auto const took = take(WireHandle{.bits = ready->token}, now); !took) {
				return took;
			}
		}
		for (auto const handle : wires_->handles()) {
			if (wires_->get(handle).quiet_by <= now) {
				return failure(Silent{.origin = origin_of(handle)});
			}
		}
		if (std::holds_alternative<Opening>(stage_)) {
			return resume<Opening>(now);
		}
		if (std::holds_alternative<Closing>(stage_)) {
			return resume<Closing>(now);
		}
		return {};
	}

	/** Which exchange a live wire carries. */
	[[nodiscard]] auto origin_of(WireHandle handle) const -> Origin {
		if (auto const waving = core::alternative<Waving>(stage_)) {
			auto const lane = std::ranges::find(waving->lanes, handle, &Lane::wire);
			// While a wave runs every live wire is one of its lanes'.
			core::invariant(lane != waving->lanes.end());
			return PromptStream{.prompt = lane->prompt};
		}
		return StatusRead{};
	}

	/** Opens a connection for `request`; its bytes move at the wire's first drain. */
	[[nodiscard]] auto launch(std::string request, Instant now) -> Result<WireHandle> {
		auto socket = net::Socket::connect(server_);
		if (!socket) {
			return failure(Unreachable{.cause = socket.error()});
		}
		auto const handle = wires_->insert(Wire{
		    .socket = std::move(*socket),
		    .request = std::move(request),
		    .sent = 0,
		    .inbound = {},
		    .quiet_by = now + protocol_.silence,
		});
		// A status read and a wave never overlap, and a wave is at most max_streams wide.
		core::invariant(handle.has_value());
		return *handle;
	}

	/**
	 * Moves a wire's bytes as far as they go now, handing each piece of its response's body to `absorb` with the
	 * response's head. A response that ended gives its head back and lets its connection go; one still open is watched.
	 */
	template<class F>
	[[nodiscard]] auto drain(WireHandle handle, Origin const& origin, Instant now, F absorb) -> Result<std::optional<Head>> {
		for (;;) {
			auto& wire = wires_->get(handle);
			auto const pushed = push(wire, origin);
			if (!pushed) {
				return std::unexpected(pushed.error());
			}
			auto const arrival = pull(wire, *buffer_, origin);
			if (!arrival) {
				return std::unexpected(arrival.error());
			}
			if (*pushed == Moved::Yes || arrival->moved == Moved::Yes) {
				wire.quiet_by = now + protocol_.silence;
			}
			auto const head = wire.inbound.head();
			if (head) {
				if (auto const absorbed = absorb(*head, std::string_view{arrival->body}); !absorbed) {
					return std::unexpected(absorbed.error());
				}
				if (arrival->extent == Extent::Complete) {
					wires_->release(handle);
					return head;
				}
			}
			if (arrival->moved == Moved::No) {
				auto const interest = wire.sent < wire.request.size() ? net::Interest::ReadWrite : net::Interest::Read;
				if (auto const watched = poller_.watch(wire.socket, handle.bits, interest); !watched) {
					return failure(Unreachable{.cause = watched.error()});
				}
				return std::nullopt;
			}
		}
	}

	/**
	 * Moves the wire `handle` names, if it is still live: a fact of one batch, or a wave's own first send, may name a
	 * wire that an earlier one already read to its end.
	 */
	[[nodiscard]] auto take(WireHandle handle, Instant now) -> Result<void> {
		if (!wires_->try_get(handle)) {
			return {};
		}
		if (std::holds_alternative<Waving>(stage_)) {
			return take_stream(handle, now);
		}
		if (std::holds_alternative<Opening>(stage_)) {
			return take_status<Opening>(handle, now);
		}
		return take_status<Closing>(handle, now);
	}

	/** Asks again once a pause is over. */
	template<class S>
	[[nodiscard]] auto resume(Instant now) -> Result<void> {
		auto& settling = std::get<S>(stage_);
		auto const pause = core::alternative<Pausing>(settling.attempt);
		if (!pause || pause->again > now) {
			return {};
		}
		auto const wire = launch(status_request(server_), now);
		if (!wire) {
			return std::unexpected(wire.error());
		}
		settling.attempt = Asking{.wire = *wire, .body = {}};
		return take_status<S>(*wire, now);
	}

	template<class S>
	[[nodiscard]] auto take_status(WireHandle handle, Instant now) -> Result<void> {
		auto& asking = std::get<Asking>(std::get<S>(stage_).attempt);
		auto const head = drain(handle, StatusRead{}, now, [&asking](Head const& /*head*/, std::string_view piece) -> Result<void> {
			if (asking.body.size() + piece.size() > max_body_bytes) {
				return failure(Malformed{.origin = StatusRead{}, .fault = Fault::BodyTooLarge});
			}
			asking.body.append(piece);
			return {};
		});
		if (!head) {
			return std::unexpected(head.error());
		}
		auto const answered = *head;
		if (!answered) {
			return {};
		}
		if (answered->status != ok) {
			return refused(StatusRead{}, answered->status, asking.body);
		}
		if (answered->media != Media::Json) {
			return failure(Malformed{.origin = StatusRead{}, .fault = Fault::UnexpectedMedia});
		}
		auto const state = read_status(reader_, asking.body);
		if (!state) {
			return failure(Malformed{.origin = StatusRead{}, .fault = state.error().fault});
		}
		auto const rest = rest_of(std::get<S>(stage_), *state);
		if (auto const counters = core::alternative<Counters>(rest)) {
			return settled(std::move(std::get<S>(stage_)), *counters);
		}
		auto& settling = std::get<S>(stage_);
		if (now >= settling.give_up) {
			return failure(Unsettled{.seen = std::get<Restless>(rest)});
		}
		settling.attempt = Pausing{.again = now + protocol_.settle_interval};
		return {};
	}

	/** The failure an answer of another status than 200 is: its refusal, or a body that is none. */
	[[nodiscard]] auto refused(Origin const& origin, std::uint16_t status, std::string_view body) -> Result<void> {
		auto refusal = read_refusal(reader_, status, body);
		if (!refusal) {
			return failure(Malformed{.origin = origin, .fault = refusal.error().fault});
		}
		return failure(Refused{.origin = origin, .refusal = std::move(*refusal)});
	}

	/** The engine is at rest before any wave: the first run begins. */
	[[nodiscard]] auto settled(Opening /*opening*/, Counters counters) -> Result<void> {
		runs_.emplace_back();
		return begin_wave(counters);
	}

	/** The engine is at rest after a wave: the wave is recorded whole, and the next begins or the session is over. */
	[[nodiscard]] auto settled(Closing closing, Counters counters) -> Result<void> {
		runs_.back().waves.push_back(WaveFacts{.streams = std::move(closing.streams), .before = closing.before, .after = counters});
		if (runs_.back().waves.size() < workload_.waves()) {
			return begin_wave(counters);
		}
		if (runs_.size() == protocol_.runs) {
			stage_ = Finished{};
			return {};
		}
		runs_.emplace_back();
		return begin_wave(counters);
	}

	/** Opens every stream of the next wave of the run under way, then sends their requests at one instant. */
	[[nodiscard]] auto begin_wave(Counters before) -> Result<void> {
		auto const wave = runs_.back().waves.size();
		auto const prompts = workload_.wave(wave);
		auto lanes = std::vector<Lane>{};
		lanes.reserve(prompts.size());
		for (auto const [index, prompt] : std::views::enumerate(prompts)) {
			auto const wire = launch(create_request(server_, prompt, protocol_.max_output_tokens), clock_.now());
			if (!wire) {
				return std::unexpected(wire.error());
			}
			lanes.push_back(Lane{
			    .prompt = (wave * prompts.size()) + core::narrow<std::size_t>(index),
			    .wire = *wire,
			    .events = {},
			    .whole = {},
			    .progress = Awaited{},
			    .extent = Extent::Open,
			});
		}
		// Stamped after the connections are open, so no stream's first token waits on another's connect.
		auto const sent = clock_.now();
		auto const wires = lanes | std::views::transform(&Lane::wire) | std::ranges::to<std::vector>();
		stage_ = Waving{.lanes = std::move(lanes), .sent = sent, .before = before};
		for (auto const wire : wires) {
			if (auto const took = take(wire, sent); !took) {
				return took;
			}
		}
		return {};
	}

	[[nodiscard]] auto take_stream(WireHandle handle, Instant now) -> Result<void> {
		auto& waving = std::get<Waving>(stage_);
		auto const found = std::ranges::find(waving.lanes, handle, &Lane::wire);
		core::invariant(found != waving.lanes.end());
		auto& lane = *found;
		auto const origin = Origin{PromptStream{.prompt = lane.prompt}};
		auto const head = drain(handle, origin, now, [this, &lane, now](Head const& response, std::string_view piece) {
			return absorb(lane, response, piece, now);
		});
		if (!head) {
			return std::unexpected(head.error());
		}
		auto const answered = *head;
		if (!answered) {
			return {};
		}
		if (answered->status != ok) {
			return refused(origin, answered->status, lane.whole);
		}
		if (!std::holds_alternative<Ended>(lane.progress)) {
			return failure(Malformed{.origin = origin, .fault = Fault::NoTerminalEvent});
		}
		lane.extent = Extent::Complete;
		if (std::ranges::all_of(waving.lanes, [](Lane const& each) {
			    return each.extent == Extent::Complete;
		    })) {
			close_wave(now);
		}
		return {};
	}

	/** Takes a piece of a stream's response: events when it is the stream asked for, else a refusal's body to keep. */
	[[nodiscard]] auto absorb(Lane& lane, Head const& head, std::string_view piece, Instant now) -> Result<void> {
		auto const origin = Origin{PromptStream{.prompt = lane.prompt}};
		if (head.status != ok) {
			if (lane.whole.size() + piece.size() > max_body_bytes) {
				return failure(Malformed{.origin = origin, .fault = Fault::BodyTooLarge});
			}
			lane.whole.append(piece);
			return {};
		}
		if (head.media != Media::EventStream) {
			return failure(Malformed{.origin = origin, .fault = Fault::UnexpectedMedia});
		}
		auto const events = lane.events.feed(piece);
		if (!events) {
			return failure(Malformed{.origin = origin, .fault = events.error().fault});
		}
		for (auto const& data : *events) {
			auto const event = read_event(reader_, data);
			if (!event) {
				return failure(Malformed{.origin = origin, .fault = event.error().fault});
			}
			if (std::holds_alternative<Ended>(lane.progress)) {
				return failure(Malformed{.origin = origin, .fault = Fault::EventAfterEnd});
			}
			++events_;
			if (auto const applied = std::visit(Apply{.lane = lane, .now = now}, *event); !applied) {
				return applied;
			}
		}
		return {};
	}

	/** What one event makes of its lane. */
	struct Apply {
		Lane& lane;
		Instant now;

		[[nodiscard]] auto operator()(Opened const& /*opened*/) const -> Result<void> {
			return {};
		}

		[[nodiscard]] auto operator()(Output const& /*output*/) const -> Result<void> {
			if (std::holds_alternative<Awaited>(lane.progress)) {
				lane.progress = Flowing{.first = now};
			}
			return {};
		}

		[[nodiscard]] auto operator()(Concluded const& concluded) const -> Result<void> {
			auto const flowing = core::alternative<Flowing>(lane.progress);
			// A response that ends before any output event has its first and last evidence of output at one instant.
			auto const first = flowing ? flowing->first : now;
			lane.progress = Ended{.first = first, .last = now, .usage = concluded.usage, .ending = concluded.ending};
			return {};
		}

		[[nodiscard]] auto operator()(Failed const& failed) const -> Result<void> {
			return failure(StreamFailed{.prompt = lane.prompt, .message = failed.message});
		}

		[[nodiscard]] auto operator()(Faulted const& faulted) const -> Result<void> {
			return failure(StreamFaulted{.prompt = lane.prompt, .code = faulted.code, .message = faulted.message});
		}
	};

	/** Every stream of the wave has ended: the engine's counters are read once it is at rest. */
	auto close_wave(Instant now) -> void {
		auto wave = std::move(std::get<Waving>(stage_));
		auto streams = std::vector<StreamFacts>{};
		streams.reserve(wave.lanes.size());
		for (auto const& lane : wave.lanes) {
			auto const& ended = std::get<Ended>(lane.progress);
			streams.push_back(StreamFacts{
			    .prompt = lane.prompt,
			    .sent = wave.sent,
			    .first = ended.first,
			    .last = ended.last,
			    .usage = ended.usage,
			    .ending = ended.ending,
			});
		}
		stage_ = Closing{
		    .attempt = Pausing{.again = now},
		    .give_up = now + protocol_.settle_patience,
		    .streams = std::move(streams),
		    .before = wave.before,
		};
	}

	static constexpr std::uint16_t ok = 200;

	net::Endpoint server_;
	Workload workload_;
	Protocol protocol_;
	C clock_;
	net::Poller poller_;
	WireReader reader_;
	std::unique_ptr<Wires> wires_;
	std::unique_ptr<std::array<char, receive_bytes>> buffer_;
	Stage stage_;
	std::vector<RunFacts> runs_;
	std::uint64_t events_ = 0;
};

/** Turns a session on the machine's clock until it is complete: what each run recorded. */
export [[nodiscard]] auto run(Session<Monotonic>& session) -> Result<std::vector<RunFacts>> {
	for (;;) {
		auto progress = session.turn(session.deadline());
		if (!progress) {
			return std::unexpected(std::move(progress.error()));
		}
		if (auto const complete = core::alternative<Complete>(*progress)) {
			return std::move(complete->runs);
		}
	}
}

}

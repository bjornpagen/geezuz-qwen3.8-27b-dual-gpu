// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:stream;

import std;
import engine.core;
import engine.http;
import engine.json;
import :engine;
import :error;
import :render;
import :text;

namespace engine::openai {

/**
 * Text a streaming client has not taken yet, bounded per stream. Past it the stream ends with a StreamBacklog error
 * instead of holding more: a client that reads this much slower than the model writes is not served by memory.
 */
export inline constexpr std::size_t max_backlog_bytes = std::size_t{1} << 20U;

/** What every piece costs beyond its text, so that many small pieces count against the bound too. */
constexpr std::size_t piece_cost = 64;

/** The stream's identity, repeated in every chunk: the completion's serial names it and its tool calls. */
struct Frame {
	std::uint64_t serial;
	std::int64_t created;
};

[[nodiscard]] auto completion_id(std::uint64_t serial) -> std::string {
	return std::format("chatcmpl-{:016x}", serial);
}

[[nodiscard]] auto call_id(std::uint64_t serial, std::uint32_t index) -> std::string {
	return std::format("call_{:016x}_{}", serial, index);
}

enum class Channel : std::uint8_t {
	Reasoning,
	Content,
};

struct Opening {};

/** Text of one channel, of which the first `sent` bytes have gone out. */
struct Text {
	Channel channel;
	std::string text;
	std::size_t sent;
};

struct CallStart {
	std::uint32_t index;
	std::string name;
};

struct CallText {
	std::uint32_t index;
	std::string text;
	std::size_t sent;
};

struct Closing {
	json::FinishReason reason;
};

struct UsageShown {
	json::Usage usage;
};

struct Failure {
	ApiError error;
};

struct Done {};

/** One event-sized unit of a stream's future, or a run of text that becomes as many events as it needs. */
using Piece = std::variant<Opening, Text, CallStart, CallText, Closing, UsageShown, Failure, Done>;

/**
 * A stream's events not yet taken by its client, in order. Adjacent text of one channel is merged as it arrives,
 * so a stalled client costs one growing string, not one piece per token, and resumes with fewer, larger chunks
 * carrying the same text.
 */
class Backlog {
public:
	[[nodiscard]] auto empty() const -> bool {
		return pieces_.empty();
	}

	[[nodiscard]] auto bytes() const -> std::size_t {
		return bytes_;
	}

	auto push_text(Channel channel, std::string_view text) -> void {
		if (!pieces_.empty()) {
			if (auto const last = core::alternative<Text>(pieces_.back()); last && last->channel == channel) {
				last->text += text;
				bytes_ += text.size();
				return;
			}
		}
		push(Text{.channel = channel, .text = std::string{text}, .sent = 0}, text.size());
	}

	auto push(Piece piece, std::size_t payload) -> void {
		pieces_.push_back(std::move(piece));
		bytes_ += piece_cost + payload;
	}

	/** Drops every piece: a stream that overflowed keeps only its ending. */
	auto clear() -> void {
		pieces_.clear();
		bytes_ = 0;
	}

	[[nodiscard]] auto front() -> Piece& {
		return pieces_.front();
	}

	/** `count` more bytes of the front piece's text went out; the piece goes once all of it has. */
	auto advance(std::size_t count) -> void {
		bytes_ -= count;
		auto const finished = std::visit(
		    [count]<class P>(P& piece) {
			    if constexpr (requires { piece.sent; }) {
				    piece.sent += count;
				    return piece.sent == piece.text.size();
			    } else {
				    return true;
			    }
		    },
		    pieces_.front());
		if (finished) {
			bytes_ -= piece_cost;
			pieces_.pop_front();
		}
	}

private:
	std::deque<Piece> pieces_;
	std::size_t bytes_ = 0;
};

/** One rendered event and how many bytes of its piece's text it carries. */
struct Rendered {
	http::Event event;
	std::size_t taken;
};

/**
 * The event for the longest prefix of `rest` that fits one event. Each refusal names its excess, and every byte of
 * text writes at least one byte, so cutting that many bytes fits unless the cut would leave nothing; then halving
 * converges. A single character always fits.
 */
template<class Make>
[[nodiscard]] auto fitted(EventWriter& writer, json::ChunkHeader const& header, std::string_view rest, Make make) -> Rendered {
	auto take = character_prefix(rest, http::max_event_bytes);
	for (;;) {
		auto rendered = writer.chunk(header, make(rest.substr(0, take)));
		if (rendered) {
			return Rendered{.event = *std::move(rendered), .taken = take};
		}
		core::invariant(take > sequence_width(rest.front()));
		auto const excess = rendered.error().needed - rendered.error().available;
		take = character_prefix(rest, excess < take ? take - excess : take / 2);
	}
}

[[nodiscard]] auto whole(core::Result<http::Event, json::Overflow> rendered) -> Rendered {
	// Only text can be long: every other chunk is bounded by the tool-name bound and the usage digits.
	core::invariant(rendered.has_value());
	return Rendered{.event = *std::move(rendered), .taken = 0};
}

struct Render {
	EventWriter& writer;
	json::ChunkHeader const& header;
	std::uint64_t serial;

	auto operator()(Opening const& /*opening*/) const -> Rendered {
		return whole(writer.chunk(header, json::Opening{}));
	}

	auto operator()(Text const& text) const -> Rendered {
		auto const rest = std::string_view{text.text}.substr(text.sent);
		switch (text.channel) {
		case Channel::Reasoning:
			return fitted(writer, header, rest, [](std::string_view part) { return json::ChunkEvent{json::ReasoningDelta{.text = part}}; });
		case Channel::Content:
			return fitted(writer, header, rest, [](std::string_view part) { return json::ChunkEvent{json::ContentDelta{.text = part}}; });
		}
		std::unreachable();
	}

	auto operator()(CallStart const& start) const -> Rendered {
		auto const id = call_id(serial, start.index);
		return whole(writer.chunk(header, json::ToolCallStart{.index = start.index, .id = id, .name = start.name}));
	}

	auto operator()(CallText const& call) const -> Rendered {
		auto const rest = std::string_view{call.text}.substr(call.sent);
		return fitted(writer, header, rest, [&call](std::string_view part) {
			return json::ChunkEvent{json::ArgumentsDelta{.index = call.index, .fragment = part}};
		});
	}

	auto operator()(Closing const& closing) const -> Rendered {
		return whole(writer.chunk(header, json::Finish{.reason = closing.reason}));
	}

	auto operator()(UsageShown const& shown) const -> Rendered {
		return whole(writer.chunk(header, json::UsageReport{.usage = shown.usage}));
	}

	auto operator()(Failure const& failure) const -> Rendered {
		return Rendered{.event = EventWriter::error(failure.error), .taken = 0};
	}

	auto operator()(Done const& /*done*/) const -> Rendered {
		return Rendered{.event = EventWriter::done(), .taken = 0};
	}
};

enum class Pumped : std::uint8_t {
	/** Everything queued went out; the stream waits for more outputs. */
	Idle,
	/** The client is behind: the rest waits for `drained`. */
	Blocked,
	/** The last event went out and the stream is finished. */
	Ended,
	/** The exchange is over. */
	Gone,
};

/** Sends the backlog in order until it is empty, the client is behind, or the exchange is over. */
template<http::BodyBounds B>
[[nodiscard]] auto pump(Backlog& backlog, Frame frame, EventWriter& writer, http::Port<B>& port, http::Exchange exchange) -> Pumped {
	auto const id = completion_id(frame.serial);
	auto const header = json::ChunkHeader{.id = id, .created = frame.created, .model = model_id};
	while (!backlog.empty()) {
		auto& piece = backlog.front();
		auto const last = std::holds_alternative<Done>(piece);
		auto rendered = std::visit(Render{.writer = writer, .header = header, .serial = frame.serial}, piece);
		switch (port.send(exchange, rendered.event)) {
		case http::Flow::Full:
			return Pumped::Blocked;
		case http::Flow::Gone:
			return Pumped::Gone;
		case http::Flow::Accepted:
			break;
		}
		backlog.advance(rendered.taken);
		if (last) {
			return port.finish(exchange) == http::Delivery::Accepted ? Pumped::Ended : Pumped::Gone;
		}
	}
	return Pumped::Idle;
}

/** Why a stream or a gathering ended, as OpenAI reports it: a turn that called tools ended with `tool_calls`. */
[[nodiscard]] constexpr auto finish_reason(Ending ending, std::size_t begun_calls) -> json::FinishReason {
	switch (ending) {
	case Ending::Stop:
		return begun_calls > 0 ? json::FinishReason::ToolCalls : json::FinishReason::Stop;
	case Ending::Length:
		return json::FinishReason::Length;
	}
	std::unreachable();
}

enum class Flowing : std::uint8_t {
	Moving,
	/** The client refused an event; nothing is sent until `drained`. */
	Blocked,
};

enum class Producer : std::uint8_t {
	Running,
	/** The engine posted the request's last output. */
	Finished,
	/** The service cancelled the request; what the engine still posts for it is dropped. */
	Cancelled,
};

/** The calls of a streamed turn: one is open exactly when more have begun than ended. */
struct Calls {
	std::uint32_t begun;
	std::uint32_t ended;

	[[nodiscard]] constexpr auto open() const -> bool {
		return begun > ended;
	}
};

/** An accepted, streamed request. */
struct Streaming {
	json::Delivery delivery;
	Backlog backlog;
	Calls calls;
	Producer producer;
	Flowing flowing;
};

/** One output of an accepted, streamed request, queued as the events it becomes. */
struct StreamStep {
	Streaming& stream;

	auto operator()(ReasoningText const& text) const -> void {
		stream.backlog.push_text(Channel::Reasoning, text.text);
	}

	auto operator()(ContentText const& text) const -> void {
		stream.backlog.push_text(Channel::Content, text.text);
	}

	auto operator()(CallBegin& begin) const -> void {
		core::invariant(!stream.calls.open() && begin.index == stream.calls.begun && !begin.name.empty() &&
		                begin.name.size() <= max_tool_name_bytes);
		auto const size = begin.name.size();
		stream.backlog.push(CallStart{.index = begin.index, .name = std::move(begin.name)}, size);
		++stream.calls.begun;
	}

	// The arguments go out as the call's one arguments delta (or as many as one event's bound needs) after its start.
	auto operator()(CallEnd const& end) const -> void {
		core::invariant(stream.calls.open() && end.index == stream.calls.ended);
		stream.backlog.push(CallText{.index = end.index, .text = end.arguments, .sent = 0}, end.arguments.size());
		++stream.calls.ended;
	}

	auto operator()(Finished const& finished) const -> void {
		core::invariant(finished.ending == Ending::Length || !stream.calls.open());
		stream.backlog.push(Closing{.reason = finish_reason(finished.ending, stream.calls.begun)}, 0);
		if (stream.delivery == json::Delivery::StreamWithUsage) {
			stream.backlog.push(UsageShown{.usage = finished.usage}, 0);
		}
		stream.backlog.push(Done{}, 0);
		stream.producer = Producer::Finished;
	}
};

}

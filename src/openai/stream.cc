// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:stream;

import std;
import engine.core;
import engine.http;
import engine.json;
import :boundary;
import :error;
import :assembly;

/**
 * A streamed response's events not yet taken by its client, and how they go out: in order, numbered from 0 as they
 * are sent, each at most one http event long. Deltas of one item that pile up while the client is behind merge into
 * one, so a slow client costs one growing string, not one piece per token, and resumes with fewer, larger events
 * carrying the same text.
 */
namespace engine::openai {

/** An event written ahead with its number: the stream's opening events, which show the response as it opened. */
struct Written {
	std::uint64_t sequence;
	std::string data;
};

/** `response.completed` or `response.incomplete`, carrying the finished response. */
struct Terminal {};

/** An `error` event: the stream ends without its response. */
struct Fault {
	ApiError error;
};

/** `response.failed`: the engine failed under the stream. */
struct Failing {
	std::string reason;
};

using Piece = std::variant<Written, Added, PartOpened, Delta, TextClosed, PartClosed, Done, Terminal, Fault, Failing>;

/** What every piece costs beyond its text, so that many small pieces count against a bound too. */
constexpr std::size_t piece_cost = 64;

/** The pieces that end a stream: after one is sent, the stream finishes. */
[[nodiscard]] auto ends_stream(Piece const& piece) -> bool {
	return std::holds_alternative<Terminal>(piece) || std::holds_alternative<Fault>(piece) || std::holds_alternative<Failing>(piece);
}

class Backlog {
public:
	[[nodiscard]] auto empty() const -> bool {
		return pieces_.empty();
	}

	/** The bytes the pieces hold: their text, and a fixed cost each. */
	[[nodiscard]] auto bytes() const -> std::size_t {
		return bytes_;
	}

	/** Whether the stream's last piece is queued: nothing may follow it. */
	[[nodiscard]] auto closed() const -> bool {
		return !pieces_.empty() && ends_stream(pieces_.back());
	}

	auto push(Piece piece) -> void {
		core::invariant(!closed());
		if (auto const delta = core::alternative<Delta>(piece); delta && !pieces_.empty()) {
			if (auto const last = core::alternative<Delta>(pieces_.back()); last && last->index == delta->index) {
				last->text += delta->text;
				bytes_ += delta->text.size();
				return;
			}
		}
		auto const delta = core::alternative<Delta>(piece);
		bytes_ += piece_cost + (delta ? delta->text.size() : 0);
		pieces_.push_back(std::move(piece));
	}

	auto push(std::vector<Step> steps) -> void {
		for (auto& step : steps) {
			push(std::visit(
			    [](auto& chosen) {
				    return Piece{std::move(chosen)};
			    },
			    step));
		}
	}

	/** Drops every piece: a stream that must end now keeps only its ending. */
	auto clear() -> void {
		pieces_.clear();
		bytes_ = 0;
	}

	[[nodiscard]] auto front() const -> Piece const& {
		return pieces_.front();
	}

	/** `count` bytes of the front piece's text went out; the piece goes once all of it has. */
	auto advance(std::size_t count) -> void {
		auto const delta = core::alternative<Delta>(pieces_.front());
		if (delta && count < delta->text.size()) {
			delta->text.erase(0, count);
			bytes_ -= count;
			return;
		}
		bytes_ -= piece_cost + (delta ? delta->text.size() : 0);
		pieces_.pop_front();
	}

private:
	std::deque<Piece> pieces_;
	std::size_t bytes_ = 0;
};

/** An event's data, when it fits one http event. */
[[nodiscard]] auto written(json::StreamEvent const& event, std::uint64_t sequence) -> std::optional<std::string> {
	auto const size = json::measure(event, sequence);
	if (size > http::max_event_bytes) {
		return std::nullopt;
	}
	auto buffer = json::Buffer{size};
	auto const appended = json::append(buffer, event, sequence);
	core::invariant(appended.has_value());
	return std::move(buffer).take();
}

/** The longest prefix of `text` of at most `bytes` bytes that ends on a character boundary. */
[[nodiscard]] auto character_prefix(std::string_view text, std::size_t bytes) -> std::size_t {
	if (bytes >= text.size()) {
		return text.size();
	}
	while (bytes > 0 && (std::bit_cast<std::uint8_t>(text[bytes]) & 0xc0U) == 0x80U) {
		bytes -= 1;
	}
	return bytes;
}

/**
 * Text one delta event always holds: escaping writes a byte as at most six, and the event's other members (its type,
 * an item id, two indexes, a number) take far less than the margin.
 */
constexpr std::size_t delta_room = (http::max_event_bytes - 4096) / 6;

/** One piece as the event it becomes now, and how many bytes of a delta's text it carries. */
struct Rendered {
	std::string data;
	std::size_t taken;
};

[[nodiscard]] constexpr auto part_of(json::Stream stream) -> json::PartKind {
	return stream == json::Stream::ReasoningText ? json::PartKind::ReasoningText : json::PartKind::OutputText;
}

[[nodiscard]] auto fault_event(ApiError const& error, std::uint64_t sequence) -> std::string {
	auto const param = error.param ? std::optional<std::string_view>{*error.param} : std::nullopt;
	auto data = written(json::ErrorEvent{.code = spelling(error.code), .message = error.message, .param = param}, sequence);
	// An error carries a sentence and a member's name, far below an event's bound.
	core::invariant(data.has_value());
	return std::move(data).value();
}

/** Renders pieces from the response they describe, numbering each. Nothing when the event would be too long. */
struct Render {
	Assembly const& assembly;
	std::uint64_t sequence;

	[[nodiscard]] auto whole(json::StreamEvent const& event) const -> std::optional<Rendered> {
		return written(event, sequence).transform([](std::string data) {
			return Rendered{.data = std::move(data), .taken = 0};
		});
	}

	[[nodiscard]] auto operator()(Written const& piece) const -> std::optional<Rendered> {
		core::invariant(piece.sequence == sequence);
		return Rendered{.data = piece.data, .taken = 0};
	}

	[[nodiscard]] auto operator()(Added const& piece) const -> std::optional<Rendered> {
		return whole(json::ItemAdded{.output_index = piece.index, .item = assembly.item(piece.index)});
	}

	[[nodiscard]] auto operator()(PartOpened const& piece) const -> std::optional<Rendered> {
		auto const [stream, id] = assembly.facts(piece.index);
		return whole(json::PartAdded{.item_id = id, .output_index = piece.index, .kind = part_of(stream)});
	}

	[[nodiscard]] auto operator()(Delta const& piece) const -> std::optional<Rendered> {
		auto const [stream, id] = assembly.facts(piece.index);
		auto const event = [&](std::string_view text) {
			return json::StreamEvent{json::TextDelta{.stream = stream, .item_id = id, .output_index = piece.index, .delta = text}};
		};
		if (auto data = written(event(piece.text), sequence)) {
			return Rendered{.data = *std::move(data), .taken = piece.text.size()};
		}
		auto const taken = character_prefix(piece.text, delta_room);
		auto data = written(event(std::string_view{piece.text}.substr(0, taken)), sequence);
		core::invariant(data.has_value() && taken > 0);
		return Rendered{.data = std::move(data).value(), .taken = taken};
	}

	[[nodiscard]] auto operator()(TextClosed const& piece) const -> std::optional<Rendered> {
		auto const [stream, id] = assembly.facts(piece.index);
		return whole(json::TextDone{.stream = stream, .item_id = id, .output_index = piece.index, .text = assembly.text_of(piece.index)});
	}

	[[nodiscard]] auto operator()(PartClosed const& piece) const -> std::optional<Rendered> {
		auto const [stream, id] = assembly.facts(piece.index);
		return whole(
		    json::PartDone{.item_id = id, .output_index = piece.index, .kind = part_of(stream), .text = assembly.text_of(piece.index)});
	}

	[[nodiscard]] auto operator()(Done const& piece) const -> std::optional<Rendered> {
		return whole(json::ItemDone{.output_index = piece.index, .item = assembly.item(piece.index)});
	}

	[[nodiscard]] auto operator()(Terminal const& /*piece*/) const -> std::optional<Rendered> {
		auto const& response = assembly.response();
		auto const lifecycle =
		    response.status == json::ResponseStatus::Completed ? json::Lifecycle::Completed : json::Lifecycle::Incomplete;
		return whole(json::ResponseEvent{.lifecycle = lifecycle, .response = response});
	}

	[[nodiscard]] auto operator()(Fault const& piece) const -> std::optional<Rendered> {
		return Rendered{.data = fault_event(piece.error, sequence), .taken = 0};
	}

	[[nodiscard]] auto operator()(Failing const& piece) const -> std::optional<Rendered> {
		return whole(
		    json::FailedEvent{.response = assembly.response(), .failure = {.code = spelling(Code::ServerError), .message = piece.reason}});
	}
};

/** The opening events of a stream, numbered 0 and 1: the response as it opened, in progress, its output empty. */
[[nodiscard]] auto opening(Assembly const& assembly) -> std::optional<std::array<Written, 2>> {
	auto const& response = assembly.response();
	core::invariant(response.output.empty() && response.status == json::ResponseStatus::InProgress);
	auto created = written(json::ResponseEvent{.lifecycle = json::Lifecycle::Created, .response = response}, 0);
	auto progressing = written(json::ResponseEvent{.lifecycle = json::Lifecycle::InProgress, .response = response}, 1);
	if (!created || !progressing) {
		return std::nullopt;
	}
	return std::array{Written{.sequence = 0, .data = *std::move(created)}, Written{.sequence = 1, .data = *std::move(progressing)}};
}

[[nodiscard]] auto too_large() -> ApiError {
	return ApiError{.code = Code::ResponseTooLarge, .message = "the response grew past what one event may carry", .param = std::nullopt};
}

export enum class Flowing : std::uint8_t {
	Moving,
	/** The client refused an event; nothing is sent until `drained`. */
	Blocked,
};

/** An open stream: what it has not sent, and the number its next event takes. */
struct Streaming {
	Backlog backlog;
	std::uint64_t sequence;
	Flowing flowing;
};

enum class Pumped : std::uint8_t {
	/** Everything queued went out; the stream waits for more. */
	Idle,
	/** The client is behind: the rest waits for `drained`. */
	Blocked,
	/** An event outgrew the bound: the backlog now holds only the `error` event that ends the stream. */
	Outgrown,
	/** The last event went out and the stream is finished. */
	Ended,
	/** The exchange is over. */
	Gone,
};

/** Sends the backlog in order until it is empty, the client is behind, an event outgrows its bound, or it ends. */
template<http::BodyBounds B>
[[nodiscard]] auto pump(Streaming& stream, Assembly const& assembly, http::Port<B>& port, http::Exchange exchange) -> Pumped {
	if (stream.flowing == Flowing::Blocked) {
		return Pumped::Blocked;
	}
	while (!stream.backlog.empty()) {
		auto const& piece = stream.backlog.front();
		auto const last = ends_stream(piece);
		auto rendered = std::visit(Render{.assembly = assembly, .sequence = stream.sequence}, piece);
		if (!rendered) {
			stream.backlog.clear();
			stream.backlog.push(Fault{.error = too_large()});
			return Pumped::Outgrown;
		}
		switch (port.send(exchange, http::Event{.data = std::move(rendered->data)})) {
		case http::Flow::Full:
			stream.flowing = Flowing::Blocked;
			return Pumped::Blocked;
		case http::Flow::Gone:
			return Pumped::Gone;
		case http::Flow::Accepted:
			break;
		}
		stream.sequence += 1;
		stream.backlog.advance(rendered->taken);
		if (last) {
			return port.finish(exchange) == http::Delivery::Accepted ? Pumped::Ended : Pumped::Gone;
		}
	}
	return Pumped::Idle;
}

}

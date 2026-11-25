// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:inbound;

import std;
import engine.core;
import :error;
import :head;
import :limits;
import :request;

namespace engine::http {

/** More bytes are needed. */
export struct Partial {
	[[nodiscard]] constexpr auto operator==(Partial const&) const -> bool = default;
};

/** The head asked for `100 Continue` and the body has not arrived: the server sends the interim response once. */
export struct Continue {
	[[nodiscard]] constexpr auto operator==(Continue const&) const -> bool = default;
};

/** One whole request, its body read to its Content-Length. The id is the client's when it sent a sane one. */
export template<BodyBounds B>
struct Assembled {
	Route<B> route;
	std::optional<RequestId> client_id;
	Persistence persistence;

	[[nodiscard]] auto operator==(Assembled const&) const -> bool = default;
};

/** The request is refused. Its framing cannot be trusted, so nothing after it on the connection is read. */
export struct Refused {
	HttpError error;

	[[nodiscard]] constexpr auto operator==(Refused const&) const -> bool = default;
};

export template<BodyBounds B>
using Assembly = std::variant<Partial, Continue, Assembled<B>, Refused>;

/** The route a head named, carrying its body; only the chat route has one (every other route's is empty). */
template<BodyBounds B>
[[nodiscard]] auto route_of(Target target, Body<B> body) -> Route<B> {
	switch (target) {
	case Target::ChatCompletions:
		return ChatCompletions<B>{.body = std::move(body)};
	case Target::Models:
		return ListModels{};
	case Target::Health:
		return Health{};
	case Target::Status:
		return ServerStatus{};
	}
	return Health{};
}

/**
 * The request side of one connection: a fixed head buffer and, while a body is read, that body. Bytes go in where
 * `space` says; `assemble` yields requests in arrival order. Requests pipelined behind the current one stay buffered,
 * so the result is the same however the bytes were split into reads.
 */
export template<BodyBounds B>
class Inbound {
public:
	/** Where the next received bytes go: the head buffer's free tail, or the body's unread tail. */
	[[nodiscard]] auto space() -> std::span<char> {
		if (auto const reading = core::alternative<Reading>(stage_)) {
			return reading->body.unread(reading->filled);
		}
		return std::span{buffer_}.subspan(buffered_);
	}

	/** Whether `space` is empty: a head filled the buffer, or a body is complete and not yet assembled. */
	[[nodiscard]] auto full() const -> bool {
		if (auto const reading = core::alternative<Reading>(stage_)) {
			return reading->filled == reading->body.length();
		}
		return buffered_ == buffer_.size();
	}

	/** `count` bytes were received into `space`. */
	auto commit(std::size_t count) -> void {
		if (auto const reading = core::alternative<Reading>(stage_)) {
			core::invariant(count <= reading->body.length() - reading->filled);
			reading->filled += count;
			return;
		}
		core::invariant(count <= buffer_.size() - buffered_);
		buffered_ += count;
	}

	/** No byte of a next request is held. */
	[[nodiscard]] auto empty() const -> bool {
		return std::holds_alternative<Heading>(stage_) && buffered_ == 0;
	}

	/** Forgets every byte held: a closing connection reads only to drain the client. */
	auto discard() -> void {
		buffered_ = 0;
		stage_ = Heading{};
	}

	[[nodiscard]] auto assemble() -> Assembly<B> {
		if (auto const reading = core::alternative<Reading>(stage_)) {
			if (reading->filled < reading->body.length()) {
				return Partial{};
			}
			auto assembled = Assembled<B>{
			    .route = route_of(reading->head.target, std::move(reading->body)),
			    .client_id = std::move(reading->head.client_id),
			    .persistence = reading->head.persistence,
			};
			stage_ = Heading{};
			return assembled;
		}

		auto const held = std::string_view{buffer_.data(), buffered_};
		auto const end = held.find("\r\n\r\n");
		if (end == std::string_view::npos) {
			if (buffered_ == buffer_.size()) {
				return Refused{.error = HttpError{HttpErrorKind::HeadTooLarge}};
			}
			return Partial{};
		}
		auto head = parse_head<B.bytes>(held.substr(0, end));
		if (!head) {
			return Refused{.error = head.error()};
		}
		consume(end + 4);

		auto body = Body<B>{head->body_length};
		auto const ready = std::min(buffered_, body.length());
		std::ranges::copy(std::span{buffer_}.first(ready), body.unread(0).begin());
		consume(ready);
		if (ready == body.length()) {
			return Assembled<B>{
			    .route = route_of(head->target, std::move(body)),
			    .client_id = std::move(head->client_id),
			    .persistence = head->persistence,
			};
		}
		auto const expectation = head->expectation;
		stage_ = Reading{.head = std::move(*head), .body = std::move(body), .filled = ready};
		if (expectation == Expectation::Continue) {
			return Continue{};
		}
		return Partial{};
	}

private:
	struct Heading {};

	struct Reading {
		Head head;
		Body<B> body;
		std::size_t filled;
	};

	/** Drops the first `count` held bytes, keeping what a pipelining client sent after them. */
	auto consume(std::size_t count) -> void {
		core::invariant(count <= buffered_);
		std::ranges::copy(std::span{buffer_}.subspan(count, buffered_ - count), buffer_.begin());
		buffered_ -= count;
	}

	std::array<char, max_head_bytes> buffer_{};
	std::size_t buffered_ = 0;
	std::variant<Heading, Reading> stage_;
};

}

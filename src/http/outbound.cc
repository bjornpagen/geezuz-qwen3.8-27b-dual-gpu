// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:outbound;

import std;
import engine.core;
import :limits;

namespace engine::http {

/**
 * The response side of one connection: a fixed buffer for small framed pieces (an interim response, a stream head,
 * ordinary events, a stream's end), bounded by max_stream_buffer_bytes, and at most one owned piece held after them
 * (a whole response, or an event too long for the buffer). The two are the backpressure window: what does not fit is
 * refused, never grown. Only the buffer is allocated with the connection; a held piece owns storage of its own size
 * and gives it back once sent, so a connection's memory follows what it has queued.
 */
class Outbound {
public:
	[[nodiscard]] auto empty() const -> bool {
		return begin_ == end_ && !holding();
	}

	/** Framed bytes held in the bounded buffer. */
	[[nodiscard]] auto queued() const -> std::size_t {
		return end_ - begin_;
	}

	[[nodiscard]] auto room() const -> std::size_t {
		return buffer_.size() - queued();
	}

	/** A piece is held after the buffer and not yet sent; the buffer takes nothing until it is. */
	[[nodiscard]] auto holding() const -> bool {
		return held_sent_ != held_.size();
	}

	/** Reserves `count` bytes at the buffer's end and lets `write` fill exactly them. */
	template<std::invocable<std::span<char>> F>
	auto emplace(std::size_t count, F write) -> void {
		// Bytes go out buffer first, then the held piece: buffering behind a held piece would send them before it.
		core::invariant(!holding());
		core::invariant(count <= room());
		if (buffer_.size() - end_ < count) {
			std::ranges::copy(std::span{buffer_}.subspan(begin_, queued()), buffer_.begin());
			end_ = queued();
			begin_ = 0;
		}
		write(std::span{buffer_}.subspan(end_, count));
		end_ += count;
	}

	auto append(std::string_view bytes) -> void {
		emplace(bytes.size(), [bytes](std::span<char> into) {
			std::ranges::copy(bytes, into.begin());
		});
	}

	/** Takes a piece to send after the buffered bytes: a whole response, or an event too long for the buffer. */
	auto hold(std::string piece) -> void {
		core::invariant(!holding());
		held_ = std::move(piece);
		held_sent_ = 0;
	}

	/** Appends `bytes` after everything queued: to the held piece while there is one, else to the buffer. */
	auto append_last(std::string_view bytes) -> void {
		if (holding()) {
			held_ += bytes;
			return;
		}
		append(bytes);
	}

	/** The next bytes to send; empty only when `empty()`. */
	[[nodiscard]] auto front() const -> std::span<char const> {
		if (begin_ != end_) {
			return std::span{buffer_}.subspan(begin_, queued());
		}
		return std::span{held_}.subspan(held_sent_);
	}

	/** `count` bytes of `front()` were sent. */
	auto consume(std::size_t count) -> void {
		if (begin_ != end_) {
			core::invariant(count <= queued());
			begin_ += count;
			if (begin_ == end_) {
				begin_ = 0;
				end_ = 0;
			}
			return;
		}
		core::invariant(count <= held_.size() - held_sent_);
		held_sent_ += count;
		if (held_sent_ == held_.size()) {
			// A held piece can be megabytes; its storage goes with it rather than staying with the connection.
			held_ = std::string{};
			held_sent_ = 0;
		}
	}

private:
	std::array<char, max_stream_buffer_bytes> buffer_{};
	std::size_t begin_ = 0;
	std::size_t end_ = 0;
	std::string held_;
	std::size_t held_sent_ = 0;
};

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:outbound;

import std;
import engine.core;
import :limits;

namespace engine::http {

/**
 * The response side of one connection: a fixed buffer for small framed pieces (an interim response, a stream head,
 * events, a stream's end), bounded by max_stream_buffer_bytes, and an owned whole response sent after them. The
 * buffer is the backpressure window: what does not fit is refused, never grown.
 */
class Outbound {
public:
	[[nodiscard]] auto empty() const -> bool {
		return begin_ == end_ && whole_sent_ == whole_.size();
	}

	/** Framed bytes held in the bounded buffer. */
	[[nodiscard]] auto queued() const -> std::size_t {
		return end_ - begin_;
	}

	[[nodiscard]] auto room() const -> std::size_t {
		return buffer_.size() - queued();
	}

	/** Reserves `count` bytes at the buffer's end and lets `write` fill exactly them. */
	template<std::invocable<std::span<char>> F>
	auto emplace(std::size_t count, F write) -> void {
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

	/** Takes a whole response to send after the buffered bytes. */
	auto adopt(std::string whole) -> void {
		core::invariant(whole_sent_ == whole_.size());
		whole_ = std::move(whole);
		whole_sent_ = 0;
	}

	/** The next bytes to send; empty only when `empty()`. */
	[[nodiscard]] auto front() const -> std::span<char const> {
		if (begin_ != end_) {
			return std::span{buffer_}.subspan(begin_, queued());
		}
		return std::span{whole_}.subspan(whole_sent_);
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
		core::invariant(count <= whole_.size() - whole_sent_);
		whole_sent_ += count;
		if (whole_sent_ == whole_.size()) {
			// A whole response can be megabytes; its storage goes with it rather than staying with the connection.
			whole_ = std::string{};
			whole_sent_ = 0;
		}
	}

private:
	std::array<char, max_stream_buffer_bytes> buffer_{};
	std::size_t begin_ = 0;
	std::size_t end_ = 0;
	std::string whole_;
	std::size_t whole_sent_ = 0;
};

}

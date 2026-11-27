// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:events;

import std;
import engine.core;
import :limits;
import :wire;

namespace engine::throughput {

constexpr std::string_view data_field = "data";

/** The longest line the decoder holds: an event's whole data on one `data: ` line. */
constexpr std::size_t max_line_bytes = max_event_bytes + data_field.size() + 2;

/**
 * An event stream decoded as its bytes arrive, by the format's own parsing rules: lines end in CRLF, LF or CR, an
 * event's data is its `data` lines joined by line feeds, a blank line dispatches it, and comments and the fields this
 * client does not read are passed over. What is pending when the stream ends is dropped, as the rules say.
 */
export class EventDecoder {
public:
	/** Takes the next bytes of the stream; the data of every event they complete, in order. */
	[[nodiscard]] auto feed(std::string_view bytes) -> core::Result<std::vector<std::string>, WireFault> {
		auto events = std::vector<std::string>{};
		while (!bytes.empty()) {
			// A CR ended the last line; the LF of a CRLF pair may only arrive now.
			if (std::exchange(after_, After::Text) == After::CarriageReturn && bytes.front() == '\n') {
				bytes.remove_prefix(1);
				continue;
			}
			auto const end = bytes.find_first_of("\r\n");
			auto const run = bytes.substr(0, end);
			if (line_.size() + run.size() > max_line_bytes) {
				return broke(Fault::EventTooLarge);
			}
			line_.append(run);
			if (end == std::string_view::npos) {
				break;
			}
			after_ = bytes[end] == '\r' ? After::CarriageReturn : After::Text;
			bytes.remove_prefix(end + 1);
			if (line_.empty()) {
				if (auto event = dispatch()) {
					events.push_back(std::move(*event));
				}
				continue;
			}
			if (auto const taken = take_line(); !taken) {
				return std::unexpected(taken.error());
			}
			line_.clear();
		}
		return events;
	}

private:
	enum class After : std::uint8_t {
		Text,
		CarriageReturn,
	};

	/** A blank line: the event gathered so far is complete, unless no data line was. */
	[[nodiscard]] auto dispatch() -> std::optional<std::string> {
		if (data_.empty()) {
			return std::nullopt;
		}
		data_.pop_back();
		return std::exchange(data_, std::string{});
	}

	[[nodiscard]] auto take_line() -> core::Result<void, WireFault> {
		auto const line = std::string_view{line_};
		auto const colon = line.find(':');
		// A line that starts with a colon is a comment; its name is empty, like any field the client passes over.
		if (line.substr(0, colon) != data_field) {
			return {};
		}
		auto value = colon == std::string_view::npos ? std::string_view{} : line.substr(colon + 1);
		if (value.starts_with(' ')) {
			value.remove_prefix(1);
		}
		// The line feed that joins this line to the next is dropped again when the event is dispatched.
		if (data_.size() + value.size() > max_event_bytes) {
			return broke(Fault::EventTooLarge);
		}
		data_.append(value);
		data_.push_back('\n');
		return {};
	}

	std::string line_;
	std::string data_;
	After after_ = After::Text;
};

}

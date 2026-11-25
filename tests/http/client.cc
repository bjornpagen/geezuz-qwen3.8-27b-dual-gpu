// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http_test_client;

import std;
import engine.net;

namespace engine::http_test_client {

/**
 * The client side of the server tests, written from the protocol and not from the server: a response reader for
 * Content-Length and chunked bodies, and an event-stream decoder that follows the WHATWG parsing rules. It is the
 * oracle the server's bytes are checked against.
 */
export struct Parsed {
	int status;
	/** Field names lowercased; a repeated field keeps its last value. */
	std::map<std::string, std::string> fields;
	std::string body;
};

export struct Responses {
	std::vector<Parsed> complete;
	std::size_t interim;
	/** Bytes after the last complete response. */
	std::size_t rest;
	/** A complete response's framing was broken. */
	bool malformed;
};

[[nodiscard]] auto lower(std::string_view text) -> std::string {
	auto out = std::string{text};
	std::ranges::transform(out, out.begin(), [](char c) {
		return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c;
	});
	return out;
}

[[nodiscard]] auto digits(std::string_view text, std::size_t base) -> std::optional<std::size_t> {
	constexpr std::string_view alphabet = "0123456789abcdef";
	if (text.empty()) {
		return std::nullopt;
	}
	auto value = std::size_t{0};
	for (auto const c : lower(text)) {
		auto const digit = alphabet.substr(0, base).find(c);
		if (digit == std::string_view::npos || value > std::numeric_limits<std::size_t>::max() / base - 1) {
			return std::nullopt;
		}
		value = (value * base) + digit;
	}
	return value;
}

struct Incomplete {};

struct Broken {};

struct Body {
	std::string data;
	/** How many bytes the body spanned on the wire. */
	std::size_t span;
};

using Framed = std::variant<Incomplete, Broken, Body>;

/** A chunked body at the start of `bytes`. */
[[nodiscard]] auto dechunk(std::string_view bytes) -> Framed {
	auto data = std::string{};
	auto offset = std::size_t{0};
	for (;;) {
		auto const line_end = bytes.find("\r\n", offset);
		if (line_end == std::string_view::npos) {
			return Incomplete{};
		}
		auto const size = digits(bytes.substr(offset, line_end - offset), 16);
		if (!size) {
			return Broken{};
		}
		auto const start = line_end + 2;
		if (bytes.size() < start + *size + 2) {
			return Incomplete{};
		}
		if (bytes.substr(start + *size, 2) != "\r\n") {
			return Broken{};
		}
		data += bytes.substr(start, *size);
		offset = start + *size + 2;
		if (*size == 0) {
			return Body{.data = std::move(data), .span = offset};
		}
	}
}

/** A body of `length` bytes at the start of `bytes`. */
[[nodiscard]] auto sized(std::string_view bytes, std::string_view length) -> Framed {
	auto const size = digits(length, 10);
	if (!size) {
		return Broken{};
	}
	if (bytes.size() < *size) {
		return Incomplete{};
	}
	return Body{.data = std::string{bytes.substr(0, *size)}, .span = *size};
}

export [[nodiscard]] auto parse_responses(std::string_view bytes) -> Responses {
	auto out = Responses{.complete = {}, .interim = 0, .rest = bytes.size(), .malformed = false};
	auto offset = std::size_t{0};
	for (;;) {
		auto const head_end = bytes.find("\r\n\r\n", offset);
		if (head_end == std::string_view::npos) {
			return out;
		}
		auto lines = std::views::split(bytes.substr(offset, head_end - offset), std::string_view{"\r\n"});
		auto line = lines.begin();
		auto const status_line = std::string_view{*line};
		auto const status = digits(status_line.substr(std::min<std::size_t>(9, status_line.size()), 3), 10);
		if (!status_line.starts_with("HTTP/1.1 ") || !status) {
			out.malformed = true;
			return out;
		}
		auto parsed = Parsed{.status = static_cast<int>(*status), .fields = {}, .body = ""};
		for (++line; line != lines.end(); ++line) {
			auto const text = std::string_view{*line};
			auto const colon = text.find(':');
			if (colon == std::string_view::npos) {
				out.malformed = true;
				return out;
			}
			auto value = text.substr(colon + 1);
			while (value.starts_with(' ')) {
				value.remove_prefix(1);
			}
			parsed.fields[lower(text.substr(0, colon))] = std::string{value};
		}
		auto const rest = bytes.substr(head_end + 4);
		if (parsed.status == 100) {
			++out.interim;
			offset = head_end + 4;
			out.rest = bytes.size() - offset;
			continue;
		}
		auto const length = parsed.fields.find("content-length");
		auto framed = length != parsed.fields.end()                 ? sized(rest, length->second)
		              : parsed.fields.contains("transfer-encoding") ? dechunk(rest)
		                                                            : Framed{Broken{}};
		if (std::holds_alternative<Broken>(framed)) {
			out.malformed = true;
			return out;
		}
		if (std::holds_alternative<Incomplete>(framed)) {
			return out;
		}
		auto& body = std::get<Body>(framed);
		parsed.body = std::move(body.data);
		offset = head_end + 4 + body.span;
		out.complete.push_back(std::move(parsed));
		out.rest = bytes.size() - offset;
	}
}

/** Decodes an event stream: the data of every dispatched event, in order. */
export [[nodiscard]] auto decode_events(std::string_view stream) -> std::vector<std::string> {
	auto events = std::vector<std::string>{};
	auto data = std::string{};
	auto const take_line = [&stream] -> std::optional<std::string_view> {
		auto const end = stream.find_first_of("\r\n");
		if (end == std::string_view::npos) {
			return std::nullopt;
		}
		auto const line = stream.substr(0, end);
		stream.remove_prefix(end + (stream.substr(end).starts_with("\r\n") ? 2 : 1));
		return line;
	};
	while (auto const line = take_line()) {
		if (line->empty()) {
			if (!data.empty()) {
				data.pop_back();
				events.push_back(std::exchange(data, std::string{}));
			}
			continue;
		}
		auto const colon = line->find(':');
		auto const name = line->substr(0, colon);
		auto value = colon == std::string_view::npos ? std::string_view{} : line->substr(colon + 1);
		if (value.starts_with(' ')) {
			value.remove_prefix(1);
		}
		if (name == "data") {
			data += value;
			data += '\n';
		}
	}
	return events;
}

/** What an event's data reads as after a round trip: every CRLF and lone CR becomes LF. */
export [[nodiscard]] auto normalized(std::string_view data) -> std::string {
	auto out = std::string{};
	for (auto index = std::size_t{0}; index < data.size(); ++index) {
		if (data[index] == '\r') {
			out += '\n';
			if (index + 1 < data.size() && data[index + 1] == '\n') {
				++index;
			}
		} else {
			out += data[index];
		}
	}
	return out;
}

export enum class Peer : std::uint8_t {
	Open,
	Closed,
};

/** A nonblocking client connection that collects what the server sends. */
export struct Client {
	net::Socket socket;
	std::string received;
	Peer peer;

	/** Reads everything the kernel holds now. */
	auto pull() -> void {
		auto buffer = std::array<char, 65536>{};
		while (peer == Peer::Open) {
			auto const got = socket.receive(buffer);
			if (std::holds_alternative<net::Transferred>(got)) {
				received.append(std::string_view{buffer}.substr(0, std::get<net::Transferred>(got).count));
			} else if (std::holds_alternative<net::WouldBlock>(got)) {
				return;
			} else {
				peer = Peer::Closed;
			}
		}
	}

	/** Sends what the kernel takes now; the count sent. */
	[[nodiscard]] auto push(std::string_view bytes) -> std::size_t {
		auto sent = std::size_t{0};
		while (sent < bytes.size()) {
			auto const outcome = socket.send(bytes.substr(sent));
			if (!std::holds_alternative<net::Transferred>(outcome)) {
				break;
			}
			sent += std::get<net::Transferred>(outcome).count;
		}
		return sent;
	}
};

export [[nodiscard]] auto connect(std::uint16_t port) -> std::optional<Client> {
	auto socket = net::Socket::connect(net::Endpoint{.address = net::loopback, .port = port});
	if (!socket) {
		return std::nullopt;
	}
	return Client{.socket = std::move(*socket), .received = "", .peer = Peer::Open};
}

}

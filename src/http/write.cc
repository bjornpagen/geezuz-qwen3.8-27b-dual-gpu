// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:write;

import std;
import engine.core;
import :error;
import :limits;
import :request;
import :status;

namespace engine::http {

/**
 * A whole response. The writer owns its framing: the status line, Content-Type, Content-Length, Connection and
 * X-Request-Id are derived, so a handler cannot frame a response wrongly or inject a field.
 */
export struct Response {
	StatusCode status;
	ContentType content_type;
	std::string body;
};

/** One server-sent event. Its data may hold line breaks; each line becomes one `data:` field. */
export struct Event {
	std::string data;
};

constexpr std::string_view interim_continue = "HTTP/1.1 100 Continue\r\n\r\n";
constexpr std::string_view stream_end = "0\r\n\r\n";

[[nodiscard]] constexpr auto connection_value(Persistence persistence) -> std::string_view {
	switch (persistence) {
	case Persistence::KeepAlive:
		return "keep-alive";
	case Persistence::Close:
		return "close";
	}
	return "close";
}

[[nodiscard]] auto encode_whole(StatusCode status, std::string_view type, std::string_view body, std::optional<std::string_view> allow,
                                Persistence persistence, RequestId const& id) -> std::string {
	auto text = std::format("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\nX-Request-Id: {}\r\n",
	                        std::to_underlying(status), reason(status), type, body.size(), connection_value(persistence), id.text());
	if (allow) {
		text += std::format("Allow: {}\r\n", *allow);
	}
	text += "\r\n";
	text += body;
	return text;
}

/** The bytes of a whole response; the caller has checked the body against max_response_body_bytes. */
[[nodiscard]] auto encode_response(Response const& response, Persistence persistence, RequestId const& id) -> std::string {
	core::invariant(response.body.size() <= max_response_body_bytes);
	return encode_whole(response.status, media_type(response.content_type), response.body, std::nullopt, persistence, id);
}

/**
 * OpenAI's error object, `{"error": {"message", "type", "param", "code"}}`: every client of this server is an
 * OpenAI client. The parts are JsonText, checked at compile time, so the document is valid JSON by construction.
 */
[[nodiscard]] auto error_document(JsonText message, JsonText type, JsonText code) -> std::string {
	return std::format(R"({{"error":{{"message":"{}","type":"{}","param":null,"code":"{}"}}}})", message.text, type.text, code.text);
}

/** A refused request's response. Its connection always closes: the bytes after a refused head cannot be framed. */
[[nodiscard]] auto encode_refusal(HttpErrorKind kind, RequestId const& id) -> std::string {
	auto const body = error_document(explanation(kind), JsonText{"invalid_request_error"}, code_text(kind));
	return encode_whole(status_of(kind), media_type(ContentType::Json), body, allowed_method(kind), Persistence::Close, id);
}

/** The 500 that replaces a whole response longer than max_response_body_bytes. */
[[nodiscard]] auto encode_oversized(Persistence persistence, RequestId const& id) -> std::string {
	auto const body =
	    error_document(JsonText{"the response is too large to send whole"}, JsonText{"server_error"}, JsonText{"response_too_large"});
	return encode_whole(StatusCode::InternalServerError, media_type(ContentType::Json), body, std::nullopt, persistence, id);
}

[[nodiscard]] auto encode_stream_head(Persistence persistence, RequestId const& id) -> std::string {
	return std::format("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: "
	                   "chunked\r\nConnection: {}\r\nX-Request-Id: {}\r\n\r\n",
	                   connection_value(persistence), id.text());
}

/** Folds `step` over the lines of `data`, split at CRLF, CR or LF as an event stream reader splits them. */
template<class T, class F>
[[nodiscard]] auto fold_lines(std::string_view data, T total, F step) -> T {
	for (;;) {
		auto const end = data.find_first_of("\r\n");
		if (end == std::string_view::npos) {
			return step(std::move(total), data);
		}
		total = step(std::move(total), data.substr(0, end));
		auto const width = data.substr(end).starts_with("\r\n") ? std::size_t{2} : std::size_t{1};
		data.remove_prefix(end + width);
	}
}

constexpr std::string_view data_field = "data: ";

[[nodiscard]] constexpr auto hex_width(std::size_t value) -> std::size_t {
	auto width = std::size_t{1};
	while (value >= 16) {
		value /= 16;
		++width;
	}
	return width;
}

[[nodiscard]] auto event_payload_size(Event const& event) -> std::size_t {
	return fold_lines(event.data, std::size_t{1}, [](std::size_t size, std::string_view line) {
		return size + data_field.size() + line.size() + 1;
	});
}

/** The framed size of an event: one chunk holding its `data:` lines and the blank line that ends it. */
[[nodiscard]] auto encoded_size(Event const& event) -> std::size_t {
	auto const payload = event_payload_size(event);
	return hex_width(payload) + 2 + payload + 2;
}

/**
 * The framed size of the longest event: every data byte a line break, each costing one `data: ` field and its
 * newline, plus the blank line and the chunk's framing.
 */
constexpr std::size_t max_encoded_event_bytes = [] {
	constexpr auto payload = (7 * max_event_bytes) + 1;
	return hex_width(payload) + 2 + payload + 2;
}();

// The longest event fits beside the stream's end in an empty buffer, so a send always proceeds once the client reads.
static_assert(max_encoded_event_bytes + stream_end.size() <= max_stream_buffer_bytes);

/** Writes the framed event into `into`, which is exactly `encoded_size(event)` bytes. */
auto encode_event(Event const& event, std::span<char> into) -> void {
	auto cursor =
	    fold_lines(event.data, std::format_to(into.begin(), "{:x}\r\n", event_payload_size(event)), [](auto out, std::string_view line) {
		    out = std::ranges::copy(data_field, out).out;
		    out = std::ranges::copy(line, out).out;
		    *out++ = '\n';
		    return out;
	    });
	*cursor++ = '\n';
	*cursor++ = '\r';
	*cursor++ = '\n';
	core::invariant(cursor == into.end());
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:head;

import std;
import engine.core;
import :error;
import :limits;
import :request;

namespace engine::http {

/** The route a head names, before its body is read. */
enum class Target : std::uint8_t {
	ChatCompletions,
	Models,
	Health,
	Status,
};

/** The client sent `Expect: 100-continue` and waits for an interim response before its body. */
enum class Expectation : std::uint8_t {
	None,
	Continue,
};

/** What a request head says once parsed: everything the server needs to read the body and answer. */
struct Head {
	Target target;
	std::optional<RequestId> client_id;
	Persistence persistence;
	/** Content-Length; zero for every GET and for a chat request that declares an empty body. */
	std::size_t body_length;
	Expectation expectation;
};

constexpr std::string_view crlf = "\r\n";

[[nodiscard]] constexpr auto is_tchar(char c) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || std::string_view{"!#$%&'*+-.^_`|~"}.contains(c);
}

[[nodiscard]] constexpr auto is_token(std::string_view text) -> bool {
	return !text.empty() && std::ranges::all_of(text, is_tchar);
}

[[nodiscard]] constexpr auto is_vchar(char c) -> bool {
	return c > '\x20' && c != '\x7f';
}

[[nodiscard]] constexpr auto is_field_char(char c) -> bool {
	return is_vchar(c) || c == ' ' || c == '\t';
}

[[nodiscard]] constexpr auto ascii_lower(char c) -> char {
	return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
}

[[nodiscard]] constexpr auto equal_fold(std::string_view left, std::string_view right) -> bool {
	return std::ranges::equal(left, right, {}, ascii_lower, ascii_lower);
}

[[nodiscard]] constexpr auto trim_ows(std::string_view text) -> std::string_view {
	auto const first = text.find_first_not_of(" \t");
	if (first == std::string_view::npos) {
		return {};
	}
	return text.substr(first, text.find_last_not_of(" \t") - first + 1);
}

[[nodiscard]] constexpr auto next_line(std::string_view& remaining) -> std::string_view {
	auto const end = remaining.find(crlf);
	if (end == std::string_view::npos) {
		return std::exchange(remaining, std::string_view{});
	}
	auto const line = remaining.substr(0, end);
	remaining.remove_prefix(end + crlf.size());
	return line;
}

[[nodiscard]] constexpr auto is_http_version(std::string_view text) -> bool {
	return text.size() == 8 && text.starts_with("HTTP/") && text[5] >= '0' && text[5] <= '9' && text[6] == '.' && text[7] >= '0' &&
	       text[7] <= '9';
}

/** One or more decimal digits that fit a size; anything else (a sign, a space, a list) is not a length. */
[[nodiscard]] constexpr auto parse_length(std::string_view text) -> std::optional<std::size_t> {
	if (text.empty() || !std::ranges::all_of(text, [](char c) {
		    return c >= '0' && c <= '9';
	    })) {
		return std::nullopt;
	}
	auto value = std::size_t{0};
	for (auto const c : text) {
		auto const digit = static_cast<std::size_t>(c - '0');
		if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
			return std::nullopt;
		}
		value = (value * 10) + digit;
	}
	return value;
}

/** Whether a comma-separated field value lists `token`, case-insensitively. */
[[nodiscard]] constexpr auto lists(std::string_view value, std::string_view token) -> bool {
	for (auto const element : std::views::split(value, ',')) {
		if (equal_fold(trim_ows(std::string_view{element}), token)) {
			return true;
		}
	}
	return false;
}

/** The last element of a comma-separated field value. */
[[nodiscard]] constexpr auto last_element(std::string_view value) -> std::string_view {
	auto const comma = value.rfind(',');
	return trim_ows(comma == std::string_view::npos ? value : value.substr(comma + 1));
}

struct Mapping {
	std::string_view path;
	std::string_view method;
	Target target;
};

constexpr auto mappings = std::array{
    Mapping{.path = "/v1/chat/completions", .method = "POST", .target = Target::ChatCompletions},
    Mapping{.path = "/v1/models", .method = "GET", .target = Target::Models},
    Mapping{.path = "/health", .method = "GET", .target = Target::Health},
    Mapping{.path = "/status", .method = "GET", .target = Target::Status},
};

/** The route for a method and request target; the query, which no route reads, is ignored. */
[[nodiscard]] constexpr auto route(std::string_view method, std::string_view target) -> core::Result<Target, HttpError> {
	auto const path = target.substr(0, target.find('?'));
	auto const found = std::ranges::find(mappings, path, &Mapping::path);
	if (found == mappings.end()) {
		return std::unexpected(HttpError{HttpErrorKind::NotFound});
	}
	if (method != found->method) {
		return std::unexpected(HttpError{found->method == "POST" ? HttpErrorKind::ExpectedPost : HttpErrorKind::ExpectedGet});
	}
	return found->target;
}

/** The fields the server reads, gathered from every header line. */
struct Fields {
	std::size_t hosts = 0;
	std::size_t lengths = 0;
	std::string_view length;
	std::optional<std::string_view> transfer_coding;
	Persistence persistence = Persistence::KeepAlive;
	std::optional<std::string_view> expect;
	std::size_t ids = 0;
	std::string_view id;
};

/**
 * Parses a request head: the bytes before its blank line, without the final CRLF CRLF. Every refusal is decided
 * here, once, so the head the server acts on is already valid. `MaxBody` is the body route's bound.
 */
template<std::size_t MaxBody>
[[nodiscard]] auto parse_head(std::string_view head) -> core::Result<Head, HttpError> {
	auto const request_line = next_line(head);
	auto const method_end = request_line.find(' ');
	auto const target_end = method_end == std::string_view::npos ? std::string_view::npos : request_line.find(' ', method_end + 1);
	if (target_end == std::string_view::npos) {
		return std::unexpected(HttpError{HttpErrorKind::Malformed});
	}
	auto const method = request_line.substr(0, method_end);
	auto const target = request_line.substr(method_end + 1, target_end - method_end - 1);
	auto const version = request_line.substr(target_end + 1);
	if (!is_token(method) || target.empty() || !std::ranges::all_of(target, is_vchar) || !is_http_version(version)) {
		return std::unexpected(HttpError{HttpErrorKind::Malformed});
	}
	if (version != "HTTP/1.1") {
		return std::unexpected(HttpError{HttpErrorKind::UnsupportedVersion});
	}

	auto fields = Fields{};
	auto count = std::size_t{0};
	while (!head.empty()) {
		auto const line = next_line(head);
		auto const colon = line.find(':');
		if (colon == std::string_view::npos) {
			return std::unexpected(HttpError{HttpErrorKind::Malformed});
		}
		auto const name = line.substr(0, colon);
		auto const value = trim_ows(line.substr(colon + 1));
		if (!is_token(name) || !std::ranges::all_of(value, is_field_char)) {
			return std::unexpected(HttpError{HttpErrorKind::Malformed});
		}
		++count;
		if (count > max_header_count) {
			return std::unexpected(HttpError{HttpErrorKind::TooManyHeaders});
		}
		if (equal_fold(name, "Host")) {
			++fields.hosts;
		} else if (equal_fold(name, "Content-Length")) {
			++fields.lengths;
			fields.length = value;
		} else if (equal_fold(name, "Transfer-Encoding")) {
			fields.transfer_coding = last_element(value);
		} else if (equal_fold(name, "Connection")) {
			fields.persistence = lists(value, "close") ? Persistence::Close : fields.persistence;
		} else if (equal_fold(name, "Expect")) {
			fields.expect = value;
		} else if (equal_fold(name, "X-Request-Id")) {
			++fields.ids;
			fields.id = value;
		}
	}

	if (fields.hosts != 1) {
		return std::unexpected(HttpError{HttpErrorKind::HostRequired});
	}
	if (fields.transfer_coding) {
		if (fields.lengths != 0) {
			return std::unexpected(HttpError{HttpErrorKind::AmbiguousFraming});
		}
		return std::unexpected(HttpError{equal_fold(*fields.transfer_coding, "chunked") ? HttpErrorKind::ChunkedUpload
		                                                                                : HttpErrorKind::UnsupportedTransferCoding});
	}
	auto const length = fields.lengths == 0 ? std::optional<std::size_t>{} : parse_length(fields.length);
	if (fields.lengths > 1 || (fields.lengths == 1 && !length)) {
		return std::unexpected(HttpError{HttpErrorKind::BadContentLength});
	}

	auto const routed = route(method, target);
	if (!routed) {
		return std::unexpected(routed.error());
	}
	if (*routed == Target::ChatCompletions && !length) {
		return std::unexpected(HttpError{HttpErrorKind::LengthRequired});
	}
	if (*routed != Target::ChatCompletions && length.value_or(0) != 0) {
		return std::unexpected(HttpError{HttpErrorKind::BodyNotAllowed});
	}
	if (length.value_or(0) > MaxBody) {
		return std::unexpected(HttpError{HttpErrorKind::BodyTooLarge});
	}
	if (fields.expect && !equal_fold(*fields.expect, "100-continue")) {
		return std::unexpected(HttpError{HttpErrorKind::UnsupportedExpectation});
	}

	return Head{
	    .target = *routed,
	    .client_id = fields.ids == 1 ? RequestId::parse(fields.id) : std::nullopt,
	    .persistence = fields.persistence,
	    .body_length = length.value_or(0),
	    .expectation = fields.expect && length.value_or(0) != 0 ? Expectation::Continue : Expectation::None,
	};
}

}

// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:status;

import std;

namespace engine::http {

/** The statuses this server sends; the reason phrase is derived, never supplied. */
export enum class StatusCode : std::uint16_t {
	Ok = 200,
	BadRequest = 400,
	NotFound = 404,
	MethodNotAllowed = 405,
	LengthRequired = 411,
	ContentTooLarge = 413,
	ExpectationFailed = 417,
	UnprocessableContent = 422,
	TooManyRequests = 429,
	HeaderFieldsTooLarge = 431,
	InternalServerError = 500,
	ServiceUnavailable = 503,
	VersionNotSupported = 505,
};

[[nodiscard]] constexpr auto reason(StatusCode status) -> std::string_view {
	switch (status) {
	case StatusCode::Ok:
		return "OK";
	case StatusCode::BadRequest:
		return "Bad Request";
	case StatusCode::NotFound:
		return "Not Found";
	case StatusCode::MethodNotAllowed:
		return "Method Not Allowed";
	case StatusCode::LengthRequired:
		return "Length Required";
	case StatusCode::ContentTooLarge:
		return "Content Too Large";
	case StatusCode::ExpectationFailed:
		return "Expectation Failed";
	case StatusCode::UnprocessableContent:
		return "Unprocessable Content";
	case StatusCode::TooManyRequests:
		return "Too Many Requests";
	case StatusCode::HeaderFieldsTooLarge:
		return "Request Header Fields Too Large";
	case StatusCode::InternalServerError:
		return "Internal Server Error";
	case StatusCode::ServiceUnavailable:
		return "Service Unavailable";
	case StatusCode::VersionNotSupported:
		return "HTTP Version Not Supported";
	}
	return "Unknown";
}

/** A whole response's media type. A stream is always text/event-stream. */
export enum class ContentType : std::uint8_t {
	Json,
	Text,
};

[[nodiscard]] constexpr auto media_type(ContentType type) -> std::string_view {
	switch (type) {
	case ContentType::Json:
		return "application/json";
	case ContentType::Text:
		return "text/plain; charset=utf-8";
	}
	return "application/octet-stream";
}

}

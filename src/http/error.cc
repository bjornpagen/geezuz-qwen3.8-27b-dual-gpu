// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:error;

import std;
import engine.core;
import :status;

namespace engine::http {

/** Why a request was refused. Each kind is one named response; none can succeed by sending the same bytes again. */
export enum class HttpErrorKind : std::uint8_t {
	/** The request line or a header field breaks the grammar (including bare LF and obsolete line folding). */
	Malformed,
	/** The head does not end within max_head_bytes. */
	HeadTooLarge,
	/** More than max_header_count header fields. */
	TooManyHeaders,
	/** Anything but HTTP/1.1. */
	UnsupportedVersion,
	/** Not exactly one Host field, which HTTP/1.1 requires. */
	HostRequired,
	/** Both Transfer-Encoding and Content-Length: the framing is ambiguous, the shape of request smuggling. */
	AmbiguousFraming,
	/** A Transfer-Encoding whose final coding is not chunked: the body's end cannot be found. */
	UnsupportedTransferCoding,
	/** A chunked upload: the server reads bodies by Content-Length only. */
	ChunkedUpload,
	/** Content-Length is not one decimal number, or appears twice. */
	BadContentLength,
	/** A chat request without Content-Length. */
	LengthRequired,
	/** A Content-Length past the route's body bound. */
	BodyTooLarge,
	/** A body on a GET route. */
	BodyNotAllowed,
	/** An Expect other than 100-continue. */
	UnsupportedExpectation,
	/** No route has this path. */
	NotFound,
	/** The path names a POST route. */
	ExpectedPost,
	/** The path names a GET route. */
	ExpectedGet,
};

export struct [[nodiscard]] HttpError {
	HttpErrorKind kind;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(HttpError const&) const -> bool = default;
};

export [[nodiscard]] constexpr auto status_of(HttpErrorKind kind) -> StatusCode {
	switch (kind) {
	case HttpErrorKind::Malformed:
	case HttpErrorKind::HostRequired:
	case HttpErrorKind::AmbiguousFraming:
	case HttpErrorKind::UnsupportedTransferCoding:
	case HttpErrorKind::BadContentLength:
	case HttpErrorKind::BodyNotAllowed:
		return StatusCode::BadRequest;
	case HttpErrorKind::HeadTooLarge:
	case HttpErrorKind::TooManyHeaders:
		return StatusCode::HeaderFieldsTooLarge;
	case HttpErrorKind::UnsupportedVersion:
		return StatusCode::VersionNotSupported;
	case HttpErrorKind::ChunkedUpload:
	case HttpErrorKind::LengthRequired:
		return StatusCode::LengthRequired;
	case HttpErrorKind::BodyTooLarge:
		return StatusCode::ContentTooLarge;
	case HttpErrorKind::UnsupportedExpectation:
		return StatusCode::ExpectationFailed;
	case HttpErrorKind::NotFound:
		return StatusCode::NotFound;
	case HttpErrorKind::ExpectedPost:
	case HttpErrorKind::ExpectedGet:
		return StatusCode::MethodNotAllowed;
	}
	return StatusCode::BadRequest;
}

/** The refusal's body: one line a person reads. */
[[nodiscard]] constexpr auto explanation(HttpErrorKind kind) -> std::string_view {
	switch (kind) {
	case HttpErrorKind::Malformed:
		return "malformed request\n";
	case HttpErrorKind::HeadTooLarge:
		return "request head too large\n";
	case HttpErrorKind::TooManyHeaders:
		return "too many header fields\n";
	case HttpErrorKind::UnsupportedVersion:
		return "only HTTP/1.1 is served\n";
	case HttpErrorKind::HostRequired:
		return "exactly one Host field is required\n";
	case HttpErrorKind::AmbiguousFraming:
		return "Transfer-Encoding and Content-Length together\n";
	case HttpErrorKind::UnsupportedTransferCoding:
		return "unsupported transfer coding\n";
	case HttpErrorKind::ChunkedUpload:
		return "chunked uploads are refused; send Content-Length\n";
	case HttpErrorKind::BadContentLength:
		return "invalid Content-Length\n";
	case HttpErrorKind::LengthRequired:
		return "Content-Length required\n";
	case HttpErrorKind::BodyTooLarge:
		return "request body too large\n";
	case HttpErrorKind::BodyNotAllowed:
		return "this route takes no body\n";
	case HttpErrorKind::UnsupportedExpectation:
		return "only 100-continue is supported\n";
	case HttpErrorKind::NotFound:
		return "not found\n";
	case HttpErrorKind::ExpectedPost:
	case HttpErrorKind::ExpectedGet:
		return "method not allowed\n";
	}
	return "bad request\n";
}

/** The Allow field a 405 carries; other refusals carry none. */
[[nodiscard]] constexpr auto allowed_method(HttpErrorKind kind) -> std::optional<std::string_view> {
	switch (kind) {
	case HttpErrorKind::ExpectedPost:
		return "POST";
	case HttpErrorKind::ExpectedGet:
		return "GET";
	case HttpErrorKind::Malformed:
	case HttpErrorKind::HeadTooLarge:
	case HttpErrorKind::TooManyHeaders:
	case HttpErrorKind::UnsupportedVersion:
	case HttpErrorKind::HostRequired:
	case HttpErrorKind::AmbiguousFraming:
	case HttpErrorKind::UnsupportedTransferCoding:
	case HttpErrorKind::ChunkedUpload:
	case HttpErrorKind::BadContentLength:
	case HttpErrorKind::LengthRequired:
	case HttpErrorKind::BodyTooLarge:
	case HttpErrorKind::BodyNotAllowed:
	case HttpErrorKind::UnsupportedExpectation:
	case HttpErrorKind::NotFound:
		return std::nullopt;
	}
	return std::nullopt;
}

}

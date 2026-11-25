// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:error;

import std;
import engine.core;
import :method;
import :status;

namespace engine::http {

/**
 * Text that a JSON string holds verbatim: printable ASCII with no quote and no backslash. The constructor is
 * consteval, so every refusal's fixed text is checked when the server is compiled and needs no escaping at run time.
 */
struct JsonText {
	std::string_view text;

	consteval explicit JsonText(std::string_view literal) : text{literal} {
		for (auto const c : literal) {
			// A character that would need escaping makes this a non-constant expression: the build fails here.
			core::invariant(c >= ' ' && c <= '~' && c != '"' && c != '\\');
		}
	}
};

/** Why a request was refused, when the reason is all a refusal says. None succeeds by sending the same bytes again. */
export enum class HttpErrorKind : std::uint8_t {
	/**
	 * The request line or a header field breaks the grammar (including bare LF and obsolete line folding), or the
	 * query does: an escape that is not '%' and two hex digits, or a parameter without a name.
	 */
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
	/** A request to create a response without Content-Length. */
	LengthRequired,
	/** A Content-Length past the route's body bound. */
	BodyTooLarge,
	/** A body on a route that takes none. */
	BodyNotAllowed,
	/** An Expect other than 100-continue. */
	UnsupportedExpectation,
	/** No route has this path. */
	NotFound,
	/** The path's id is not 1 to max_resource_id_bytes letters, digits, '-' or '_'. */
	InvalidResourceId,
	/** More than max_query_parameters query parameters. */
	TooManyQueryParameters,
	/** A query parameter on a route that reads none. */
	QueryNotAllowed,
};

/** The path names a route, but not for this method: a 405 whose Allow field lists `allowed`, the path's methods. */
export struct WrongMethod {
	Methods allowed;

	[[nodiscard]] constexpr auto operator==(WrongMethod const&) const -> bool = default;
};

export struct [[nodiscard]] HttpError {
	std::variant<HttpErrorKind, WrongMethod> cause;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(HttpError const&) const -> bool = default;
};

[[nodiscard]] constexpr auto status_of(HttpErrorKind kind) -> StatusCode {
	switch (kind) {
	case HttpErrorKind::Malformed:
	case HttpErrorKind::HostRequired:
	case HttpErrorKind::AmbiguousFraming:
	case HttpErrorKind::UnsupportedTransferCoding:
	case HttpErrorKind::BadContentLength:
	case HttpErrorKind::BodyNotAllowed:
	case HttpErrorKind::InvalidResourceId:
	case HttpErrorKind::TooManyQueryParameters:
	case HttpErrorKind::QueryNotAllowed:
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
	}
	return StatusCode::BadRequest;
}

[[nodiscard]] constexpr auto status_of(WrongMethod const&) -> StatusCode {
	return StatusCode::MethodNotAllowed;
}

/** The refusal's status. */
export [[nodiscard]] constexpr auto status_of(HttpError const& error) -> StatusCode {
	return std::visit(
	    [](auto const& cause) {
		    return status_of(cause);
	    },
	    error.cause);
}

[[nodiscard]] constexpr auto code_text(HttpErrorKind kind) -> JsonText {
	switch (kind) {
	case HttpErrorKind::Malformed:
		return JsonText{"malformed_request"};
	case HttpErrorKind::HeadTooLarge:
		return JsonText{"request_head_too_large"};
	case HttpErrorKind::TooManyHeaders:
		return JsonText{"too_many_header_fields"};
	case HttpErrorKind::UnsupportedVersion:
		return JsonText{"http_version_not_supported"};
	case HttpErrorKind::HostRequired:
		return JsonText{"host_required"};
	case HttpErrorKind::AmbiguousFraming:
		return JsonText{"ambiguous_framing"};
	case HttpErrorKind::UnsupportedTransferCoding:
		return JsonText{"unsupported_transfer_coding"};
	case HttpErrorKind::ChunkedUpload:
		return JsonText{"chunked_upload_refused"};
	case HttpErrorKind::BadContentLength:
		return JsonText{"invalid_content_length"};
	case HttpErrorKind::LengthRequired:
		return JsonText{"length_required"};
	case HttpErrorKind::BodyTooLarge:
		return JsonText{"request_body_too_large"};
	case HttpErrorKind::BodyNotAllowed:
		return JsonText{"body_not_allowed"};
	case HttpErrorKind::UnsupportedExpectation:
		return JsonText{"unsupported_expectation"};
	case HttpErrorKind::NotFound:
		return JsonText{"route_not_found"};
	case HttpErrorKind::InvalidResourceId:
		return JsonText{"invalid_resource_id"};
	case HttpErrorKind::TooManyQueryParameters:
		return JsonText{"too_many_query_parameters"};
	case HttpErrorKind::QueryNotAllowed:
		return JsonText{"query_not_allowed"};
	}
	return JsonText{"malformed_request"};
}

[[nodiscard]] constexpr auto code_text(WrongMethod const&) -> JsonText {
	return JsonText{"method_not_allowed"};
}

[[nodiscard]] constexpr auto code_text(HttpError const& error) -> JsonText {
	return std::visit(
	    [](auto const& cause) {
		    return code_text(cause);
	    },
	    error.cause);
}

/** The refusal's stable OpenAI error `code`, one per kind: clients and logs may match on it. */
export [[nodiscard]] constexpr auto error_code(HttpError const& error) -> std::string_view {
	return code_text(error).text;
}

/** The refusal's `message`: a sentence a person reads. */
[[nodiscard]] constexpr auto explanation(HttpErrorKind kind) -> JsonText {
	switch (kind) {
	case HttpErrorKind::Malformed:
		return JsonText{"the request line, its query or a header field is malformed"};
	case HttpErrorKind::HeadTooLarge:
		return JsonText{"the request head is too large"};
	case HttpErrorKind::TooManyHeaders:
		return JsonText{"the request has too many header fields"};
	case HttpErrorKind::UnsupportedVersion:
		return JsonText{"only HTTP/1.1 is served"};
	case HttpErrorKind::HostRequired:
		return JsonText{"exactly one Host field is required"};
	case HttpErrorKind::AmbiguousFraming:
		return JsonText{"Transfer-Encoding and Content-Length must not appear together"};
	case HttpErrorKind::UnsupportedTransferCoding:
		return JsonText{"the final transfer coding must be chunked"};
	case HttpErrorKind::ChunkedUpload:
		return JsonText{"chunked uploads are refused; send Content-Length"};
	case HttpErrorKind::BadContentLength:
		return JsonText{"Content-Length must be one decimal number"};
	case HttpErrorKind::LengthRequired:
		return JsonText{"creating a response needs Content-Length"};
	case HttpErrorKind::BodyTooLarge:
		return JsonText{"the request body is too large"};
	case HttpErrorKind::BodyNotAllowed:
		return JsonText{"this route takes no request body"};
	case HttpErrorKind::UnsupportedExpectation:
		return JsonText{"only Expect: 100-continue is supported"};
	case HttpErrorKind::NotFound:
		return JsonText{"no route has this path"};
	case HttpErrorKind::InvalidResourceId:
		return JsonText{"the id in the path is not letters, digits, - or _ of a sane length"};
	case HttpErrorKind::TooManyQueryParameters:
		return JsonText{"the query has too many parameters"};
	case HttpErrorKind::QueryNotAllowed:
		return JsonText{"this route takes no query parameters"};
	}
	return JsonText{"the request is malformed"};
}

[[nodiscard]] constexpr auto explanation(WrongMethod const&) -> JsonText {
	return JsonText{"this path does not take this method; Allow lists the methods it takes"};
}

[[nodiscard]] constexpr auto explanation(HttpError const& error) -> JsonText {
	return std::visit(
	    [](auto const& cause) {
		    return explanation(cause);
	    },
	    error.cause);
}

/** The methods a 405's Allow field lists; other refusals carry no Allow. */
[[nodiscard]] constexpr auto allowed_methods(HttpError const& error) -> std::optional<Methods> {
	if (auto const wrong = core::alternative<WrongMethod>(error.cause)) {
		return wrong->allowed;
	}
	return std::nullopt;
}

}

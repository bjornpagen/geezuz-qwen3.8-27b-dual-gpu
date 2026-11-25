// SPDX-License-Identifier: AGPL-3.0-only
export module engine.http:request;

import std;
import engine.core;
import :limits;

namespace engine::http {

[[nodiscard]] constexpr auto is_id_char(char c) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == ':';
}

/**
 * The id that tags one request in logs and in its response's X-Request-Id: 1 to max_request_id_bytes of letters,
 * digits, '-', '_', '.' or ':', so it can be echoed into a header and a log line verbatim.
 */
export class RequestId {
public:
	/** The client's id when it is sane; otherwise the server issues one. */
	[[nodiscard]] static auto parse(std::string_view text) -> std::optional<RequestId> {
		if (text.empty() || text.size() > max_request_id_bytes || !std::ranges::all_of(text, is_id_char)) {
			return std::nullopt;
		}
		return RequestId{std::string{text}};
	}

	[[nodiscard]] auto text() const -> std::string_view {
		return text_;
	}

	[[nodiscard]] auto operator==(RequestId const&) const -> bool = default;

private:
	explicit RequestId(std::string text) : text_{std::move(text)} {}

	std::string text_;

	friend class IdIssuer;
};

/** A bijection on 64-bit words (splitmix64's finalizer): distinct inputs give distinct ids. */
[[nodiscard]] constexpr auto mix(std::uint64_t value) -> std::uint64_t {
	value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
	value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
	return value ^ (value >> 31U);
}

/** Issues request ids for one server: sixteen hex digits, never repeating within 2^64 issues. */
class IdIssuer {
public:
	explicit IdIssuer(std::uint64_t seed) : seed_{seed} {}

	[[nodiscard]] auto issue() -> RequestId {
		// The golden-ratio step is odd, so the counter's images are distinct, and mix keeps them distinct.
		auto const word = mix(seed_ + (next_ * 0x9e37'79b9'7f4a'7c15ULL));
		++next_;
		return RequestId{std::format("{:016x}", word)};
	}

private:
	std::uint64_t seed_;
	std::uint64_t next_ = 0;
};

/**
 * The bounds of a request body, compile-time parameters of the route that carries one: at most `bytes` bytes
 * (Content-Length past it is refused 413), stored with `padding` readable bytes after the last, so a parser that
 * reads past the end in place needs no copy. The application instantiates them from its parser's own constants.
 */
export struct BodyBounds {
	std::size_t bytes;
	std::size_t padding;
};

export template<BodyBounds B>
class Inbound;

/** A request body, owned, with B.padding readable bytes after it (their contents carry no meaning). */
export template<BodyBounds B>
class Body {
public:
	static_assert(B.bytes > 0);

	/** A padded copy of `text`: for a service or a test that builds a request itself. */
	[[nodiscard]] static auto copy_of(std::string_view text) -> Body {
		core::invariant(text.size() <= B.bytes);
		auto body = Body{text.size()};
		std::ranges::copy(text, body.storage_.begin());
		return body;
	}

	[[nodiscard]] auto length() const -> std::size_t {
		return length_;
	}

	/** The body alone. */
	[[nodiscard]] auto text() const -> std::string_view {
		return std::string_view{storage_}.substr(0, length_);
	}

	/** The body followed by B.padding readable bytes: `length() + B.padding` bytes in all. */
	[[nodiscard]] auto padded() const -> std::span<char const> {
		return storage_;
	}

	[[nodiscard]] auto operator==(Body const& other) const -> bool {
		return text() == other.text();
	}

private:
	explicit Body(std::size_t length) : storage_(length + B.padding, '\0'), length_{length} {}

	/** Where the body's bytes are read into. */
	[[nodiscard]] auto unread(std::size_t filled) -> std::span<char> {
		return std::span{storage_}.subspan(filled, length_ - filled);
	}

	std::string storage_;
	std::size_t length_;

	friend class Inbound<B>;
};

/** POST /v1/chat/completions: the body is the client's json, unparsed. */
export template<BodyBounds B>
struct ChatCompletions {
	Body<B> body;

	[[nodiscard]] auto operator==(ChatCompletions const&) const -> bool = default;
};

/** GET /v1/models. */
export struct ListModels {
	[[nodiscard]] constexpr auto operator==(ListModels const&) const -> bool = default;
};

/** GET /health. */
export struct Health {
	[[nodiscard]] constexpr auto operator==(Health const&) const -> bool = default;
};

/** GET /status. */
export struct ServerStatus {
	[[nodiscard]] constexpr auto operator==(ServerStatus const&) const -> bool = default;
};

/** Exactly the routes this server serves; a request that names no route never reaches a service. */
export template<BodyBounds B>
using Route = std::variant<ChatCompletions<B>, ListModels, Health, ServerStatus>;

/** A request the service handles. It owns every byte it carries. */
export template<BodyBounds B>
struct Request {
	RequestId id;
	Route<B> route;

	[[nodiscard]] auto operator==(Request const&) const -> bool = default;
};

/** Whether the connection carries another request after this exchange. */
export enum class Persistence : std::uint8_t {
	KeepAlive,
	Close,
};

}

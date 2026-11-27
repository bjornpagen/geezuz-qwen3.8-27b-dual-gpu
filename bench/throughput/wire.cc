// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:wire;

import std;
import engine.core;
import :limits;

/**
 * The response side of http/1.1 as this client reads it: one response per connection, its head parsed once, its body
 * through its framing (a length or chunks), whatever pieces the bytes arrive in. What the server never sends is
 * refused by name, not read leniently: an interim response, a body delimited by the connection's end, a chunk
 * extension, a trailer.
 */
namespace engine::throughput {

/** How a response broke the protocol or passed a bound. None reads differently when the same bytes are sent again. */
export enum class Fault : std::uint8_t {
	/** The head does not end within max_head_bytes. */
	HeadTooLarge,
	/** The status line or a field breaks the grammar, or the status is an interim one nobody asked for. */
	BadHead,
	/** The body's end cannot be found: no length and no chunking, both, a length that is no number, a broken chunk. */
	BadFraming,
	/** Bytes follow the response on a connection that carries one. */
	TrailingBytes,
	/** A whole body is longer than max_body_bytes. */
	BodyTooLarge,
	/** An event's data, or one line of the event stream, is longer than max_event_bytes. */
	EventTooLarge,
	/** A 200 answer is not of the media type its route answers with. */
	UnexpectedMedia,
	/** A body or an event's data is not JSON of the shape the api gives it. */
	BadJson,
	/** An event whose type is none of the api's. */
	UnknownEvent,
	/** An event after the stream's terminal one. */
	EventAfterEnd,
	/** The stream ended without a terminal event. */
	NoTerminalEvent,
};

export struct [[nodiscard]] WireFault {
	Fault fault;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(WireFault const&) const -> bool = default;
};

[[nodiscard]] constexpr auto broke(Fault fault) -> std::unexpected<WireFault> {
	return std::unexpected(WireFault{.fault = fault});
}

/** What a response's Content-Type says its body is, as far as the client tells bodies apart. */
export enum class Media : std::uint8_t {
	Json,
	EventStream,
	Other,
};

/** What a response's head says of it, beyond its framing. */
export struct Head {
	std::uint16_t status;
	Media media;

	[[nodiscard]] constexpr auto operator==(Head const&) const -> bool = default;
};

/** Whether a response's body has arrived whole. */
export enum class Extent : std::uint8_t {
	Open,
	Complete,
};

[[nodiscard]] constexpr auto ascii_lower(char c) -> char {
	return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
}

[[nodiscard]] constexpr auto equal_fold(std::string_view left, std::string_view right) -> bool {
	return std::ranges::equal(left, right, {}, ascii_lower, ascii_lower);
}

[[nodiscard]] constexpr auto trim(std::string_view text) -> std::string_view {
	auto const first = text.find_first_not_of(" \t");
	if (first == std::string_view::npos) {
		return {};
	}
	return text.substr(first, text.find_last_not_of(" \t") - first + 1);
}

[[nodiscard]] constexpr auto is_digit(char c) -> bool {
	return c >= '0' && c <= '9';
}

/** One or more decimal digits that fit a size. */
[[nodiscard]] constexpr auto length_of(std::string_view text) -> std::optional<std::size_t> {
	if (text.empty() || !std::ranges::all_of(text, is_digit)) {
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

[[nodiscard]] constexpr auto media_of(std::string_view content_type) -> Media {
	auto const names = [content_type](std::string_view media) {
		return content_type.size() >= media.size() && equal_fold(content_type.substr(0, media.size()), media);
	};
	if (names("application/json")) {
		return Media::Json;
	}
	if (names("text/event-stream")) {
		return Media::EventStream;
	}
	return Media::Other;
}

/** The body is the next `length` bytes. */
struct Sized {
	std::size_t length;
};

/** The body is chunks, ended by the empty one. */
struct Chunked {};

using Framing = std::variant<Sized, Chunked>;

struct Framed {
	Head head;
	Framing framing;
};

/** Parses a response head: the bytes before its blank line. */
[[nodiscard]] constexpr auto parse_head(std::string_view text) -> core::Result<Framed, WireFault> {
	constexpr auto version = std::string_view{"HTTP/1.1 "};
	// Splitting no text yields no line at all, not one empty line.
	if (text.empty()) {
		return broke(Fault::BadHead);
	}
	auto lines = std::views::split(text, std::string_view{"\r\n"});
	auto line = lines.begin();
	auto const status_line = std::string_view{*line};
	auto const code = status_line.substr(std::min(version.size(), status_line.size()), 3);
	if (!status_line.starts_with(version) || code.size() != 3 || !std::ranges::all_of(code, is_digit) ||
	    (status_line.size() > version.size() + 3 && status_line[version.size() + 3] != ' ')) {
		return broke(Fault::BadHead);
	}
	auto const status = static_cast<std::uint16_t>(((code[0] - '0') * 100) + ((code[1] - '0') * 10) + (code[2] - '0'));
	// An interim response answers an Expect this client never sends, and no status is below 100.
	if (status < 200) {
		return broke(Fault::BadHead);
	}

	auto lengths = std::size_t{0};
	auto length = std::string_view{};
	auto coding = std::optional<std::string_view>{};
	auto media = Media::Other;
	for (++line; line != lines.end(); ++line) {
		auto const field = std::string_view{*line};
		auto const colon = field.find(':');
		if (colon == std::string_view::npos || colon == 0 || field.substr(0, colon).find_first_of(" \t") != std::string_view::npos) {
			return broke(Fault::BadHead);
		}
		auto const name = field.substr(0, colon);
		auto const value = trim(field.substr(colon + 1));
		if (equal_fold(name, "Content-Length")) {
			++lengths;
			length = value;
		} else if (equal_fold(name, "Transfer-Encoding")) {
			coding = value;
		} else if (equal_fold(name, "Content-Type")) {
			media = media_of(value);
		}
	}

	auto const head = Head{.status = status, .media = media};
	if (coding) {
		if (lengths != 0 || !equal_fold(*coding, "chunked")) {
			return broke(Fault::BadFraming);
		}
		return Framed{.head = head, .framing = Chunked{}};
	}
	auto const sized = lengths == 1 ? length_of(length) : std::nullopt;
	if (!sized) {
		return broke(Fault::BadFraming);
	}
	return Framed{.head = head, .framing = Sized{.length = *sized}};
}

/** A chunk's size is being read: hex digits up to its CR. */
struct SizeDigits {
	std::size_t value;
	std::size_t digits;
};

/** A size line's CR was read; its LF follows. */
struct SizeEnd {
	std::size_t size;
};

struct ChunkData {
	std::size_t remaining;
};

/** A chunk's data is over; its CR and LF follow. */
struct DataCr {};

struct DataLf {};

/** The empty chunk's size line was read: the blank line that ends the body follows, with no trailer before it. */
struct LastCr {};

struct LastLf {};

struct ChunksOver {};

using ChunkStage = std::variant<SizeDigits, SizeEnd, ChunkData, DataCr, DataLf, LastCr, LastLf, ChunksOver>;

[[nodiscard]] constexpr auto hex_value(char c) -> std::optional<std::size_t> {
	if (is_digit(c)) {
		return static_cast<std::size_t>(c - '0');
	}
	auto const lower = ascii_lower(c);
	if (lower >= 'a' && lower <= 'f') {
		return static_cast<std::size_t>(lower - 'a') + 10;
	}
	return std::nullopt;
}

/** The stage after `byte` of a chunk's framing; nullopt when the byte cannot follow. */
struct FramingByte {
	char byte;

	[[nodiscard]] constexpr auto operator()(SizeDigits const& size) const -> std::optional<ChunkStage> {
		if (auto const digit = hex_value(byte)) {
			// Sixteen hex digits fill a size; one more cannot be a length this machine holds.
			if (size.digits == 2 * sizeof(std::size_t)) {
				return std::nullopt;
			}
			return SizeDigits{.value = (size.value * 16) + *digit, .digits = size.digits + 1};
		}
		if (byte == '\r' && size.digits > 0) {
			return SizeEnd{.size = size.value};
		}
		return std::nullopt;
	}

	[[nodiscard]] constexpr auto operator()(SizeEnd const& end) const -> std::optional<ChunkStage> {
		if (byte != '\n') {
			return std::nullopt;
		}
		if (end.size == 0) {
			return LastCr{};
		}
		return ChunkData{.remaining = end.size};
	}

	[[nodiscard]] constexpr auto operator()(DataCr const& /*stage*/) const -> std::optional<ChunkStage> {
		return follows('\r', DataLf{});
	}

	[[nodiscard]] constexpr auto operator()(DataLf const& /*stage*/) const -> std::optional<ChunkStage> {
		return follows('\n', SizeDigits{.value = 0, .digits = 0});
	}

	[[nodiscard]] constexpr auto operator()(LastCr const& /*stage*/) const -> std::optional<ChunkStage> {
		return follows('\r', LastLf{});
	}

	[[nodiscard]] constexpr auto operator()(LastLf const& /*stage*/) const -> std::optional<ChunkStage> {
		return follows('\n', ChunksOver{});
	}

	// Data bytes are taken in runs and nothing follows the end: neither stage reads a framing byte.
	[[nodiscard]] constexpr auto operator()(ChunkData const& /*stage*/) const -> std::optional<ChunkStage> {
		return std::nullopt;
	}

	[[nodiscard]] constexpr auto operator()(ChunksOver const& /*stage*/) const -> std::optional<ChunkStage> {
		return std::nullopt;
	}

	[[nodiscard]] constexpr auto follows(char expected, ChunkStage next) const -> std::optional<ChunkStage> {
		if (byte != expected) {
			return std::nullopt;
		}
		return next;
	}
};

/** What some bytes of a chunked body held. */
struct Dechunked {
	std::string data;
	/** How many of the bytes belonged to the body; fewer than were given only once the body is complete. */
	std::size_t consumed;
	Extent extent;
};

/** A chunked body, read across however many pieces it arrives in. */
class Chunks {
public:
	[[nodiscard]] constexpr auto feed(std::string_view bytes) -> core::Result<Dechunked, WireFault> {
		auto out = Dechunked{.data = {}, .consumed = 0, .extent = Extent::Open};
		while (out.consumed < bytes.size() && !std::holds_alternative<ChunksOver>(stage_)) {
			if (auto const data = core::alternative<ChunkData>(stage_)) {
				auto const taken = std::min(data->remaining, bytes.size() - out.consumed);
				out.data.append(bytes.substr(out.consumed, taken));
				out.consumed += taken;
				data->remaining -= taken;
				if (data->remaining == 0) {
					stage_ = DataCr{};
				}
				continue;
			}
			auto const next = std::visit(FramingByte{.byte = bytes[out.consumed]}, stage_);
			if (!next) {
				return broke(Fault::BadFraming);
			}
			stage_ = *next;
			++out.consumed;
		}
		if (std::holds_alternative<ChunksOver>(stage_)) {
			out.extent = Extent::Complete;
		}
		return out;
	}

private:
	ChunkStage stage_ = SizeDigits{.value = 0, .digits = 0};
};

/** The body bytes some received bytes completed, and whether the response is over. */
export struct Fed {
	std::string body;
	Extent extent;

	[[nodiscard]] auto operator==(Fed const&) const -> bool = default;
};

/**
 * One response as its bytes arrive. The body comes out as it is received, unframed, so an event stream is read while
 * it is still being written; what the caller does with the pieces (keep a whole body, decode events) is its own.
 */
export class Inbound {
public:
	/** Takes the next received bytes, in order. */
	[[nodiscard]] constexpr auto feed(std::string_view bytes) -> core::Result<Fed, WireFault> {
		auto const heading = core::alternative<Heading>(stage_);
		if (!heading) {
			return body(bytes);
		}
		constexpr auto blank_line = std::string_view{"\r\n\r\n"};
		heading->held.append(bytes);
		auto const end = std::string_view{heading->held}.find(blank_line);
		if (end == std::string_view::npos) {
			if (heading->held.size() >= max_head_bytes) {
				return broke(Fault::HeadTooLarge);
			}
			return Fed{.body = {}, .extent = Extent::Open};
		}
		if (end + blank_line.size() > max_head_bytes) {
			return broke(Fault::HeadTooLarge);
		}
		auto const framed = parse_head(std::string_view{heading->held}.substr(0, end));
		if (!framed) {
			return std::unexpected(framed.error());
		}
		// Owned before the stage changes: the head's buffer goes with its stage.
		auto const rest = heading->held.substr(end + blank_line.size());
		if (auto const sized = core::alternative<Sized>(framed->framing)) {
			stage_ = Counting{.head = framed->head, .remaining = sized->length};
		} else {
			stage_ = Chunking{.head = framed->head, .chunks = {}};
		}
		return body(rest);
	}

	/** The head, once its blank line has arrived. */
	[[nodiscard]] constexpr auto head() const -> std::optional<Head> {
		if (auto const counting = core::alternative<Counting>(stage_)) {
			return counting->head;
		}
		if (auto const chunking = core::alternative<Chunking>(stage_)) {
			return chunking->head;
		}
		if (auto const over = core::alternative<Over>(stage_)) {
			return over->head;
		}
		return std::nullopt;
	}

private:
	struct Heading {
		std::string held;
	};

	struct Counting {
		Head head;
		std::size_t remaining;
	};

	struct Chunking {
		Head head;
		Chunks chunks;
	};

	struct Over {
		Head head;
	};

	[[nodiscard]] constexpr auto body(std::string_view bytes) -> core::Result<Fed, WireFault> {
		auto fed = Fed{.body = {}, .extent = Extent::Open};
		auto used = std::size_t{0};
		if (auto const counting = core::alternative<Counting>(stage_)) {
			used = std::min(counting->remaining, bytes.size());
			fed.body = bytes.substr(0, used);
			counting->remaining -= used;
			if (counting->remaining == 0) {
				stage_ = Over{.head = counting->head};
			}
		} else if (auto const chunking = core::alternative<Chunking>(stage_)) {
			auto dechunked = chunking->chunks.feed(bytes);
			if (!dechunked) {
				return std::unexpected(dechunked.error());
			}
			used = dechunked->consumed;
			fed.body = std::move(dechunked->data);
			if (dechunked->extent == Extent::Complete) {
				stage_ = Over{.head = chunking->head};
			}
		}
		if (used != bytes.size()) {
			return broke(Fault::TrailingBytes);
		}
		if (std::holds_alternative<Over>(stage_)) {
			fed.extent = Extent::Complete;
		}
		return fed;
	}

	std::variant<Heading, Counting, Chunking, Over> stage_ = Heading{.held = {}};
};

}

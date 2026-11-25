// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:write;

import std;
import engine.core;
import engine.simdjson;
import :read;

namespace engine::json {

/** Where a writer's bytes go. */
export template<class O>
concept Sink = requires(O& out, std::string_view text) { out.put(text); };

/** A sink that only counts: the first pass of every bounded append, so a failed append writes nothing. */
export class Measure {
public:
	auto put(std::string_view text) -> void {
		bytes_ += text.size();
	}

	[[nodiscard]] auto bytes() const -> std::size_t {
		return bytes_;
	}

private:
	std::size_t bytes_ = 0;
};

/** The escape for a byte that a JSON string cannot hold literally, or nothing for a byte it can. */
[[nodiscard]] constexpr auto short_escape(char byte) -> std::optional<std::string_view> {
	switch (byte) {
	case '"':
		return "\\\"";
	case '\\':
		return "\\\\";
	case '\b':
		return "\\b";
	case '\f':
		return "\\f";
	case '\n':
		return "\\n";
	case '\r':
		return "\\r";
	case '\t':
		return "\\t";
	default:
		return std::nullopt;
	}
}

[[nodiscard]] constexpr auto needs_escape(char byte) -> bool {
	return byte == '"' || byte == '\\' || std::bit_cast<std::uint8_t>(byte) < 0x20;
}

/**
 * Writes `text` as a JSON string. Only the quote, the backslash and the controls below U+0020 are escaped; every
 * other code point is copied as its UTF-8 bytes, which RFC 8259 allows and keeps output compact. The engine only
 * produces UTF-8, so text that is not is a bug upstream of the writer.
 */
template<Sink O>
auto put_string(O& out, std::string_view text) -> void {
	core::invariant(simdjson::valid_utf8(text));
	out.put("\"");
	auto at = std::size_t{0};
	while (at < text.size()) {
		auto const run_begin = at;
		while (at < text.size() && !needs_escape(text[at])) {
			++at;
		}
		if (at > run_begin) {
			out.put(text.substr(run_begin, at - run_begin));
		}
		if (at == text.size()) {
			break;
		}
		if (auto const escape = short_escape(text[at])) {
			out.put(*escape);
		} else {
			auto constexpr hex = std::string_view{"0123456789abcdef"};
			auto const value = std::uint32_t{std::bit_cast<std::uint8_t>(text[at])};
			auto const code = std::array{'\\', 'u', '0', '0', hex[value >> 4U], hex[value & 0xFU]};
			out.put(std::string_view{code.data(), code.size()});
		}
		++at;
	}
	out.put("\"");
}

template<Sink O, std::integral I>
auto put_integer(O& out, I value) -> void {
	auto digits = std::array<char, 24>{};
	auto const written = std::to_chars(digits.data(), std::to_address(digits.end()), value);
	core::invariant(!std::make_error_code(written.ec));
	out.put(std::string_view{digits.data(), written.ptr});
}

/**
 * Writes a finite double as the shortest text that reads back as exactly the same double, with `.0` added when that
 * text has neither a fraction nor an exponent, so it reads back as a double and not as an integer. JSON has no
 * spelling for an infinity or a NaN, so writing one is a bug in the caller.
 */
template<Sink O>
auto put_real(O& out, double value) -> void {
	core::invariant(std::isfinite(value));
	auto digits = std::array<char, 32>{};
	auto const written = std::to_chars(digits.data(), std::to_address(digits.end()), value);
	core::invariant(!std::make_error_code(written.ec));
	auto const text = std::string_view{digits.data(), written.ptr};
	out.put(text);
	if (text.find_first_of(".e") == std::string_view::npos) {
		out.put(".0");
	}
}

/** The integer types a writer prints exactly; `bool` is a JSON literal, not a number. */
export template<class I>
concept Integer = std::integral<I> && !std::same_as<I, bool>;

export template<Sink O>
class Elements;

/**
 * Writes the members of one JSON object, in call order. It cannot be left open: `object` and `array` hand a nested
 * writer to a callable and close the container when it returns. Member names are the caller's to keep unique.
 */
export template<Sink O>
class Members {
public:
	explicit Members(O& out) : out_{out} {}

	auto string(std::string_view name, std::string_view value) -> void {
		key(name);
		put_string(out_, value);
	}

	template<Integer I>
	auto integer(std::string_view name, I value) -> void {
		key(name);
		put_integer(out_, value);
	}

	auto real(std::string_view name, double value) -> void {
		key(name);
		put_real(out_, value);
	}

	auto boolean(std::string_view name, bool value) -> void {
		key(name);
		out_.put(value ? "true" : "false");
	}

	auto null(std::string_view name) -> void {
		key(name);
		out_.put("null");
	}

	auto json(std::string_view name, JsonText const& value) -> void {
		key(name);
		out_.put(value.text());
	}

	template<std::invocable<Members<O>&> F>
	auto object(std::string_view name, F const& fill) -> void {
		key(name);
		out_.put("{");
		auto inner = Members<O>{out_};
		fill(inner);
		out_.put("}");
	}

	template<std::invocable<Elements<O>&> F>
	auto array(std::string_view name, F const& fill) -> void {
		key(name);
		out_.put("[");
		auto inner = Elements<O>{out_};
		fill(inner);
		out_.put("]");
	}

private:
	auto key(std::string_view name) -> void {
		if (written_ > 0) {
			out_.put(",");
		}
		++written_;
		put_string(out_, name);
		out_.put(":");
	}

	O& out_;
	std::size_t written_ = 0;
};

/** Writes the elements of one JSON array, in call order; as `Members`. */
export template<Sink O>
class Elements {
public:
	explicit Elements(O& out) : out_{out} {}

	auto string(std::string_view value) -> void {
		next();
		put_string(out_, value);
	}

	template<Integer I>
	auto integer(I value) -> void {
		next();
		put_integer(out_, value);
	}

	auto real(double value) -> void {
		next();
		put_real(out_, value);
	}

	auto boolean(bool value) -> void {
		next();
		out_.put(value ? "true" : "false");
	}

	auto null() -> void {
		next();
		out_.put("null");
	}

	auto json(JsonText const& value) -> void {
		next();
		out_.put(value.text());
	}

	template<std::invocable<Members<O>&> F>
	auto object(F const& fill) -> void {
		next();
		out_.put("{");
		auto inner = Members<O>{out_};
		fill(inner);
		out_.put("}");
	}

	template<std::invocable<Elements<O>&> F>
	auto array(F const& fill) -> void {
		next();
		out_.put("[");
		auto inner = Elements<O>{out_};
		fill(inner);
		out_.put("]");
	}

private:
	auto next() -> void {
		if (written_ > 0) {
			out_.put(",");
		}
		++written_;
	}

	O& out_;
	std::size_t written_ = 0;
};

/** Writes one JSON object as a whole document. */
export template<Sink O, std::invocable<Members<O>&> F>
auto write_object(O& out, F const& fill) -> void {
	out.put("{");
	auto members = Members<O>{out};
	fill(members);
	out.put("}");
}

/** Writes one JSON array as a whole document. */
export template<Sink O, std::invocable<Elements<O>&> F>
auto write_array(O& out, F const& fill) -> void {
	out.put("[");
	auto elements = Elements<O>{out};
	fill(elements);
	out.put("]");
}

/** An append did not fit: it needed `needed` bytes where `available` remained. Nothing was written. */
export struct [[nodiscard]] Overflow {
	std::size_t needed;
	std::size_t available;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(Overflow const&) const -> bool = default;
};

/**
 * A byte buffer whose capacity its owner fixes at setup: appending never reallocates. A connection keeps one and
 * appends each frame's payload to it; an append that does not fit fails whole and leaves the buffer as it was.
 */
export class Buffer {
public:
	explicit Buffer(std::size_t capacity) : capacity_{capacity} {
		bytes_.reserve(capacity);
	}

	[[nodiscard]] auto text() const -> std::string_view {
		return bytes_;
	}

	[[nodiscard]] auto available() const -> std::size_t {
		return capacity_ - bytes_.size();
	}

	auto clear() -> void {
		bytes_.clear();
	}

	/** Appends raw bytes the caller has checked fit; past the capacity is a bug in the caller. */
	auto put(std::string_view text) -> void {
		core::invariant(text.size() <= available());
		bytes_.append(text);
	}

	/**
	 * Runs `write` once against a `Measure` and, when its bytes fit, once against this buffer; `write` must be a
	 * pure function of its sink, writing the same bytes to both.
	 */
	template<class W>
	    requires std::invocable<W const&, Measure&> && std::invocable<W const&, Buffer&>
	[[nodiscard]] auto append(W const& write) -> core::Result<void, Overflow> {
		auto measure = Measure{};
		write(measure);
		if (measure.bytes() > available()) {
			return std::unexpected(Overflow{.needed = measure.bytes(), .available = available()});
		}
		auto const before = bytes_.size();
		write(*this);
		core::invariant(bytes_.size() - before == measure.bytes());
		return {};
	}

	/** The written bytes, ending the buffer's life. */
	[[nodiscard]] auto take() && -> std::string {
		return std::move(bytes_);
	}

private:
	std::string bytes_;
	std::size_t capacity_;
};

/** Writes into a string reserved to exactly the size `write` produces. */
export template<class W>
    requires std::invocable<W const&, Measure&> && std::invocable<W const&, Buffer&>
[[nodiscard]] auto encode(W const& write) -> std::string {
	auto measure = Measure{};
	write(measure);
	auto buffer = Buffer{measure.bytes()};
	auto const appended = buffer.append(write);
	core::invariant(appended.has_value());
	return std::move(buffer).take();
}

}

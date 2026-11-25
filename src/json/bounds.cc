// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:bounds;

import std;

namespace engine::json {

/**
 * The bounds a read enforces, each refused with its own fault when exceeded. Different inputs have different
 * sizes (a chat request, a safetensors header, a model config), so each reader declares its own `Bounds` as a
 * constant and reads with it as a template argument: the bounds are compile-time facts, never runtime options.
 */
export struct Bounds {
	/** The longest text, in bytes; the reader's parser and copy buffer are sized to it once. */
	std::size_t document_bytes;
	/** The deepest nesting of arrays and objects; the outermost container is depth 1. */
	std::size_t depth;
	/** The longest string after unescaping, in bytes; member names included. */
	std::size_t string_bytes;
	/** The most elements in one array, or members in one object. */
	std::size_t members;
	/** The longest number literal, in characters. */
	std::size_t number_chars;
};

/** Bounds a reader can enforce: each admits at least one of what it bounds. */
export template<Bounds B>
concept Enforceable = B.document_bytes > 0 && B.depth > 0 && B.string_bytes > 0 && B.members > 0 && B.number_chars > 0;

/**
 * The bounds of an OpenAI chat request. The document holds 64 bytes per token of the model's 262,144-token context:
 * room for a full context sent with every non-ASCII character escaped as `\uXXXX`, plus the tool schemas. A string
 * holds four times the context at four bytes per token, so no message that could fit the context is refused and one
 * string cannot take the whole document. Every serializer writes a double in at most 25 characters; the number
 * bound leaves room for exact decimal expansions of doubles of moderate magnitude.
 */
export inline constexpr Bounds request_bounds{
    .document_bytes = std::size_t{16} << 20U,
    .depth = 64,
    .string_bytes = std::size_t{4} << 20U,
    .members = std::size_t{1} << 16U,
    .number_chars = 128,
};

static_assert(Enforceable<request_bounds>);

}

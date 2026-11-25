// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:codec;

import std;
import engine.core;
import engine.tokenizer_image;
import :added;
import :bpe;
import :image;
import :split;
import :unicode;
import :utf8;

namespace engine::tokenizer {

/** How an added token's spelling inside text is read. */
export enum class Spelling : std::uint8_t {
	/** As that added token: how Hugging Face reads a rendered chat template, and what the model was trained on. */
	Token,
	/** As the characters it is made of, so text from a client cannot forge a control token. */
	Characters,
};

/** Utf-8 text to encode, borrowed for the call. */
export struct Text {
	std::string_view utf8;
	Spelling spelling;
};

/**
 * One piece of an input: text, or an added token placed by the caller. Consecutive text pieces are read as one run
 * of text (the pretokenizer sees across their boundary); an added token, placed or spelled, ends the run. A spelling
 * is recognized only inside one `Spelling::Token` text, never across two pieces.
 */
export using Piece = std::variant<Text, Added>;

/** The input is not utf-8: piece `piece`, from byte `offset`, is not a valid sequence. */
export struct [[nodiscard]] EncodeError {
	std::size_t piece;
	std::size_t offset;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(EncodeError const&) const -> bool = default;
};

/** The id names no token: the model's lm_head is padded past the tokenizer's vocabulary. */
export struct [[nodiscard]] DecodeError {
	Id id;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(DecodeError const&) const -> bool = default;
};

/**
 * Qwen's byte-level byte-pair tokenizer, exactly as Hugging Face `tokenizers` runs the checkpoint's tokenizer.json:
 * added tokens split the input; each run between them is normalized to NFC, split by the pretokenizer's pattern, and
 * each piece's utf-8 bytes are merged by rank.
 *
 * Encoding is O(n log n) in the input length n: normalization and the split are linear (table lookups are binary
 * searches over fixed tables), and merging a word of k bytes is O(k log k). It allocates a fixed number of O(n)
 * buffers per run of text between added tokens and reuses them across words, never allocating per word.
 */
export class Tokenizer {
public:
	/** Reads and checks a tokenizer image. */
	[[nodiscard]] static auto parse(std::span<std::byte const> image) -> core::Result<Tokenizer, LoadError> {
		auto tables = read_image(image);
		if (!tables) {
			return std::unexpected(tables.error());
		}
		return Tokenizer{std::move(*tables)};
	}

	Tokenizer(Tokenizer const&) = delete;
	auto operator=(Tokenizer const&) -> Tokenizer& = delete;
	Tokenizer(Tokenizer&&) noexcept = default;
	auto operator=(Tokenizer&&) noexcept -> Tokenizer& = default;
	~Tokenizer() = default;

	[[nodiscard]] auto encode(std::span<Piece const> pieces) const -> core::Result<std::vector<Id>, EncodeError> {
		auto scratch = Scratch{};
		auto run = std::vector<char32_t>{};
		for (auto index = std::size_t{0}; index < pieces.size(); ++index) {
			if (std::holds_alternative<Added>(pieces[index])) {
				flush(scratch, run);
				scratch.ids.push_back(id(std::get<Added>(pieces[index])));
				continue;
			}
			auto const text = std::get<Text>(pieces[index]);
			auto from = std::size_t{0};
			while (true) {
				auto const spelled = text.spelling == Spelling::Token ? find_spelled(text.utf8, from) : std::optional<Spelled>{};
				auto const until = spelled ? spelled->begin : text.utf8.size();
				auto const decoded = scalars(text.utf8.substr(from, until - from));
				if (!decoded) {
					return std::unexpected(EncodeError{.piece = index, .offset = from + decoded.error().offset});
				}
				run.insert(run.end(), decoded->begin(), decoded->end());
				if (!spelled) {
					break;
				}
				flush(scratch, run);
				scratch.ids.push_back(id(spelled->token));
				from = spelled->begin + entry(spelled->token).spelling.size();
			}
		}
		flush(scratch, run);
		return std::move(scratch.ids);
	}

	/** The bytes a token decodes to, if the id names a token. */
	[[nodiscard]] auto token(Id id) const -> std::optional<std::string_view> {
		if (id < bpe_vocabulary) {
			return tables_.token(id);
		}
		if (auto const named = added(id)) {
			return entry(*named).spelling;
		}
		return std::nullopt;
	}

	/**
	 * The text of a token sequence: every token's bytes, concatenated, with each maximal invalid utf-8 subpart replaced
	 * by U+FFFD, as Hugging Face's byte-level decoder does. Added tokens, special ones included, decode to their
	 * spelling.
	 */
	[[nodiscard]] auto decode(std::span<Id const> ids) const -> core::Result<std::string, DecodeError>;

private:
	explicit Tokenizer(Tables tables) : tables_{std::move(tables)} {}

	/** Encodes one run of text between added tokens and empties it. */
	auto flush(Scratch& scratch, std::vector<char32_t>& run) const -> void {
		if (run.empty()) {
			return;
		}
		auto const normalized = nfc(tables_.unicode, run);
		run.clear();
		auto kinds = std::vector<Kind>{};
		kinds.reserve(normalized.size());
		for (auto const point : normalized) {
			kinds.push_back(kind_of(tables_.unicode, point));
		}
		auto word = std::vector<Id>{};
		split(normalized, kinds, [&](std::size_t begin, std::size_t end) {
			word.clear();
			for (auto const point : std::span{normalized}.subspan(begin, end - begin)) {
				for (auto const byte : utf8(point).view()) {
					word.push_back(tables_.byte_tokens.at(std::bit_cast<std::uint8_t>(byte)));
				}
			}
			merge_word(tables_.merges, scratch, word);
		});
	}

	Tables tables_;
};

/** The engine's tokenizer, read from the image built into the binary. */
export [[nodiscard]] auto load() -> core::Result<Tokenizer, LoadError> {
	return Tokenizer::parse(tokenizer_image::bytes());
}

/**
 * Turns a stream of token ids into text as they arrive. A token can end inside a multi-byte character; those bytes
 * are held back until the character completes, so every piece returned is utf-8 and the pieces concatenate to exactly
 * what `Tokenizer::decode` returns for the whole sequence. At most three bytes are ever held.
 */
export class Detokenizer {
public:
	/** The text the token completes, possibly empty. On failure nothing changes. */
	[[nodiscard]] auto push(Tokenizer const& tokenizer, Id id) -> core::Result<std::string, DecodeError> {
		auto const bytes = tokenizer.token(id);
		if (!bytes) {
			return std::unexpected(DecodeError{.id = id});
		}
		pending_.append(*bytes);
		auto out = std::string{};
		auto const view = std::string_view{pending_};
		auto at = std::size_t{0};
		while (at < view.size()) {
			auto const next = step(view.substr(at));
			if (std::holds_alternative<Incomplete>(next)) {
				break;
			}
			if (std::holds_alternative<Invalid>(next)) {
				out.append(replacement);
				at += std::get<Invalid>(next).length;
				continue;
			}
			auto const length = std::get<Scalar>(next).length;
			out.append(view.substr(at, length));
			at += length;
		}
		pending_.erase(0, at);
		core::invariant(pending_.size() < 4);
		return out;
	}

	/** Ends the stream: bytes still held can no longer complete, so they become one U+FFFD. */
	[[nodiscard]] auto finish() -> std::string {
		auto out = pending_.empty() ? std::string{} : std::string{replacement};
		pending_.clear();
		return out;
	}

private:
	/** A proper prefix of one character's encoding. */
	std::string pending_;
};

auto Tokenizer::decode(std::span<Id const> ids) const -> core::Result<std::string, DecodeError> {
	auto stream = Detokenizer{};
	auto out = std::string{};
	for (auto const token : ids) {
		auto const piece = stream.push(*this, token);
		if (!piece) {
			return std::unexpected(piece.error());
		}
		out.append(*piece);
	}
	out.append(stream.finish());
	return out;
}

}

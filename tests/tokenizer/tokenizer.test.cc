// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.testing;
import engine.tokenizer;
import engine.tokenizer_image;
import engine_test_tokenizer.fixtures;

namespace {

namespace tok = engine::tokenizer;
using tok::Added;
using tok::Id;
using tok::Piece;
using tok::Spelling;
using tok::Text;
using tok::Tokenizer;

[[nodiscard]] auto encode(Tokenizer const& tokenizer, std::string_view text, Spelling spelling) -> std::optional<std::vector<Id>> {
	auto const pieces = std::array{Piece{Text{.utf8 = text, .spelling = spelling}}};
	auto result = tokenizer.encode(pieces);
	if (!result) {
		return std::nullopt;
	}
	return std::move(*result);
}

[[nodiscard]] auto decode(Tokenizer const& tokenizer, std::span<Id const> ids) -> std::optional<std::string> {
	auto result = tokenizer.decode(ids);
	if (!result) {
		return std::nullopt;
	}
	return std::move(*result);
}

/** The utf-8 encoding of a scalar value, written from the definition: a second implementation to check against. */
[[nodiscard]] auto utf8_of(char32_t point) -> std::string {
	auto const byte = [](std::uint32_t bits) {
		return std::bit_cast<char>(engine::core::narrow<std::uint8_t>(bits));
	};
	auto const value = std::uint32_t{point};
	if (value < 0x80) {
		return {byte(value)};
	}
	if (value < 0x800) {
		return {byte(0xc0U | (value >> 6U)), byte(0x80U | (value & 0x3fU))};
	}
	if (value < 0x10000) {
		return {byte(0xe0U | (value >> 12U)), byte(0x80U | ((value >> 6U) & 0x3fU)), byte(0x80U | (value & 0x3fU))};
	}
	return {byte(0xf0U | (value >> 18U)), byte(0x80U | ((value >> 12U) & 0x3fU)), byte(0x80U | ((value >> 6U) & 0x3fU)),
	        byte(0x80U | (value & 0x3fU))};
}

/** Whether text is utf-8, judged by the encoder's refusal (which the two-byte and scalar cases pin down). */
[[nodiscard]] auto is_utf8(Tokenizer const& tokenizer, std::string_view text) -> bool {
	auto const pieces = std::array{Piece{Text{.utf8 = text, .spelling = Spelling::Characters}}};
	return tokenizer.encode(pieces).has_value();
}

[[nodiscard]] auto show(std::string_view text) -> std::string {
	auto out = std::string{};
	for (auto const letter : text.substr(0, 100)) {
		auto const byte = std::bit_cast<std::uint8_t>(letter);
		out += byte >= 0x20 && byte < 0x7f ? std::string(1, letter) : std::format("\\x{:02x}", byte);
	}
	return out;
}

[[nodiscard]] auto encodes_every_fixture_exactly(Tokenizer const& tokenizer, std::span<engine_test_tokenizer::EncodeCase const> cases)
    -> bool {
	auto failures = std::size_t{0};
	for (auto const& sample : cases) {
		if (encode(tokenizer, sample.text, sample.spelling) != sample.ids && ++failures <= 5) {
			std::println("  encode mismatch ({}): \"{}\"", sample.spelling == Spelling::Token ? "token" : "characters", show(sample.text));
		}
	}
	return !cases.empty() && failures == 0;
}

[[nodiscard]] auto decodes_every_fixture_exactly(Tokenizer const& tokenizer, std::span<engine_test_tokenizer::DecodeCase const> cases)
    -> bool {
	auto failures = std::size_t{0};
	for (auto const& sample : cases) {
		if (decode(tokenizer, sample.ids) != sample.text && ++failures <= 5) {
			std::println("  decode mismatch: want \"{}\"", show(sample.text));
		}
	}
	return !cases.empty() && failures == 0;
}

// The streamed pieces are each utf-8 and concatenate to the whole decode, for every split of the reference's id
// sequences into tokens (each case is streamed one token at a time).
[[nodiscard]] auto streams_every_fixture(Tokenizer const& tokenizer, std::span<engine_test_tokenizer::DecodeCase const> cases) -> bool {
	auto holds = !cases.empty();
	for (auto const& sample : cases) {
		auto stream = tok::Detokenizer{};
		auto text = std::string{};
		for (auto const id : sample.ids) {
			auto const piece = stream.push(tokenizer, id);
			holds = holds && piece && is_utf8(tokenizer, *piece);
			text += piece.value_or("");
		}
		text += stream.finish();
		holds = holds && text == sample.text;
	}
	return holds;
}

/** The characters that stress each stage: contractions and their folds, every kind of space, numbers, marks,
 * normalization pairs, hangul, emoji, and added-token spellings and fragments. */
constexpr auto alphabet = std::array<std::string_view, 42>{
    "a",  "Z",  "'",   "s",      "S",  "ſ",          "re",  "LL", " ",  "  ",      "\t",         "\n",    "\r\n", "\r",
    " ",  "　", " ", "\u0085", "1",  "٣",          "½",   "!",  "((", "́",        "̖",           "é",     "é",    "Å",
    "ᄀ", "ᅡ",  "ᆨ",   "가",     "中", "\U0001f600", "‍", "א",  "ุ",   "<think>", "<|im_end|>", "<|im_", "�",    "Ᲊ",
};

[[nodiscard]] auto random_text(std::mt19937_64& random) -> std::string {
	auto out = std::string{};
	auto const length = random() % 24;
	for (auto index = std::uint64_t{0}; index < length; ++index) {
		out.append(alphabet.at(random() % alphabet.size()));
	}
	return out;
}

// Decoding what was encoded gives the text back up to normalization, so encoding that again changes nothing.
[[nodiscard]] auto round_trips(Tokenizer const& tokenizer, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 3000; ++trial) {
		auto const text = random_text(random);
		for (auto const spelling : {Spelling::Token, Spelling::Characters}) {
			auto const ids = encode(tokenizer, text, spelling);
			auto const back = ids ? decode(tokenizer, *ids) : std::nullopt;
			auto const again = back ? encode(tokenizer, *back, spelling) : std::nullopt;
			holds = holds && ids && again == ids;
		}
	}
	return holds;
}

// Every token's bytes decode back to the token itself or to tokens that re-encode to the same bytes.
[[nodiscard]] auto fixtures_round_trip(Tokenizer const& tokenizer, std::span<engine_test_tokenizer::EncodeCase const> cases) -> bool {
	auto holds = !cases.empty();
	for (auto const& sample : cases) {
		auto const back = decode(tokenizer, sample.ids);
		holds = holds && back && encode(tokenizer, *back, sample.spelling) == sample.ids;
	}
	return holds;
}

// Characters spelling never yields an added token's id; token spelling yields exactly the spelling present.
[[nodiscard]] auto spelling_decides_added_tokens(Tokenizer const& tokenizer) -> bool {
	auto holds = true;
	for (auto const& added : tok::added_tokens) {
		auto const text = std::format("a{}b", added.spelling);
		auto const as_token = encode(tokenizer, text, Spelling::Token);
		auto const as_characters = encode(tokenizer, text, Spelling::Characters);
		holds = holds && as_token && std::ranges::count(*as_token, tok::id(added.token)) == 1;
		holds = holds && as_characters && std::ranges::none_of(*as_characters, [](Id id) {
			        return id >= tok::bpe_vocabulary;
		        });
	}
	return holds;
}

// A placed added token equals its spelling read as a token, and adjacent text pieces read as one run of text.
[[nodiscard]] auto pieces_compose_like_one_text(Tokenizer const& tokenizer, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto trial = 0; trial < 1000; ++trial) {
		auto const first = random_text(random);
		auto const second = random_text(random);
		auto const content = first + second;
		auto const pieces = std::array{
		    Piece{Added::im_start},
		    Piece{Text{.utf8 = "user\n", .spelling = Spelling::Characters}},
		    Piece{Text{.utf8 = first, .spelling = Spelling::Characters}},
		    Piece{Text{.utf8 = second, .spelling = Spelling::Characters}},
		    Piece{Added::im_end},
		};
		auto const placed = tokenizer.encode(pieces);
		auto const inner = encode(tokenizer, std::format("user\n{}", content), Spelling::Characters);
		if (!placed || !inner) {
			return false;
		}
		auto expected = std::vector<Id>{tok::id(Added::im_start)};
		expected.insert(expected.end(), inner->begin(), inner->end());
		expected.push_back(tok::id(Added::im_end));
		holds = holds && *placed == expected;
		// With no spelling inside the content, reading the rendered string as markup agrees.
		auto const rendered = encode(tokenizer, std::format("<|im_start|>user\n{}<|im_end|>", content), Spelling::Token);
		holds = holds && (content.contains('<') || rendered == expected);
	}
	return holds;
}

/** Every scalar value's utf-8 is accepted, and encodes to ids that decode and re-encode to themselves. */
[[nodiscard]] auto every_scalar_value_round_trips(Tokenizer const& tokenizer) -> bool {
	auto holds = true;
	for (auto point = char32_t{0}; point < 0x110000; ++point) {
		if (point >= 0xd800 && point < 0xe000) {
			continue;
		}
		auto const bytes = utf8_of(point);
		auto const ids = encode(tokenizer, bytes, Spelling::Characters);
		auto const back = ids ? decode(tokenizer, *ids) : std::nullopt;
		auto const again = back ? encode(tokenizer, *back, Spelling::Characters) : std::nullopt;
		holds = holds && ids && again == ids;
	}
	return holds;
}

[[nodiscard]] auto as_text(std::initializer_list<std::uint8_t> bytes) -> std::string {
	auto out = std::string{};
	for (auto const byte : bytes) {
		out.push_back(std::bit_cast<char>(byte));
	}
	return out;
}

/** The valid two-byte strings are exactly two ascii bytes or the two-byte encodings of U+0080..U+07FF. */
[[nodiscard]] auto every_two_byte_string_is_judged(Tokenizer const& tokenizer) -> bool {
	auto valid = std::set<std::string>{};
	for (auto point = char32_t{0x80}; point < 0x800; ++point) {
		valid.insert(utf8_of(point));
	}
	auto holds = true;
	for (auto first = 0U; first < 256U; ++first) {
		for (auto second = 0U; second < 256U; ++second) {
			auto const text = as_text({engine::core::narrow<std::uint8_t>(first), engine::core::narrow<std::uint8_t>(second)});
			auto const expected = (first < 0x80 && second < 0x80) || valid.contains(text);
			holds = holds && encode(tokenizer, text, Spelling::Characters).has_value() == expected;
		}
	}
	return holds;
}

/** The single-byte token of each byte, found through the public surface. */
[[nodiscard]] auto byte_tokens(Tokenizer const& tokenizer) -> std::array<Id, 256> {
	auto out = std::array<Id, 256>{};
	for (auto id = Id{0}; id < tok::bpe_vocabulary; ++id) {
		auto const bytes = tokenizer.token(id);
		if (bytes && bytes->size() == 1) {
			out.at(std::bit_cast<std::uint8_t>(bytes->front())) = id;
		}
	}
	return out;
}

/** Leads, continuations and the bytes at the edges of utf-8's excluded ranges. */
constexpr auto edge_bytes = std::array<std::uint8_t, 18>{0x00, 0x41, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf,
                                                         0xc0, 0xc2, 0xdf, 0xe0, 0xed, 0xef, 0xf0, 0xf4, 0xf5};

// Random byte strings, streamed one byte token at a time: encoding accepts a string exactly when the stream passes
// it through unrepaired, the stream's pieces are always utf-8, and they concatenate to the whole decode.
[[nodiscard]] auto invalid_utf8_is_refused_and_repaired(Tokenizer const& tokenizer, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const singles = byte_tokens(tokenizer);
	auto holds = true;
	for (auto trial = 0; trial < 20000; ++trial) {
		auto bytes = std::string{};
		auto ids = std::vector<Id>{};
		auto const length = random() % 9;
		for (auto index = std::uint64_t{0}; index < length; ++index) {
			auto const byte =
			    random() % 2 == 0 ? edge_bytes.at(random() % edge_bytes.size()) : engine::core::narrow<std::uint8_t>(random() % 256);
			bytes.push_back(std::bit_cast<char>(byte));
			ids.push_back(singles.at(byte));
		}
		auto stream = tok::Detokenizer{};
		auto repaired = std::string{};
		for (auto const id : ids) {
			auto const piece = stream.push(tokenizer, id);
			holds = holds && piece && is_utf8(tokenizer, *piece);
			repaired += piece.value_or("");
		}
		repaired += stream.finish();
		holds = holds && repaired == decode(tokenizer, ids);
		holds = holds && encode(tokenizer, bytes, Spelling::Characters).has_value() == (repaired == bytes);
	}
	return holds;
}

[[nodiscard]] auto refusal_names_the_piece_and_offset(Tokenizer const& tokenizer) -> bool {
	auto const broken = std::string{"ok<think>x"} + as_text({0xe2, 0x28});
	auto const pieces =
	    std::array{Piece{Text{.utf8 = "fine", .spelling = Spelling::Token}}, Piece{Text{.utf8 = broken, .spelling = Spelling::Token}}};
	auto const result = tokenizer.encode(pieces);
	return !result && result.error() == tok::EncodeError{.piece = 1, .offset = 10} &&
	       result.error().retry() == engine::core::Retry::Permanent;
}

[[nodiscard]] auto ids_past_the_vocabulary_are_refused(Tokenizer const& tokenizer) -> bool {
	auto holds = true;
	for (auto const id : {tok::vocabulary, tok::vocabulary + 1, Id{248319}, std::numeric_limits<Id>::max()}) {
		auto stream = tok::Detokenizer{};
		auto const ids = std::array{Id{0}, id};
		holds = holds && !tokenizer.token(id) && tokenizer.decode(ids).error() == tok::DecodeError{.id = id};
		holds = holds && stream.push(tokenizer, id).error() == tok::DecodeError{.id = id};
	}
	return holds;
}

[[nodiscard]] auto added_tokens_are_named_by_id() -> bool {
	auto holds = true;
	for (auto id = Id{0}; id < tok::vocabulary + 4; ++id) {
		auto const named = tok::added(id);
		holds = holds && named.has_value() == (id >= tok::bpe_vocabulary && id < tok::vocabulary);
		holds = holds && (!named || tok::id(*named) == id);
	}
	auto spellings = std::set<std::string_view>{};
	for (auto const& added : tok::added_tokens) {
		spellings.insert(added.spelling);
	}
	return holds && spellings.size() == tok::added_tokens.size();
}

// Pathological inputs: long runs of one kind of character. A quadratic merge or split would not finish inside the
// test's deadline; each must also round trip.
[[nodiscard]] auto long_runs_encode_and_round_trip(Tokenizer const& tokenizer) -> bool {
	auto holds = true;
	for (auto const [unit, count] : std::array{std::pair{std::string_view{"a"}, 400000}, std::pair{std::string_view{" "}, 400000},
	                                           std::pair{std::string_view{"\n"}, 200000}, std::pair{std::string_view{"7"}, 200000},
	                                           std::pair{std::string_view{"!"}, 400000}, std::pair{std::string_view{"ab "}, 100000},
	                                           std::pair{std::string_view{"́"}, 100000}, std::pair{std::string_view{"é"}, 100000},
	                                           std::pair{std::string_view{"\U0001f600"}, 100000}}) {
		auto text = std::string{};
		for (auto index = 0; index < count; ++index) {
			text.append(unit);
		}
		auto const ids = encode(tokenizer, text, Spelling::Characters);
		auto const back = ids ? decode(tokenizer, *ids) : std::nullopt;
		auto const again = back ? encode(tokenizer, *back, Spelling::Characters) : std::nullopt;
		holds = holds && ids && again == ids;
	}
	return holds;
}

template<class Mutate>
[[nodiscard]] auto refuses(std::span<std::byte const> image, tok::Defect defect, Mutate mutate) -> bool {
	auto bytes = std::vector<std::byte>{image.begin(), image.end()};
	mutate(bytes);
	auto const result = Tokenizer::parse(bytes);
	return !result && result.error() == tok::LoadError{.defect = defect} && result.error().retry() == engine::core::Retry::Permanent;
}

auto put(std::vector<std::byte>& bytes, std::size_t at, std::uint32_t value) -> void {
	for (auto shift = 0U; shift < 32U; shift += 8U) {
		bytes[at + shift / 8] = std::byte{engine::core::narrow<std::uint8_t>((value >> shift) & 0xffU)};
	}
}

[[nodiscard]] auto get(std::span<std::byte const> bytes, std::size_t at) -> std::uint32_t {
	auto value = std::uint32_t{0};
	for (auto shift = 0U; shift < 32U; shift += 8U) {
		value |= std::to_integer<std::uint32_t>(bytes[at + shift / 8]) << shift;
	}
	return value;
}

/** Each kind of damage to the image is refused with its own defect. */
[[nodiscard]] auto damaged_images_are_refused(std::span<std::byte const> image) -> bool {
	constexpr auto header = std::size_t{8 + 9 * 4};
	auto const word = [](std::size_t field) {
		return 8 + field * 4;
	};
	auto const token_bytes = get(image, word(2));
	auto const merges_at = header + std::size_t{tok::bpe_vocabulary} * 4 + (std::size_t{token_bytes} + 3) / 4 * 4;
	auto const classes_at = merges_at + tok::merge_count * 12;
	auto holds = true;
	holds = holds && refuses(image, tok::Defect::Size, [](auto& bytes) {
		        bytes.resize(20);
	        });
	holds = holds && refuses(image, tok::Defect::Size, [](auto& bytes) {
		        bytes.pop_back();
	        });
	holds = holds && refuses(image, tok::Defect::Size, [](auto& bytes) {
		        bytes.push_back(std::byte{0});
	        });
	holds = holds && refuses(image, tok::Defect::Magic, [](auto& bytes) {
		        bytes[0] = std::byte{'G'};
	        });
	holds = holds && refuses(image, tok::Defect::Version, [&](auto& bytes) {
		        put(bytes, word(0), 2);
	        });
	holds = holds && refuses(image, tok::Defect::Shape, [&](auto& bytes) {
		        put(bytes, word(1), tok::bpe_vocabulary - 1);
	        });
	holds = holds && refuses(image, tok::Defect::Shape, [&](auto& bytes) {
		        put(bytes, word(3), tok::merge_count + 1);
	        });
	holds = holds && refuses(image, tok::Defect::Bound, [&](auto& bytes) {
		        put(bytes, word(4), 1U << 30U);
	        });
	holds = holds && refuses(image, tok::Defect::Bound, [&](auto& bytes) {
		        put(bytes, word(2), 0xffffffffU);
	        });
	// An empty token: the second token ends where the first does.
	holds = holds && refuses(image, tok::Defect::Token, [&](auto& bytes) {
		        put(bytes, header + 4, get(bytes, header));
	        });
	// A merge whose result is not its pair's concatenation, and one naming an unknown token.
	holds = holds && refuses(image, tok::Defect::Merge, [&](auto& bytes) {
		        put(bytes, merges_at + 8, get(bytes, merges_at + 8) + 1);
	        });
	holds = holds && refuses(image, tok::Defect::Merge, [&](auto& bytes) {
		        put(bytes, merges_at, tok::vocabulary);
	        });
	// A repeated pair: the second merge rule restates the first.
	holds = holds && refuses(image, tok::Defect::Merge, [&](auto& bytes) {
		        for (auto offset = std::size_t{0}; offset < 12; offset += 4) {
			        put(bytes, merges_at + 12 + offset, get(bytes, merges_at + offset));
		        }
	        });
	// Class runs that do not start at codepoint 0, and runs out of order.
	holds = holds && refuses(image, tok::Defect::Runs, [&](auto& bytes) {
		        put(bytes, classes_at, 1U << 3U);
	        });
	holds = holds && refuses(image, tok::Defect::Runs, [&](auto& bytes) {
		        put(bytes, classes_at + 4, get(bytes, classes_at + 8));
	        });
	return holds;
}

}

auto main() -> int {
	auto const tokenizer = tok::load();
	if (!tokenizer) {
		std::println("FAIL: the built-in tokenizer image loads (defect {})", std::to_underlying(tokenizer.error().defect));
		return 1;
	}
	auto const encode_cases = engine_test_tokenizer::encode_cases();
	auto const decode_cases = engine_test_tokenizer::decode_cases();
	auto const cases = std::array{
	    engine::testing::Case{.name = "encodes every reference fixture to exactly its ids, in both spellings",
		                      .holds = encodes_every_fixture_exactly(*tokenizer, encode_cases)},
	    engine::testing::Case{.name = "decodes every reference fixture to exactly its text",
		                      .holds = decodes_every_fixture_exactly(*tokenizer, decode_cases)},
	    engine::testing::Case{.name = "streaming every reference fixture yields utf-8 pieces that add up to its text",
		                      .holds = streams_every_fixture(*tokenizer, decode_cases)},
	    engine::testing::Case{.name = "decoding a fixture's ids and encoding again gives the same ids",
		                      .holds = fixtures_round_trip(*tokenizer, encode_cases)},
	    engine::testing::Case{.name = "encode, decode, encode is stable (seed 1)", .holds = round_trips(*tokenizer, 1)},
	    engine::testing::Case{.name = "encode, decode, encode is stable (seed 2)", .holds = round_trips(*tokenizer, 2)},
	    engine::testing::Case{.name = "the spelling decides whether added-token spellings become added tokens",
		                      .holds = spelling_decides_added_tokens(*tokenizer)},
	    engine::testing::Case{.name = "placed added tokens and split text pieces encode like one rendered text (seed 3)",
		                      .holds = pieces_compose_like_one_text(*tokenizer, 3)},
	    engine::testing::Case{.name = "every scalar value's utf-8 is accepted and round-trips through encode and decode",
		                      .holds = every_scalar_value_round_trips(*tokenizer)},
	    engine::testing::Case{.name = "every two-byte string is accepted exactly when it is utf-8",
		                      .holds = every_two_byte_string_is_judged(*tokenizer)},
	    engine::testing::Case{.name = "random bytes: refused exactly when decoding repairs them; streaming agrees with decode (seed 4)",
		                      .holds = invalid_utf8_is_refused_and_repaired(*tokenizer, 4)},
	    engine::testing::Case{.name = "a refusal names the piece and byte where utf-8 breaks",
		                      .holds = refusal_names_the_piece_and_offset(*tokenizer)},
	    engine::testing::Case{.name = "ids past the vocabulary are refused by decode and by the stream",
		                      .holds = ids_past_the_vocabulary_are_refused(*tokenizer)},
	    engine::testing::Case{.name = "added tokens are named by exactly their ids and spelled uniquely",
		                      .holds = added_tokens_are_named_by_id()},
	    engine::testing::Case{.name = "long runs of one kind of character encode in time and round trip",
		                      .holds = long_runs_encode_and_round_trip(*tokenizer)},
	    engine::testing::Case{.name = "each kind of image damage is refused with its own defect",
		                      .holds = damaged_images_are_refused(engine::tokenizer_image::bytes())},
	};
	return engine::testing::report(cases);
}

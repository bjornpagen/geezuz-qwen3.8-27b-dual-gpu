// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.file;
import engine.testing;
import engine.tokenizer;
import engine.tokenizer_image;

namespace {

namespace tok = engine::tokenizer;
using tok::Id;

// A small strict json reader, enough to read tokenizer.json independently of the converter that produced the image.

struct Value;

struct Member;

struct Null {
	[[nodiscard]] auto operator==(Null const&) const -> bool = default;
};

struct Number {
	std::string text;
};

using Array = std::vector<Value>;

struct Object {
	std::vector<Member> members;

	[[nodiscard]] auto find(std::string_view key) const -> std::optional<Value const&>;
};

struct Value {
	std::variant<Null, bool, Number, std::string, Array, Object> data;
};

struct Member {
	std::string key;
	Value value;
};

auto Object::find(std::string_view key) const -> std::optional<Value const&> {
	auto const found = std::ranges::find(members, key, &Member::key);
	if (found == members.end()) {
		return std::nullopt;
	}
	return found->value;
}

[[nodiscard]] auto utf8_of(std::uint32_t value) -> std::string {
	auto const byte = [](std::uint32_t bits) {
		return std::bit_cast<char>(engine::core::narrow<std::uint8_t>(bits));
	};
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

class Parser {
public:
	explicit Parser(std::string_view text) : text_{text} {}

	[[nodiscard]] auto document() -> std::optional<Value> {
		auto value = parse(0);
		skip_space();
		if (!value || at_ != text_.size()) {
			return std::nullopt;
		}
		return value;
	}

private:
	static constexpr std::size_t max_depth = 64;

	auto skip_space() -> void {
		while (at_ < text_.size() && (text_[at_] == ' ' || text_[at_] == '\n' || text_[at_] == '\r' || text_[at_] == '\t')) {
			++at_;
		}
	}

	[[nodiscard]] auto take(char expected) -> bool {
		skip_space();
		if (at_ < text_.size() && text_[at_] == expected) {
			++at_;
			return true;
		}
		return false;
	}

	[[nodiscard]] auto literal(std::string_view word) -> bool {
		if (text_.substr(at_).starts_with(word)) {
			at_ += word.size();
			return true;
		}
		return false;
	}

	[[nodiscard]] auto hex4() -> std::optional<std::uint32_t> {
		if (at_ + 4 > text_.size()) {
			return std::nullopt;
		}
		auto value = std::uint32_t{0};
		for (auto const digit : text_.substr(at_, 4)) {
			auto const nibble = digit >= '0' && digit <= '9'   ? digit - '0'
			                    : digit >= 'a' && digit <= 'f' ? digit - 'a' + 10
			                    : digit >= 'A' && digit <= 'F' ? digit - 'A' + 10
			                                                   : -1;
			if (nibble < 0) {
				return std::nullopt;
			}
			value = value * 16 + engine::core::narrow<std::uint32_t>(nibble);
		}
		at_ += 4;
		return value;
	}

	[[nodiscard]] auto string() -> std::optional<std::string> {
		if (!take('"')) {
			return std::nullopt;
		}
		auto out = std::string{};
		while (at_ < text_.size()) {
			auto const letter = text_[at_++];
			if (letter == '"') {
				return out;
			}
			if (letter != '\\') {
				out.push_back(letter);
				continue;
			}
			if (at_ == text_.size()) {
				return std::nullopt;
			}
			auto const escape = text_[at_++];
			switch (escape) {
			case '"':
			case '\\':
			case '/':
				out.push_back(escape);
				break;
			case 'b':
				out.push_back('\b');
				break;
			case 'f':
				out.push_back('\f');
				break;
			case 'n':
				out.push_back('\n');
				break;
			case 'r':
				out.push_back('\r');
				break;
			case 't':
				out.push_back('\t');
				break;
			case 'u': {
				auto unit = hex4();
				if (!unit) {
					return std::nullopt;
				}
				// A high surrogate escape is followed by its low half; together they name one scalar value.
				if (*unit >= 0xd800 && *unit < 0xdc00) {
					if (!literal("\\u")) {
						return std::nullopt;
					}
					auto const low = hex4();
					if (!low || *low < 0xdc00 || *low >= 0xe000) {
						return std::nullopt;
					}
					unit = 0x10000 + ((*unit - 0xd800) << 10U) + (*low - 0xdc00);
				}
				out += utf8_of(*unit);
				break;
			}
			default:
				return std::nullopt;
			}
		}
		return std::nullopt;
	}

	[[nodiscard]] auto parse(std::size_t depth) -> std::optional<Value> {
		skip_space();
		if (depth > max_depth || at_ == text_.size()) {
			return std::nullopt;
		}
		auto const first = text_[at_];
		if (first == '"') {
			auto text = string();
			return text ? std::optional{Value{std::move(*text)}} : std::nullopt;
		}
		if (first == '{') {
			++at_;
			auto object = Object{};
			if (take('}')) {
				return Value{std::move(object)};
			}
			do {
				auto key = string();
				if (!key || !take(':')) {
					return std::nullopt;
				}
				auto value = parse(depth + 1);
				if (!value) {
					return std::nullopt;
				}
				object.members.push_back(Member{.key = std::move(*key), .value = std::move(*value)});
			} while (take(','));
			return take('}') ? std::optional{Value{std::move(object)}} : std::nullopt;
		}
		if (first == '[') {
			++at_;
			auto array = Array{};
			if (take(']')) {
				return Value{std::move(array)};
			}
			do {
				auto value = parse(depth + 1);
				if (!value) {
					return std::nullopt;
				}
				array.push_back(std::move(*value));
			} while (take(','));
			return take(']') ? std::optional{Value{std::move(array)}} : std::nullopt;
		}
		if (literal("null")) {
			return Value{Null{}};
		}
		if (literal("true")) {
			return Value{true};
		}
		if (literal("false")) {
			return Value{false};
		}
		auto const begin = at_;
		while (at_ < text_.size() && std::string_view{"0123456789+-.eE"}.contains(text_[at_])) {
			++at_;
		}
		if (at_ == begin) {
			return std::nullopt;
		}
		return Value{Number{.text = std::string{text_.substr(begin, at_ - begin)}}};
	}

	std::string_view text_;
	std::size_t at_ = 0;
};

[[nodiscard]] auto object(Value const& value) -> std::optional<Object const&> {
	if (!std::holds_alternative<Object>(value.data)) {
		return std::nullopt;
	}
	return std::get<Object>(value.data);
}

/** The value at a path of object keys. */
[[nodiscard]] auto at(Value const& root, std::initializer_list<std::string_view> path) -> std::optional<Value const&> {
	auto current = std::optional<Value const&>{root};
	for (auto const key : path) {
		auto const holder = object(*current);
		if (!holder) {
			return std::nullopt;
		}
		current = holder->find(key);
		if (!current) {
			return std::nullopt;
		}
	}
	return current;
}

[[nodiscard]] auto is_null(std::optional<Value const&> value) -> bool {
	return value && std::holds_alternative<Null>(value->data);
}

[[nodiscard]] auto text(std::optional<Value const&> value) -> std::optional<std::string_view> {
	if (!value || !std::holds_alternative<std::string>(value->data)) {
		return std::nullopt;
	}
	return std::get<std::string>(value->data);
}

[[nodiscard]] auto flag(std::optional<Value const&> value) -> std::optional<bool> {
	if (!value || !std::holds_alternative<bool>(value->data)) {
		return std::nullopt;
	}
	return std::get<bool>(value->data);
}

[[nodiscard]] auto number(std::optional<Value const&> value) -> std::optional<std::uint64_t> {
	if (!value || !std::holds_alternative<Number>(value->data)) {
		return std::nullopt;
	}
	auto const& digits = std::get<Number>(value->data).text;
	if (digits.empty() || digits.size() > 18 || !std::ranges::all_of(digits, [](char digit) {
		    return digit >= '0' && digit <= '9';
	    })) {
		return std::nullopt;
	}
	auto out = std::uint64_t{0};
	for (auto const digit : digits) {
		out = out * 10 + engine::core::narrow<std::uint64_t>(digit - '0');
	}
	return out;
}

/** The byte-level alphabet: the character each byte is written as inside tokenizer.json's vocabulary and merges. */
[[nodiscard]] auto byte_of_character() -> std::map<std::uint32_t, std::uint8_t> {
	auto printable = std::vector<std::uint32_t>{};
	for (auto const [low, high] : {std::pair{0x21U, 0x7eU}, std::pair{0xa1U, 0xacU}, std::pair{0xaeU, 0xffU}}) {
		for (auto byte = low; byte <= high; ++byte) {
			printable.push_back(byte);
		}
	}
	auto out = std::map<std::uint32_t, std::uint8_t>{};
	auto shifted = std::uint32_t{0};
	for (auto byte = 0U; byte < 256U; ++byte) {
		auto const kept = std::ranges::contains(printable, byte);
		out.emplace(kept ? byte : 256 + shifted++, engine::core::narrow<std::uint8_t>(byte));
	}
	return out;
}

/** A vocabulary entry's bytes: its characters, each mapped back through the byte-level alphabet. */
[[nodiscard]] auto token_bytes(std::string_view entry, std::map<std::uint32_t, std::uint8_t> const& alphabet)
    -> std::optional<std::string> {
	auto out = std::string{};
	auto at = std::size_t{0};
	while (at < entry.size()) {
		auto const lead = std::bit_cast<std::uint8_t>(entry[at]);
		auto const length = lead < 0x80 ? 1U : lead < 0xe0 ? 2U : lead < 0xf0 ? 3U : 4U;
		auto value = length == 1 ? std::uint32_t{lead} : std::uint32_t{lead} & (0x7fU >> length);
		for (auto index = 1U; index < length && at + index < entry.size(); ++index) {
			value = (value << 6U) | (std::uint32_t{std::bit_cast<std::uint8_t>(entry[at + index])} & 0x3fU);
		}
		auto const found = alphabet.find(value);
		if (found == alphabet.end()) {
			return std::nullopt;
		}
		out.push_back(std::bit_cast<char>(found->second));
		at += length;
	}
	return out;
}

[[nodiscard]] auto le32(std::span<std::byte const> bytes, std::size_t at) -> std::uint32_t {
	auto value = std::uint32_t{0};
	for (auto shift = 0U; shift < 32U; shift += 8U) {
		value |= std::to_integer<std::uint32_t>(bytes[at + shift / 8]) << shift;
	}
	return value;
}

struct Checkpoint {
	Value json;
	std::map<std::string, Id, std::less<>> vocabulary;
};

[[nodiscard]] auto no_baked_truncation_or_padding(Value const& json) -> bool {
	return is_null(at(json, {"truncation"})) && is_null(at(json, {"padding"}));
}

// The pipeline the hand-written stages implement: any other normalizer, pattern or byte-level option fails here.
[[nodiscard]] auto pipeline_is_the_one_implemented(Value const& json) -> bool {
	auto const byte_level = [&](std::optional<Value const&> stage) {
		return stage && text(at(*stage, {"type"})) == "ByteLevel" && flag(at(*stage, {"add_prefix_space"})) == false &&
		       flag(at(*stage, {"trim_offsets"})) == false && flag(at(*stage, {"use_regex"})) == false;
	};
	auto const stages = at(json, {"pre_tokenizer", "pretokenizers"});
	if (!stages || !std::holds_alternative<Array>(stages->data) || std::get<Array>(stages->data).size() != 2) {
		return false;
	}
	auto const& split = std::get<Array>(stages->data)[0];
	auto const model = at(json, {"model"});
	if (!model) {
		return false;
	}
	return text(at(json, {"normalizer", "type"})) == "NFC" && text(at(json, {"pre_tokenizer", "type"})) == "Sequence" &&
	       text(at(split, {"type"})) == "Split" && text(at(split, {"pattern", "Regex"})) == tok::pattern &&
	       text(at(split, {"behavior"})) == "Isolated" && flag(at(split, {"invert"})) == false &&
	       byte_level(std::get<Array>(stages->data)[1]) && byte_level(at(json, {"post_processor"})) && byte_level(at(json, {"decoder"})) &&
	       text(at(*model, {"type"})) == "BPE" && is_null(at(*model, {"dropout"})) && is_null(at(*model, {"unk_token"})) &&
	       text(at(*model, {"continuing_subword_prefix"})) == "" && text(at(*model, {"end_of_word_suffix"})) == "" &&
	       flag(at(*model, {"fuse_unk"})) == false && flag(at(*model, {"byte_fallback"})) == false &&
	       flag(at(*model, {"ignore_merges"})) == false;
}

[[nodiscard]] auto vocabulary_matches(Checkpoint const& checkpoint, tok::Tokenizer const& tokenizer) -> bool {
	auto const alphabet = byte_of_character();
	auto holds = checkpoint.vocabulary.size() == tok::bpe_vocabulary;
	auto seen = std::vector<bool>(tok::bpe_vocabulary);
	for (auto const& [entry, id] : checkpoint.vocabulary) {
		holds = holds && id < tok::bpe_vocabulary && token_bytes(entry, alphabet) == tokenizer.token(id);
		if (id < tok::bpe_vocabulary) {
			seen[id] = true;
		}
	}
	return holds && std::ranges::all_of(seen, std::identity{});
}

// The image's merge section holds (left, right, result) per rank; ranks decide encoding, so the order is checked too.
[[nodiscard]] auto merges_match(Checkpoint const& checkpoint) -> bool {
	auto const merges = at(checkpoint.json, {"model", "merges"});
	if (!merges || !std::holds_alternative<Array>(merges->data) || std::get<Array>(merges->data).size() != tok::merge_count) {
		return false;
	}
	auto const image = engine::tokenizer_image::bytes();
	auto const token_bytes_size = le32(image, 8 + 2 * 4);
	auto const section = std::size_t{8 + 9 * 4} + std::size_t{tok::bpe_vocabulary} * 4 + (std::size_t{token_bytes_size} + 3) / 4 * 4;
	auto const lookup = [&](std::string_view entry) -> std::optional<Id> {
		auto const found = checkpoint.vocabulary.find(entry);
		return found == checkpoint.vocabulary.end() ? std::nullopt : std::optional{found->second};
	};
	auto holds = true;
	for (auto const [rank, merge] : std::views::enumerate(std::get<Array>(merges->data))) {
		auto left = std::string{};
		auto right = std::string{};
		if (auto const spelled = text(merge)) {
			auto const space = spelled->find(' ');
			left = spelled->substr(0, space);
			right = space == std::string_view::npos ? std::string{} : std::string{spelled->substr(space + 1)};
		} else if (std::holds_alternative<Array>(merge.data) && std::get<Array>(merge.data).size() == 2) {
			left = text(std::get<Array>(merge.data)[0]).value_or("");
			right = text(std::get<Array>(merge.data)[1]).value_or("");
		}
		auto const at_rank = section + static_cast<std::size_t>(rank) * 12;
		holds = holds && lookup(left) == le32(image, at_rank) && lookup(right) == le32(image, at_rank + 4) &&
		        lookup(left + right) == le32(image, at_rank + 8);
	}
	return holds;
}

[[nodiscard]] auto added_tokens_match(Value const& json) -> bool {
	auto const listed = at(json, {"added_tokens"});
	if (!listed || !std::holds_alternative<Array>(listed->data)) {
		return false;
	}
	auto const& entries = std::get<Array>(listed->data);
	auto holds = entries.size() == tok::added_tokens.size();
	for (auto index = std::size_t{0}; holds && index < entries.size(); ++index) {
		auto const& entry = entries[index];
		auto const& expected = tok::added_tokens.at(index);
		holds = number(at(entry, {"id"})) == tok::id(expected.token) && text(at(entry, {"content"})) == expected.spelling &&
		        flag(at(entry, {"special"})) == (expected.role == tok::Role::Special) && flag(at(entry, {"single_word"})) == false &&
		        flag(at(entry, {"lstrip"})) == false && flag(at(entry, {"rstrip"})) == false && flag(at(entry, {"normalized"})) == false;
	}
	return holds;
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 2) {
		std::println("FAIL: usage: <checkpoint directory>");
		return 1;
	}
	auto const mapping = engine::file::Mapping::open(std::filesystem::path{arguments[1]} / "tokenizer.json");
	auto const tokenizer = tok::load();
	if (!mapping || !tokenizer) {
		std::println("FAIL: tokenizer.json maps and the built-in image loads");
		return 1;
	}
	auto const bytes = mapping->bytes();
	auto source = std::string{};
	source.reserve(bytes.size());
	for (auto const byte : bytes) {
		source.push_back(std::bit_cast<char>(byte));
	}
	auto json = Parser{source}.document();
	if (!json || !object(*json)) {
		std::println("FAIL: tokenizer.json is a json object");
		return 1;
	}
	auto checkpoint = Checkpoint{.json = std::move(*json), .vocabulary = {}};
	auto const vocabulary = at(checkpoint.json, {"model", "vocab"});
	if (auto const entries = vocabulary ? object(*vocabulary) : std::nullopt) {
		for (auto const& member : entries->members) {
			checkpoint.vocabulary.emplace(member.key, engine::core::narrow<Id>(number(member.value).value_or(tok::vocabulary)));
		}
	}
	auto const cases = std::array{
	    engine::testing::Case{.name = "tokenizer.json bakes in no truncation and no padding",
		                      .holds = no_baked_truncation_or_padding(checkpoint.json)},
	    engine::testing::Case{.name = "tokenizer.json's normalizer, pattern, byte-level stages and bpe options are the ones implemented",
		                      .holds = pipeline_is_the_one_implemented(checkpoint.json)},
	    engine::testing::Case{.name = "every vocabulary entry's bytes are the image's bytes for its id",
		                      .holds = vocabulary_matches(checkpoint, *tokenizer)},
	    engine::testing::Case{.name = "every merge, by rank, is the image's merge at that rank", .holds = merges_match(checkpoint)},
	    engine::testing::Case{.name = "the added tokens are the compiled table: ids, spellings, roles, no strip or normalization",
		                      .holds = added_tokens_match(checkpoint.json)},
	};
	return engine::testing::report(cases);
}
